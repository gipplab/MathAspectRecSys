"""
Train AchGNN (Secs. 3.2 and 4.2) and evaluate it on GoldRiM and the SilverRiM test split.

    positives   SilverRiM training pairs (75% of the seeds; --fold 0..4 selects one of five random splits)
    negatives   per positive, one hard negative (nearest candidate to the seed by f_text cosine similarity that is
                not one of its recommendations) and --easy-negatives random candidates resampled every step
    model       heterogeneous GraphSAGE (L=2, 15 paper / 5 author neighbours), aspect-conditioned scorer and
                aspect classifier; L = L_rank (BPR) + lambda * L_aspect, lambda = 0.2

Single process (CPU or one GPU):
    python achgnn/train.py --work work --run-name achgnn_fold0 --fold 0

All GPUs of a node (DistributedDataParallel):
    torchrun --standalone --nproc_per_node=4 achgnn/train.py --work work --run-name achgnn_fold0 --fold 0 --amp
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from common import ASPECTS, Work, log, save_json, setup_logging
from distributed import Context
from engine import (evaluate, infer_all_papers, load_pairs, log_metrics, make_fanouts, mine_distributed, prefetch,
                    sample_batch, to_device)
from graph import HeteroGraph, open_features
from model import AchGNN


def build_triplets(ctx, train, candidates, feats, cache_path, random_neg=False, seed=0):
    """
    Arrays (query, positive, hard negative, aspect) for all training pairs. The i-th positive of a seed is paired
    with its i-th most similar candidate by f_text cosine similarity, excluding the seed and its training
    recommendations. Hard negatives are mined once (split over all ranks) and cached.
    """
    q, p, a = (train[c].values.astype(np.int64) for c in ("seed_idx", "rec_idx", "aspect_id"))
    positives = train.groupby("seed_idx")["rec_idx"].apply(set).to_dict()
    rng = np.random.default_rng(seed)

    def random_negative(s):
        while True:
            c = int(rng.choice(candidates))
            if c != s and c not in positives[s]:
                return c

    if random_neg:
        return q, p, np.array([random_negative(s) for s in q.tolist()], dtype=np.int64), a
    if os.path.exists(cache_path):
        n = np.load(cache_path)
        if len(n) == len(q):
            log.info("using cached hard negatives %s", cache_path)
            return q, p, n, a
    seeds, counts = np.unique(q, return_counts=True)
    log.info("mining hard negatives for %d seeds over %d candidates", len(seeds), len(candidates))
    negs = mine_distributed(ctx, feats, seeds, dict(zip(seeds.tolist(), counts.tolist())), positives, candidates)
    taken = {s: 0 for s in seeds.tolist()}
    n = np.empty(len(q), dtype=np.int64)
    for i, s in enumerate(q.tolist()):
        lst = negs[s]
        n[i] = lst[taken[s]] if taken[s] < len(lst) else random_negative(s)
        taken[s] += 1
    if ctx.is_main:
        np.save(cache_path, n)
    ctx.barrier()
    return q, p, n, a


def batches(q, p, n, a, order, batch_size, easy, candidates, graph, feats, fanouts, num_layers, rng, pin):
    """Yield ready-to-run training batches (sampled blocks and gathered features on the CPU)."""
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        m = len(idx)
        negs = np.concatenate([n[idx]] + [rng.choice(candidates, m) for _ in range(easy)])
        nodes, inv = np.unique(np.concatenate([q[idx], p[idx], negs]), return_inverse=True)
        h0, blocks = sample_batch(graph, feats, nodes, fanouts, num_layers, rng, pin)
        yield h0, blocks, torch.from_numpy(inv.astype(np.int64)), m, torch.from_numpy(a[idx].astype(np.int64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--run-name", default="achgnn", help="name of the run directory under <work>/runs")
    ap.add_argument("--fold", type=int, default=0, help="which of the five random 75/25 SilverRiM splits (0-4)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N (single process)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512, help="training pairs per step and process")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--aspect-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2, help="GNN depth L")
    ap.add_argument("--paper-neighbors", type=int, default=15)
    ap.add_argument("--author-neighbors", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.2, help="weight lambda of the aspect loss (Eq. 10)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--final-activation", type=int, default=1, help="apply sigma after the last GNN layer (Eq. 2)")
    ap.add_argument("--normalize", type=int, default=0, help="L2-normalise the final paper embeddings")
    ap.add_argument("--random-negatives", action="store_true", help="random instead of hard negatives")
    ap.add_argument("--easy-negatives", type=int, default=1, help="random negatives per positive and step")
    ap.add_argument("--drop-aspects", nargs="*", default=[], help="ablation: remove supervision of these aspects")
    ap.add_argument("--no-authors", action="store_true", help="ablation: remove authorship edges")
    ap.add_argument("--random-authors", action="store_true", help="ablation: randomise authorship edges")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    ap.add_argument("--features-on-gpu", choices=["auto", "on", "off"], default="auto",
                    help="keep all node features in GPU memory (auto: if they use < 60%% of it)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimisation steps")
    ap.add_argument("--eval-k", type=int, default=10)
    ap.add_argument("--eval-pool", type=int, default=200,
                    help="candidates re-ranked by the aspect scorer per query (0 = all candidates)")
    ap.add_argument("--pool-source", choices=["text", "gnn"], default="text",
                    help="origin of the re-ranked candidates: f_text cosine or AchGNN cosine neighbours")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint of this run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging()
    ctx = Context(args.device)
    device = ctx.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    work = Work(args.work)
    if ctx.is_main:
        log.info("device %s, %d process(es)", device, ctx.world)

    graph = HeteroGraph(work, drop_authorship=args.no_authors, random_authorship=args.random_authors, seed=args.seed)
    feats = open_features(work, device, args.features_on_gpu, log if ctx.is_main else None)
    candidates = np.load(work.candidates)
    pairs = load_pairs(work, args.fold)
    train = pairs[pairs["split"] == "train"]
    for name in args.drop_aspects:
        if name not in ASPECTS:
            raise SystemExit(f"unknown aspect {name!r}; choose from {ASPECTS}")
        train = train[train["aspect"] != name]
    if ctx.is_main:
        log.info("pairs: %s", pairs["split"].value_counts().to_dict())
    if len(train) == 0:
        raise SystemExit("no training pairs")

    q, p, n, a = build_triplets(ctx, train, candidates, feats, work.path(f"hard_negatives_fold{args.fold}.npy"),
                                args.random_negatives, args.seed)
    fanouts = make_fanouts(args.paper_neighbors, args.author_neighbors)

    model = AchGNN(feats.dim, args.hidden_dim, args.layers, len(ASPECTS), args.aspect_dim, args.dropout,
                   bool(args.final_activation), bool(args.normalize)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt_path = work.run(args.run_name, "model.pt")
    start_epoch, step, history = 0, 0, []
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optim_state"])
        start_epoch, step, history = ckpt["epoch"], ckpt["step"], ckpt.get("history", [])
        if ctx.is_main:
            log.info("resumed from epoch %d", start_epoch)
    net = model
    if ctx.enabled:
        net = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)

    if ctx.is_main:
        log.info("training on %d pairs, %d parameters, %d pairs per step", len(q),
                 sum(x.numel() for x in model.parameters()), args.batch_size * ctx.world)
    sample_rng = np.random.default_rng(args.seed + 1000 * (ctx.rank + 1))
    per_rank = len(q) // ctx.world
    done = False
    for epoch in range(start_epoch, args.epochs):
        net.train()
        perm = np.random.default_rng(args.seed + epoch).permutation(len(q))  # identical on every rank
        order = perm[ctx.rank * per_rank:(ctx.rank + 1) * per_rank]
        sums, nb, t0 = np.zeros(3), 0, time.time()
        stream = prefetch(batches(q, p, n, a, order, args.batch_size, args.easy_negatives, candidates, graph, feats,
                                  fanouts, args.layers, sample_rng, device.type == "cuda"))
        for h0, blocks, inv, m, aspect in stream:
            h0, blocks = to_device(h0, blocks, device, feats)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
                loss, l_rank, l_asp = net(h0, blocks, inv.to(device), m, aspect.to(device), args.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums += [loss.item(), l_rank.item(), l_asp.item()]
            nb, step = nb + 1, step + 1
            if ctx.is_main and step % 50 == 0:
                log.info("epoch %d step %d loss %.4f (rank %.4f, aspect %.4f)", epoch + 1, step, *(sums / nb))
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        nb = max(nb, 1)
        history.append({"epoch": epoch + 1, "loss": float(sums[0] / nb), "rank_loss": float(sums[1] / nb),
                        "aspect_loss": float(sums[2] / nb), "seconds": time.time() - t0})
        if ctx.is_main:
            log.info("epoch %d done: %s", epoch + 1, history[-1])
            torch.save({"model_state": model.state_dict(), "optim_state": opt.state_dict(),
                        "model_config": model.config, "epoch": epoch + 1, "step": step, "history": history,
                        "args": vars(args), "aspects": ASPECTS,
                        "fanouts": [args.paper_neighbors, args.author_neighbors]}, ckpt_path)
        ctx.barrier()
        if done:
            break
    if ctx.is_main:
        save_json(history, work.run(args.run_name, "history.json"))
        log.info("model saved to %s", ckpt_path)

    if not args.no_eval:
        run_evaluation(ctx, model, graph, feats, candidates, pairs, fanouts, work, args)
    ctx.close()


def run_evaluation(ctx, model, graph, feats, candidates, pairs, fanouts, work, args):
    H = infer_all_papers(model, graph, feats, fanouts, ctx, os.path.dirname(work.run(args.run_name, "x")),
                         amp=args.amp)
    if ctx.is_main:
        torch.save(H, work.run(args.run_name, "paper_embeddings.pt"))
    results = {}
    for split, name in [("gold", "GoldRiM"), ("test", "SilverRiM-test")]:
        sub = pairs[pairs["split"] == split]
        if len(sub) == 0:
            continue
        metrics, retrieved = evaluate(model, H, candidates, sub, args.eval_k, args.eval_pool, ctx, feats,
                                      args.pool_source)
        results[name] = metrics
        if ctx.is_main:
            log_metrics(f"{name} (k={args.eval_k})", metrics)
            save_json(retrieved, work.run(args.run_name, f"retrieved_{name}.json"))
    if ctx.is_main:
        save_json(results, work.run(args.run_name, "metrics.json"))


if __name__ == "__main__":
    main()

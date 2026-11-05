"""
Step 5: train AchGNN (paper, Secs. 3.2 and 4.2) and evaluate it on GoldRiM and the SilverRiM test split.

    positives   SilverRiM training pairs (75% of the seeds; --fold 0..4 picks one of five random splits)
    negatives   hard negatives: for each positive, a top-ranked candidate by cosine similarity of the f_text
                (Qwen3) embeddings, excluding documents that appear in the training data
    model       2-layer heterogeneous GraphSAGE (15 paper / 5 author neighbours), aspect-conditioned scorer and
                aspect classifier; loss L = L_rank (BPR) + lambda * L_aspect with lambda = 0.2

Runs on GPU when available (--device auto), otherwise on CPU.

    python achgnn/train.py --work work --run achgnn_fold0 --fold 0 --epochs 10
    python achgnn/train.py --work work_sample --run smoke --epochs 2 --device cpu
"""
import argparse
import os
import time

import numpy as np
import torch

from common import ASPECTS, Work, get_device, log, save_json, setup_logging
from engine import (embed_batch, evaluate, infer_all_papers, load_pairs, log_metrics, make_fanouts,
                    mine_hard_negatives)
from graph import FeatureStore, HeteroGraph
from model import AchGNN


def build_triplets(train, candidates, feats, device, cache_path, random_neg=False, seed=0):
    """
    Return arrays (query, positive, negative, aspect) for all training pairs. Each positive gets one negative:
    the next most similar candidate to its seed by f_text cosine similarity, never the seed itself or one of the
    seed's training recommendations ("excluding documents that appear in the training data", Sec. 4.2).
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
    negs = mine_hard_negatives(feats, seeds, dict(zip(seeds.tolist(), counts.tolist())), positives, candidates,
                               device)
    taken = {s: 0 for s in seeds.tolist()}
    n = np.empty(len(q), dtype=np.int64)
    for i, s in enumerate(q.tolist()):  # i-th positive of a seed is paired with its i-th hardest negative
        lst = negs[s]
        n[i] = lst[taken[s]] if taken[s] < len(lst) else random_negative(s)
        taken[s] += 1
    np.save(cache_path, n)
    return q, p, n, a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--run", default="achgnn", help="name of the run directory under <work>/runs")
    ap.add_argument("--fold", type=int, default=0, help="which of the 5 random 75/25 SilverRiM splits (0-4)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512, help="training triplets per step")
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
    ap.add_argument("--random-negatives", action="store_true", help="use random instead of hard negatives")
    ap.add_argument("--easy-negatives", type=int, default=1,
                    help="random negatives per positive, resampled every step, in addition to the hard negative. "
                         "With hard negatives only the model never learns that unrelated papers are irrelevant")
    ap.add_argument("--drop-aspects", nargs="*", default=[], help="ablation: remove supervision of these aspects")
    ap.add_argument("--no-authors", action="store_true", help="ablation: remove authorship edges")
    ap.add_argument("--random-authors", action="store_true", help="ablation: randomise authorship edges")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (quick tests)")
    ap.add_argument("--eval-k", type=int, default=10)
    ap.add_argument("--eval-pool", type=int, default=200,
                    help="candidates re-ranked by the aspect scorer per query (0 = score every candidate)")
    ap.add_argument("--pool-source", choices=["text", "gnn"], default="text",
                    help="where the re-ranked candidates come from: f_text cosine (default) or AchGNN cosine")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint of this run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging()
    work = Work(args.work)
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    log.info("device: %s", device)

    graph = HeteroGraph(work, drop_authorship=args.no_authors, random_authorship=args.random_authors, seed=args.seed)
    feats = FeatureStore(work)
    papers_candidate = np.load(work.candidates)
    pairs = load_pairs(work, args.fold)
    train = pairs[pairs["split"] == "train"]
    for name in args.drop_aspects:
        if name not in ASPECTS:
            raise SystemExit(f"unknown aspect {name!r}; choose from {ASPECTS}")
        train = train[train["aspect"] != name]
    log.info("pairs: %s", pairs["split"].value_counts().to_dict())
    if len(train) == 0:
        raise SystemExit("no training pairs")

    q, p, n, a = build_triplets(train, papers_candidate, feats, device,
                                work.path(f"hard_negatives_fold{args.fold}.npy"), args.random_negatives, args.seed)
    fanouts = make_fanouts(args.paper_neighbors, args.author_neighbors)

    model = AchGNN(feats.dim, args.hidden_dim, args.layers, len(ASPECTS), args.aspect_dim, args.dropout,
                   bool(args.final_activation), bool(args.normalize)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt_path = work.run(args.run, "model.pt")
    start_epoch, step = 0, 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optim_state"])
        start_epoch, step = ckpt["epoch"], ckpt["step"]
        log.info("resumed from epoch %d", start_epoch)

    log.info("training on %d triplets, %d parameters", len(q), sum(x.numel() for x in model.parameters()))
    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        perm = rng.permutation(len(q))
        tot = rank_tot = asp_tot = 0.0
        t0 = time.time()
        nb = 0
        for b in range(0, len(perm), args.batch_size):
            idx = perm[b:b + args.batch_size]
            m = len(idx)
            reps = 1 + args.easy_negatives
            negs = np.concatenate([n[idx]] + [rng.choice(papers_candidate, m) for _ in range(args.easy_negatives)])
            nodes, inv = np.unique(np.concatenate([q[idx], p[idx], negs]), return_inverse=True)
            h = embed_batch(model, graph, feats, nodes, fanouts, rng, device)
            hq, hp, hn = h[inv[:m]], h[inv[m:2 * m]], h[inv[2 * m:]]
            aspect = torch.from_numpy(a[idx]).long().to(device)
            # every positive is contrasted with its hard negative and its easy negatives (BPR, Eq. 9)
            loss, l_rank, l_asp = model.loss(hq.repeat(reps, 1), hp.repeat(reps, 1), hn, aspect.repeat(reps), args.lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, rank_tot, asp_tot, nb, step = tot + loss.item(), rank_tot + l_rank.item(), asp_tot + l_asp.item(), nb + 1, step + 1
            if step % 50 == 0:
                log.info("epoch %d step %d loss %.4f (rank %.4f, aspect %.4f)", epoch + 1, step, tot / nb,
                         rank_tot / nb, asp_tot / nb)
            if args.max_steps and step >= args.max_steps:
                break
        history.append({"epoch": epoch + 1, "loss": tot / nb, "rank_loss": rank_tot / nb, "aspect_loss": asp_tot / nb,
                        "seconds": time.time() - t0})
        log.info("epoch %d done: %s", epoch + 1, history[-1])
        torch.save({"model_state": model.state_dict(), "optim_state": opt.state_dict(), "model_config": model.config,
                    "epoch": epoch + 1, "step": step, "args": vars(args), "aspects": ASPECTS,
                    "fanouts": [args.paper_neighbors, args.author_neighbors]}, ckpt_path)
        if args.max_steps and step >= args.max_steps:
            break
    save_json(history, work.run(args.run, "history.json"))
    log.info("model saved to %s", ckpt_path)

    if args.no_eval:
        return
    H = infer_all_papers(model, graph, feats, fanouts, device)
    torch.save(H, work.run(args.run, "paper_embeddings.pt"))
    results = {}
    for split, name in [("gold", "GoldRiM"), ("test", "SilverRiM-test")]:
        sub = pairs[pairs["split"] == split]
        if len(sub) == 0:
            continue
        metrics, retrieved = evaluate(model, H, papers_candidate, sub, args.eval_k, args.eval_pool, device,
                                      feats, args.pool_source)
        log_metrics(f"{name} (k={args.eval_k})", metrics)
        results[name] = metrics
        save_json(retrieved, work.run(args.run, f"retrieved_{name}.json"))
    save_json(results, work.run(args.run, "metrics.json"))


if __name__ == "__main__":
    main()

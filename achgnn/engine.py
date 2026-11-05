"""Training/inference building blocks shared by train.py, evaluate.py and recommend.py."""
import os
import queue
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import ASPECTS, log, seed_split
from distributed import Context, gather_rows
from graph import NODE_TYPES, RELATIONS
from model import AchGNN


def make_fanouts(paper_neighbors, author_neighbors):
    """Sampled neighbours per layer: paper neighbours for citations and author->paper, author neighbours."""
    return {RELATIONS[0]: paper_neighbors, RELATIONS[1]: author_neighbors, RELATIONS[2]: paper_neighbors}


def sample_batch(graph, feats, papers, fanouts, num_layers, rng, pin=False):
    """
    Sample the computation graph of a batch of papers. Input features are gathered here (CPU, optionally pinned)
    unless the feature store is GPU-resident, in which case the node ids are returned and to_device gathers them.
    """
    input_nodes, blocks = graph.sample_blocks(papers, fanouts, num_layers, rng)
    if feats.resident is not None:
        return {t: ("ids", input_nodes[t]) for t in NODE_TYPES}, blocks
    h0 = {t: feats.gather(t, input_nodes[t], "cpu") for t in NODE_TYPES}
    if pin:
        h0 = {t: v.pin_memory() for t, v in h0.items()}
    return h0, blocks


def to_device(h0, blocks, device, feats=None):
    h0 = {t: feats.gather(t, v[1], device) if isinstance(v, tuple) else v.to(device, non_blocking=True)
          for t, v in h0.items()}
    blocks = [{"num_dst": b["num_dst"],
               "edges": {r: (s.to(device, non_blocking=True), d.to(device, non_blocking=True))
                         for r, (s, d) in b["edges"].items()}} for b in blocks]
    return h0, blocks


def prefetch(generator, depth=4):
    """Run a batch generator in a background thread (neighbour sampling and feature reads overlap the GPU)."""
    q = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in generator:
                q.put(item)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item


def embed_batch(model, graph, feats, papers, fanouts, rng, device):
    """Final GNN embeddings h for a batch of papers, using sampled neighbourhoods."""
    h0, blocks = sample_batch(graph, feats, papers, fanouts, len(model.layers), rng)
    h0, blocks = to_device(h0, blocks, device, feats)
    return model.encode(h0, blocks)


@torch.no_grad()
def infer_all_papers(model, graph, feats, fanouts, ctx: Context, out_dir, batch_size=8192, seed=0, amp=False):
    """
    Contextual embeddings for all papers (Sec. 3.2.6), computed layer by layer over the whole graph. Each layer's
    output is written to a memory-mapped file in out_dir; with several processes every rank computes an
    interleaved share of the node chunks. Returns a float16 CPU tensor [num_papers, hidden] on every rank.
    """
    model.eval()
    device = ctx.device
    hidden = model.config["hidden_dim"]
    prev = None
    num_layers = len(model.layers)
    for l in range(num_layers):
        paths = {t: os.path.join(out_dir, f"layer{l + 1}_{t}.npy") for t in NODE_TYPES}
        if ctx.is_main:
            for t in NODE_TYPES:
                np.lib.format.open_memmap(paths[t], mode="w+", dtype=np.float16,
                                          shape=(graph.num_nodes[t], hidden)).flush()
        ctx.barrier()
        out = {t: np.load(paths[t], mmap_mode="r+") for t in NODE_TYPES}
        for t in NODE_TYPES:
            n = graph.num_nodes[t]
            starts = ctx.shard(list(range(0, n, batch_size)))
            for s in tqdm(starts, desc=f"inference layer {l + 1}/{num_layers} ({t})", unit="batch",
                          disable=not ctx.is_main, leave=False):
                chunk = np.arange(s, min(s + batch_size, n), dtype=np.int64)
                rng = np.random.default_rng([seed, l, NODE_TYPES.index(t), s])  # independent of the world size
                input_nodes, (block,) = graph.sample_blocks({t: chunk}, fanouts, 1, rng)
                if prev is None:
                    h_src = {u: feats.gather(u, input_nodes[u], device) for u in NODE_TYPES}
                else:
                    h_src = {u: torch.from_numpy(gather_rows(prev[u], input_nodes[u])).to(device)
                             for u in NODE_TYPES}
                _, (block,) = to_device({}, [block], device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp and device.type == "cuda"):
                    res = model.layer_forward(l, h_src, block)[t]
                out[t][s:s + len(chunk)] = res.float().cpu().numpy().astype(np.float16)
            out[t].flush()
        del out
        ctx.barrier()
        prev = {t: np.load(paths[t], mmap_mode="r") for t in NODE_TYPES}
    H = torch.from_numpy(np.array(prev["paper"]))  # copy into RAM, detached from the file
    del prev
    ctx.barrier()
    if ctx.is_main:
        for l in range(num_layers):  # intermediate files are no longer needed
            for t in NODE_TYPES:
                try:
                    os.remove(os.path.join(out_dir, f"layer{l + 1}_{t}.npy"))
                except OSError:
                    pass
    return H


@torch.no_grad()
def mine_hard_negatives(feats, seeds, needed, exclude, candidates, device, seed_block=4096, cand_block=250_000):
    """
    For each seed, the `needed[seed]` top-ranked candidates by cosine similarity of the f_text embeddings,
    skipping the seed itself and the ids in `exclude[seed]`. Exact chunked search; fp16 matmuls on GPU.
    Returns {seed: [ids]}.
    """
    seeds = np.asarray(seeds, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(seeds) == 0:
        return {}
    k = min(int(max(needed[s] + len(exclude.get(s, ())) + 1 for s in needed)), len(candidates))
    half = device.type == "cuda"
    best_s = torch.full((len(seeds), k), -2.0)
    best_i = torch.zeros((len(seeds), k), dtype=torch.long)
    q_all = F.normalize(feats.gather("paper", seeds, "cpu"), dim=1)
    for cs in tqdm(range(0, len(candidates), cand_block), desc="cosine search", unit="block", leave=False):
        cand = candidates[cs:cs + cand_block]
        c = F.normalize(feats.gather("paper", cand, device), dim=1)
        c = c.half() if half else c
        cand_t = torch.from_numpy(cand)
        for qs in range(0, len(seeds), seed_block):
            q = q_all[qs:qs + seed_block].to(device)
            sim = (q.half() if half else q) @ c.T
            s, i = sim.float().topk(min(k, sim.size(1)), dim=1)
            s, i = s.cpu(), cand_t[i.cpu()]
            merged_s = torch.cat([best_s[qs:qs + seed_block], s], 1)
            merged_i = torch.cat([best_i[qs:qs + seed_block], i], 1)
            top = merged_s.topk(k, dim=1)
            best_s[qs:qs + seed_block] = top.values
            best_i[qs:qs + seed_block] = merged_i.gather(1, top.indices)
    out = {}
    for j, q in enumerate(seeds.tolist()):
        skip = set(exclude.get(q, ())) | {q}
        out[q] = [c for c in best_i[j].tolist() if c not in skip][: needed[q]]
    return out


def mine_distributed(ctx: Context, feats, seeds, needed, exclude, candidates):
    """mine_hard_negatives with the seeds split over all ranks; every rank receives the full result."""
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    mine = ctx.shard(seeds)
    part = mine_hard_negatives(feats, mine, {int(s): needed[int(s)] for s in mine}, exclude, candidates, ctx.device)
    return ctx.gather_dicts(part)


def text_pools(ctx: Context, feats, seeds, candidates, pool):
    """Top-`pool` candidates per seed by cosine similarity of the f_text embeddings."""
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    return mine_distributed(ctx, feats, seeds, {int(q): pool for q in seeds}, {}, candidates)


def load_pairs(work, fold):
    """Pairs with a `split` column: gold (test only), train, test (SilverRiM 75/25 split of seeds)."""
    pairs = pd.read_csv(work.pairs)
    silver = pairs[pairs["silver"].astype(bool)]
    test_seeds = seed_split(silver["seed_idx"].values, fold)
    pairs["split"] = np.where(pairs["gold"].astype(bool), "gold",
                              np.where(pairs["seed_idx"].isin(test_seeds), "test", "train"))
    return pairs


def build_queries(pairs):
    """Per aspect and 'General': {setting: {seed_idx: set(relevant rec_idx)}} (Sec. 4.2)."""
    queries = {"General": pairs.groupby("seed_idx")["rec_idx"].apply(set).to_dict()}
    for a_id, name in enumerate(ASPECTS):
        sub = pairs[pairs["aspect_id"] == a_id]
        queries[name] = sub.groupby("seed_idx")["rec_idx"].apply(set).to_dict()
    return queries


@torch.no_grad()
def retrieve(model, H, candidates, query_seeds, aspect_id, k, pool, device, pools=None, batch_size=256):
    """
    Rank candidates for each query paper by the aspect-conditioned score s^(as)_{q,c} (Eq. 11). For the
    aspect-agnostic 'General' setting (aspect_id None) the best score over all aspects is used.

    Candidates scored per query:
      pools given  -> pools[q] (e.g. the top f_text cosine neighbours, see text_pools)
      pool > 0     -> the top `pool` candidates by cosine similarity of the AchGNN embeddings H
      pool == 0    -> every candidate (exact Eq. 11)
    Returns {seed: [top-k paper ids]}.
    """
    model.eval()
    candidates = np.unique(np.asarray(candidates, dtype=np.int64))
    cand = torch.from_numpy(candidates)
    Hc_n = F.normalize(H[cand].to(device).float(), dim=1) if pools is None and pool else None
    results = {}
    for s in range(0, len(query_seeds), batch_size):
        q_ids = np.asarray(query_seeds[s:s + batch_size], dtype=np.int64)
        hq = H[torch.from_numpy(q_ids)].to(device).float()
        if pools is not None:
            lists = [torch.tensor([c for c in pools[int(q)] if c != q], dtype=torch.long) for q in q_ids]
        elif pool:
            sim = F.normalize(hq, dim=1) @ Hc_n.T
            pos = np.minimum(np.searchsorted(candidates, q_ids), len(candidates) - 1)
            is_self = candidates[pos] == q_ids
            sim[torch.from_numpy(np.nonzero(is_self)[0]), torch.from_numpy(pos[is_self])] = -2
            top = sim.topk(min(pool, len(candidates) - 1), dim=1).indices.cpu()
            lists = [cand[t] for t in top]
        else:
            lists = [cand[cand != int(q)] for q in q_ids]
        if pools is not None or pool:
            # score the whole query batch at once: [b, P] padded candidate matrix
            width = max(len(x) for x in lists) if lists else 0
            if width == 0:
                results.update({int(q): [] for q in q_ids})
                continue
            ids = torch.zeros(len(lists), width, dtype=torch.long)
            valid = torch.zeros(len(lists), width, dtype=torch.bool)
            for j, x in enumerate(lists):
                ids[j, :len(x)], valid[j, :len(x)] = x, True
            hc = H[ids.reshape(-1)].to(device).float()
            hqq = hq.repeat_interleave(width, dim=0)
            if aspect_id is None:
                scores = model.score_all_aspects(hqq, hc).max(dim=1).values
            else:
                scores = model.score(hqq, hc, torch.full((len(hqq),), aspect_id, device=device, dtype=torch.long))
            scores = scores.view(len(lists), width).masked_fill(~valid.to(device), float("-inf"))
            order = scores.topk(min(k, width), dim=1)
            for j, q in enumerate(q_ids):
                keep = order.values[j].isfinite().cpu()
                results[int(q)] = ids[j, order.indices[j].cpu()[keep]].tolist()
        else:
            for j, q in enumerate(q_ids):
                ids = lists[j]
                scores = torch.empty(len(ids), device=device)
                for c0 in range(0, len(ids), 1_000_000):
                    hc = H[ids[c0:c0 + 1_000_000]].to(device).float()
                    hqq = hq[j].expand(len(hc), -1)
                    if aspect_id is None:
                        scores[c0:c0 + len(hc)] = model.score_all_aspects(hqq, hc).max(dim=1).values
                    else:
                        scores[c0:c0 + len(hc)] = model.score(
                            hqq, hc, torch.full((len(hc),), aspect_id, device=device, dtype=torch.long))
                results[int(q)] = ids[scores.topk(min(k, len(ids))).indices.cpu()].tolist()
    return results


def metrics_at_k(retrieved, relevant, k):
    p, r, mrr = [], [], []
    for q, rel in relevant.items():
        ranked = retrieved.get(q, [])[:k]
        hits = [i for i, d in enumerate(ranked) if d in rel]
        p.append(len(hits) / k)
        r.append(len(hits) / len(rel))
        mrr.append(1.0 / (hits[0] + 1) if hits else 0.0)
    n = len(relevant)
    return {"P": float(np.mean(p)) if n else 0.0, "R": float(np.mean(r)) if n else 0.0,
            "MRR": float(np.mean(mrr)) if n else 0.0, "queries": n}


def evaluate(model, H, candidates, pairs, k, pool, ctx: Context, feats=None, pool_source="text"):
    """
    Precision@k, Recall@k and MRR per setting (General + each aspect) for the given pairs, plus the retrieved
    lists. Queries are split over all ranks. pool_source "text": re-rank the top-`pool` f_text neighbours of each
    seed; "gnn": the top-`pool` neighbours by AchGNN embedding cosine (pool 0: all candidates).
    """
    out, retrieved_all = {}, {}
    pools = None
    if pool_source == "text" and pool:
        pools = text_pools(ctx, feats, pairs["seed_idx"].values, candidates, pool)
    for setting, relevant in build_queries(pairs).items():
        if not relevant:
            continue
        aspect_id = None if setting == "General" else ASPECTS.index(setting)
        mine = ctx.shard(sorted(relevant))
        retrieved = ctx.gather_dicts(retrieve(model, H, candidates, mine, aspect_id, k, pool, ctx.device, pools))
        out[setting] = metrics_at_k(retrieved, relevant, k)
        retrieved_all[setting] = {str(q): v for q, v in retrieved.items()}
    return out, retrieved_all


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = AchGNN(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def format_metrics(metrics):
    lines = [f"{'setting':32s} {'P@k':>7s} {'R@k':>7s} {'MRR':>7s} {'#q':>6s}"]
    for setting, m in metrics.items():
        lines.append(f"{setting:32s} {m['P']:7.3f} {m['R']:7.3f} {m['MRR']:7.3f} {m['queries']:6d}")
    return "\n".join(lines)


def log_metrics(title, metrics):
    log.info("%s\n%s", title, format_metrics(metrics))

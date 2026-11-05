"""Training/inference building blocks shared by train.py, evaluate.py and recommend.py."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import ASPECTS, log, seed_split
from graph import NODE_TYPES, RELATIONS
from model import AchGNN


def make_fanouts(paper_neighbors, author_neighbors):
    """Paper, Sec. 4.2: 15 sampled paper neighbours and 5 author neighbours per layer."""
    return {RELATIONS[0]: paper_neighbors, RELATIONS[1]: author_neighbors, RELATIONS[2]: paper_neighbors}


def embed_batch(model, graph, feats, papers, fanouts, rng, device):
    """Final GNN embeddings h for a batch of papers, using sampled neighbourhoods."""
    input_nodes, blocks = graph.sample_blocks(papers, fanouts, len(model.layers), rng)
    h0 = {t: feats.gather(t, input_nodes[t], device) for t in NODE_TYPES}
    return model.encode(h0, blocks)


@torch.no_grad()
def infer_all_papers(model, graph, feats, fanouts, device, batch_size=4096, seed=0):
    """
    Contextual embeddings for all papers (paper, Sec. 3.2.6), computed layer by layer over the whole graph so
    every node is processed once per layer. Returns a float16 CPU tensor [num_papers, hidden].
    """
    model.eval()
    rng = np.random.default_rng(seed)
    prev = None  # outputs of the previous layer, per node type (CPU float16)
    for l in range(len(model.layers)):
        out = {t: torch.empty(graph.num_nodes[t], model.config["hidden_dim"], dtype=torch.float16)
               for t in NODE_TYPES}
        for t in NODE_TYPES:
            n = graph.num_nodes[t]
            for s in tqdm(range(0, n, batch_size), desc=f"inference layer {l + 1} ({t})", unit="batch", leave=False):
                chunk = np.arange(s, min(s + batch_size, n), dtype=np.int64)
                input_nodes, (block,) = graph.sample_blocks({t: chunk}, fanouts, 1, rng)
                if prev is None:
                    h_src = {u: feats.gather(u, input_nodes[u], device) for u in NODE_TYPES}
                else:
                    h_src = {u: prev[u][torch.from_numpy(input_nodes[u])].to(device).float() for u in NODE_TYPES}
                out[t][s:s + len(chunk)] = model.layer_forward(l, h_src, block)[t].half().cpu()
        prev = out
    return prev["paper"]


@torch.no_grad()
def mine_hard_negatives(feats, seeds, needed, exclude, candidates, device, seed_block=2048, cand_block=200_000):
    """
    Hard negatives (paper, Sec. 4.2): for each training seed, the `needed[seed]` top-ranked candidates by cosine
    similarity of the f_text embeddings, skipping the seed itself and the ids in `exclude[seed]` (its known
    recommendations). Returns {seed: [neg ids]}.
    """
    seeds = np.asarray(seeds, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    k = int(max(needed[s] + len(exclude.get(s, ())) + 1 for s in needed))
    k = min(k, len(candidates))
    best_s = torch.full((len(seeds), k), -2.0)
    best_i = torch.zeros((len(seeds), k), dtype=torch.long)
    q_all = F.normalize(feats.gather("paper", seeds, "cpu"), dim=1)
    for cs in tqdm(range(0, len(candidates), cand_block), desc="mining hard negatives", unit="block"):
        cand = candidates[cs:cs + cand_block]
        c = F.normalize(feats.gather("paper", cand, device), dim=1)
        cand_t = torch.from_numpy(cand)
        for qs in range(0, len(seeds), seed_block):
            q = q_all[qs:qs + seed_block].to(device)
            sim = q @ c.T
            s, i = sim.topk(min(k, sim.size(1)), dim=1)
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


def load_pairs(work, fold):
    """Pairs with a `split` column: gold (test only), train, test (SilverRiM 75/25 split of seeds)."""
    pairs = pd.read_csv(work.pairs)
    silver = pairs[pairs["silver"].astype(bool)]
    test_seeds = seed_split(silver["seed_idx"].values, fold)
    pairs["split"] = np.where(pairs["gold"].astype(bool), "gold",
                              np.where(pairs["seed_idx"].isin(test_seeds), "test", "train"))
    return pairs


def build_queries(pairs):
    """Per aspect and 'General': {setting: {seed_idx: set(relevant rec_idx)}} (paper, Sec. 4.2)."""
    queries = {"General": pairs.groupby("seed_idx")["rec_idx"].apply(set).to_dict()}
    for a_id, name in enumerate(ASPECTS):
        sub = pairs[pairs["aspect_id"] == a_id]
        queries[name] = sub.groupby("seed_idx")["rec_idx"].apply(set).to_dict()
    return queries


def text_pools(feats, seeds, candidates, pool, device):
    """Top-`pool` candidates per seed by cosine similarity of the f_text (Qwen3) embeddings."""
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    return mine_hard_negatives(feats, seeds, {int(q): pool for q in seeds}, {}, candidates, device)


@torch.no_grad()
def retrieve(model, H, candidates, query_seeds, aspect_id, k, pool, device, pools=None, batch_size=256):
    """
    Rank candidates for each query paper by the aspect-conditioned score s^(as)_{q,c} (Eq. 11). For the
    aspect-agnostic 'General' setting (aspect_id None) the best score over all aspects is used.

    Which candidates are scored:
      pools given  -> pools[q], e.g. the top candidates by f_text cosine similarity (see text_pools)
      pool > 0     -> the top `pool` candidates by cosine similarity of the AchGNN embeddings H
      pool == 0    -> every candidate (exact Eq. 11; slow for large corpora)
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
            cand_lists = [torch.tensor(pools[int(q)], dtype=torch.long) for q in q_ids]
        elif pool:
            sim = F.normalize(hq, dim=1) @ Hc_n.T
            pos = np.minimum(np.searchsorted(candidates, q_ids), len(candidates) - 1)
            is_self = candidates[pos] == q_ids
            sim[torch.from_numpy(np.nonzero(is_self)[0]), torch.from_numpy(pos[is_self])] = -2  # skip the query
            top = sim.topk(min(pool + 1, len(candidates)), dim=1).indices.cpu()
            cand_lists = [cand[t] for t in top]
        else:
            cand_lists = [cand] * len(q_ids)
        for j, q in enumerate(q_ids):
            ids = cand_lists[j]
            ids = ids[ids != int(q)]  # never recommend the query itself
            if len(ids) == 0:
                results[int(q)] = []
                continue
            hc = H[ids].to(device).float()
            hqq = hq[j].expand(len(ids), -1)
            if aspect_id is None:
                scores = model.score_all_aspects(hqq, hc).max(dim=1).values
            else:
                scores = model.score(hqq, hc, torch.full((len(ids),), aspect_id, device=device, dtype=torch.long))
            order = scores.topk(min(k, len(ids))).indices.cpu()
            results[int(q)] = ids[order].tolist()
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


def evaluate(model, H, candidates, pairs, k, pool, device, feats=None, pool_source="text"):
    """
    Metrics per setting (General + each aspect) for the given pairs, plus the retrieved lists.
    pool_source "text": re-rank the top-`pool` f_text neighbours of each seed (needs feats); "gnn": the top-`pool`
    neighbours by AchGNN embedding cosine (pool 0: all candidates).
    """
    out, retrieved_all = {}, {}
    pools = None
    if pool_source == "text" and pool:
        pools = text_pools(feats, pairs["seed_idx"].values, candidates, pool, device)
    for setting, relevant in build_queries(pairs).items():
        if not relevant:
            continue
        aspect_id = None if setting == "General" else ASPECTS.index(setting)
        retrieved = retrieve(model, H, candidates, list(relevant), aspect_id, k, pool, device, pools)
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

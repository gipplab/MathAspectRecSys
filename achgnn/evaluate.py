"""
Evaluate a trained AchGNN on GoldRiM and the SilverRiM test split: Precision@k, Recall@k and MRR per aspect and in
the aspect-agnostic General setting (Sec. 4.2).

    python achgnn/evaluate.py --work work --run-name achgnn_fold0
    torchrun --standalone --nproc_per_node=4 achgnn/evaluate.py --work work --run-name achgnn_fold0
"""
import argparse
import os

import numpy as np
import torch

from common import Work, save_json, setup_logging
from distributed import Context
from engine import evaluate, infer_all_papers, load_model, load_pairs, log_metrics, make_fanouts
from graph import HeteroGraph, open_features


def load_or_infer(work, run, model, ckpt, ctx, recompute=False, feats=None):
    """AchGNN embeddings of all papers, cached as <run>/paper_embeddings.pt."""
    path = work.run(run, "paper_embeddings.pt")
    if os.path.exists(path) and not recompute:
        return torch.load(path)
    a = ckpt["args"]
    graph = HeteroGraph(work, drop_authorship=a["no_authors"], random_authorship=a["random_authors"], seed=a["seed"])
    feats = feats or open_features(work, ctx.device)
    H = infer_all_papers(model, graph, feats, make_fanouts(*ckpt["fanouts"]), ctx, os.path.dirname(path),
                         amp=a.get("amp", False))
    if ctx.is_main:
        torch.save(H, path)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--run-name", default="achgnn")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N (single process)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=200, help="re-ranked candidates per query (0 = all)")
    ap.add_argument("--pool-source", choices=["text", "gnn"], default="text",
                    help="re-rank the f_text cosine neighbours or the AchGNN cosine neighbours")
    ap.add_argument("--recompute", action="store_true", help="recompute the paper embeddings")
    args = ap.parse_args()
    setup_logging()
    ctx = Context(args.device)
    work = Work(args.work)
    model, ckpt = load_model(work.run(args.run_name, "model.pt"), ctx.device)
    feats = open_features(work, ctx.device)
    H = load_or_infer(work, args.run_name, model, ckpt, ctx, args.recompute, feats)
    pairs = load_pairs(work, ckpt["args"]["fold"])
    candidates = np.load(work.candidates)
    results = {}
    for split, name in [("gold", "GoldRiM"), ("test", "SilverRiM-test")]:
        sub = pairs[pairs["split"] == split]
        if len(sub):
            metrics, retrieved = evaluate(model, H, candidates, sub, args.k, args.pool, ctx, feats, args.pool_source)
            results[name] = metrics
            if ctx.is_main:
                log_metrics(f"{name} (k={args.k})", metrics)
                save_json(retrieved, work.run(args.run_name, f"retrieved_{name}.json"))
    if ctx.is_main:
        save_json(results, work.run(args.run_name, "metrics.json"))
    ctx.close()


if __name__ == "__main__":
    main()

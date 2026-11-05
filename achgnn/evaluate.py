"""
Step 6: evaluate a trained AchGNN on GoldRiM and the SilverRiM test split: Precision@k, Recall@k and MRR, per
aspect and in the aspect-agnostic 'General' setting (paper, Sec. 4.2).

    python achgnn/evaluate.py --work work --run achgnn_fold0
"""
import argparse
import os

import numpy as np
import torch

from common import Work, get_device, save_json, setup_logging
from engine import evaluate, infer_all_papers, load_model, load_pairs, log_metrics, make_fanouts
from graph import FeatureStore, HeteroGraph


def load_or_infer(work, run, model, ckpt, device, recompute=False):
    path = work.run(run, "paper_embeddings.pt")
    if os.path.exists(path) and not recompute:
        return torch.load(path)
    a = ckpt["args"]
    graph = HeteroGraph(work, drop_authorship=a["no_authors"], random_authorship=a["random_authors"], seed=a["seed"])
    H = infer_all_papers(model, graph, FeatureStore(work), make_fanouts(*ckpt["fanouts"]), device)
    torch.save(H, path)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--run", default="achgnn")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=200, help="re-ranked candidates per query (0 = all)")
    ap.add_argument("--pool-source", choices=["text", "gnn"], default="text",
                    help="re-rank the f_text (Qwen3) cosine neighbours (default) or the AchGNN cosine neighbours")
    ap.add_argument("--recompute", action="store_true", help="recompute paper embeddings")
    args = ap.parse_args()
    setup_logging()
    work, device = Work(args.work), get_device(args.device)
    model, ckpt = load_model(work.run(args.run, "model.pt"), device)
    H = load_or_infer(work, args.run, model, ckpt, device, args.recompute)
    pairs = load_pairs(work, ckpt["args"]["fold"])
    candidates = np.load(work.candidates)
    feats = FeatureStore(work)
    results = {}
    for split, name in [("gold", "GoldRiM"), ("test", "SilverRiM-test")]:
        sub = pairs[pairs["split"] == split]
        if len(sub):
            metrics, retrieved = evaluate(model, H, candidates, sub, args.k, args.pool, device, feats,
                                          args.pool_source)
            log_metrics(f"{name} (k={args.k})", metrics)
            results[name] = metrics
            save_json(retrieved, work.run(args.run, f"retrieved_{name}.json"))
    save_json(results, work.run(args.run, "metrics.json"))


if __name__ == "__main__":
    main()

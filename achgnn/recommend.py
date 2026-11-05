"""
Aspect-specific recommendations for one or more papers with a trained AchGNN (Eq. 11).

    python achgnn/recommend.py --work work --run-name achgnn_fold0 --zbl 0940.14001
    python achgnn/recommend.py --work work --run-name achgnn_fold0 --zbl 0940.14001 --aspect "Results" --top 5
"""
import argparse

import numpy as np
import pandas as pd

from common import ASPECTS, Work, normalize_zbl, setup_logging
from distributed import Context
from engine import load_model, retrieve, text_pools
from evaluate import load_or_infer
from graph import open_features


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--run-name", default="achgnn")
    ap.add_argument("--zbl", nargs="+", required=True, help="Zbl number(s) or zbMATH document id(s) of query papers")
    ap.add_argument("--aspect", choices=ASPECTS + ["General", "all"], default="all")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--pool", type=int, default=200, help="re-ranked candidates per query (0 = all)")
    ap.add_argument("--pool-source", choices=["text", "gnn"], default="text")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    setup_logging()
    ctx = Context(args.device)
    work, device = Work(args.work), ctx.device
    model, ckpt = load_model(work.run(args.run_name, "model.pt"), device)
    feats = open_features(work, device)
    H = load_or_infer(work, args.run_name, model, ckpt, ctx, feats=feats)
    papers = pd.read_csv(work.papers, dtype=str, keep_default_na=False, usecols=["doc_id", "zbl_id", "title"])
    by_zbl = {z: i for i, z in enumerate(papers["zbl_id"]) if z}
    by_id = {d: i for i, d in enumerate(papers["doc_id"])}
    candidates = np.load(work.candidates)
    aspects = ASPECTS + ["General"] if args.aspect == "all" else [args.aspect]

    for z in args.zbl:
        q = by_zbl.get(normalize_zbl(z), by_id.get(z))
        if q is None:
            print(f"\n{z}: not in the graph")
            continue
        pools = text_pools(ctx, feats, [q], candidates, args.pool) if args.pool_source == "text" and args.pool else None
        print(f"\n=== {papers.zbl_id[q]}  {papers.title[q]}  (https://zbmath.org/?q=an:{papers.zbl_id[q]})")
        for aspect in aspects:
            aid = None if aspect == "General" else ASPECTS.index(aspect)
            recs = retrieve(model, H, candidates, [q], aid, args.top, args.pool, device, pools)[q]
            print(f"--- {aspect}")
            for rank, r in enumerate(recs, 1):
                print(f"  {rank:2d}. {papers.zbl_id[r] or papers.doc_id[r]:12s} {papers.title[r][:100]}")


if __name__ == "__main__":
    main()

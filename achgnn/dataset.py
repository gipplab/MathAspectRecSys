"""
Step 1: read the GoldRiM/SilverRiM CSV (https://drive.google.com/file/d/1npaipBwA606KAUKmxikjZjvjj3VMQV0Q/view),
normalise the Zbl numbers and write the pairs plus the list of document ids needed for the graph.

    python achgnn/dataset.py --csv mathRec_dataset.csv --work work
    python achgnn/dataset.py --csv mathRec_dataset.csv --work work_sample --sample-seeds 300   # small test set
"""
import argparse

import numpy as np
import pandas as pd

from common import ASPECTS, Work, log, normalize_zbl, setup_logging


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="mathRec_dataset.csv")
    ap.add_argument("--work", default="work")
    ap.add_argument("--sample-seeds", type=int, default=0,
                    help="keep only N random SilverRiM seeds (all GoldRiM pairs are kept); 0 = everything")
    ap.add_argument("--no-gold", action="store_true", help="with --sample-seeds: drop GoldRiM as well")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging()
    work = Work(args.work)

    df = pd.read_csv(args.csv, dtype=str)
    rows = []
    for r in df.itertuples(index=False):
        for rec in str(r.Recommendation).split(";"):  # the column may hold several ids
            if rec.strip() and rec.strip() != "nan":
                rows.append((normalize_zbl(r.Seed), normalize_zbl(rec), r.Aspect, r._3, r.GoldRiM == "1",
                             r.SilverRiM == "1"))
    pairs = pd.DataFrame(rows, columns=["seed", "rec", "aspect", "aspect_label", "gold", "silver"])
    unknown = set(pairs.aspect) - set(ASPECTS)
    if unknown:
        raise SystemExit(f"unexpected aspect names {unknown}")
    pairs = pairs[pairs.seed != pairs.rec].drop_duplicates(["seed", "rec", "aspect", "gold", "silver"])

    if args.sample_seeds:
        silver_seeds = pairs.loc[pairs.silver, "seed"].unique()
        rng = np.random.default_rng(args.seed)
        keep = set(rng.choice(silver_seeds, min(args.sample_seeds, len(silver_seeds)), replace=False))
        mask = pairs.seed.isin(keep) & pairs.silver
        if not args.no_gold:
            mask |= pairs.gold
        pairs = pairs[mask]

    pairs.to_csv(work.pairs_raw, index=False)
    ids = sorted(set(pairs.seed) | set(pairs.rec))
    with open(work.dataset_ids, "w", encoding="utf8") as f:
        f.write("\n".join(ids) + "\n")
    log.info("%d pairs (%d GoldRiM, %d SilverRiM), %d seeds, %d distinct documents -> %s",
             len(pairs), pairs.gold.sum(), pairs.silver.sum(), pairs.seed.nunique(), len(ids), work.dataset_ids)
    log.info("aspect distribution:\n%s", pairs.groupby(["gold", "aspect"]).size().to_string())


if __name__ == "__main__":
    main()

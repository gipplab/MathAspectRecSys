"""
Average the metrics of several runs (e.g. the five cross-validation folds): mean ± standard deviation of P@k, R@k
and MRR per dataset and setting. Writes <work>/runs/summary.json and prints a table.

    python achgnn/summarize.py --work work --runs achgnn_fold0 achgnn_fold1 achgnn_fold2 achgnn_fold3 achgnn_fold4
"""
import argparse
import os

import numpy as np

from common import Work, load_json, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    work = Work(args.work)

    found = {r: load_json(work.run(r, "metrics.json")) for r in args.runs
             if os.path.exists(work.run(r, "metrics.json"))}
    if not found:
        raise SystemExit("no metrics.json found for the given runs")
    summary = {}
    for dataset in sorted({d for m in found.values() for d in m}):
        settings = [s for s in next(m[dataset] for m in found.values() if dataset in m)]
        summary[dataset] = {}
        for setting in settings:
            vals = [m[dataset][setting] for m in found.values() if setting in m.get(dataset, {})]
            summary[dataset][setting] = {
                key: {"mean": float(np.mean([v[key] for v in vals])), "std": float(np.std([v[key] for v in vals]))}
                for key in ("P", "R", "MRR")
            }
    save_json({"runs": list(found), "metrics": summary}, work.path("runs", "summary.json"))

    print(f"runs: {', '.join(found)}")
    for dataset, settings in summary.items():
        print(f"\n{dataset}\n{'setting':32s} {'P@k':>17s} {'R@k':>17s} {'MRR':>17s}")
        for setting, m in settings.items():
            cells = [f"{m[k]['mean']:.3f} +/- {m[k]['std']:.3f}" for k in ("P", "R", "MRR")]
            print(f"{setting:32s} {cells[0]:>17s} {cells[1]:>17s} {cells[2]:>17s}")


if __name__ == "__main__":
    main()

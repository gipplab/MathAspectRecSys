#!/usr/bin/env bash
#SBATCH --job-name=achgnn-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#
# Step 2: dataset pairs, zbMATH network with abstracts, AchGNN graph.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/config.sh"
activate_env

python "$REPO/achgnn/dataset.py" --csv "$DATASET_CSV" --work "$WORK_DIR"

# dataset documents missing from the crawl (requires internet; skipped documents are reported later)
(cd "$REPO/zbmath_hetnet" && python fetch_ids.py --ids-file "$WORK_DIR/dataset_ids.txt" --out-dir "$ZBMATH_DIR") \
    || echo "fetch_ids.py failed (no internet on this node?); continuing with the crawled documents"

(cd "$REPO/zbmath_hetnet" && python build_graph.py --out-dir "$ZBMATH_DIR" --with-text)

EXTRA=()
[ "$ENGLISH_ONLY" = 1 ] && EXTRA+=(--english-only)
python "$REPO/achgnn/prepare_graph.py" --graph-dir "$ZBMATH_DIR/graph" --work "$WORK_DIR" ${EXTRA[@]+"${EXTRA[@]}"}

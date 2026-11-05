#!/usr/bin/env bash
#SBATCH --job-name=zbmath-crawl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#
# Step 1: harvest all zbMATH Open documents and authors (needs internet access from the node).
# The crawl is resumable: resubmit the job if it reaches the time limit.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/config.sh"
activate_env

cd "$REPO/zbmath_hetnet"
python crawl.py all --out-dir "$ZBMATH_DIR" --workers "${CRAWL_WORKERS:-4}"

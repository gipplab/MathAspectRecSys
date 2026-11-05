#!/usr/bin/env bash
#SBATCH --job-name=achgnn-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#
# Step 5: evaluate trained runs and summarise the cross-validation folds.
#   RUNS="achgnn_fold0 achgnn_fold1" sbatch slurm/05_evaluate.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/config.sh"
activate_env
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NUM_GPUS ))

RUNS="${RUNS:-achgnn_fold0 achgnn_fold1 achgnn_fold2 achgnn_fold3 achgnn_fold4}"
for RUN in $RUNS; do
    [ -f "$WORK_DIR/runs/$RUN/model.pt" ] || { echo "skipping $RUN (no model.pt)"; continue; }
    torchrun --standalone --nproc_per_node="$NUM_GPUS" "$REPO/achgnn/evaluate.py" \
        --work "$WORK_DIR" --run-name "$RUN" ${EVAL_ARGS:-}
done
python "$REPO/achgnn/summarize.py" --work "$WORK_DIR" --runs $RUNS

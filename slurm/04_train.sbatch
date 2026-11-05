#!/usr/bin/env bash
#SBATCH --job-name=achgnn-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm/logs/%x-%A_%a.out
#
# Step 4: train and evaluate AchGNN with DistributedDataParallel on all GPUs of the node.
# One array task per cross-validation fold (0-4); submit with --array=0 for a single fold.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/config.sh"
activate_env

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NUM_GPUS ))

torchrun --standalone --nproc_per_node="$NUM_GPUS" "$REPO/achgnn/train.py" \
    --work "$WORK_DIR" --run-name "achgnn_fold$FOLD" --fold "$FOLD" \
    --epochs "$TRAIN_EPOCHS" --batch-size "$TRAIN_BATCH" --amp --resume $TRAIN_ARGS

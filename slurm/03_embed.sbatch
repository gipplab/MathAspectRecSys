#!/usr/bin/env bash
#SBATCH --job-name=achgnn-embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#
# Step 3: Qwen3 node features for all papers and authors, one worker per GPU. Resumable.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/config.sh"
activate_env
nvidia-smi --query-gpu=name,memory.total --format=csv

export HF_HUB_OFFLINE=1
python "$REPO/achgnn/embed.py" --work "$WORK_DIR" --encoder hf --model "$EMBED_MODEL" \
    --devices all --batch-size "$EMBED_BATCH" --max-length 512

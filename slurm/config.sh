#!/usr/bin/env bash
# Shared configuration for all SLURM jobs. Edit the values below for your cluster, then submit from the
# repository root (see README.md, section "Running on a SLURM cluster").

# --- scheduler -------------------------------------------------------------------------------------------
export SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"        # partition with the A100 node
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"               # project/account, if your cluster requires one
export CPU_PARTITION="${CPU_PARTITION:-$SBATCH_PARTITION}" # partition for the CPU-only steps
export NUM_GPUS="${NUM_GPUS:-4}"                          # GPUs on the node

# --- software environment --------------------------------------------------------------------------------
# Commands that make Python and the "achgnn" environment available inside a job.
activate_env() {
    # module load cuda/12.4            # uncomment/adapt if your cluster uses environment modules
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV:-achgnn}"
    elif [ -f "${VENV_DIR:-$REPO/.venv}/bin/activate" ]; then
        source "${VENV_DIR:-$REPO/.venv}/bin/activate"
    fi
}

# --- paths -----------------------------------------------------------------------------------------------
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-$REPO/experiments}"                   # large outputs (~100 GB)
export DATASET_CSV="${DATASET_CSV:-$REPO/data/mathRec_dataset.csv}"  # GoldRiM/SilverRiM CSV
export ZBMATH_DIR="${ZBMATH_DIR:-$DATA_ROOT/zbmath}"                 # crawl + zbMATH network
export WORK_DIR="${WORK_DIR:-$DATA_ROOT/work}"                       # AchGNN inputs, features and runs
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
export EMBED_MODEL="${EMBED_MODEL:-$DATA_ROOT/models/Qwen3-Embedding-8B}"  # local copy of the encoder

# --- experiment ------------------------------------------------------------------------------------------
export ENGLISH_ONLY="${ENGLISH_ONLY:-1}"     # retrieval corpus restricted to English documents
export EMBED_BATCH="${EMBED_BATCH:-64}"      # texts per forward pass and GPU
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
export TRAIN_BATCH="${TRAIN_BATCH:-512}"     # training pairs per step and GPU
export TRAIN_ARGS="${TRAIN_ARGS:-}"          # extra arguments for achgnn/train.py

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$REPO/slurm/logs" "$DATA_ROOT"

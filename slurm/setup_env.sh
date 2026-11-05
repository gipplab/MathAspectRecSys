#!/usr/bin/env bash
# One-time setup on a login node (needs internet): Python environment, dataset and encoder weights.
#
#   bash slurm/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

ENV_NAME="${CONDA_ENV:-achgnn}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda env list | grep -qE "^$ENV_NAME\s" || conda create -y -n "$ENV_NAME" python=3.11
    conda activate "$ENV_NAME"
else
    python3 -m venv "${VENV_DIR:-$REPO/.venv}"
    source "${VENV_DIR:-$REPO/.venv}/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install "torch==2.6.0" --index-url "$TORCH_INDEX"
python -m pip install -r "$REPO/achgnn/requirements.txt" -r "$REPO/zbmath_hetnet/requirements.txt"
# FlashAttention-2 speeds up the Qwen3 encoder; the pipeline falls back to PyTorch SDPA without it.
python -m pip install flash-attn --no-build-isolation || echo "flash-attn not installed; using SDPA attention"

if [ ! -f "$DATASET_CSV" ]; then
    python -m pip install gdown
    mkdir -p "$(dirname "$DATASET_CSV")"
    gdown 1npaipBwA606KAUKmxikjZjvjj3VMQV0Q -O "$DATASET_CSV"
fi

if [ ! -f "$EMBED_MODEL/config.json" ]; then
    mkdir -p "$EMBED_MODEL"
    huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir "$EMBED_MODEL"
fi

python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| transformers", transformers.__version__)
EOF
echo "Environment ready."

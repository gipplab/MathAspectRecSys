#!/bin/bash

set -e  # Exit immediately if any command fails

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Define output paths 
ACHGNN_INDEX="$ROOT_DIR/data/achgnn.index"

RETRIEVED_ACHGNN="$ROOT_DIR/data/retrieved_achgnn.json"

# Ensure output dir exists
mkdir -p "$ROOT_DIR/data"

# Helper function to run a pipeline for one model
run_pipeline() {
    local index_script="$1"
    local index_path="$2"
    local retrieved_path="$3"
    local model_name="$4"

    echo "Running pipeline for $model_name..."

    # Step 1: Build index
    python3 "$index_script" --output "$index_path"

    # Step 2: Retrieve top-k
    python3 "$SRC_DIR/retrieval_/top_k_retrieval.py" --index "$index_path" --output "$retrieved_path"

    # Step 3: Evaluate retrieval
    python3 "$SRC_DIR/retrieval_/top_k_eval.py" --input "$retrieved_path"

    # Step 4: Aspect-wise evaluation
    python3 "$SRC_DIR/retrieval_/aspect_eval.py" --input "$retrieved_path"

    echo "Done: $model_name"
    echo "----------------------------------------"
}

run_pipeline "$SRC_DIR/graph_based/index_AchGNN.py"       "$ACHGNN_INDEX"       "$RETRIEVED_ACHGNN"       "AchGNN"

echo "All scores are reported!"
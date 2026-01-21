#!/bin/bash

set -e  # Exit immediately if any command fails

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"

# Define output paths
SPECTER_INDEX="$SCRIPT_DIR/data/specter.index"
SCIBERT_INDEX="$SCRIPT_DIR/data/scibert.index"
SCIBERT_FT_INDEX="$SCRIPT_DIR/data/scibert_ft.index"
QWEN2_INDEX="$SCRIPT_DIR/data/qwen2_7b.index"
QWEN3_INDEX="$SCRIPT_DIR/data/qwen3_7b.index"
MEMTRON_INDEX="$SCRIPT_DIR/data/memtron.index"
GRAPGHCL_INDEX="$SCRIPT_DIR/data/grapHCL.index"
ACHGNN_INDEX="$SCRIPT_DIR/data/achgnn.index"

RETRIEVED_SPECTER="$SCRIPT_DIR/data/retrieved_specter.json"
RETRIEVED_SCIBERT="$SCRIPT_DIR/data/retrieved_scibert.json"
RETRIEVED_SCIBERT_FT="$SCRIPT_DIR/data/retrieved_scibert_ft.json"
RETRIEVED_QWEN2="$SCRIPT_DIR/data/retrieved_qwen2.json"
RETRIEVED_QWEN3="$SCRIPT_DIR/data/retrieved_qwen3.json"
RETRIEVED_MEMTRON="$SCRIPT_DIR/data/retrieved_memtron.json"
RETRIEVED_GRAPGHCL="$SCRIPT_DIR/data/retrieved_grapHCL.json"
RETRIEVED_ACHGNN="$SCRIPT_DIR/data/retrieved_achgnn.json"

# Ensure output dir exists
mkdir -p "$SCRIPT_DIR/data"

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

# Run all pipelines
run_pipeline "$SRC_DIR/create_indexes_/index_specter.py"      "$SPECTER_INDEX"      "$RETRIEVED_SPECTER"      "SPECTER"
run_pipeline "$SRC_DIR/create_indexes_/index_sciBERT.py"      "$SCIBERT_INDEX"      "$RETRIEVED_SCIBERT"      "sciBERT"
run_pipeline "$SRC_DIR/create_indexes_/index_sciBERT_FT.py"   "$SCIBERT_FT_INDEX"   "$RETRIEVED_SCIBERT_FT"   "sciBERT-FT"
run_pipeline "$SRC_DIR/create_indexes_/index_Qwen2_7B.py"     "$QWEN2_INDEX"        "$RETRIEVED_QWEN2"        "Qwen2-7B"
run_pipeline "$SRC_DIR/create_indexes_/index_Qwen3_7B.py"     "$QWEN3_INDEX"        "$RETRIEVED_QWEN3"        "Qwen3-7B"
run_pipeline "$SRC_DIR/create_indexes_/index_memtron.py"      "$MEMTRON_INDEX"      "$RETRIEVED_MEMTRON"      "Memtron"
run_pipeline "$SRC_DIR/graph_based/index_grapCL.py"      "$GRAPGHCL_INDEX"     "$RETRIEVED_GRAPGHCL"     "grapHCL"
run_pipeline "$SRC_DIR/graph_based/index_AchGNN.py"       "$ACHGNN_INDEX"       "$RETRIEVED_ACHGNN"       "AchGNN"

echo "All scores are reported!"
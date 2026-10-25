#!/bin/bash

# Ablation study runner for ACHGNN
# Hyperparameters:
# - num_authors: [1, 5, 10]
# - num_neighbors: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# - num_layers: [1, 2, 3, 4]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="ablation_results"
mkdir -p "$OUTPUT_DIR"

# Define hyperparameter ranges
AUTHORS=(1 5 10)
NEIGHBORS=(1 5 10 15 20 25 30 35 40 45 50)
LAYERS=(1 2 3 4)

# Total combinations
TOTAL_COMBOS=$(( ${#AUTHORS[@]} * ${#NEIGHBORS[@]} * ${#LAYERS[@]} ))
echo "Total combinations: $TOTAL_COMBOS"
echo "Starting ablation study..."

COUNTER=0
for auth in "${AUTHORS[@]}"; do
    for neigh in "${NEIGHBORS[@]}"; do
        for layer in "${LAYERS[@]}"; do
            COUNTER=$((COUNTER + 1))
            echo "[$COUNTER/$TOTAL_COMBOS] Running: authors=$auth, neighbors=$neigh, layers=$layer"
            
            # Check if result already exists
            RESULT_FILE="$OUTPUT_DIR/ablation_nauth${auth}_nneigh${neigh}_nlayers${layer}.json"
            if [ -f "$RESULT_FILE" ]; then
                echo "  → Already exists, skipping..."
                continue
            fi
            
            # Run the experiment
            python "$SCRIPT_DIR/graph_based/ablated_achGNN.py" \
                --num_authors $auth \
                --num_neighbors $neigh \
                --num_layers $layer \
                --k_folds 5 \
                --output_dir "$OUTPUT_DIR"
            
            echo "  → Completed!"
        done
    done
done

echo "Ablation study completed! Results in $OUTPUT_DIR/"
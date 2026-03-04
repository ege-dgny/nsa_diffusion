#!/bin/bash
# Run all 5 baseline methods sequentially.
# Usage: bash scripts/run_baselines.sh [--num_steps 100000] [--batch_size 64]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NUM_STEPS="${1:-100000}"
BATCH_SIZE="${2:-64}"
RANK_RATIO="${3:-0.25}"

METHODS=("lowrank_kd" "standard_nsa" "nsa_diff" "fitnets" "gramian")

for method in "${METHODS[@]}"; do
    echo "========================================"
    echo "Running method: $method"
    echo "========================================"
    python "$SCRIPT_DIR/train.py" \
        --method "$method" \
        --num_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --rank_ratio "$RANK_RATIO" \
        --run_name "$method" \
        --use_wandb true

    echo "Done: $method"
    echo ""
done

echo "All baselines complete."

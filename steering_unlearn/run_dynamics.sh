#!/bin/bash
# Run script for Controlled Region-to-Attractor Transport Unlearning
# 
# Method: Controlled Region-to-Attractor Transport for LLM Unlearning
# 
# Usage:
#   ./run_dynamics.sh [options]
#
# Options:
#   --force-extract    Force re-extraction of activations
#   --force-retrain    Force retraining of probes and attractors
#   --eval-only        Run evaluation only (requires existing controller)

set -e

# Change to script directory
cd "$(dirname "$0")"

# Default values
FORCE_EXTRACT=false
FORCE_RETRAIN=false
EVAL_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-extract)
            FORCE_EXTRACT=true
            shift
            ;;
        --force-retrain)
            FORCE_RETRAIN=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Controlled Region-to-Attractor Transport"
echo "for LLM Unlearning"
echo "=============================================="
echo ""
echo "Options:"
echo "  Force Extract: $FORCE_EXTRACT"
echo "  Force Retrain: $FORCE_RETRAIN"
echo "  Eval Only: $EVAL_ONLY"
echo ""

# Set environment variables
export HF_ENDPOINT='https://hf-mirror.com'
export CUDA_VISIBLE_DEVICES=0

# Run the main script
if [ "$EVAL_ONLY" = true ]; then
    echo "Running evaluation only..."
    python run_dynamics_unlearn.py \
        +force_retrain=false \
        +eval_only=true
else
    echo "Running full pipeline..."
    python run_dynamics_unlearn.py \
        +force_retrain=$FORCE_RETRAIN
fi

echo ""
echo "=============================================="
echo "Pipeline completed!"
echo "=============================================="
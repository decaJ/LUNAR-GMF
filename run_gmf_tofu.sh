#!/bin/bash
# Gated Manifold Flow for LLM Unlearning
# Run script for GMF unlearning on TOFU dataset

# Author: GMF Research Team

# Use GPU 4 (most free memory)
export CUDA_VISIBLE_DEVICES=4

# Activate the conda environment if needed
# source activate lunar  # Uncomment if using conda

# Default configuration
CONFIG_FILE="config/forget_gmf_tofu.yaml"

# Model path (use the TOFU fine-tuned model)
MODEL_PATH="models_finetune/tofu_llama2_7b"

# Output directory
OUTPUT_DIR="outputs/gmf_tofu"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Gated Manifold Flow for LLM Unlearning"
echo "=========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo ""

# Run GMF unlearning
python run_gmf.py \
    --config-name forget_gmf_tofu \
    model_path=$MODEL_PATH \
    save_path=$OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/gmf_run.log

# Save the return code
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "GMF Unlearning completed successfully!"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "GMF Unlearning failed with exit code: $EXIT_CODE"
        echo "=========================================="
    fi
    
    exit $EXIT_CODE
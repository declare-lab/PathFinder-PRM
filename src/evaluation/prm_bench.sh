#!/bin/bash
set -euo pipefail

# Set config file paths
ACCELERATE_CONFIG="./PRMBench/mr_eval/scripts/examples/accelerate_configs/deepspeed_run_1.yaml"
MODEL_CONFIG="./PRMBench/mr_eval/scripts/configs/disc_prm_eval.yaml"

# Set environment variables
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG}"
export OMP_NUM_THREADS=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

# HF cache dirs
export HF_DATASETS_CACHE="/tmp/cache"
export HF_HOME="/tmp/transformers_cache"
export PYTHONUNBUFFERED=1
export PRM_DEBUG=1

echo "Using accelerate config: ${ACCELERATE_CONFIG}"
echo "Using model config: ${MODEL_CONFIG}"

accelerate launch --config_file ${ACCELERATE_CONFIG} \
    -m mr_eval --config ${MODEL_CONFIG}

echo "Evaluation complete!"
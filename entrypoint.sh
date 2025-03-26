#!/bin/bash

# Set environment variables to avoid TensorFlow AVX/SSE issues
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0

# Disable JIT compilation which can cause issues on ARM emulation
export XLA_FLAGS="--xla_cpu_enable_fast_math=false"

if [ "$1" = "test" ]; then
    echo "Running tests..."
    python run_tests.py
elif [ "$1" = "train" ]; then
    echo "Starting training..."
    python train.py
else
    echo "Invalid command. Use 'test' or 'train'"
    exit 1
fi

#!/bin/bash
# Rank-based budget ablation (5 cache rates, 20 images)
# Usage: bash scripts/ablation.sh [gpu_id]

GPU="${1:-3}"
cd "$(dirname "$0")/.."

/opt/conda/envs/fast3d/bin/python scripts/ablation.py \
    --factors 8.0,11.2,14.0,16.8,19.6 \
    --max_images 20 \
    --gpu "$GPU"

#!/bin/bash
set -euo pipefail

# Offline-only single-image inference.
# Usage: bash scripts/inference.sh [image_path] [output_name] [cache_ratio]

IMAGE="${1:-assets/example_image/typical_creature_elephant.png}"
NAME="${2:-tensor_cache_output}"
CACHE_RATIO="${3:-0.5}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -m tensor_cache_impl.example \
  --use_tensor_cache \
  --image_path "$IMAGE" \
  --output_name "$NAME" \
  --tensor_cache_budget_mode ratio \
  --tensor_cache_target_ratio "$CACHE_RATIO"

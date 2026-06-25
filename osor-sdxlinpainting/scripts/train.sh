#!/usr/bin/env bash
# OSOR-SDXL training launcher.
# Usage:
#   bash scripts/train.sh 1     # Phase I (hard-blend pretraining, 15k steps)
#   bash scripts/train.sh 2     # Phase II (alpha-aware, 5k steps, needs Phase I ckpt)
#
# Hardware assumption: 8 GPUs. Phase config uses batch_size=2 and grad_accum=1,
# giving a global batch of 16. Adjust NPROC_PER_NODE if you have a different GPU count
# (and re-scale batch_size in the config to keep the global batch near 16).

set -euo pipefail

PHASE="${1:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

case "$PHASE" in
  1) CONFIG=configs/phase1.yaml ;;
  2) CONFIG=configs/phase2.yaml ;;
  *) echo "Usage: bash scripts/train.sh <1|2>"; exit 1 ;;
esac

# Resolve repo root (one level up from scripts/) so the script works from anywhere.
cd "$(dirname "$0")/.."

echo "=== OSOR-SDXL Phase $PHASE ==="
echo "Config: $CONFIG"
echo "GPUs:   $NPROC_PER_NODE"
echo

# Pre-compute the fixed prompt embeddings on first run; the trainers load this file
# instead of the heavy text encoders. Safe to skip if it already exists.
PROMPT_CACHE="pretrained_weights/fixed_prompt_embeds.pt"
if [ ! -f "$PROMPT_CACHE" ]; then
  echo "$PROMPT_CACHE not found; generating it via scripts/cache_prompts.py..."
  python3 scripts/cache_prompts.py --config "$CONFIG"
fi

accelerate launch \
  --num_processes "$NPROC_PER_NODE" \
  --multi_gpu \
  scripts/train.py --config "$CONFIG"

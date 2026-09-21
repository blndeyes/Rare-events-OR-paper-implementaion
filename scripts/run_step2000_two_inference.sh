#!/usr/bin/env bash
# Generate either the two-sample MMOR or 4D-OR step-2000 evaluation set.

set -uo pipefail

DATASET="${1:-}"
REPO="${REPO_ROOT:-$PWD}"
TRAINER=/scratch/irtaza/upstreams/LTX-Video-Trainer
PYTHON="$TRAINER/.venv/bin/python"
RUN=/scratch/irtaza/or-repro-artifacts/reduced-30train-6heldout-2000step
CHECKPOINT="$RUN/training/checkpoints/lora_weights_step_02000.safetensors"

case "$DATASET" in
  mmor)
    OUT="$RUN/evaluation/mmor-two"
    SPLIT=heldout
    ;;
  4dor)
    OUT="$RUN/evaluation/4dor-two"
    SPLIT=table1_ood
    ;;
  *)
    echo "Usage: $0 {mmor|4dor}" >&2
    exit 2
    ;;
esac

PAIR_MANIFEST="$OUT/pair-manifest.json"
LOG="$OUT/inference-step2000.log"

if [[ ! -d "$REPO/src/or_video_reproduction" ]]; then
  echo "Launch this script from the Rare-events-OR-paper-implementaion repository" >&2
  echo "or set REPO_ROOT to its absolute path." >&2
  exit 1
fi

for REQUIRED in "$PYTHON" "$CHECKPOINT" "$PAIR_MANIFEST"; do
  if [[ ! -e "$REQUIRED" ]]; then
    echo "Missing required input: $REQUIRED" >&2
    exit 1
  fi
done

if [[ -e "$OUT/inference-manifest.json" ]]; then
  echo "Refusing to overwrite: $OUT/inference-manifest.json" >&2
  exit 1
fi

if find "$OUT/samples" -maxdepth 1 -type f -name '*.mp4' -print -quit 2>/dev/null |
  grep -q .; then
  echo "Refusing to overwrite existing generated videos in $OUT/samples" >&2
  exit 1
fi

export HF_HOME=/scratch/irtaza/huggingface
export HF_HUB_CACHE=/scratch/irtaza/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/scratch/irtaza/huggingface/hub
export HF_ASSETS_CACHE=/scratch/irtaza/huggingface/assets
export TRANSFORMERS_CACHE=/scratch/irtaza/huggingface/transformers
export TORCH_HOME=/scratch/irtaza/huggingface/torch
export UV_CACHE_DIR=/scratch/irtaza/uv-cache
export XDG_CACHE_HOME=/scratch/irtaza/cache
export CUDA_CACHE_PATH=/scratch/irtaza/cache/nvidia/ComputeCache
export TRITON_CACHE_DIR=/scratch/irtaza/cache/triton
export MPLCONFIGDIR=/scratch/irtaza/cache/matplotlib
export TMPDIR=/scratch/irtaza/tmp
export WANDB_DIR=/scratch/irtaza/wandb
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO/src:$TRAINER/src"

mkdir -p \
  "$HF_HUB_CACHE" \
  "$HF_ASSETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$CUDA_CACHE_PATH" \
  "$TRITON_CACHE_DIR" \
  "$MPLCONFIGDIR" \
  "$TMPDIR" \
  "$WANDB_DIR"

cd "$REPO" || exit 1

"$PYTHON" -c 'import or_video_reproduction, ltxv_trainer' || exit 1

"$PYTHON" -m or_video_reproduction.evaluation.corrected_inference \
  --trainer-root "$TRAINER" \
  --pair-manifest "$PAIR_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT" \
  --prompt "Fixed overhead surveillance view of an operating room." \
  --seed 42 \
  --split "$SPLIT" \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
echo "${DATASET} step-2000 inference exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"

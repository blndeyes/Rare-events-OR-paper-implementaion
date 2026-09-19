#!/usr/bin/env bash
set -euo pipefail

# Sequential six-case edited-only single-ellipse evaluation at step 2000.
# Run only on irtazastone after checking out the exact pushed commit.
# Do not generate original-conditioned companion videos.
# Do not run cases concurrently.

REPO_ROOT="${REPO_ROOT:-/home/irtaza/Rare-events-OR-paper-implementaion}"
RUN_ROOT="${RUN_ROOT:-/scratch/irtaza/or-repro-artifacts/reduced-30train-6heldout-2000step}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/evaluation/ellipse-control-single-six}"
MMOR_PAIR_MANIFEST="${MMOR_PAIR_MANIFEST:-/scratch/irtaza/or-repro-artifacts/reduced-30train-6heldout-600step/geometry/pair-manifest.json}"
FOURDOR_PAIR_MANIFEST="${FOURDOR_PAIR_MANIFEST:-/scratch/irtaza/or-repro-artifacts/4dor-table1-six/geometry/pair-manifest.json}"
TRAINER_ROOT="${TRAINER_ROOT:-/scratch/irtaza/upstreams/LTX-Video-Trainer}"
SAM2_ROOT="${SAM2_ROOT:-/scratch/irtaza/upstreams/sam2}"
REPO_PYTHON="${REPO_PYTHON:-/home/irtaza/.venv/bin/python}"
TRAINER_PYTHON="${TRAINER_PYTHON:-/scratch/irtaza/upstreams/LTX-Video-Trainer/.venv/bin/python}"
SAM2_PYTHON="${SAM2_PYTHON:-$TRAINER_PYTHON}"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/training/checkpoints/lora_weights_step_02000.safetensors}"
TRAINING_REPORT="${TRAINING_REPORT:-$RUN_ROOT/training-report.json}"
EXPECTED_SHA256="${EXPECTED_SHA256:-e0b7118a8a41196df84688925181480ea4e50afddbd02be370ec0f8af78c1ad9}"
PROMPT="${PROMPT:-Fixed overhead surveillance view of an operating room.}"
SEED="${SEED:-42}"
LOG_ROOT="$OUTPUT_ROOT/logs"

export HF_HOME=/scratch/irtaza/huggingface
export TORCH_HOME=/scratch/irtaza/huggingface/torch
export TFHUB_CACHE_DIR=/scratch/irtaza/huggingface/tfhub
export UV_CACHE_DIR=/scratch/irtaza/uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE

if [[ ! -f "$TRAINING_REPORT" ]]; then
  for candidate in \
    "$RUN_ROOT/training-report.json" \
    "$RUN_ROOT/training/training-report.json" \
    "$RUN_ROOT/training/report.json" \
    "$RUN_ROOT/training/audit.json"; do
    if [[ -f "$candidate" ]]; then
      TRAINING_REPORT="$candidate"
      break
    fi
  done
fi

for required in \
  "$REPO_PYTHON" \
  "$TRAINER_PYTHON" \
  "$SAM2_PYTHON" \
  "$MMOR_PAIR_MANIFEST" \
  "$FOURDOR_PAIR_MANIFEST" \
  "$CHECKPOINT" \
  "$TRAINING_REPORT" \
  "$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT_ROOT" ]] && [[ -n "$(ls -A "$OUTPUT_ROOT" 2>/dev/null || true)" ]]; then
  echo "Refusing to overwrite non-empty experiment directory: $OUTPUT_ROOT" >&2
  PYTHONPATH="$REPO_ROOT/src" "$REPO_PYTHON" \
    -m or_video_reproduction.evaluation.ellipse_control_suite inventory --output-dir "$OUTPUT_ROOT" \
    | tee "$OUTPUT_ROOT/inventory.json" || true
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cp "$REPO_ROOT/scripts/run_ellipse_control_single_six.sh" "$OUTPUT_ROOT/reproduce-all.sh"
git -C "$REPO_ROOT" rev-parse HEAD | tee "$OUTPUT_ROOT/repository-commit.txt"
git -C "$TRAINER_ROOT" rev-parse HEAD > "$OUTPUT_ROOT/trainer-commit.txt"
date --iso-8601=seconds | tee "$OUTPUT_ROOT/launch-timestamp.txt"
hostname | tee "$OUTPUT_ROOT/hostname.txt"
echo "tmux_session=${TMUX_SESSION_NAME:-unspecified}" | tee "$OUTPUT_ROOT/tmux-metadata.txt"
nvidia-smi | tee "$LOG_ROOT/nvidia-smi-before.log"
df -h /scratch | tee "$LOG_ROOT/disk-before.log"

PYTHONPATH="$REPO_ROOT/src" "$REPO_PYTHON" \
  -c 'from pathlib import Path
from or_video_reproduction.evaluation.checkpoint_audit import audited_checkpoint_step
import json, sys
result = audited_checkpoint_step(
    training_report=Path(sys.argv[1]),
    checkpoint=Path(sys.argv[2]),
    expected_step=2000,
    expected_sha256=sys.argv[3],
)
Path(sys.argv[4]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))' \
  "$TRAINING_REPORT" "$CHECKPOINT" "$EXPECTED_SHA256" "$OUTPUT_ROOT/checkpoint-audit.json" \
  | tee "$LOG_ROOT/checkpoint-audit.log"

PYTHONPATH="$REPO_ROOT/src" "$REPO_PYTHON" \
  -m or_video_reproduction.evaluation.ellipse_control_suite prepare-suite \
  --mmor-pair-manifest "$MMOR_PAIR_MANIFEST" \
  --four-dor-pair-manifest "$FOURDOR_PAIR_MANIFEST" \
  --output-dir "$OUTPUT_ROOT" \
  --seed "$SEED" \
  | tee "$LOG_ROOT/prepare-suite.log"

mapfile -t CASE_DIRS < <(
  PYTHONPATH="$REPO_ROOT/src" "$REPO_PYTHON" -c 'import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload["cases"]:
    print(row["case_dir"])' "$OUTPUT_ROOT/experiment_manifest.json"
)

if [[ "${#CASE_DIRS[@]}" -ne 6 ]]; then
  echo "Expected six prepared cases, found ${#CASE_DIRS[@]}" >&2
  exit 2
fi

for CASE_DIR in "${CASE_DIRS[@]}"; do
  CASE_ID="$(basename "$CASE_DIR")"
  echo "===== $CASE_ID =====" | tee -a "$LOG_ROOT/generation.log"
  SOURCE_JSON="$CASE_DIR/source.json"
  CLIP_ID="$("$REPO_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["clip_id"])' "$SOURCE_JSON")"
  PAIR_MANIFEST="$("$REPO_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["pair_manifest"])' "$SOURCE_JSON")"
  CASE_LOG_DIR="$CASE_DIR/logs"
  mkdir -p "$CASE_LOG_DIR"

  PYTHONPATH="$REPO_ROOT/src:$TRAINER_ROOT/src" "$TRAINER_PYTHON" \
    -m or_video_reproduction.evaluation.trajectory_experiment \
    --trainer-root "$TRAINER_ROOT" \
    --source-pair-manifest "$PAIR_MANIFEST" \
    --clip-id "$CLIP_ID" \
    --edited-conditioning "$CASE_DIR/edited-control.mp4" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$CASE_DIR/inference-run" \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --edited-only \
    --expected-checkpoint-step 2000 \
    --training-report "$TRAINING_REPORT" \
    --expected-sha256 "$EXPECTED_SHA256" \
    2>&1 | tee "$CASE_LOG_DIR/generation.log" | tee -a "$LOG_ROOT/generation.log"

  GENERATED="$("$REPO_PYTHON" -c 'import json,sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload["samples"]
if len(rows) != 1:
    raise SystemExit(f"expected exactly one edited generation, got {len(rows)}")
print(rows[0]["generated_video"])' "$CASE_DIR/inference-run/inference/inference-manifest.json")"
  cp "$GENERATED" "$CASE_DIR/edited-generated.mp4"
  cp "$CASE_DIR/inference-run/inference/effective-inference-config.yaml" "$CASE_DIR/inference-config.yaml" || true
  cp "$CASE_DIR/inference-run/inference/first-frames/"*.png "$CASE_DIR/initial-rgb.png"

  PYTHONPATH="$REPO_ROOT/src:$SAM2_ROOT" "$SAM2_PYTHON" \
    -m or_video_reproduction.evaluation.trajectory_control \
    --edited-only \
    --edited-generated "$CASE_DIR/edited-generated.mp4" \
    --sam2-root "$SAM2_ROOT" \
    --sam2-checkpoint "$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt" \
    --edit-manifest "$CASE_DIR/requested-trajectory.json" \
    --edited-metadata "$CASE_DIR/edit/metadata.json" \
    --original-conditioning "$CASE_DIR/original-control.mp4" \
    --edited-conditioning "$CASE_DIR/edited-control.mp4" \
    --output-dir "$CASE_DIR" \
    2>&1 | tee "$CASE_LOG_DIR/evaluation.log" | tee -a "$LOG_ROOT/evaluation.log"
done

PYTHONPATH="$REPO_ROOT/src" "$REPO_PYTHON" \
  -m or_video_reproduction.evaluation.ellipse_control_suite aggregate \
  --output-dir "$OUTPUT_ROOT" \
  | tee "$LOG_ROOT/aggregate.log"

nvidia-smi | tee "$LOG_ROOT/nvidia-smi-after.log"
df -h /scratch | tee "$LOG_ROOT/disk-after.log"
echo "Completed edited-only six-case experiment: $OUTPUT_ROOT"
echo "Summary: $OUTPUT_ROOT/results-summary.md"

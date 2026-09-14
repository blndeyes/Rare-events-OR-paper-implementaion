#!/usr/bin/env bash
set -euo pipefail

# Reproduce the step-600 original-versus-edited trajectory control experiment.
# Run this only on irtazastone after pulling the implementation commit.

REPO_ROOT="${REPO_ROOT:-/home/irtaza/Rare-events-OR-paper-implementaion}"
RUN_ROOT="${RUN_ROOT:-/scratch/irtaza/or-repro-artifacts/reduced-30train-6heldout-600step}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/irtaza/or-repro-artifacts/trajectory-control-step600}"
TRAINER_ROOT="${TRAINER_ROOT:-/scratch/irtaza/upstreams/LTX-Video-Trainer}"
SAM2_ROOT="${SAM2_ROOT:-/scratch/irtaza/upstreams/sam2}"
REPO_PYTHON="${REPO_PYTHON:-/home/irtaza/.venv/bin/python}"
TRAINER_PYTHON="${TRAINER_PYTHON:-/scratch/irtaza/upstreams/LTX-Video-Trainer/.venv/bin/python}"
CLIP_ID="${CLIP_ID:-mmor-001_PKA-camera01-timestamp000210}"
SELECTED_INSTANCE="${SELECTED_INSTANCE:-12:nurse}"
TARGET_INSTANCE="${TARGET_INSTANCE:-1:instrument_table}"
PROMPT="${PROMPT:-Fixed overhead surveillance view of an operating room.}"
SEED="${SEED:-42}"

PAIR_MANIFEST="$RUN_ROOT/geometry/pair-manifest.json"
SOURCE_ROOT="$RUN_ROOT/geometry/$CLIP_ID"
SOURCE_METADATA="$SOURCE_ROOT/conditioning/metadata.json"
SOURCE_CONDITIONING="$SOURCE_ROOT/conditioning/conditioning.mp4"
SOURCE_LABELS="$SOURCE_ROOT/labels.npz"
CHECKPOINT="$RUN_ROOT/training/checkpoints/lora_weights_step_00600.safetensors"
TRAJECTORY_INPUT="$OUTPUT_ROOT/trajectory-input.json"
EDIT_ROOT="$OUTPUT_ROOT/edit"
LOG_ROOT="$OUTPUT_ROOT/logs"

export HF_HOME=/scratch/irtaza/huggingface
export TORCH_HOME=/scratch/irtaza/huggingface/torch
export TFHUB_CACHE_DIR=/scratch/irtaza/huggingface/tfhub
export UV_CACHE_DIR=/scratch/irtaza/uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE

for required in \
  "$REPO_PYTHON" \
  "$TRAINER_PYTHON" \
  "$PAIR_MANIFEST" \
  "$SOURCE_METADATA" \
  "$SOURCE_CONDITIONING" \
  "$SOURCE_LABELS" \
  "$CHECKPOINT" \
  "$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT_ROOT/experiment-manifest.json" || -e "$OUTPUT_ROOT/evaluation/control-metrics.json" ]]; then
  echo "Refusing to mix a new run with completed artifacts in $OUTPUT_ROOT" >&2
  echo "Move the old directory or set OUTPUT_ROOT to a new path." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$EDIT_ROOT" "$LOG_ROOT"
cp "$REPO_ROOT/scripts/run_trajectory_control_step600.sh" "$OUTPUT_ROOT/reproduce-all.sh"
git -C "$REPO_ROOT" rev-parse HEAD > "$OUTPUT_ROOT/repository-commit.txt"
git -C "$TRAINER_ROOT" rev-parse HEAD > "$OUTPUT_ROOT/trainer-commit.txt"
nvidia-smi | tee "$LOG_ROOT/nvidia-smi.log"
df -h /scratch | tee "$LOG_ROOT/disk-before.log"

PYTHONPATH="$REPO_ROOT/src" "$REPO_PYTHON" \
  -c 'import json, math, sys
from pathlib import Path

metadata_path, output_path, selected_key, target_key = sys.argv[1:]
metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

def centroid(frame_index, key):
    rows = [row for row in metadata["frames"][frame_index]["instances"] if row["key"] == key]
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one {key!r} in frame {frame_index}, found {len(rows)}")
    ellipse = rows[0]["ellipse"]
    return float(ellipse["center_x"]), float(ellipse["center_y"])

start = centroid(0, selected_key)
table_end = centroid(96, target_key)
delta = (table_end[0] - start[0], table_end[1] - start[1])
end = (start[0] + 0.85 * delta[0], start[1] + 0.85 * delta[1])
mid = (start[0] + 0.45 * delta[0], start[1] + 0.45 * delta[1] - 12.0)
displacement = math.dist(start, end)
if not 150.0 <= displacement <= 250.0:
    raise SystemExit(
        f"Planned displacement {displacement:.3f}px is outside the predeclared 150-250px range"
    )
payload = {
    "schema_version": 1,
    "mode": "replace",
    "simplify_epsilon_pixels": 2.0,
    "waypoints": [list(start), list(mid), list(end)],
    "construction": {
        "selected_instance": selected_key,
        "target_instance": target_key,
        "target_fraction": 0.85,
        "midpoint_vertical_offset_pixels": -12.0,
        "commanded_endpoint_displacement_pixels": displacement,
        "assumption": "replacement polyline toward the target table; exact author semantics undisclosed",
    },
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))' \
  "$SOURCE_METADATA" "$TRAJECTORY_INPUT" "$SELECTED_INSTANCE" "$TARGET_INSTANCE" \
  | tee "$LOG_ROOT/trajectory-input.log"

PYTHONPATH="$REPO_ROOT/src" "$REPO_PYTHON" \
  -m or_video_reproduction.geometry.trajectory \
  --metadata "$SOURCE_METADATA" \
  --source-conditioning "$SOURCE_CONDITIONING" \
  --trajectory "$TRAJECTORY_INPUT" \
  --instance-id "$SELECTED_INSTANCE" \
  --output-dir "$EDIT_ROOT" \
  --seed "$SEED" \
  | tee "$LOG_ROOT/render-edit.log"

PYTHONPATH="$REPO_ROOT/src:$TRAINER_ROOT/src" "$TRAINER_PYTHON" \
  -m or_video_reproduction.evaluation.trajectory_experiment \
  --trainer-root "$TRAINER_ROOT" \
  --source-pair-manifest "$PAIR_MANIFEST" \
  --clip-id "$CLIP_ID" \
  --edited-conditioning "$EDIT_ROOT/conditioning-edited.mp4" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT" \
  --prompt "$PROMPT" \
  --seed "$SEED" \
  2>&1 | tee "$LOG_ROOT/inference.log"

mapfile -t GENERATED_VIDEOS < <(
  "$REPO_PYTHON" -c 'import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
by_id = {row["id"].rsplit("-", 1)[-1]: row["generated_video"] for row in payload["samples"]}
print(by_id["original"])
print(by_id["edited"])' "$OUTPUT_ROOT/inference/inference-manifest.json"
)

PYTHONPATH="$REPO_ROOT/src:$SAM2_ROOT" "$REPO_PYTHON" \
  -m or_video_reproduction.evaluation.trajectory_control \
  --original-generated "${GENERATED_VIDEOS[0]}" \
  --edited-generated "${GENERATED_VIDEOS[1]}" \
  --source-labels "$SOURCE_LABELS" \
  --sam2-root "$SAM2_ROOT" \
  --sam2-checkpoint "$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt" \
  --edit-manifest "$EDIT_ROOT/trajectory-edit-manifest.json" \
  --edited-metadata "$EDIT_ROOT/metadata.json" \
  --original-conditioning "$OUTPUT_ROOT/conditioning/original.mp4" \
  --edited-conditioning "$OUTPUT_ROOT/conditioning/edited.mp4" \
  --output-dir "$OUTPUT_ROOT/evaluation" \
  2>&1 | tee "$LOG_ROOT/evaluation.log"

df -h /scratch | tee "$LOG_ROOT/disk-after.log"
echo "Completed trajectory-control experiment: $OUTPUT_ROOT"
echo "Metrics: $OUTPUT_ROOT/evaluation/control-metrics.json"
echo "Contact sheet: $OUTPUT_ROOT/evaluation/synchronized-contact-sheet.png"

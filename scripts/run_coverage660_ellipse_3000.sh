#!/usr/bin/env bash
# Stone only. Invoke inside one uniquely named tmux session after training completes.
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set the staged Stone bundle root}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set a new Stone output root}"
CHECKPOINT="${CHECKPOINT:?set the audited step-3000 generator LoRA path}"
PYTHON="${PYTHON:-/scratch/irtaza/upstreams/LTX-Video-Trainer/.venv/bin/python}"
INFER_SCRIPT="${INFER_SCRIPT:-/scratch/irtaza/or_synth_native_video_20260921_v1/infer_validate.py}"

[[ "$BUNDLE_ROOT" == /scratch/irtaza/* && "$OUTPUT_ROOT" == /scratch/irtaza/* ]] || {
  echo 'Bundle and output must be under /scratch/irtaza' >&2; exit 2;
}
[[ "$CHECKPOINT" == *lora_weights_step_03000.safetensors ]] || {
  echo 'Refusing a non-step-3000 LoRA' >&2; exit 2;
}
[[ -f "$CHECKPOINT" && -f "$INFER_SCRIPT" && -f "$BUNDLE_ROOT/input-audit.json" ]] || {
  echo 'Checkpoint, inference script, or bundle audit missing' >&2; exit 2;
}
[[ -f "$BUNDLE_ROOT/jobs/g42.json" && -f "$BUNDLE_ROOT/jobs/g43.json" ]] || {
  echo 'Expected both frozen seed job files' >&2; exit 2;
}
"$PYTHON" - "$BUNDLE_ROOT" "$OUTPUT_ROOT" "$CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path
from safetensors import safe_open

bundle, output, checkpoint = map(Path, sys.argv[1:])
audit = json.loads((bundle / "input-audit.json").read_text())
if audit["stone_root"] != str(bundle) or not audit["staged"]:
    raise SystemExit("Bundle was not staged for this Stone root")
if audit["counts"] != {"challenge": 72, "dev": 18}:
    raise SystemExit("Wrong request counts")
if checkpoint.stat().st_size < 1_000_000_000:
    raise SystemExit("Step-3000 generator LoRA is implausibly small")
with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    rank_shapes = [handle.get_slice(key).get_shape() for key in keys if "lora_A" in key]
    if not rank_shapes or any(len(shape) != 2 or shape[0] != 128 for shape in rank_shapes):
        raise SystemExit("Checkpoint is not an audited rank-128 generator LoRA")
    if not any("lora_B" in key for key in keys):
        raise SystemExit("Checkpoint has no generator LoRA B tensors")
print(f"generator_lora_tensors={len(keys)} rank=128 bytes={checkpoint.stat().st_size}")
for seed, expected in ((42, 54), (43, 36)):
    jobs = json.loads((bundle / "jobs" / f"g{seed}.json").read_text())
    if len(jobs) != expected or any(not Path(j["reference"]).is_file() or not Path(j["target"]).is_file() for j in jobs):
        raise SystemExit(f"Missing or wrong g{seed} inputs")
if output.exists() and any(output.iterdir()):
    for child in output.iterdir():
        if child.name not in {"g42", "g43", "checkpoint.sha256"}:
            raise SystemExit(f"Unexpected existing output: {child}")
PY
(cd "$BUNDLE_ROOT" && sha256sum -c sha256sums.txt >/dev/null)
sha256sum "$CHECKPOINT"
mkdir -p "$OUTPUT_ROOT"
checkpoint_hash="$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)"
if [[ -f "$OUTPUT_ROOT/checkpoint.sha256" ]]; then
  recorded_hash="$(cut -d ' ' -f 1 "$OUTPUT_ROOT/checkpoint.sha256")"
  [[ "$recorded_hash" == "$checkpoint_hash" ]] || {
    echo 'Existing output belongs to another checkpoint' >&2; exit 2;
  }
else
  sha256sum "$CHECKPOINT" > "$OUTPUT_ROOT/checkpoint.sha256"
fi

for seed in 42 43; do
  run_dir="$OUTPUT_ROOT/g$seed"
  if [[ -d "$run_dir" ]]; then
    "$PYTHON" - "$run_dir/run.json" "$seed" <<'PY'
import json
import sys
from pathlib import Path

path, seed = Path(sys.argv[1]), int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"Existing run has no manifest: {path}")
state = json.loads(path.read_text())
expected = 54 if seed == 42 else 36
if state.get("state") != "complete" or len(state.get("results", [])) != expected:
    raise SystemExit(f"Existing g{seed} run is incomplete; preserve and inspect it")
PY
    echo "g$seed already complete; preserving it"
    continue
  fi
  "$PYTHON" "$INFER_SCRIPT" \
    --jobs "$BUNDLE_ROOT/jobs/g$seed.json" \
    --out "$run_dir" \
    --lora "$CHECKPOINT" \
    --steps 50 --guidance 3.5 --seed "$seed"
done

echo "Both seed groups completed. Copy outputs directly to irtazapc, then score."

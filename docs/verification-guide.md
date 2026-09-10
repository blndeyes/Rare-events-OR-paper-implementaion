# Verification checkpoints

Do not start the 8,000-step training run yet. Verification is staged so that a
bad dataset assumption or geometric encoding cannot consume hours of GPU time.

## Run now: repository and dataset audit

On `IRTAZAPC`, update the repository and run:

```bash
cd /home/irtaza/Rare-events-OR-paper-implementaion
git pull --ff-only
OR_PYTHON=/home/irtaza/anaconda3/bin/python
PYTHONPATH=src "$OR_PYTHON" -m unittest discover -s tests -v
PYTHONPATH=src "$OR_PYTHON" -m or_video_reproduction.data.semantics
PYTHONPATH=src "$OR_PYTHON" -m or_video_reproduction.data.inventory mmor \
  --root /home/irtaza/dump/or-datasets/MM-OR_processed \
  --output /tmp/mmor-inventory.json \
  --report /tmp/mmor-inventory.md
PYTHONPATH=src "$OR_PYTHON" -m or_video_reproduction.data.inventory 4dor \
  --root /home/irtaza/dump/or-datasets/4D-OR_full \
  --output /tmp/4dor-inventory.json \
  --report /tmp/4dor-inventory.md
cat /tmp/mmor-inventory.md
cat /tmp/4dor-inventory.md
```

The tests must pass, and the semantic command must report 21 entities, 15
predicates, and 36 labels. Inventory reports should agree with the committed
dataset audit. Differences are a stop condition, not something to train through.

Build the paper-shaped smoke clip manifest without copying dataset frames:

```bash
OR_PYTHON=/home/irtaza/anaconda3/bin/python
PYTHONPATH=src "$OR_PYTHON" -m or_video_reproduction.data.clips \
  --root /home/irtaza/dump/or-datasets/MM-OR_processed \
  --take 001_PKA --camera 1 --start-timestamp 0 --split smoke \
  --output /tmp/mmor-smoke-clip.json
```

The output must contain five distinct RGB keyframes assigned to generated-frame
indices 0, 24, 48, 72, and 96, plus an existing first-frame ground-truth mask.

Create an auditable LTX interpolation plan before executing it:

```bash
PYTHONPATH=src "$OR_PYTHON" -m or_video_reproduction.preprocessing.ltx_interpolation \
  --manifest /tmp/mmor-smoke-clip.json \
  --dataset-root /home/irtaza/dump/or-datasets/MM-OR_processed \
  --ltx-root /home/irtaza/Work/or-reproduction-upstreams/ltx-video \
  --python /home/irtaza/Work/or-reproduction-envs/ltx097/bin/python \
  --output-dir /tmp/mmor-ltx-interpolation \
  --prompt "Fixed overhead surveillance view of an operating room." \
  --seed 42 --plan-output /tmp/mmor-ltx-interpolation-plan.json
```

The prompt and seed are explicit smoke-test hypotheses, not values disclosed by the
paper. Add `--execute` only after reviewing the plan and confirming model weights are
available; the launcher then uses the pinned official 0.9.7-dev pipeline with CPU
offload. The BF16 checkpoint cannot load on the audited 24 GiB RTX 4090 (23.48 GiB
was already in use when a further 32 MiB allocation failed). For this host, add
`--precision fp8` to select the official 0.9.7-dev FP8 checkpoint. This is an explicit
hardware accommodation, not a paper-disclosed precision choice; it leaves the model
version, five-keyframe contract, resolution, frame count, and frame rate unchanged.
The pinned legacy launcher also moves every component to CUDA before its own offload
logic can run. Apply `patches/ltx-video-0.9.7-sequential-offload.patch` to the clean,
pinned LTX checkout. The patch activates the pipeline's already-declared sequential
offload order only when `--offload_to_cpu` is set; non-offloaded behavior is unchanged.

## Run now that geometry has landed: inspect actual pictures

Create a one-frame plumbing preview from MMOR camera 1 (replace `FRAME` with an
annotated frame such as `000329`):

```bash
OR_PYTHON=/home/irtaza/anaconda3/bin/python
MMOR_SAMPLE_ROOT=/home/irtaza/dump/or-datasets/MM-OR_processed/001_PKA
FRAME=000329
PYTHONPATH=src "$OR_PYTHON" -m or_video_reproduction.geometry.preview \
  --rgb "$MMOR_SAMPLE_ROOT/colorimage/camera01_colorimage-$FRAME.jpg" \
  --mask "$MMOR_SAMPLE_ROOT/segmentation_export_1/camera01_colorimage-$FRAME.png" \
  --depth "$MMOR_SAMPLE_ROOT/depthimage/camera01_depthimage-$FRAME.tiff" \
  --output-dir /tmp/or-geometry-preview
```

`conditioning.png` is the model input: a black canvas containing only filled
ellipses. Inspect `comparison.png` and `overlay.png` only as diagnostics for mask
alignment; they must never enter the diffusion pipeline. This smoke
test deliberately uses MMOR sensor depth to validate alignment and rendering; it
does **not** replace Video Depth Anything in the reproduction. Reject the output
if ellipses badly mismatch masks, class colors change, depth ordering is reversed,
or any predicate appears as a standalone ellipse. A multi-frame temporal preview
will be the next gate after the one-frame output passes.

After the one-frame preview passes, render five consecutive 1-fps annotations:

```bash
OR_PYTHON=/home/irtaza/anaconda3/bin/python
PYTHONPATH=src "$OR_PYTHON" -m or_video_reproduction.geometry.sequence_preview \
  --root /home/irtaza/dump/or-datasets/MM-OR_processed \
  --procedure 001_PKA --camera 1 --start-frame 329 --count 5 \
  --output-dir /tmp/or-geometry-sequence
```

Inspect `conditioning.gif` and `conditioning_contact_sheet.png` as the actual
conditioning representation. `overlay.gif` and
`diagnostic_overlay_contact_sheet.png` exist only to verify alignment against RGB.
`sequence_metadata.json` reports per-class center, axis, and angle changes between
neighboring annotations. These are diagnostic values rather than hard pass/fail
thresholds because the paper does not specify temporal smoothing and 1-fps motion
can legitimately be large.

## Run after model integration: cheap GPU checks

In order: one-batch forward/backward, inference from one real conditioning clip,
then a tiny-subset overfit. Check that loss is finite, LoRA parameters receive
gradients, frozen base weights do not, and the overfit output visibly responds to
geometry. Only then enable the optional PatchGAN path.

## Run last: full experiment

Start the 8,000-step training only after the clip manifest, geometry rendering,
one-batch test, and tiny-overfit gate all pass. Freeze the twelve evaluation
videos before using FVD, SSIM, PSNR, or LPIPS.

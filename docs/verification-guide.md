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

Inspect `/tmp/or-geometry-preview/comparison.png` and `metadata.json`. This smoke
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

Inspect `overlay.gif`, `conditioning.gif`, and `temporal_contact_sheet.png`.
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

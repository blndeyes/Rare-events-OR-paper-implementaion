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
pinned LTX checkout, followed by
`patches/ltx-video-0.9.7-explicit-component-offload.patch`. The patches retain the
launcher's intended explicit sequence—text encoder, conditioning VAE, transformer,
decoding VAE—while ensuring each component is moved to CUDA only for its own phase.
Non-offloaded behavior is unchanged. The factory patch also routes the official
`float8_e4m3fn` precision through the pipeline's existing BF16 autocast block.
Without it, the first attention projection receives BF16 activations and FP8 weights
and fails before denoising step one.
Apply `patches/ltx-video-0.9.7-latent-normalization-device.patch` as the final host
patch. After first-pass offload, the multi-scale upsampler otherwise combines CUDA
latents with VAE normalization buffers on CPU. The patch copies only those small
statistics to the latent device and does not alter their values.
The launcher sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless the caller
already chose an allocator configuration. The component patch also offloads the
spatial upsampler immediately after use. Together these reclaim the approximately
1.3 GiB needed by the full-resolution second-pass feed-forward activation.

The `001_PKA` camera-1 smoke clip now passes this gate: 97 decoded frames, 24 fps,
1024x768, five visibly aligned conditioning anchors, anchor PSNR 28.82--32.94 dB,
and anchor SSIM 0.9038--0.9517. Do not run 338 separate launcher processes; the
production preprocessing runner must keep the loaded pipeline alive across clips,
resume from validated outputs, and record per-clip failures and timings.
For FP8 multi-keyframe interpolation, also apply
`patches/ltx-video-0.9.7-disable-multikeyframe-prompt-enhancement.patch`. The upstream
pipeline reports that enhancement is unsupported with multiple conditioning items and
returns the original prompt, but otherwise retains the unused enhancer models on GPU.
Disabling that dead path preserves the effective prompt and frees about 10.7 GiB.

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

The one-batch runner is `or-run-ic-lora-one-step`. It consumes data already encoded by
the pinned official trainer and writes a machine-readable report even when CUDA runs
out of memory. On the audited RTX 4090, unchanged BF16 cannot place the base model;
INT8 and INT4 also exhaust memory during the step. The following integration-only
fallback completed, including an AdamW update and LoRA checkpoint save:

```bash
TRAINER_ROOT=/home/irtaza/Work/or-reproduction-upstreams/LTX-Video-Trainer
TRAINER_PYTHON="$TRAINER_ROOT/.venv/bin/python"
PYTHONPATH="src:$TRAINER_ROOT/src" "$TRAINER_PYTHON" \
  -m or_video_reproduction.training.one_step \
  --paper-config configs/paper_table1.yaml \
  --trainer-root "$TRAINER_ROOT" \
  --precomputed-root /tmp/mmor-ic-lora-paired/.precomputed \
  --output-dir /tmp/mmor-ic-lora-one-step-int2 \
  --report /tmp/mmor-ic-lora-one-step-int2-report.json \
  --profile rtx4090_integration_int2
```

INT2 is a hardware plumbing test, not the paper-faithful training configuration. Do
not start the 8,000-step run with it and report the result as a faithful reproduction.
On suitable hardware, switch the same gate to the paper-compatible configuration with
`--profile faithful_bf16`. The named profiles live in
`configs/training_profiles.yaml` and are reusable by the full training launcher; the
faithful profile is validated to reject quantization and non-BF16 dtypes.

## Run last: full experiment

Start the 8,000-step training only after the clip manifest, geometry rendering,
one-batch test, and tiny-overfit gate all pass. Freeze the twelve evaluation
videos before using FVD, SSIM, PSNR, or LPIPS.

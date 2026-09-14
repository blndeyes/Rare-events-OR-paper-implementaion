# Interactive ellipse trajectory conditioning

This module implements the counterfactual conditioning interface described in
Section 3 of arXiv:2602.21365. It consumes the existing 97-frame
`conditioning/metadata.json`, edits one stable instance key, and renders another
1024×768, 24-fps, 97-frame ellipse-only MP4.

## Explicit reproduction assumptions

The paper says freehand waypoints are interpolated over the complete video and
applied as centroid translations, but does not disclose simplification, temporal
parameterization, endpoint handling, or clipping semantics. This implementation
therefore records the following local convention in every edit manifest:

- Coordinates are image `(x, y)` pixels with a top-left origin, `+x` right, and
  `+y` down.
- Ramer-Douglas-Peucker simplification uses a default 2-pixel tolerance.
- The simplified polyline is sampled at constant Euclidean arc-length speed at
  exactly 97 inclusive samples. It is piecewise linear, not a spline.
- The initial click may occur anywhere inside the selected ellipse, but frame zero
  is anchored to the original ellipse centroid. This preserves the initial
  conditioning scene used with the same first RGB frame.
- Default `replace` mode makes the selected centroid follow the sampled user path.
  Optional `offset` mode adds the sampled displacement to the original centroid
  track.
- Width, height, rotation, class channels, depth, and all non-selected instances
  remain unchanged. If needed, the selected centroid is clamped so the complete
  rotated ellipse stays on-canvas. An ellipse too large to fit is uniformly scaled,
  and every such adjustment is recorded.

These are reproducible engineering choices, not claims of author-equivalent
semantics.

## Reproducible CLI

A trajectory file has this schema:

```json
{
  "schema_version": 1,
  "mode": "replace",
  "simplify_epsilon_pixels": 2.0,
  "waypoints": [
    {"x": 230.0, "y": 310.0},
    {"x": 330.0, "y": 350.0},
    {"x": 430.0, "y": 390.0}
  ]
}
```

The first point must lie inside the selected frame-zero ellipse. Exact selectable
IDs and frame-zero geometry can be inspected first:

```bash
or-list-ellipse-instances \
  --metadata /path/to/conditioning/metadata.json
```

This reports each stable instance ID, class, exact centroid, dimensions, rotation,
and depth, so a remote trajectory can start at the selected centroid without
estimating it from pixels. Then render by stable instance key:

```bash
PYTHONPATH=src python -m or_video_reproduction.geometry.trajectory \
  --metadata /path/to/conditioning/metadata.json \
  --source-conditioning /path/to/conditioning/conditioning.mp4 \
  --trajectory trajectory.json \
  --instance-id '12:circulating_nurse' \
  --output-dir /path/to/edit \
  --seed 42
```

Selection by click is also deterministic:

```bash
... --click 230 310
```

If ellipses overlap, the visible/topmost ellipse under the click is selected using
the renderer's depth order. Outputs are:

- `conditioning-edited.mp4`
- `conditioning-edited-contact-sheet.png`
- `metadata.json`
- `trajectory-edit-manifest.json`

## Interactive GUI

Install the optional GUI dependencies and launch:

```bash
python -m pip install -e '.[gui]'
or-edit-ellipse-trajectory-gui \
  --metadata /path/to/conditioning/metadata.json \
  --source-conditioning /path/to/conditioning/conditioning.mp4 \
  --output-dir /path/to/edit
```

The GUI uses Tkinter file dialogs, a Pygame event/display loop, and OpenCV drawing.
Click an ellipse and drag the freehand path. Press Enter to render, `R` to redraw,
or Escape to cancel. Supplying `--instance-id` preselects the entity.

## Step-600 control evaluation

Generate the original and edited samples in one corrected-inference invocation with
the same target path repeated twice and only `conditioning_video` changed. Keep the
prompt, seed, guidance scale, 50 denoising steps, checkpoint, first RGB frame, and
all other settings identical. Then propagate the same first-frame mask through both
generated videos:

```bash
PYTHONPATH=src python -m or_video_reproduction.evaluation.trajectory_control \
  --original-generated /path/to/original-generated.mp4 \
  --edited-generated /path/to/edited-generated.mp4 \
  --source-labels /path/to/source-labels.npz \
  --sam2-root /scratch/irtaza/upstreams/sam2 \
  --sam2-checkpoint /scratch/irtaza/upstreams/sam2/checkpoints/sam2.1_hiera_large.pt \
  --edit-manifest /path/to/edit/trajectory-edit-manifest.json \
  --edited-metadata /path/to/edit/metadata.json \
  --original-conditioning /path/to/conditioning.mp4 \
  --edited-conditioning /path/to/edit/conditioning-edited.mp4 \
  --output-dir /path/to/evaluation
```

The primary measurements are selected-entity centroid distance, endpoint error,
movement vector/direction/displacement, and counterfactual movement relative to the
original-conditioned output. Bounding-box and mask IoU against the conditioning
ellipse are reported only as abstraction-alignment proxies. Non-selected SAM2 tracks
and generated pixels outside the selected masks measure scene stability. PSNR/SSIM
against the original ground-truth motion are intentionally not used as the primary
controllability test.

`--first-mask /path/to/first-frame-mask.png` may be used instead. With
`--source-labels`, the evaluator recovers frame zero from the existing SAM2 label
archive and saves the exact prompt image alongside the metrics.

For the frozen step-600 MMOR control experiment on `irtazastone`, the checked-in
`scripts/run_trajectory_control_step600.sh` performs the render, paired inference,
and SAM2 evaluation without retraining. Its defaults select
`mmor-001_PKA-camera01-timestamp000210`, move `12:nurse` 85% of the centroid-to-
`1:instrument_table` vector (and require that displacement to be 150–250 pixels),
and write the required artifacts under
`/scratch/irtaza/or-repro-artifacts/trajectory-control-step600/`.
The runner uses the lightweight repository Python for rendering and defaults
`SAM2_PYTHON` to the PyTorch-enabled trainer environment; either interpreter may
be overridden explicitly without changing the recorded experiment settings.

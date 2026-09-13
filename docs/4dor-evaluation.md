# 4D-OR out-of-domain evaluation

The target paper trains the LoRA on MMOR and reports Table 1 results on six
4D-OR videos. It does not disclose the six clip identities, camera choice,
manual SAM2 prompts, or interpolation settings. This adapter therefore records
the selection and all annotations as reproduction hypotheses.

The paper explicitly reports FVD, SSIM, PSNR, and LPIPS for the six 4D-OR
videos. BB IoU and Seg IoU are available as secondary structural diagnostics,
but they are not columns in the paper's 4D-OR Table 1 comparison.

## 1. Create the deterministic six-clip hypothesis

The default uses official 4D-OR test takes 2 and 6, camera 1, non-overlapping
five-row windows, and seed 42. The five-keyframe construction is inferred from
the disclosed 1 fps source and 97-frame/24-fps output; it is not stated directly.

```bash
cd ~/Rare-events-OR-paper-implementaion
RUN=/scratch/irtaza/or-repro-artifacts/4dor-table1-six
PYTHON=/scratch/irtaza/upstreams/LTX-Video-Trainer/.venv/bin/python

PYTHONPATH=src "$PYTHON" -m or_video_reproduction.data.four_dor \
  --root /scratch/irtaza/4D-OR_full \
  --takes 2,6 --cameras 1 --stride-rows 5 --count 6 --seed 42 \
  --output-dir "$RUN/split" \
  --accept-undisclosed-selection-hypothesis
```

This writes `batch.json`, six clip manifests, and six incomplete manual prompt
templates. It does not alter the dataset.

## 2. Complete every point-prompt template

Open the first source image named in each template under `$RUN/split/prompts`.
Populate `instances` using normalized coordinates. Each physical instance gets
a unique ID even if two instances share the same semantic class:

```json
{
  "object_id": 1,
  "four_dor_class": "head_surgeon",
  "mmor_class": "head_surgeon",
  "points": [[0.48, 0.42], [0.20, 0.15]],
  "point_labels": [1, 0]
}
```

Positive points use label 1 and negative correction points use label 0. Generic
`human` and `object` labels are deliberately rejected; assign a supported role
or entity explicitly. The accepted mapping is versioned in
`configs/4dor_semantic_mapping.yaml`.

## 3. Interpolate and render the conditioning videos

Run the existing persistent LTX batch, then the shared geometry batch. The
geometry batch detects the point manifests automatically and reuses one SAM2
and one Video Depth Anything model:

```bash
PYTHONPATH=src "$PYTHON" -m or_video_reproduction.preprocessing.ltx_batch \
  --batch-manifest "$RUN/split/batch.json" \
  --dataset-root /scratch/irtaza/4D-OR_full \
  --ltx-root /scratch/irtaza/upstreams/ltx-video \
  --output-root "$RUN/ltx-targets" \
  --prompt "Fixed overhead surveillance view of an operating room." \
  --seed 42 --precision bf16 --reload-pipeline-after-each-clip --fail-fast

PYTHONPATH=src "$PYTHON" -m or_video_reproduction.preprocessing.geometry_batch \
  --batch-manifest "$RUN/split/batch.json" \
  --ltx-output-root "$RUN/ltx-targets" \
  --dataset-root /scratch/irtaza/4D-OR_full \
  --output-root "$RUN/geometry" \
  --vda-root /scratch/irtaza/upstreams/Video-Depth-Anything \
  --vda-checkpoint /scratch/irtaza/models/video_depth_anything_vitl.pth \
  --sam2-root /scratch/irtaza/upstreams/sam2 \
  --sam2-checkpoint /scratch/irtaza/models/sam2.1_hiera_large.pt \
  --fail-fast
```

Confirm the actual checkpoint filenames before running. A completed geometry
batch produces `$RUN/geometry/pair-manifest.json`.

## 4. Generate with the MMOR-trained LoRA

Use corrected inference with both the 4D-OR target's first RGB frame and the
complete ellipse video. Pass `--split table1_ood`. The LoRA must be the MMOR
checkpoint; do not train on 4D-OR.

```bash
PYTHONPATH=src "$PYTHON" -m or_video_reproduction.evaluation.corrected_inference \
  --trainer-root /scratch/irtaza/upstreams/LTX-Video-Trainer \
  --pair-manifest "$RUN/geometry/pair-manifest.json" \
  --checkpoint /path/to/mmor_step_000600.safetensors \
  --output-dir "$RUN/inference" \
  --prompt "Fixed overhead surveillance view of an operating room." \
  --seed 42 --split table1_ood
```

## 5. Track generated instances and evaluate

The first command applies the exact same manual prompts to generated videos,
preserving object IDs for mask and bounding-box comparison. The second computes
the paper's Table 1 metrics:

```bash
PYTHONPATH=src "$PYTHON" -m or_video_reproduction.evaluation.four_dor \
  --inference-manifest "$RUN/inference/inference-manifest.json" \
  --pair-manifest "$RUN/geometry/pair-manifest.json" \
  --output-root "$RUN/evaluation" \
  --sam2-root /scratch/irtaza/upstreams/sam2 \
  --sam2-checkpoint /scratch/irtaza/models/sam2.1_hiera_large.pt

PYTHONPATH=src "$PYTHON" -m or_video_reproduction.evaluation.runner \
  --manifest "$RUN/evaluation/evaluation-manifest.json" \
  --output "$RUN/evaluation/table1-metrics.json" \
  --metrics psnr,ssim,lpips,fvd \
  --google-research-root /scratch/irtaza/upstreams/google-research \
  --device cuda --batch-size 16
```

Run `--metrics box_iou,mask_iou` separately if secondary structural diagnostics
are desired. Do not present those as reported 4D-OR Table 1 columns.

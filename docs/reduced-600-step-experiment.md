# Reduced 30-video / 600-step experiment

This run is a **reproduction hypothesis**, not the paper's 338-video / 8,000-step
result. The paper does not disclose clip identities, camera selection, prompt, seed,
or split construction. This protocol fixes those choices for repeatability and keeps
PatchGAN disabled until the baseline and corrected inference pass.

## 1. Safe remote preflight

On `irtazastone`, inspect storage and both worktrees before changing anything:

```bash
df -h /scratch /scratch/irtaza
du -sh /scratch/irtaza/or-repro-artifacts/* 2>/dev/null | sort -h

cd /home/irtaza/Rare-events-OR-paper-implementaion
git status --short --branch
git rev-parse HEAD

cd /scratch/irtaza/upstreams/LTX-Video-Trainer
git status --short --branch
git rev-parse HEAD
```

Do not delete anything based only on age or size. The corrupt 20-video step-400 file
is known obsolete, but remove it only after resolving its exact path and confirming
that it is the `322,174,976`-byte file. If the repository worktree is clean, update it:

```bash
cd /home/irtaza/Rare-events-OR-paper-implementaion
git pull --ff-only
```

If it is not clean, preserve the changes and inspect them before pulling.

## 2. Freeze a take-disjoint split

The following hypothesis uses camera 1, five-second candidate stride, seed 42, thirty
distinct training takes, and six distinct held-out takes. The selector fails if a
logical take appears in both sets.

```bash
RUN=/scratch/irtaza/or-repro-artifacts/reduced-30train-6heldout-600step
REPO=/home/irtaza/Rare-events-OR-paper-implementaion
PYTHON=/home/irtaza/.venv/bin/python

cd "$REPO"
PYTHONPATH=src "$PYTHON" -m or_video_reproduction.data.selection \
  --root /scratch/irtaza/MM-OR_processed \
  --takes all --cameras 1 --stride-seconds 5 --seed 42 \
  --train-count 30 --ablation-count 6 --split-mode take-disjoint \
  --output-dir "$RUN/split" \
  --accept-undisclosed-selection-hypothesis
```

Review `$RUN/split/batch.json`. It must report an empty `policy.take_overlap`, thirty
training rows, and six held-out rows.

## 3. Resume preprocessing and validate all pairs

Use the existing persistent LTX and geometry runners. Run the LTX batch without a
clip limit only after the first one or two clips have passed on this exact split.

```bash
export HF_HOME=/scratch/irtaza/huggingface
export TORCH_HOME=/scratch/irtaza/huggingface/torch
export UV_CACHE_DIR=/scratch/irtaza/uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=src "$PYTHON" -m or_video_reproduction.preprocessing.ltx_batch \
  --batch-manifest "$RUN/split/batch.json" \
  --dataset-root /scratch/irtaza/MM-OR_processed \
  --ltx-root /scratch/irtaza/upstreams/ltx-video \
  --output-root "$RUN/ltx-targets" \
  --prompt "Fixed overhead surveillance view of an operating room." \
  --seed 42 --precision bf16 --reload-pipeline-after-each-clip --fail-fast

PYTHONPATH=src "$PYTHON" -m or_video_reproduction.preprocessing.geometry_batch \
  --batch-manifest "$RUN/split/batch.json" \
  --ltx-output-root "$RUN/ltx-targets" \
  --dataset-root /scratch/irtaza/MM-OR_processed \
  --output-root "$RUN/geometry" \
  --vda-root /scratch/irtaza/upstreams/Video-Depth-Anything \
  --vda-checkpoint /scratch/irtaza/upstreams/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth \
  --sam2-root /scratch/irtaza/upstreams/sam2 \
  --sam2-checkpoint /scratch/irtaza/upstreams/sam2/checkpoints/sam2.1_hiera_large.pt \
  --fail-fast

PYTHONPATH=src "$PYTHON" -m or_video_reproduction.preprocessing.validate_pairs \
  --pair-manifest "$RUN/geometry/pair-manifest.json" \
  --output "$RUN/geometry/pair-validation.json"
```

The optional SAM2 CUDA-extension warning is an environment fallback; propagation must
still produce all 97 frames. Pair validation must report `36 passed, 0 failed`.

## 4. Precompute official trainer inputs

The fixed caption is an undisclosed-choice hypothesis. The dataset builder refuses a
train/held-out take overlap and emits only the thirty training pairs. It stages
hardlinks beneath `trainer-data` because the pinned official preprocessor rejects
media paths outside the directory containing `dataset.json`; hardlinks avoid copying
the videos.

```bash
TRAINER=/scratch/irtaza/upstreams/LTX-Video-Trainer
TRAINER_PYTHON="$TRAINER/.venv/bin/python"
mkdir -p "$RUN/trainer-data"

PYTHONPATH=src "$TRAINER_PYTHON" \
  -m or_video_reproduction.preprocessing.trainer_dataset \
  --pair-manifest "$RUN/geometry/pair-manifest.json" \
  --output "$RUN/trainer-data/dataset.json" \
  --caption "Fixed overhead surveillance view of an operating room."

"$TRAINER_PYTHON" "$TRAINER/scripts/preprocess_dataset.py" \
  "$RUN/trainer-data/dataset.json" \
  --resolution-buckets "1024x768x97" \
  --caption-column caption --video-column media_path \
  --reference-column reference_path --model-source LTXV_13B_097_DEV
```

Before training, count and open the condition, target-latent, and reference-latent
files. Each group must contain thirty readable files.

## 5. Launch fresh training under tmux

The launcher forces fresh LoRA initialization, BF16, no quantization, rank/alpha 128,
600 steps, learning rate `2e-4`, batch size 1, no PatchGAN, and checkpoint retention at
every 100 steps. It requires at least 15 GiB free and fsyncs every optimizer-step loss
and learning-rate row to `loss-history.csv`.

```bash
cd "$REPO"
tmux new-session -d -s or-reduced-600 \
  "bash -lc 'set -o pipefail; export HF_HOME=/scratch/irtaza/huggingface TORCH_HOME=/scratch/irtaza/huggingface/torch UV_CACHE_DIR=/scratch/irtaza/uv-cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PYTHONPATH=src:$TRAINER/src $TRAINER_PYTHON -m or_video_reproduction.training.reduced_run --trainer-root $TRAINER --precomputed-root $RUN/trainer-data/.precomputed --output-dir $RUN/training --report $RUN/training-report.json 2>&1 | tee $RUN/training.log'"

tmux attach -t or-reduced-600
```

The final report passes only if CSV rows are exactly steps 1 through 600 and all six
checkpoints are at least 1 GB and readable through `safetensors`.

## 6. Corrected held-out inference and metrics

First apply the small validation correction to the pinned trainer. It fixes the
width/height check and exports generated frames only, rather than a side-by-side
reference comparison. Check applicability before changing the upstream worktree:

```bash
cd "$TRAINER"
git apply --check "$REPO/patches/ltx-trainer-correct-validation-conditioning.patch"
git apply "$REPO/patches/ltx-trainer-correct-validation-conditioning.patch"
```

Then generate each checkpoint with the same prompt, seed, first RGB frame, and ellipse
video. Example for step 600:

```bash
cd "$REPO"
PYTHONPATH="src:$TRAINER/src" "$TRAINER_PYTHON" \
  -m or_video_reproduction.evaluation.corrected_inference \
  --trainer-root "$TRAINER" \
  --pair-manifest "$RUN/geometry/pair-manifest.json" \
  --checkpoint "$RUN/training/checkpoints/lora_weights_step_00600.safetensors" \
  --output-dir "$RUN/mmor-step600-corrected" \
  --prompt "Fixed overhead surveillance view of an operating room." --seed 42

PYTHONPATH=src "$PYTHON" -m or_video_reproduction.evaluation.runner \
  --manifest "$RUN/mmor-step600-corrected/inference-manifest.json" \
  --output "$RUN/mmor-step600-corrected/paired-metrics.json" \
  --metrics psnr,ssim,lpips --device cuda --batch-size 8
```

Repeat without changing ordering for steps 100 through 500. Treat checkpoint choice on
these held-out clips as model selection, not final Table 1 evaluation. Shifted-ellipse
tests come only after these corrected baseline outputs pass the video contract.

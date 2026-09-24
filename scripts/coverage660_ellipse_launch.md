# Coverage660 ellipse step-3000 launch (not yet started)

Use only after the dual-host PatchGAN run reaches **exactly step 3000**, the
generator LoRA is audited, and Stone's GPU is free. The frozen source is
`/home/irtaza/dump/gray/eval_coverage660_v1/requests/requests_v1.json` on
irtazapc. This run has 18 development reconstructions and 72 challenge requests;
the six base requests are excluded. It uses the ellipse controls, not mask controls.

The local repository must first be committed and pushed through GitHub, then
pulled at that exact commit on both hosts after training exits. Do not change
the code checkout while the trainer is running. Never copy code or media through
Windows. Use one unique root per attempt; never overwrite partial outputs.

## 1. Checkpoint and machines

On Stone, verify that the generator and worker have exited, no other GPU job
is using the card, and the following *generator-side* adapter exists:

```bash
CHECKPOINT=/scratch/irtaza/or-ellipse-coverage660-ce506f0-20260923T1405Z/generator-v2/checkpoints/lora_weights_step_03000.safetensors
test -f "$CHECKPOINT" && sha256sum "$CHECKPOINT"
nvidia-smi
```

Do not treat the 2000-step adapter, discriminator state, or a partial 3000-step
file as final. Record training/code commit, LoRA rank/alpha, BF16 precision,
base model, tensor count, checkpoint byte size/hash, and final log step.

## 2. Stage inputs on irtazapc

Choose a new identifier and use it consistently. This command audits every
frozen source hash, writes exactly 54 seed-42 and 36 seed-43 jobs, and copies
only the needed inputs to a new PC bundle. It refuses to overwrite.

```bash
RUN_ID=competitor-ellipse-s3000-001
PC_BUNDLE=/home/irtaza/or-coverage660-$RUN_ID/bundle
STONE_BUNDLE=/scratch/irtaza/or-coverage660-$RUN_ID/bundle
cd /home/irtaza/Rare-events-OR-paper-implementaion
python3 scripts/prepare_coverage660_ellipse.py \
  --requests /home/irtaza/dump/gray/eval_coverage660_v1/requests/requests_v1.json \
  --freeze /home/irtaza/dump/gray/eval_coverage660_v1/challenge/FREEZE.json \
  --output "$PC_BUNDLE" --stone-root "$STONE_BUNDLE" --stage
```

Copy the immutable input bundle **directly PC → Stone**, then verify its hashes
on Stone. Never route it through Windows. Check Stone free disk first.

```bash
RUN_ID=competitor-ellipse-s3000-001
ssh irtazastone 'df -h /scratch/irtaza'
ssh irtazastone "mkdir -p /scratch/irtaza/or-coverage660-$RUN_ID"
rsync -a --partial "$PC_BUNDLE/" "irtazastone:$STONE_BUNDLE/"
ssh irtazastone "cd '$STONE_BUNDLE' && sha256sum -c sha256sums.txt"
```

## 3. Generate on Stone

After checking `nvidia-smi` and existing tmux sessions, launch **one** session.
The script runs seed 42 then seed 43 sequentially with BF16 13B LTX, the
audited LoRA, 50 steps, guidance 3.5, 97 frames at 1024×768 and 24 fps.
The existing `infer_validate.py` emits lossless `libx264rgb` CRF-0 videos and
records hashes/configuration. It refuses a busy GPU (<40 GiB free). The launch
script refuses a different checkpoint or partial output on resume.

```bash
RUN_ID=competitor-ellipse-s3000-001
cd /scratch/irtaza/dualhost-patchgan-code-272c163
tmux new-session -d -s "coverage660-ellipse-$RUN_ID" \
  "BUNDLE_ROOT=/scratch/irtaza/or-coverage660-$RUN_ID/bundle OUTPUT_ROOT=/scratch/irtaza/or-coverage660-$RUN_ID/outputs CHECKPOINT=/scratch/irtaza/or-ellipse-coverage660-ce506f0-20260923T1405Z/generator-v2/checkpoints/lora_weights_step_03000.safetensors bash scripts/run_coverage660_ellipse_3000.sh > /scratch/irtaza/or-coverage660-$RUN_ID/generation.log 2>&1"
```

Inspect `generation.log` and each `outputs/g{42,43}/run.json`; require states
`complete` and result counts 54/36. Do not regenerate a poor-looking output.

## 4. Score on irtazapc

After verifying all 90 lossless videos, copy the outputs **directly Stone → PC**,
preserving paths and hashes. On irtazapc:

```bash
RUN_ID=competitor-ellipse-s3000-001
PC_ROOT=/home/irtaza/or-coverage660-$RUN_ID
rsync -a --partial "irtazastone:/scratch/irtaza/or-coverage660-$RUN_ID/outputs/" "$PC_ROOT/outputs/"
rsync -a "irtazastone:/scratch/irtaza/or-ellipse-coverage660-ce506f0-20260923T1405Z/generator-v2/checkpoints/lora_weights_step_03000.safetensors" "$PC_ROOT/checkpoint.safetensors"
cd /home/irtaza/dump/gray/eval_coverage660_v1
source env.sh
$EVAL_PY -m harness.make_submission \
  --requests requests/requests_v1.json --method competitor_ellipse \
  --sets challenge,dev --root "$PC_ROOT/outputs" \
  --layout 'g{gen_seed}/{key_dir}/generated_lossless.mp4' \
  --representation ellipse_depth --adapter "$PC_ROOT/checkpoint.safetensors" \
  --out "$PC_ROOT/submission.json"
$EVAL_PY -m harness.score \
  --requests requests/requests_v1.json --submission "$PC_ROOT/submission.json" \
  --out "$PC_ROOT/scoring" --stages recon,dover,sam2,iou,follow,marks,setlevel \
  --device cuda --workers 24
$EVAL_PY -m harness.report \
  --requests requests/requests_v1.json \
  --runs runs/ours_A_s3000 runs/ours_B_s3000 "$PC_ROOT/scoring" \
  --out "$PC_ROOT/report"
```

The supervisor harness reports its own FVD implementation. For paper-method
comparability, compute the repository's separately pinned Google TF I3D FVD
on the **18 development reconstructions** as an additional result, and label
the two implementations separately. The 72 authored challenge cases are not
paired reconstructions of the real footage; do not invent SSIM/PSNR/LPIPS
against a counterfactual target. The paper itself reports FVD, SSIM, PSNR,
LPIPS, DOVER, IS, Seg IoU and BB IoU; the handover specifies the matching
implementations and aggregation.

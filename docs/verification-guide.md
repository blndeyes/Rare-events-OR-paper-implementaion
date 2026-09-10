# Verification checkpoints

Do not start the 8,000-step training run yet. Verification is staged so that a
bad dataset assumption or geometric encoding cannot consume hours of GPU time.

## Run now: repository and dataset audit

On `IRTAZAPC`, update the repository and run:

```bash
cd /home/irtaza/Rare-events-OR-paper-implementaion
git pull --ff-only
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m or_video_reproduction.data.semantics
PYTHONPATH=src python3 -m or_video_reproduction.data.inventory mmor \
  --root /home/irtaza/dump/or-datasets/MM-OR_processed \
  --output /tmp/mmor-inventory.json \
  --report /tmp/mmor-inventory.md
PYTHONPATH=src python3 -m or_video_reproduction.data.inventory 4dor \
  --root /home/irtaza/dump/or-datasets/4D-OR_full \
  --output /tmp/4dor-inventory.json \
  --report /tmp/4dor-inventory.md
cat /tmp/mmor-inventory.md
cat /tmp/4dor-inventory.md
```

The tests must pass, and the semantic command must report 21 entities, 15
predicates, and 36 labels. Inventory reports should agree with the committed
dataset audit. Differences are a stop condition, not something to train through.

## Run after geometry lands: inspect actual pictures

The next user-visible checkpoint is one short MMOR clip. Verify the fitted
ellipses over the source frames, then inspect the rendered red/green class and
blue depth channels. Reject the output if ellipses jump between frames, class
colors change, depth ordering is reversed, or any predicate appears as a
standalone ellipse. The exact command will land with the geometry module.

## Run after model integration: cheap GPU checks

In order: one-batch forward/backward, inference from one real conditioning clip,
then a tiny-subset overfit. Check that loss is finite, LoRA parameters receive
gradients, frozen base weights do not, and the overfit output visibly responds to
geometry. Only then enable the optional PatchGAN path.

## Run last: full experiment

Start the 8,000-step training only after the clip manifest, geometry rendering,
one-batch test, and tiny-overfit gate all pass. Freeze the twelve evaluation
videos before using FVD, SSIM, PSNR, or LPIPS.

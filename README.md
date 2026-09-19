# Reproducing Controllable OR Video Synthesis

This repository reproduces the **Ours** method from *Towards Controllable Video
Synthesis of Routine and Rare OR Events* (arXiv:2602.21365v1).

The first target is Table 1:

| Dataset | FVD (lower) | SSIM (higher) | PSNR (higher) | LPIPS (lower) |
| --- | ---: | ---: | ---: | ---: |
| MMOR (in-domain) | 689.88 | 0.86 | 23.21 | 0.13 |
| 4DOR (out-of-domain) | 265.25 | 0.90 | 25.87 | 0.07 |

Latest **step-600 reproduction** results:

| Dataset | FVD (lower) | SSIM (higher) | PSNR (higher) | LPIPS (lower) |
| --- | ---: | ---: | ---: | ---: |
| MMOR (in-domain, 6 clips) | 286.02 | 0.8868 | 24.17 | 0.0822 |
| 4DOR (out-of-domain, 6 clips) | 275.39 | 0.8939 | 24.37 | 0.0759 |

These reproduction scores use strict native-resolution frame means for the paired
metrics and the official Google Kinetics-400 I3D at 224x224 for one set-level FVD
score per dataset. These are explicit repository conventions because the paper does
not disclose its exact metric implementations or evaluation clips.

## Current status

Geometry, reduced 600-step training, Table 1 evaluation, trajectory-control tooling,
and the frozen fine-grained evaluation adapters are on `main` (`ccd9231`, including
source commit `b4158f5`). The local test suite is **162 passed**.

**Table 1 (step-600).** Six-clip MMOR and 4D-OR scores above are the current
reproduction numbers. MMOR FVD is far below the paper; 4D-OR is close. These are
not the authors' undisclosed clips or metric code.

**Fine-grained evaluation (code merged; remote extras on `irtazapc`).** One frozen
split, one frame sampler, video-level pairing. MMOR and 4D-OR are never ranked
against each other. WAN take 9 is diagnostic only.

| Layer | Status |
| --- | --- |
| CLIP CMMD / polynomial KID, DINOv2, DINOv3 | Implemented and executed |
| Anatomy + control reaggregation | Clip-scoped detection rates; control is person-only plus independent depth; class consistency is `unavailable` |
| Hands (MediaPipe Tasks `HandLandmarker`) | Ran; 4D-OR has no-hand frames; MMOR support is a single clip |
| FVMD `fvmd==1.0.0` | Ran: 4D-OR 180.92, MMOR 128.10, WAN 149.01 (diagnostic) |
| JEDi `videojedi==1.1.0` + official V-JEPA checkpoints | Provisioned on `irtazapc`; **execution-blocked** (no same-clip multi-method comparison; do not emit an absolute score) |
| VBench-2.0 Human Anatomy | Stopped mid-provision; partial env preserved; not resumed |

Remote inputs, weights, environments, and result directories stay outside Git.

**Still open.** Exact paper clip identities, PatchGAN details, and author metric
implementations remain undisclosed. Full 8,000-step Table 1 training has not been
run.

## Local development

```bash
python -m pip install -e ".[dev]"
pytest
```

Fine-grained evaluation:

```bash
python -m or_video_reproduction.evaluation.finegrained.runner --help
```

## Dataset inventory

The inventory commands read filenames and metadata only; they do not alter the
datasets.

```bash
python -m or_video_reproduction.data.inventory mmor \
  --root /home/irtaza/dump/or-datasets/MM-OR_processed \
  --output /tmp/mmor-inventory.json \
  --report /tmp/mmor-inventory.md

python -m or_video_reproduction.data.inventory 4dor \
  --root /home/irtaza/dump/or-datasets/4D-OR_full \
  --output /tmp/4dor-inventory.json \
  --report /tmp/4dor-inventory.md
```

The reports include sequence gaps, modality counts, JSON validity, and
timestamp-to-frame correspondence. Keep outputs outside the repository because they
contain machine-specific paths and dataset-derived metadata.

Data, model weights, generated media, cached latents, credentials, and local `docs/`
notes must never be committed.

# Reproducing Controllable OR Video Synthesis

This repository reproduces the **Ours** method from *Towards Controllable Video
Synthesis of Routine and Rare OR Events* (arXiv:2602.21365v1).

The first target is Table 1:

| Dataset | FVD (lower) | SSIM (higher) | PSNR (higher) | LPIPS (lower) |
| --- | ---: | ---: | ---: | ---: |
| MMOR (in-domain) | 689.88 | 0.86 | 23.21 | 0.13 |
| 4DOR (out-of-domain) | 265.25 | 0.90 | 25.87 | 0.07 |

## Current status

Dataset/source auditing is complete enough to enter the geometry phase. A
one-frame MMOR conditioning preview and numerical geometry tests are available.
Full preprocessing and training remain gated on the checks described in
[the reproduction plan](docs/reproduction-plan.md).

## Local development

```bash
python -m pip install -e ".[dev]"
pytest
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

Data, model weights, generated media, cached latents, and credentials must never be
committed.

## Project documents

- [Reproduction plan](docs/reproduction-plan.md)
- [Source audit](docs/source-audit.md)
- [Dataset audit](docs/dataset-audit.md)
- [Geometry audit](docs/geometry-audit.md)
- [Ambiguities register](docs/ambiguities.md)
- [Supervisor questions](docs/supervisor-questions.md)
- [Verification checkpoints](docs/verification-guide.md)
- [Agent project brief](docs/agent-prompt.md)

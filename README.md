# Reproducing Controllable OR Video Synthesis

This repository reproduces the **Ours** method from *Towards Controllable Video
Synthesis of Routine and Rare OR Events* (arXiv:2602.21365v1).

The first target is Table 1:

| Dataset | FVD (lower) | SSIM (higher) | PSNR (higher) | LPIPS (lower) |
| --- | ---: | ---: | ---: | ---: |
| MMOR (in-domain) | 689.88 | 0.86 | 23.21 | 0.13 |
| 4DOR (out-of-domain) | 265.25 | 0.90 | 25.87 | 0.07 |

## Current status

Phase 1 is in progress: source audit, dataset inventory, and reproduction-spec
development. Full preprocessing and training are deliberately gated on passing the
small correctness checks described in [the reproduction plan](docs/reproduction-plan.md).

## Local development

```bash
python -m pip install -e ".[dev]"
pytest
```

## Dataset inventory

The inventory command reads filenames and metadata only; it does not alter the
dataset.

```bash
python -m or_video_reproduction.data.inventory mmor \
  --root /home/irtaza/dump/or-datasets/MM-OR_processed \
  --output artifacts/inventory/mmor.json
```

Data, model weights, generated media, cached latents, and credentials must never be
committed.

## Project documents

- [Reproduction plan](docs/reproduction-plan.md)
- [Source audit](docs/source-audit.md)
- [Ambiguities register](docs/ambiguities.md)
- [Agent project brief](docs/agent-prompt.md)

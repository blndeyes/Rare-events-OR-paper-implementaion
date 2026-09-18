# Fine-grained video evaluation

This protocol is a **new** evaluation layer. It does not replace the paper Table 1
FVD/SSIM/PSNR/LPIPS runner. It freezes one shared split, one frame sampler, and
one implementation of each metric so 4D-OR, MMOR, WAN, and later methods remain
comparable only when they actually share that split.

## What is compared

| Group | Clips | Role |
| --- | ---: | --- |
| `4dor_step600` | 6 | distributional, vs 4D-OR references; control fidelity vs ellipse videos |
| `mmor_step600` | 6 | distributional, vs MMOR references |
| `wan_latest_diagnostic` | 1 | diagnostic only until WAN covers the same six 4D-OR identities |

MMOR and 4D-OR are never merged into one score. The current WAN file is a take-9
clip and is **not** a member of the frozen 4D-OR takes 2 and 6.

Pairing uses `inference-manifest.json` clip identities and `step_000600_N`
basenames. Directory alphabetical order is not used.

## Shared frame sampler

`uniform_inclusive_endpoints`: 16 frames from the 97-frame contract, including
frame 0 and frame 96. Real and generated members of a pair use the same indices.
The recorded seed `42` is consumed by KID subsets, DINOv3 coreset initialisation,
and video-level bootstrap, not by the linspace itself.

FVMD and JEDi keep their official temporal protocols and add video-level
uncertainty separately.

## Metrics

- CLIP ViT-L/14@336 (`openai/clip-vit-large-patch14-336`): bicubic warp to
  336×336, CLIP mean/std, `get_image_features`, unbiased RBF MMD, σ=10, ×1000.
- Polynomial KID on the same CLIP embeddings: \(k(x,y)=(x^\top y/d+1)^3\), 100
  subsets of size \(n/2\), \(n\) = embeddings per set.
- DINOv2 Large CLS: Frechet, density, coverage with \(k=5\).
- DINOv3 Large patches: 3×3 average pool, greedy k-center coreset, RBF MMD.
  DINOv2 is never substituted.
- Hands: MediaPipe Tasks `HandLandmarker` (`hand_landmarker.task`) detection
  rates plus CLIP crop embeddings. `mediapipe.solutions.hands` is never used.
  Failures are counted; scores are not 0 or 1 when no hands are detected.
  Pass `--hand-landmarker-model`.
- Anatomy: official VBench-2.0 `evaluate.py` `Human_Anatomy` if `--vbench-root`
  points at a VBench-2.0 checkout (optional `--vbench-python`); DETR person
  detection rates; MPJPE only when same-frame IoU matching is defensible.
  Undetected people do not receive zero MPJPE. VBench-1.0 is rejected.
- Control fidelity: parse ellipse red/green/blue encoding, recover entities from
  generated RGB with an independent detector and Depth Anything V2 (not Video
  Depth Anything). No PSNR/SSIM against the black ellipse canvas.
- Motion: official FVMD via `--fvmd-python -m fvmd` (primary); official JEDi
  ranking-only via `--jedi-python`, `--jedi-model-dir` (`vith16.pth.tar`,
  `ssv2-probe.pth.tar`), and `--jedi-config`. No FVD/CLIP/DINO/I3D substitute.

## Running

Develop and test on the local repository. Execute model-dependent stages on
`irtazapc` only. Do not copy videos or weights to the laptop, and do not write
results into Git.

```bash
PYTHONPATH=src python -m or_video_reproduction.evaluation.finegrained.runner \
  --input-root /home/irtaza/or-metrics-inputs-20260918 \
  --output-root /home/irtaza/or-metrics-results-20260918 \
  --stage preflight \
  --hf-cache /home/irtaza/.cache/huggingface \
  --torch-cache /home/irtaza/.cache/torch

PYTHONPATH=src python -m or_video_reproduction.evaluation.finegrained.runner \
  --input-root /home/irtaza/or-metrics-inputs-20260918 \
  --output-root /home/irtaza/or-metrics-results-20260918 \
  --stage run \
  --device cuda \
  --hand-landmarker-model /path/to/hand_landmarker.task \
  --fvmd-python /path/to/fvmd-venv/bin/python \
  --jedi-python /path/to/jedi-venv/bin/python \
  --jedi-model-dir /path/to/vjepa-checkpoints \
  --jedi-config /path/to/vith16_ssv2_16x2x3.yaml \
  --vbench-root /path/to/VBench/VBench-2.0 \
  --vbench-python /path/to/vbench-venv/bin/python \
  --hf-cache /home/irtaza/.cache/huggingface \
  --torch-cache /home/irtaza/.cache/torch
```

If an official implementation is missing, that metric is recorded as
`unavailable` instead of being replaced.

## Statistical limits

Six videos are a weak distributional sample. Frame embeddings from one clip are
correlated, so confidence intervals resample videos. A one-video WAN score is
diagnostic and must not be ranked against the six-video groups.

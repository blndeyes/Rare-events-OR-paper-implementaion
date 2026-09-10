# Reproduction plan

## Objective

Reproduce the **Ours** row in Table 1 of arXiv:2602.21365v1 on the six-video
MMOR and 4DOR evaluations. Training uses MMOR only; 4DOR is out-of-domain
evaluation data.

## Paper-to-code map

| Paper component | Paper description | Planned implementation | Verification |
| --- | --- | --- | --- |
| Geometric abstraction | SAM2 masks, Video Depth Anything, mask-fitted ellipses | `geometry` and preprocessing modules | synthetic-mask tests and visual overlays |
| Class encoding | 36 MMOR semantic labels encoded in red/green | versioned `paper_36` vocabulary: 21 entities + 15 predicates | count, uniqueness, and round-trip tests |
| Depth encoding | normalized relative depth in blue | configurable mask aggregation and normalization | monotonicity and range tests |
| Conditioning | 97 rendered frames paired with target video | IC-LoRA reference video | paired-shape and alignment checks |
| Diffusion | LTX-Video IC-LoRA | pinned official legacy trainer | one-batch forward/backward test |
| First frame | used in 20% of training; always at inference | trainer conditioning configuration | sampling-frequency test |
| PatchGAN | adversarial loss added to baseline fine-tuning | isolated optional integration | discriminator/generator gradient tests |
| Evaluation | FVD, SSIM, PSNR, LPIPS | one frozen evaluation entry point | identity/degradation sanity tests |

## Stages and gates

1. **Audit:** pin official upstream repositories, historical revisions, model variant,
   and dataset semantics.
2. **Inventory:** produce machine-readable MMOR/4DOR inventories without modifying
   source data.
3. **Geometry:** validate ellipse fitting, palette encoding, depth encoding, and exact
   frame alignment on one clip.
4. **Official baseline:** run the unmodified IC-LoRA path on one tiny paired example.
5. **Integration:** use rendered geometry as reference-video conditioning and overfit a
   tiny subset.
6. **PatchGAN:** integrate behind a configuration flag and compare controlled short
   runs.
7. **Full run:** train 8,000 steps only after all earlier gates pass.
8. **Evaluation:** freeze a twelve-video manifest and compute all four Table 1 metrics.

## Compute arrangement

- Source code: local Git repository, synchronized through the configured GitHub remote.
- Training host: `irtaza@IRTAZAPC`.
- GPU: RTX 4090, 24,564 MiB.
- Host memory: 125 GiB; dataset filesystem has approximately 1.3 TiB free.
- Remote data: approximately 489 GiB MMOR and 153 GiB 4DOR.

The remote MMOR extraction contains 22 top-level procedure directories and five
camera streams with approximately 63,535 frames per camera. These are contiguous
source-frame sequences, not prebuilt 97-frame, 24-fps clips. Reproducing the authors'
338 examples therefore requires a separate, still-undetermined event-clipping and
interpolation recipe.

The first complete inventory is recorded in [the dataset audit](dataset-audit.md), and
the commands to run at each gate are in [the verification guide](verification-guide.md).
Timestamp-to-file integrity passes for both datasets. Exact clip construction and the
paper's undisclosed red/green mapping remain hard gates before bulk preprocessing.
The class-count gate uses the explicit `paper_36` interpretation; predicates remain
relation labels and do not generate ellipse instances.

The paper used one NVIDIA A100 but does not state its memory capacity. Any 4090
memory accommodation must preserve the paper's effective optimization settings and be
recorded with each run.

## Reproducibility rules

- Unknown paper details stay explicit; they are never replaced with unmarked guesses.
- Test-video selection is frozen before final evaluation.
- The final test data is not used for tuning or checkpoint selection.
- Every experiment records configuration, source revision, upstream revision, seed,
  hardware, runtime, metrics, and produced checkpoint.
- Dataset files, weights, credentials, cached latents, and generated videos stay outside
  Git.

## Upstream baseline

The reproducible trainer baseline is the official legacy
`Lightricks/LTX-Video-Trainer` at commit
`e055182fa36dba6f48eb0919aef09d277da30fbd`, the last commit before the paper's
24 February 2026 release. Its `ltxv_13b_ic_lora.yaml` configuration matches all
paper-stated trainer values and identifies `LTXV_13B_097_DEV`. This is compelling
evidence for the 13B 0.9.7 development checkpoint, but remains an inference because
the paper does not state the checkpoint or source revision.

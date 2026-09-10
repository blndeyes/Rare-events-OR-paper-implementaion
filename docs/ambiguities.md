# Reproduction ambiguities

Status values: `open`, `hypothesis`, `resolved`, or `blocked`.

| Item | Status | Current evidence / next action |
| --- | --- | --- |
| LTX model variant and checkpoint | hypothesis | Official `ltxv_13b_ic_lora.yaml` matches every stated trainer value and specifies `LTXV_13B_097_DEV`; the paper itself omits the checkpoint. |
| Trainer revision | resolved | Reproducible snapshot pinned to `e055182fa36dba6f48eb0919aef09d277da30fbd`, the last official legacy-trainer commit before paper release. Exact author commit remains unknown. |
| 338 training clips | open | Five 1-fps keyframes explain 97 frames at 24 fps. Simple next-action grouping yields 336 three-view clips; countdown-aware grouping does not match 338. This close but non-exact result is insufficient to freeze a manifest. |
| Six MMOR and six 4DOR Table 1 clips | open | Not disclosed. Must choose before final evaluation and label as a reproduction deviation. |
| Camera views | open | Paper does not identify cameras. Inspect examples and dataset conventions. |
| 1 fps to 24 fps interpolation | open | Paper cites LTX keyframe interpolation but omits sampling and boundary details. |
| Class palette | hypothesis | Selected `paper_36`: 21 supplementary entity labels plus 15 non-proximity predicates. A deterministic 6×6 red/green lattice is testable but not author-disclosed; predicates cannot create standalone ellipses. |
| Ellipse fit | hypothesis | Filled-mask second moments; diameter is 4×sqrt(covariance eigenvalue), angle is major-axis degrees clockwise from +x in image coordinates. Paper gives no fitting convention. |
| Depth normalization | hypothesis | Mean valid depth inside each instance, min-max over visible instances per frame, with near encoded as blue 255. VDA direction must be verified before preprocessing. |
| PatchGAN | blocked | Architecture, receptive field, inputs, loss, coefficient, and optimization schedule are absent. Escalate questions to the user's supervisor if public evidence is insufficient. |
| Optimizer details | resolved | Pinned trainer passes only LR to PyTorch AdamW: betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`. LinearLR decays 1.0 to 0.1 over all steps with no warmup. These are upstream defaults, not paper facts. |
| Metric implementations | open | Exact FVD backbone/library, spatial preprocessing, temporal sampling, and per-video aggregation are absent. |
| 4090 memory plan | open | The inferred 13B model doubles tokens for IC-LoRA and may exceed 24 GB despite gradient checkpointing. Measure preprocessing and one batch before altering effective optimization. |

## Newly established constraints

- **Clip duration hypothesis:** 97 frames at 24 fps span exactly four seconds, matching
  five consecutive 1-fps MMOR keyframes at output indices 0, 24, 48, 72, and 96. This
  is strong arithmetic evidence, not a disclosed author recipe.
- **Selected class-count interpretation:** the `paper_36` profile combines the MMOR
  supplementary material's 21 entities with 15 non-proximity predicates. Official
  scene-graph source contains 22 entities and 16 predicates; its additional entity is
  `secondary_table`, and its additional automatically derived spatial predicate is
  `closeTo`. Panoptic source additionally contains `unrelated_person` and `cementer`.
  The profile is now frozen and executable, but remains a documented reproduction
  hypothesis because the target paper does not publish its mapping.
- **Rendering constraint:** predicates describe relations such as
  `nurse --holding--> instrument`; they do not own masks. The 36-label vocabulary may
  reserve red/green codes for all labels, but only the 21 entity labels may create
  ellipse instances unless later paper evidence discloses a relation-rendering rule.
- **Memory lower bound:** the inferred 13B checkpoint is roughly 26 GB in bfloat16,
  before activations or optimizer state, and therefore cannot fit unmodified on the
  24.6 GB 4090. Quantized frozen base weights or CPU offload will be required and must
  be labeled as a reproduction deviation.

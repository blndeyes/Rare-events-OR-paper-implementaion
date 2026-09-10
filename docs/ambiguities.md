# Reproduction ambiguities

Status values: `open`, `hypothesis`, `resolved`, or `blocked`.

| Item | Status | Current evidence / next action |
| --- | --- | --- |
| LTX model variant and checkpoint | hypothesis | Official `ltxv_13b_ic_lora.yaml` matches every stated trainer value and specifies `LTXV_13B_097_DEV`; the paper itself omits the checkpoint. |
| Trainer revision | resolved | Reproducible snapshot pinned to `e055182fa36dba6f48eb0919aef09d277da30fbd`, the last official legacy-trainer commit before paper release. Exact author commit remains unknown. |
| 338 training clips | open | Remote data contains contiguous 1-fps camera frames, not 97-frame clips. Derive the event-clipping/interpolation construction, then freeze a manifest. |
| Six MMOR and six 4DOR Table 1 clips | open | Not disclosed. Must choose before final evaluation and label as a reproduction deviation. |
| Camera views | open | Paper does not identify cameras. Inspect examples and dataset conventions. |
| 1 fps to 24 fps interpolation | open | Paper cites LTX keyframe interpolation but omits sampling and boundary details. |
| Class palette | open | Paper states unique red/green values for 36 classes but gives no mapping. |
| Ellipse fit | open | Centroid, height, width, angle are stated; fitting algorithm and conventions are omitted. |
| Depth normalization | open | Averaging within instance masks is stated; normalization scope and direction are omitted. |
| PatchGAN | blocked | Architecture, receptive field, inputs, loss, coefficient, and optimization schedule are absent. Escalate questions to the user's supervisor if public evidence is insufficient. |
| Optimizer details | resolved | Pinned trainer passes only LR to PyTorch AdamW: betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`. LinearLR decays 1.0 to 0.1 over all steps with no warmup. These are upstream defaults, not paper facts. |
| Metric implementations | open | Exact FVD backbone/library, spatial preprocessing, temporal sampling, and per-video aggregation are absent. |
| 4090 memory plan | open | The inferred 13B model doubles tokens for IC-LoRA and may exceed 24 GB despite gradient checkpointing. Measure preprocessing and one batch before altering effective optimization. |

## Newly established constraints

- **Clip duration hypothesis:** 97 frames at 24 fps span exactly four seconds, matching
  five consecutive 1-fps MMOR keyframes at output indices 0, 24, 48, 72, and 96. This
  is strong arithmetic evidence, not a disclosed author recipe.
- **Class-count conflict:** the target paper explicitly says 36 MMOR semantic classes.
  The MMOR supplementary Table 5 lists 21 entity classes and 15 predicates (36 in
  total), while the current official panoptic code defines 23 foreground entity classes
  plus background. Predicates cannot label ellipses. The implementation must not freeze
  a 36-entry entity palette until this discrepancy is resolved or a documented mapping
  is selected.
- **Memory lower bound:** the inferred 13B checkpoint is roughly 26 GB in bfloat16,
  before activations or optimizer state, and therefore cannot fit unmodified on the
  24.6 GB 4090. Quantized frozen base weights or CPU offload will be required and must
  be labeled as a reproduction deviation.

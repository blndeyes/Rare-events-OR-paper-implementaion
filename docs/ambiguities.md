# Reproduction ambiguities

Status values: `open`, `hypothesis`, `resolved`, or `blocked`.

| Item | Status | Current evidence / next action |
| --- | --- | --- |
| LTX model variant and checkpoint | open | Paper says LTX-Video but not 2B/13B or release. Inspect historical trainer configs and model dates. |
| Trainer revision | hypothesis | Legacy official LTX-Video-Trainer is consistent with the paper and gained IC-LoRA in July 2025. Pin a commit predating the experiment freeze. |
| 338 training clips | open | Paper gives count but no IDs. Derive candidate construction, then freeze a manifest. |
| Six MMOR and six 4DOR Table 1 clips | open | Not disclosed. Must choose before final evaluation and label as a reproduction deviation. |
| Camera views | open | Paper does not identify cameras. Inspect examples and dataset conventions. |
| 1 fps to 24 fps interpolation | open | Paper cites LTX keyframe interpolation but omits sampling and boundary details. |
| Class palette | open | Paper states unique red/green values for 36 classes but gives no mapping. |
| Ellipse fit | open | Centroid, height, width, angle are stated; fitting algorithm and conventions are omitted. |
| Depth normalization | open | Averaging within instance masks is stated; normalization scope and direction are omitted. |
| PatchGAN | blocked | Architecture, receptive field, inputs, loss, coefficient, and optimization schedule are absent. Escalate questions to the user's supervisor if public evidence is insufficient. |
| Optimizer details | open | AdamW is stated; batch size, betas, weight decay, scheduler, warmup, clipping, and seed are omitted. Prefer pinned trainer defaults and record them. |
| Metric implementations | open | Exact FVD backbone/library, spatial preprocessing, temporal sampling, and per-video aggregation are absent. |
| 4090 memory plan | open | Establish the exact model first; then measure one batch before selecting checkpointing or accumulation. |


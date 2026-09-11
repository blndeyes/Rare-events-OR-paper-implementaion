# Reproduction progress

## Status

- **Overall:** `in_progress`
- **Current phase:** MMOR clip reconstruction and 97-frame preprocessing smoke test
- **Latest verified accomplishment:** official pinned LTX 0.9.7-dev interpolation and
  Video Depth Anything ViT-L inference both pass on the same real 97-frame MMOR clip.
- **Current blocker:** the authors' exact 338 training clips, 50 ablation clips, camera
  choices, and event boundaries are undisclosed.
- **Next action:** propagate the first MMOR ground-truth entity mask through the passed
  interpolated clip with pinned SAM2.1 Hiera Large and visually inspect the labels.

## Area status

| Area | Status | Evidence | Next step |
| --- | --- | --- | --- |
| Source and dataset audit | `passed` | `docs/source-audit.md`; `docs/dataset-audit.md` | Recheck only if sources or datasets change |
| Paper-shaped smoke clip | `passed` | `/tmp/mmor-smoke-clip.json` on `IRTAZAPC`; `tests/test_clips.py` | Retain as preprocessing gate |
| Exact 338/50 clip construction | `blocked` | Public metadata rules yield 336 or 277, not 338 | Obtain supervisor guidance or freeze a labeled reproduction policy |
| Frozen train/evaluation manifests | `not_started` | `configs/paper_table1.yaml` still has `split_manifest: null` | Create after clip policy is settled |
| Ellipse and semantic rendering | `passed` | `docs/geometry-audit.md`; geometry tests and real-frame review | Revalidate on the final 97-frame representation |
| LTX temporal interpolation | `passed` | 97 frames at 24 fps and 1024x768; anchor PSNR 28.82--32.94 dB and SSIM 0.9038--0.9517 | Build persistent resumable batch runner |
| SAM2 propagation | `in_progress` | Pinned adapter and unit tests implemented | Execute and inspect one 97-frame propagation |
| Video Depth Anything | `passed` | ViT-L float32 `(97, 768, 1024)` output; finite values and valid 97-frame visualization | Feed result to sequence geometry renderer |
| IC-LoRA training | `not_started` | Trainer/checkpoint hypothesis pinned | Wait for preprocessing correctness gates |
| PatchGAN | `blocked` | Paper omits architecture, inputs, loss, weight, and schedule | Seek supervisor guidance; keep optional and isolated |
| Table 1 evaluation | `not_started` | Exact test clips and metric implementations are unresolved | Freeze manifests and metric contract before evaluation |
| Experiment registry | `not_started` | No registry file or run records in the repository | Add schema before the first model execution |

## Completed work

- Repository scaffold, paper configuration, source audit, and read-only MMOR/4DOR
  inventory are complete.
- Remote inventory established dataset organization, counts, timestamp integrity, and
  official split evidence without modifying source data.
- The `paper_36` semantic profile, mask-to-ellipse geometry, relative-depth color
  encoding, deterministic occlusion, and one-/five-frame diagnostics are implemented.
- A real five-keyframe smoke clip manifest was generated from MMOR take `001_PKA`,
  camera 1, timestamps 0-4 (Azure frames 000329-000333).
- Official upstream snapshots are pinned for LTX-Video inference, LTX-Video Trainer,
  SAM2, Video Depth Anything, MMOR, and 4DOR.
- All 22 repository tests passed on `IRTAZAPC` on 2026-09-10 at implementation commit
  `74d98f1692de80f745cbceebcb4abfd8f6999277`.

## Work in progress

- Validate first-frame entity-mask propagation with official pinned SAM2.1 Hiera Large.
- Keep the smoke clip separate from any future frozen training or evaluation split.

## Next three prioritized actions

1. Execute and visually inspect pinned SAM2 propagation on the 97-frame smoke clip.
2. Render 97 frames of ellipse-only geometry from SAM2 labels and VDA relative depth.
3. Build a persistent, resumable preprocessing runner before scaling toward 338 clips.

## Blockers and questions requiring supervisor input

- Exact identities and construction rules for the 338 training, 50 ablation, six MMOR
  test, and six 4DOR test videos.
- Camera selection, event boundaries, and the exact interpolation prompt/sampling
  settings.
- Author palette, ellipse fit, and depth normalization details.
- PatchGAN architecture, objective, coefficient, inputs, optimizer, and schedule.
- Metric implementations and preprocessing for FVD, SSIM, PSNR, and LPIPS.

The concise question set is maintained in `docs/supervisor-questions.md`; all unresolved
or inferred details remain classified in `docs/ambiguities.md`.

## Decisions and evidence

| Decision | Classification | Evidence |
| --- | --- | --- |
| Use five 1-fps source frames at output indices 0/24/48/72/96 | Evidence-supported inference | 97 frames at 24 fps span exactly four seconds; paper cites keyframe interpolation |
| Use legacy LTX-Video Trainer commit `e055182f...` and infer `LTXV_13B_097_DEV` | Evidence-supported inference | Official 13B IC-LoRA configuration matches every disclosed trainer value |
| Use LTX inference commit `20799e51...` | Local engineering choice | Pinned pre-paper-compatible 0.9.7-dev multi-keyframe interface |
| Use 21 entity plus 15 predicate `paper_36` vocabulary | Evidence-supported inference | MMOR supplementary vocabulary; predicates are edges and never ellipse instances |
| Use moment-matched filled ellipses and near-as-bright-blue | Local engineering choice | Testable implementation isolated behind configuration; paper omits convention |

## Experiments and results

Two preprocessing models have run successfully on the smoke clip: LTX interpolation
and Video Depth Anything. No IC-LoRA training checkpoint or Table 1 metric has been
produced. The experiment registry must distinguish execution success, correctness
checks, qualitative quality, and quantitative reproduction success.

## Reproduction risks

- Undisclosed clip selection can dominate both training distribution and reported
  metrics; any locally chosen manifest is a reproduction deviation.
- The inferred 13B checkpoint is larger than the RTX 4090's VRAM before activations and
  optimizer state; quantization or CPU offload may be necessary and must be recorded.
- The paper's undisclosed PatchGAN and metric details prevent exact equivalence without
  additional evidence.
- Geometry hypotheses may be internally correct but visually incompatible with the
  authors' encoding; final-sequence visual checks remain mandatory.

## Relevant paths

- Paper configuration: `configs/paper_table1.yaml`
- Upstream pins: `configs/upstreams.yaml`
- Reproduction plan: `docs/reproduction-plan.md`
- Dataset and source evidence: `docs/dataset-audit.md`, `docs/source-audit.md`
- Geometry evidence: `docs/geometry-audit.md`
- Ambiguity register: `docs/ambiguities.md`
- Verification commands: `docs/verification-guide.md`
- Remote smoke manifest: `/tmp/mmor-smoke-clip.json`
- Remote interpolation output: `/tmp/mmor-ltx-interpolation-fp8`
- Remote VDA output: `/tmp/mmor-vda-vitl`

## Latest update

- **Date:** 2026-09-11 (Asia/Karachi)
- **Verified implementation commit:** `4b94ad1` before the current SAM2 adapter
- **Remote verification:** `IRTAZAPC` produced and validated the 97-frame LTX and VDA
  artifacts. SAM2 execution is the active gate.

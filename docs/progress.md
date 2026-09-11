# Reproduction progress

## Status

- **Overall:** `in_progress`
- **Current phase:** BF16 tiny-overfit validation and clip-policy construction
- **Latest verified accomplishment:** the resumable VDA/SAM2/ellipse batch completed a
  real 97-frame sample, passed strict pair validation, and skipped all work on rerun.
- **Current blockers:** the tiny-overfit checkpoint's generated response is not yet
  verified, and the authors' exact 338/50 clip identities remain undisclosed.
- **Next action:** finish tiny-overfit inference on irtazastone, and review an
  explicitly hypothetical clip-selection policy before freezing any 338/50 split.

## Area status

| Area | Status | Evidence | Next step |
| --- | --- | --- | --- |
| Source and dataset audit | `passed` | `docs/source-audit.md`; `docs/dataset-audit.md` | Recheck only if sources or datasets change |
| Paper-shaped smoke clip | `passed` | `/tmp/mmor-smoke-clip.json` on `IRTAZAPC`; `tests/test_clips.py` | Retain as preprocessing gate |
| Exact 338/50 clip construction | `blocked` | Action-run theory yields 336 before file checks but only 216 usable clips; paper omits selection | Obtain author guidance or approve a labeled hypothesis |
| Candidate train/ablation manifests | `passed` | Deterministic selector finds 4,608 eligible one-camera, five-second-stride windows | Review cameras/takes/stride/seed before explicit acknowledgement |
| Frozen train/evaluation manifests | `not_started` | `configs/paper_table1.yaml` still has `split_manifest: null` | Freeze only after the hypothesis is accepted |
| Ellipse and semantic rendering | `passed` | 97-frame 1024x768 black-canvas conditioning MP4 and contact sheet | Preserve exact representation in training loader |
| LTX temporal interpolation | `passed` | 97 frames at 24 fps and 1024x768; anchor PSNR 28.82--32.94 dB and SSIM 0.9038--0.9517 | Use persistent resumable batch runner after manifests freeze |
| SAM2 propagation | `passed` | Reusable SAM2.1 Hiera Large runner produces 97 masks at about 18 fps | Run through resumable geometry batch |
| Video Depth Anything | `passed` | Reusable ViT-L runner produces finite float32 `(97, 768, 1024)` depth | Run through resumable geometry batch |
| Full geometry batch | `passed` | Real sample completed and strict rerun returned `skipped_valid_existing` | Execute on accepted clip manifest |
| IC-LoRA training | `in_progress` | User-provided stone log loads BF16 LoRA checkpoint step 80; inference traceback is incomplete | Complete tiny-overfit response test |
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
- A persistent resumable LTX batch runner now loads the interpolation pipeline once
  instead of repeating model initialization for every clip.
- Reusable VDA and SAM2 adapters plus a resumable geometry batch now avoid model
  initialization for each of hundreds of clips.
- Strict target/conditioning validation checks both videos, label/depth arrays,
  ellipse-only metadata, palette membership, and predicate exclusion.
- A deterministic split builder can construct 338/50 manifests, but writes them only
  after explicit acknowledgement that the undisclosed selection is a hypothesis.
- Official IC-LoRA preprocessing and a real one-step integration gate are complete.
- All 49 repository tests passed on `IRTAZAPC` on 2026-09-11 at implementation commit
  `ef2176d`.

## Work in progress

- Finish the BF16 tiny-overfit inference/conditioning-response gate on irtazastone.
- Review the candidate pool policy; do not equate a deterministic local selection with
  the authors' unavailable manifest.
- Keep the smoke clip separate from any future frozen training or evaluation split.

## Next three prioritized actions

1. Complete inference from the BF16 step-80 tiny-overfit checkpoint and verify response
   to changed ellipse trajectories.
2. Approve or revise the explicit takes/cameras/stride/seed selection hypothesis.
3. Run the resumable LTX and geometry batches on a small selected subset before scaling.

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

The complete preprocessing chain has run successfully on one smoke clip: LTX
interpolation, reusable Video Depth Anything, reusable SAM2, ellipse-only sequence
rendering, strict pair validation, and official target/reference latent encoding.
The prior RTX 4090 required INT2 for an optimizer-step integration test. A user-provided
irtazastone log shows an unquantized BF16 tiny-overfit checkpoint at step 80 loading
successfully; generation still exits with an incomplete traceback, so the behavioral
overfit gate has not passed. No Table 1 metric has been produced.

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
- Remote SAM2 output: `/tmp/mmor-sam2-vitl`
- Remote ellipse conditioning: `/tmp/mmor-ellipse-conditioning`
- Remote IC-LoRA construction report: `/tmp/ic-lora-construction-smoke.json`
- Remote paired trainer data: `/tmp/mmor-ic-lora-paired`
- Remote one-step reports: `/tmp/mmor-ic-lora-one-step-report.json`,
  `/tmp/mmor-ic-lora-one-step-int4-report.json`, and
  `/tmp/mmor-ic-lora-one-step-int2-report.json`
- Remote strict pair report: `/tmp/mmor-smoke-pair-validation.json`
- Remote resumable geometry smoke: `/tmp/mmor-geometry-batch-smoke`

## Latest update

- **Date:** 2026-09-11 (Asia/Karachi)
- **Verified implementation commit:** `ef2176d`
- **Remote verification:** `IRTAZAPC` passed all 49 tests; the real geometry batch
  completed once and skipped valid outputs on rerun.

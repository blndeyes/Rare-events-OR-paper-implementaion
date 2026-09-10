# Reproduction progress

## Status

- **Overall:** `in_progress`
- **Current phase:** MMOR clip reconstruction and 97-frame preprocessing smoke test
- **Latest verified accomplishment:** a real MMOR smoke manifest maps five consecutive
  1-fps keyframes to frames 0, 24, 48, 72, and 96 of the paper-shaped 97-frame,
  24-fps contract. The auditable pinned-LTX interpolation launcher is implemented.
- **Current blocker:** the authors' exact 338 training clips, 50 ablation clips, camera
  choices, and event boundaries are undisclosed. Video Depth Anything is also not yet
  installed on the training host.
- **Next action:** generate and review the LTX interpolation plan for the existing smoke
  manifest, then execute one 97-frame interpolation and inspect it before integrating
  SAM2 or Video Depth Anything.

## Area status

| Area | Status | Evidence | Next step |
| --- | --- | --- | --- |
| Source and dataset audit | `passed` | `docs/source-audit.md`; `docs/dataset-audit.md` | Recheck only if sources or datasets change |
| Paper-shaped smoke clip | `passed` | `/tmp/mmor-smoke-clip.json` on `IRTAZAPC`; `tests/test_clips.py` | Run one LTX interpolation |
| Exact 338/50 clip construction | `blocked` | Public metadata rules yield 336 or 277, not 338 | Obtain supervisor guidance or freeze a labeled reproduction policy |
| Frozen train/evaluation manifests | `not_started` | `configs/paper_table1.yaml` still has `split_manifest: null` | Create after clip policy is settled |
| Ellipse and semantic rendering | `passed` | `docs/geometry-audit.md`; geometry tests and real-frame review | Revalidate on the final 97-frame representation |
| LTX temporal interpolation | `in_progress` | Pinned launcher at `src/or_video_reproduction/preprocessing/ltx_interpolation.py`; no plan/output yet | Plan, execute, and visually inspect one smoke clip |
| SAM2 propagation | `not_started` | Pinned upstream exists remotely | Integrate after interpolation output passes |
| Video Depth Anything | `not_started` | Pinned choice recorded; remote upstream absent | Install pinned ViT-L relative-depth source and verify direction |
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

- Validate the five-keyframe-to-97-frame interpolation hypothesis with the official
  pinned LTX 0.9.7-dev inference path.
- Keep the smoke clip separate from any future frozen training or evaluation split.

## Next three prioritized actions

1. Produce, review, and execute one LTX interpolation plan from
   `/tmp/mmor-smoke-clip.json`; record runtime, GPU memory, output path, and visual
   correctness separately from execution success.
2. Add a version-controlled experiment registry schema and record the interpolation
   smoke run as the first experiment.
3. Install the pinned Video Depth Anything snapshot, infer relative depth for the smoke
   sequence, and verify near/far direction before geometry rendering uses it.

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

No model experiment has run yet. The smoke manifest is a preprocessing artifact, not
an experiment result. No interpolation output, training checkpoint, generated video,
or Table 1 metric has been produced. The experiment registry must distinguish
execution success, correctness checks, qualitative quality, and quantitative
reproduction success.

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
- Expected remote interpolation plan: `/tmp/mmor-ltx-interpolation-plan.json`
- Expected remote interpolation output: `/tmp/mmor-ltx-interpolation`

## Latest update

- **Date:** 2026-09-10 (Asia/Karachi)
- **Verified implementation commit:** `74d98f1692de80f745cbceebcb4abfd8f6999277`
- **Local branch at verification:** `main`, clean, synchronized with `origin/main`
- **Remote verification:** `IRTAZAPC` contains the smoke manifest and passed all 22
  tests; no interpolation plan/output or experiment registry was present.

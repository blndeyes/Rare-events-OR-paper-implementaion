# Reproduction progress

## Status

- **Overall:** `in_progress`
- **Current phase:** training-hardware decision and frozen clip-manifest construction
- **Latest verified accomplishment:** one real target/ellipse pair passes official
  latent preprocessing and a complete IC-LoRA optimizer step, including checkpoint
  saving, with the documented INT2-Quanto integration fallback.
- **Current blockers:** faithful BF16 training does not fit the 24 GiB RTX 4090, and the
  authors' exact 338 training clips, camera choices, and event boundaries are undisclosed.
- **Next action:** select larger/multi-GPU or offloaded hardware for faithful BF16, and
  freeze a clearly labeled reproduction policy for the 338 training clips.

## Area status

| Area | Status | Evidence | Next step |
| --- | --- | --- | --- |
| Source and dataset audit | `passed` | `docs/source-audit.md`; `docs/dataset-audit.md` | Recheck only if sources or datasets change |
| Paper-shaped smoke clip | `passed` | `/tmp/mmor-smoke-clip.json` on `IRTAZAPC`; `tests/test_clips.py` | Retain as preprocessing gate |
| Exact 338/50 clip construction | `blocked` | Public metadata rules yield 336 or 277, not 338 | Obtain supervisor guidance or freeze a labeled reproduction policy |
| Frozen train/evaluation manifests | `not_started` | `configs/paper_table1.yaml` still has `split_manifest: null` | Create after clip policy is settled |
| Ellipse and semantic rendering | `passed` | 97-frame 1024x768 black-canvas conditioning MP4 and contact sheet | Preserve exact representation in training loader |
| LTX temporal interpolation | `passed` | 97 frames at 24 fps and 1024x768; anchor PSNR 28.82--32.94 dB and SSIM 0.9038--0.9517 | Use persistent resumable batch runner after manifests freeze |
| SAM2 propagation | `passed` | 97 masks at about 18 fps; anchor entity IoU 0.868--1.000 | Integrate into resumable batch runner |
| Video Depth Anything | `passed` | ViT-L float32 `(97, 768, 1024)` output; finite values and valid 97-frame visualization | Feed result to sequence geometry renderer |
| IC-LoRA training | `hardware_blocked` | Official pair preprocessing passes; INT2 completes one optimizer step, while INT8/INT4 OOM | Move faithful BF16 run to larger/multi-GPU or validated offload |
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
- Official IC-LoRA preprocessing and a real one-step integration gate are complete.
- All 32 repository tests passed on `IRTAZAPC` on 2026-09-11 at implementation commit
  `54fcb11` before packaging the one-step runner.

## Work in progress

- Package the verified one-step hardware gate and keep its quantized modes explicitly
  separated from the faithful BF16 experiment.
- Keep the smoke clip separate from any future frozen training or evaluation split.

## Next three prioritized actions

1. Choose faithful-training hardware: at least a larger GPU, multiple GPUs, or a
   separately validated offload strategy.
2. Freeze a labeled 338-clip reproduction manifest after resolving the undisclosed
   clip-selection policy.
3. Run a tiny-subset overfit before any 8,000-step experiment.

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
interpolation, Video Depth Anything, SAM2, ellipse-only sequence rendering, and
official target/reference latent encoding. A single INT2-Quanto IC-LoRA optimizer
step produced a checkpoint; INT8 and INT4 both reached model execution but exhausted
24 GiB VRAM. No faithful BF16 training checkpoint or Table 1 metric has been produced.

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

## Latest update

- **Date:** 2026-09-11 (Asia/Karachi)
- **Verified implementation commit:** `54fcb11` (one-step runner packaging pending)
- **Remote verification:** `IRTAZAPC` passed all 32 tests, produced validated 97-frame
  preprocessing artifacts, and completed one quantized IC-LoRA optimizer step.

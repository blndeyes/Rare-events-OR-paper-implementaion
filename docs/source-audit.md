# Source audit

## Paper

- *Towards Controllable Video Synthesis of Routine and Rare OR Events*
- arXiv:2602.21365v1, released 24 February 2026
- All 14 pages reviewed; implementation-relevant content is concentrated in Sections
  3 and 4, Figures 1-3, and Tables 1 and 4.

The paper supplies no supplementary implementation appendix or released source link.
Its PatchGAN description is limited to naming the added loss and citing pix2pix. Table
4 establishes that the full Base+ellipse+depth+PatchGAN variant improves FVD from
532.05 to 487.20 on the 50-video ablation, but does not disclose how the loss is built.

## Official LTX sources

- Trainer: <https://github.com/Lightricks/LTX-Video-Trainer>
- Pinned pre-paper commit: `e055182fa36dba6f48eb0919aef09d277da30fbd`
- Configuration: `configs/ltxv_13b_ic_lora.yaml`
- Base checkpoint alias: `LTXV_13B_097_DEV`

The official configuration independently matches the paper on LoRA rank/alpha 128,
reference-video conditioning, first-frame probability 0.2, learning rate `2e-4`,
AdamW, bfloat16, 50 inference steps, and guidance scale 3.5. It supplies likely
defaults omitted by the paper: batch size 1, accumulation 1, clipping 1.0, linear
scheduler, gradient checkpointing, shifted-logit-normal timestep sampling, and seed 42.

Direct inspection of the pinned source resolved the remaining optimizer mechanics:

- `torch.optim.AdamW` is constructed with only the configured learning rate, so the
  PyTorch defaults apply: betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `0.01`.
- `LinearLR` starts at the configured learning rate and decays to 10% over the full
  configured step count. There is no warmup.
- IC-LoRA concatenates clean reference-video latents before noisy target latents.
  Reference tokens have timestep zero and are excluded from the flow-matching MSE.
  When first-frame conditioning is sampled, the first target-frame tokens are also
  clean, timestep zero, and excluded from loss.
- The lockfile resolves PyTorch 2.6.0, torchvision 0.21.0, diffusers 0.33.1,
  accelerate 1.2.1, PEFT 0.14.0, and transformers 4.52.2.

The paper changes at least two values from the template: training increases from 2,000
to 8,000 steps, and video dimensions become 1024x768 with 97 frames. Therefore each
template value is tracked as an upstream-default inference rather than a paper fact.

## Official dataset and preprocessing sources

- MMOR: <https://github.com/egeozsoy/MM-OR>
- 4DOR: <https://github.com/egeozsoy/4D-OR>
- SAM2: <https://github.com/facebookresearch/sam2>
- Video Depth Anything: <https://github.com/DepthAnything/Video-Depth-Anything>

Exact author-used revisions are undisclosed. Reproducible audit snapshots are pinned
below; SAM2 and Video Depth Anything model variants still require selection.

Audited source snapshots:

- MM-OR: `defe55b855d603a3a64b219bfee2d3b729875a6d` (27 August 2025)
- 4D-OR: `2685e61539897409ea8df59f7097faf1a41ca3f1` (29 March 2025)
- LTX-Video inference: `20799e51cd739986d98d9b1aab55cc2067c1eabb`
  (8 July 2025), a legacy snapshot retaining the 13B 0.9.7-dev configuration and
  multi-keyframe conditioning interface.
- Video Depth Anything: `4f5ae23172ba60fd7bc11ef671cca678842c7072`
  (7 October 2025); relative-depth ViT-L is selected from the official default CLI.
- SAM2: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
  (15 December 2024); SAM 2.1 Hiera Large is selected from the official video example.

These are reproducible audit snapshots, not claims about the unreleased author code.
The VDA and SAM2 sizes are official-default hypotheses because the target paper only
says the models are used out of the box.

The official MM-OR source at `defe55b855d603a3a64b219bfee2d3b729875a6d`
documents 22 physical directories representing 39 logical takes. Its published
panoptic split is train 10 takes, validation 3, test 4, plus 22 short clips; this is
useful evidence but does not prove that the video-synthesis paper used the same split.
The official 4D-OR source currently uses take split train `[1,3,5,7,9,10]`, validation
`[4,8]`, test `[2,6]`.

## Remote audit, 10 September 2026

- Training host: Ubuntu 22.04, Linux 6.8
- CPU: Intel Core i9-13900K, 32 logical CPUs
- RAM: 125 GiB
- GPU: RTX 4090, 24,564 MiB, driver 560.35.03
- Dataset filesystem: approximately 1.3 TiB free
- MMOR extraction: approximately 489 GiB
- 4DOR extraction: approximately 153 GiB
- Python/Conda exists under `/home/irtaza/anaconda3` but is not initialized in the
  non-interactive SSH shell.
- The official LTX 13B 0.9.7-dev BF16 checkpoint fails during transformer placement
  on this GPU: 23.48 GiB was in use and the next 32 MiB allocation failed. The
  official 0.9.7-dev FP8 checkpoint is therefore the documented smoke-test fallback;
  the target paper does not disclose its interpolation checkpoint precision.
- The legacy inference factory transfers the transformer, VAE, and text encoder to
  CUDA before its runtime `offload_to_cpu` path. The tracked sequential-offload patch
  keeps construction on CPU, while the explicit-component patch transfers only the
  text encoder, conditioning VAE, transformer, or decoding VAE during its own phase.
  These are host-compatibility patches against pinned source, not author-code claims.
- The 0.9.7 FP8 config is not included in the inference code's mixed-precision test,
  producing a BF16-input/FP8-weight mismatch at the first attention projection on the
  RTX 4090. The tracked factory patch enables the pipeline's existing BF16 autocast
  for the official FP8 precision value; a direct CUDA linear-layer probe passes.
- First-pass FP8 denoising completes, then the multi-scale upsampler reads per-channel
  VAE statistics after that VAE has been offloaded. The tracked latent-normalization
  patch moves those unchanged statistics to the latent device before arithmetic.
- Full-resolution second-pass denoising initially failed on a 1.31 GiB activation with
  610 MiB free and 781 MiB reserved. The host path now uses expandable CUDA segments
  and offloads the already-consumed spatial upsampler before that pass.
- Multi-keyframe inference explicitly rejects prompt enhancement and returns the
  original prompt, yet the 0.9.7 config loads and retains both enhancer models. The
  tracked FP8 config patch disables this ineffective path, preserving the prompt while
  avoiding roughly 10.7 GiB of otherwise stranded GPU allocation.

The first MMOR filename inventory found 22 top-level procedure directories and five
camera streams totaling approximately 63,535 contiguous frames per camera. Panoptic
annotations are present for cameras 1, 4, and 5. The extracted dataset is source data;
the paper's 338 videos of 97 frames each have not yet been reconstructed.

## LTX interpolation smoke result, 11 September 2026

The five consecutive camera-1 frames `000329` through `000333` from `001_PKA` were
conditioned at output indices 0, 24, 48, 72, and 96. The patched official 0.9.7-dev
FP8 pipeline completed on the RTX 4090. Its first multi-scale pass took 78 seconds
(27 executed steps), and its second pass took 182 seconds (13 executed steps).

The H.264 artifact decodes as exactly 97 frames at 24 fps and 1024x768, with duration
4.042 seconds. Against raw source frames resized to the output aspect ratio, anchor
PSNR values were `[32.94, 31.94, 28.82, 30.41, 29.88]` dB and SSIM values were
`[0.9517, 0.9417, 0.9038, 0.9226, 0.9197]`. Visual inspection confirms matching
room geometry, equipment, and person positions at all five anchors; mild VAE softness
and color shift remain. This passes the interpolation smoke gate but does not identify
the undisclosed author clip, prompt, seed, or checkpoint precision.

The pinned Video Depth Anything checkout is run in the shared preprocessing runtime
(Torch 2.6/CUDA 12.4, xFormers 0.0.29.post3) rather than its older Torch 2.1.1
requirements pin. Matplotlib 3.11 removed `cm.get_cmap`; the tracked compatibility
patch uses the equivalent `cm.inferno` object and avoids constructing a colormap for
grayscale output. This affects visualization saving only, not predicted depths.

The official ViT-L relative-depth model completed the same 97-frame smoke clip in
five chunks. The saved float32 array has shape `(97, 768, 1024)`, contains only finite
values, and ranges from 12.609 to 3146.822. Its visualization decodes as 97 frames at
24 fps and 1024x768. Visual inspection shows temporally stable room structure with
people and nearby equipment separated from the background, so the VDA smoke gate
passes. Absolute values are model-relative and must not be interpreted as metric depth.

## SAM2 and ellipse-conditioning smoke result, 11 September 2026

Official SAM2.1 Hiera Large propagated the resized first-frame ground-truth mask
through all 97 interpolated frames at about 18 frames per second. The output is uint8
with shape `(97, 768, 1024)` and contains background plus three paper entities:
instrument table, MPS station, and tracker. Against the four later MMOR annotations at
the LTX anchor frames, per-entity IoU ranges from 0.868 to 0.968; frame zero is preserved
exactly and has IoU 1.0. The optional SAM2 CUDA connected-components extension was
disabled, so official hole-filling post-processing is skipped as documented upstream.

Combining those masks with VDA relative depth produces an H.264 conditioning video of
exactly 97 frames, 24 fps, and 1024x768. Visual inspection confirms that every frame is
a black canvas containing only three filled ellipses. Larger VDA output is treated as
nearer: on frame zero, entity mean VDA predictions increase as the corresponding MMOR
sensor distances generally decrease. This direction and per-frame min-max normalization
remain explicitly recorded reproduction hypotheses because the paper omits both details.

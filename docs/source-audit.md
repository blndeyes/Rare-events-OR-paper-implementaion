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
- Multi-keyframe inference explicitly rejects prompt enhancement and returns the
  original prompt, yet the 0.9.7 config loads and retains both enhancer models. The
  tracked FP8 config patch disables this ineffective path, preserving the prompt while
  avoiding roughly 10.7 GiB of otherwise stranded GPU allocation.

The first MMOR filename inventory found 22 top-level procedure directories and five
camera streams totaling approximately 63,535 contiguous frames per camera. Panoptic
annotations are present for cameras 1, 4, and 5. The extracted dataset is source data;
the paper's 338 videos of 97 frames each have not yet been reconstructed.

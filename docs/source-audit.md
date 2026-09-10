# Source audit

## Paper

- *Towards Controllable Video Synthesis of Routine and Rare OR Events*
- arXiv:2602.21365v1, released 24 February 2026
- Implementation-relevant content: Sections 3 and 4, Figures 1-3, Tables 1 and 4

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

The paper changes at least two values from the template: training increases from 2,000
to 8,000 steps, and video dimensions become 1024x768 with 97 frames. Therefore each
template value is tracked as an upstream-default inference rather than a paper fact.

## Official dataset and preprocessing sources

- MMOR: <https://github.com/egeozsoy/MM-OR>
- 4DOR: <https://github.com/egeozsoy/4D-OR>
- SAM2: <https://github.com/facebookresearch/sam2>
- Video Depth Anything: <https://github.com/DepthAnything/Video-Depth-Anything>

Exact revisions for these sources remain to be selected after inspecting dataset
metadata and release chronology.

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

The first MMOR filename inventory found 22 top-level procedure directories and five
camera streams totaling approximately 63,535 contiguous frames per camera. Panoptic
annotations are present for cameras 1, 4, and 5. The extracted dataset is source data;
the paper's 338 videos of 97 frames each have not yet been reconstructed.

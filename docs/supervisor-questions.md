# Questions for supervisor discussion

These questions target details that are not recoverable from the paper or current
official upstream code. They are ordered by impact on the Table 1 reproduction.

1. Can the authors' 338 training, 50 ablation, six MMOR Table 1, and six 4DOR Table 1
   clip manifests be obtained internally, including take, camera, and source timestamps?
2. Were 97-frame clips produced from five 1-fps keyframes at output frames 0, 24, 48,
   72, and 96? Which LTX checkpoint/workflow, prompts, seed, guidance, and denoising
   settings were used for temporal interpolation?
3. The paper says 36 MMOR semantic classes, but MMOR supplementary Table 5 has 21
   entity classes plus 15 predicates, while official panoptic code has 23 foreground
   entity classes. Which entity list and exact red/green palette were used?
4. Which camera views were used for training and evaluation, and were clips from the
   same physical procedure kept in a single split?
5. Which LTX-Video checkpoint and trainer revision were used? Was it the 13B 0.9.7-dev
   checkpoint with the legacy `ltxv_13b_ic_lora.yaml` style-transfer configuration?
6. For PatchGAN, what were the discriminator architecture/receptive field, spatial or
   spatiotemporal input, real/fake inputs, adversarial objective, loss coefficient,
   optimizer, and discriminator-to-generator update ratio? Was the generator loss
   applied to decoded predicted-clean frames or another representation?
7. Which implementations and preprocessing were used for FVD, SSIM, PSNR, and LPIPS,
   including FVD backbone, resize/crop, pixel range, temporal sampling, and aggregation?
8. How were ellipse axes and angles fitted, how was overlap drawing order chosen, and
   over what domain and direction was relative depth normalized?

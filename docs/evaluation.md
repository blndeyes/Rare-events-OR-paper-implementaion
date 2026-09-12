# Evaluation protocol

The [target paper](https://arxiv.org/html/2602.21365#S4.SS1) reports FVD, SSIM, PSNR, LPIPS, DOVER, Inception Score,
bounding-box IoU, segmentation-mask IoU, downstream accuracy, and downstream
recall. It does not disclose the exact packages, model snapshots, image
preprocessing, or aggregation rules. `configs/evaluation.yaml` therefore records
the choices in this repository as **reproduction hypotheses**, not author-confirmed
details.

The structural comparison follows the paper's stated procedure: prompt each
instance in the real first frame, propagate the corresponding instance through
both real and generated sequences with SAM2, then compare masks and their tight
axis-aligned boxes. Consequently, each nonzero label in the two mask archives
must refer to the same prompted instance.

## Manifest

Create a JSON manifest. Paths may be absolute or relative to the manifest:

```json
{
  "schema_version": 1,
  "samples": [
    {
      "id": "take_001",
      "reference_video": "real/take_001.mp4",
      "generated_video": "generated/take_001.mp4",
      "reference_masks": "masks/real_take_001.npz",
      "generated_masks": "masks/generated_take_001.npz"
    }
  ],
  "classification": {
    "targets": [1, 0],
    "predictions": [1, 0],
    "positive_label": 1
  }
}
```

Mask archives must contain a `labels` array shaped `[T,H,W]` (or `[H,W]` for
one image). IDs must agree between reference and generated masks; zero is
background.

## Running

Install the repository's evaluation extra, FFmpeg, and the separately pinned
official FVD and DOVER environments. Then run:

```text
or-evaluate-videos \
  --manifest evaluation.json \
  --output metrics.json \
  --metrics psnr,ssim,lpips,fvd,is,dover,box_iou,mask_iou,classification \
  --google-research-root /path/to/google-research \
  --dover-root /path/to/DOVER
```

FVD needs at least two reference and two generated videos. Paired metrics reject
different frame counts, native resolutions, or frame rates. The official Google
FVD graph fixes its extraction batch at 16 and the final batch is padded only for
inference; padded embeddings are discarded. DOVER is parsed from the official
`evaluate_one_video.py -f` normalized fused score.

For the paper comparison, evaluate the same held-out 12-video set for every
method. The paper does not publish those 12 identities, so comparisons are not
strictly author-equivalent until that split is recovered.

# Dataset audit

This report records the read-only inventory run on the training host on 10 September
2026. Counts describe the extracted source data; no dataset files were changed. The
machine-readable reports remain outside Git because they contain host-specific paths
and dataset-derived metadata.

## MMOR

- The extraction contains 22 physical procedure directories representing 39 logical
  takes (17 long recordings and 22 short clips in the official metadata).
- Twenty physical directories contain five Azure Kinect RGB-D streams. Procedures
  `005_TKA` and `009_TKA` contain SimStation views instead of Azure `colorimage` and
  `depthimage` directories.
- Parsed RGB totals are 63,536 frames for camera 1 and 63,535 for cameras 2-5. Each
  physical Azure sequence has a contiguous integer frame range. Depth identifiers
  match RGB identifiers exactly.
- All 39 `take_jsons` files parse successfully. Across Azure procedures they contain
  55,668 logical timestamps. Every referenced Azure frame exists. Extra physical
  frames occur before or after logical-take ranges and must not be treated as samples
  without consulting the take manifest.
- Procedure `004_PKA` has five repeated Azure identifiers in its timestamp manifest;
  this must be preserved or explicitly de-duplicated in a versioned clip policy.
- Room-camera RGB is 2048x1536, depth is 1024x768 16-bit TIFF, raw segmentation
  exports are 2048x1536 grayscale, and derived panoptic masks are 2048x1536 RGB.
- Ground-truth/derived panoptic data exists for Azure cameras 1, 4, and 5. The masks
  are sparse relative to the complete RGB streams, consistent with the MMOR paper's
  mixture of dense high-action annotation and every-fifth-frame low-action annotation.

The official MMOR panoptic code publishes train/validation/test splits of 10/3/4
logical takes plus a separate set of 22 short clips. This is evidence for leakage-safe
grouping, but there is no evidence that the video-synthesis paper used the identical
split to construct its 338/50 videos.

## 4DOR

- The extraction contains all 10 official takes and 6,734 timestamp rows at 1 fps.
- Each timepoint references six RGB and six depth views. Every referenced file is
  present. Parsed totals are 6,734 frames for cameras 1-4 and 6, 6,733 for camera 5;
  take 2 references one camera-5 RGB/depth identifier twice, explaining the one-file
  difference without a missing timestamp input.
- There are 40,403 RGB and 40,403 depth images, 6,735 point clouds, and 5,229 human
  pose annotation JSON files. Take 4 has one more point cloud than timestamp rows.
- `2D_keypoint_annotations.json` is absent in takes 2 and 6, which are the held-out
  test takes in the official 4DOR split. Other inspected metadata JSON files parse.
- RGB samples are 2048x1536. Their integer camera frame identifiers are sampled from
  higher-rate source streams and therefore contain large expected index gaps; timestamp
  correspondence, not integer contiguity, is the correct completeness check.
- The official split is train takes 1, 3, 5, 7, 9, 10; validation takes 4, 8; and test
  takes 2, 6. The video-synthesis paper does not say whether its six 4DOR clips obeyed
  this split.

## Consequences for clip construction

A 97-frame video at 24 fps spans exactly four seconds: `(97 - 1) / 24 = 4`. Since
MMOR is synchronized at 1 fps, five consecutive source frames exactly delimit such a
clip. This makes five source keyframes placed at output frames 0, 24, 48, 72, and 96 a
strong construction hypothesis. It does not explain which intervals produced the 338
training and 50 ablation videos, so clip selection remains open and must be frozen in a
manifest before preprocessing.

The MMOR next-action metadata was also tested as a possible event-boundary source. On
the official panoptic training takes, grouping only by consecutive action label gives
132 runs, of which 112 contain at least five 1-fps samples. Replicating those across
three annotated views would give 336 videos, intriguingly close to but not equal to
338. Treating countdown resets as distinct action episodes gives 393 episodes, of which
277 contain at least five samples. Neither rule reproduces 338, so the metadata does
not justify silently reconstructing the authors' selection.

## Acceptance status

- **Passed:** dataset roots and take organization; source-sequence counts; JSON
  readability; RGB-depth identifier alignment; timestamp-referenced file presence;
  official dataset split recovery.
- **Not yet passed:** exact 338/50 clip identities; camera selection; semantic palette;
  per-file decode/corruption scan; real-clip geometry visualization; 4DOR panoptic
  mapping for evaluation.

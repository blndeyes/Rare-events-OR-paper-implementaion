# Geometry conditioning audit

## Implemented contract

- Output resolution is 1024×768.
- MMOR `segmentation_export_*` masks are decoded using official raw grayscale
  label IDs and resized from 2048×1536 with nearest-neighbor sampling.
- Each supported semantic mask is replaced by a filled ellipse matching its
  centroid and covariance. Full axis diameter is four times the square root of
  the corresponding covariance eigenvalue.
- The major-axis angle is measured clockwise from positive x in image
  coordinates and normalized to `[0, 180)`.
- Red/green jointly encode the versioned `paper_36` semantic label. Only entity
  labels create ellipses; predicate codes remain reserved vocabulary entries.
- Blue encodes min-max-normalized mean instance depth, with nearer mapped to 255.
- Ellipses are drawn far-to-near with deterministic class/key tie breaking.

The ellipse fit, red/green palette, normalization scope, and near/blue direction
are reproduction hypotheses because the target paper does not disclose those
details. They remain isolated and configuration-controlled.

## Synthetic acceptance checks

The geometry suite verifies:

- recovery of centroid, major/minor diameters, and angle from a synthetic rotated
  filled ellipse;
- rejection of empty/tiny masks;
- 36 unique, round-trippable red/green pairs;
- masked depth aggregation excluding zero and non-finite samples;
- near-to-bright-blue normalization;
- deterministic near-over-far occlusion;
- complete preview artifact creation;
- correct temporal angle differences across the 180-degree axis wraparound.

All 16 repository tests passed locally and on IRTAZAPC at commit `1293572`.

## Real MMOR frame audit

Input: `001_PKA`, camera 1, frame 000329.

The official mask contains three supported labels: `instrument_table`,
`mps_station`, and `tracker`. No unknown or excluded labels were encountered.
The RGB, semantic mask, sensor depth, and rendered conditioning frame align at
1024×768. Right-edge ellipses are intentionally clipped because their source
masks extend beyond the image boundary.

This audit uses the supplied MMOR 16-bit sensor depth only to validate I/O,
alignment, aggregation, and rendering. The target pipeline must replace it with
Video Depth Anything relative depth.

## Five-frame temporal audit

Input: frames 000329–000333 at the dataset's source rate of 1 fps.

The audit produced 12 same-class adjacent-frame comparisons. The maximum center
movement was 305.21 pixels for `nurse` from frame 000331 to 000332; inspection of
the RGB frames confirms that the person genuinely crosses the scene. Other
notable changes occur as the Mako robot and people enter the camera view.

This is evidence against inventing ellipse smoothing. Temporal densification must
come from the paper-specified keyframe interpolation and SAM2 propagation path.
The current five-frame result passes the geometry-core gate but does not yet pass
the 97-frame/24-fps conditioning gate.

## Remaining geometry work

1. Freeze a defensible source-clip manifest.
2. Integrate the official Video Depth Anything inference path and verify its
   relative-depth direction.
3. Integrate first-frame ground-truth initialization and SAM2 mask propagation.
4. Reconstruct a complete 97-frame, 24-fps conditioning sequence.
5. Repeat visual and temporal checks on that final representation.

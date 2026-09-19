"""Evaluate whether a generated entity follows an edited ellipse trajectory."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from or_video_reproduction.data.clips import OUTPUT_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH
from or_video_reproduction.geometry.ellipse import Ellipse, rasterize_ellipse
from or_video_reproduction.preprocessing.sam2_propagation import Sam2VideoRunner

from .video import decode_rgb_video, probe_video


def load_labels(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "labels" not in archive:
            raise ValueError(f"SAM2 archive has no labels array: {path}")
        labels = np.asarray(archive["labels"])
    if labels.shape != (OUTPUT_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH):
        raise ValueError(f"SAM2 labels have unexpected shape: {labels.shape}")
    return labels


def instance_label(instance_id: str) -> int:
    try:
        value = int(instance_id.split(":", 1)[0])
    except ValueError as error:
        raise ValueError(
            "MMOR trajectory evaluation requires an instance ID beginning with its integer label"
        ) from error
    if not 1 <= value <= 255:
        raise ValueError("Selected instance label must be in [1, 255]")
    return value


def mask_track(labels: np.ndarray, label: int) -> dict[str, object]:
    centroids: list[list[float] | None] = []
    boxes: list[list[int] | None] = []
    areas: list[int] = []
    for frame in labels:
        ys, xs = np.nonzero(frame == label)
        areas.append(len(xs))
        if not len(xs):
            centroids.append(None)
            boxes.append(None)
        else:
            centroids.append([float(xs.mean()), float(ys.mean())])
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return {
        "label": label,
        "centroids": centroids,
        "boxes_xyxy_exclusive": boxes,
        "areas_pixels": areas,
        "missing_frames": [index for index, value in enumerate(centroids) if value is None],
    }


def _paired_points(
    observed: Sequence[list[float] | None], commanded: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    indices = [index for index, point in enumerate(observed) if point is not None]
    if not indices:
        raise ValueError("Selected entity is missing from every tracked frame")
    actual = np.asarray([observed[index] for index in indices], dtype=np.float64)
    target = np.asarray([commanded[index] for index in indices], dtype=np.float64)
    return actual, target, indices


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0 else float(a @ b / denominator)


def _box_iou(a: Sequence[int] | None, b: Sequence[int] | None) -> float | None:
    if a is None or b is None:
        return None
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else None


def _ellipse_from_row(row: Mapping[str, object]) -> Ellipse:
    value = row["ellipse"]
    return Ellipse(
        center_x=float(value["center_x"]),
        center_y=float(value["center_y"]),
        major_diameter=float(value["major_diameter"]),
        minor_diameter=float(value["minor_diameter"]),
        angle_degrees=float(value["angle_degrees"]),
        source_pixels=int(value["source_pixels"]),
    )


def ellipse_alignment(
    generated_labels: np.ndarray,
    *,
    label: int,
    edited_metadata: Mapping[str, object],
    instance_id: str,
) -> dict[str, object]:
    box_values: list[float] = []
    mask_values: list[float] = []
    observations = 0
    for frame_index, (labels, frame) in enumerate(
        zip(generated_labels, edited_metadata["frames"], strict=True)
    ):
        rows = [row for row in frame["instances"] if row.get("key") == instance_id]
        if len(rows) != 1:
            raise ValueError(
                f"Edited metadata does not uniquely contain {instance_id} at {frame_index}"
            )
        ellipse_mask = rasterize_ellipse(_ellipse_from_row(rows[0]), labels.shape)
        generated_mask = labels == label
        if not generated_mask.any():
            continue
        observations += 1
        generated_y, generated_x = np.nonzero(generated_mask)
        ellipse_y, ellipse_x = np.nonzero(ellipse_mask)
        generated_box = [
            int(generated_x.min()),
            int(generated_y.min()),
            int(generated_x.max()) + 1,
            int(generated_y.max()) + 1,
        ]
        ellipse_box = [
            int(ellipse_x.min()),
            int(ellipse_y.min()),
            int(ellipse_x.max()) + 1,
            int(ellipse_y.max()) + 1,
        ]
        value = _box_iou(generated_box, ellipse_box)
        assert value is not None
        box_values.append(value)
        union = np.count_nonzero(generated_mask | ellipse_mask)
        mask_values.append(float(np.count_nonzero(generated_mask & ellipse_mask) / union))
    if not observations:
        raise ValueError("Selected generated entity has no masks for alignment")
    return {
        "bounding_box_iou_to_conditioning_ellipse": float(np.mean(box_values)),
        "segmentation_iou_to_conditioning_ellipse": float(np.mean(mask_values)),
        "observed_frames": observations,
        "interpretation": (
            "proxy alignment only: an ellipse is a spatial abstraction, not an expected exact "
            "human segmentation silhouette"
        ),
    }


def unrelated_track_stability(
    original_labels: np.ndarray, edited_labels: np.ndarray, *, selected_label: int
) -> dict[str, object]:
    labels = sorted(
        {int(value) for value in np.union1d(original_labels, edited_labels)} - {0, selected_label}
    )
    per_label: dict[str, object] = {}
    all_distances: list[float] = []
    for label in labels:
        original = mask_track(original_labels, label)
        edited = mask_track(edited_labels, label)
        distances = []
        for before, after in zip(original["centroids"], edited["centroids"], strict=True):
            if before is not None and after is not None:
                distances.append(float(np.linalg.norm(np.asarray(after) - np.asarray(before))))
        if distances:
            all_distances.extend(distances)
        per_label[str(label)] = {
            "mean_centroid_change_pixels": float(np.mean(distances)) if distances else None,
            "max_centroid_change_pixels": float(np.max(distances)) if distances else None,
            "paired_frames": len(distances),
            "original_missing_frames": len(original["missing_frames"]),
            "edited_missing_frames": len(edited["missing_frames"]),
        }
    return {
        "mean_centroid_change_pixels": float(np.mean(all_distances)) if all_distances else None,
        "max_centroid_change_pixels": float(np.max(all_distances)) if all_distances else None,
        "per_label": per_label,
    }


def background_pixel_stability(
    original_video: Path,
    edited_video: Path,
    original_labels: np.ndarray,
    edited_labels: np.ndarray,
    *,
    selected_label: int,
) -> dict[str, object]:
    original = decode_rgb_video(original_video).astype(np.float64)
    edited = decode_rgb_video(edited_video).astype(np.float64)
    keep = ~((original_labels == selected_label) | (edited_labels == selected_label))
    differences = original - edited
    squared = differences**2
    absolute = np.abs(differences)
    mse_values, mae_values = [], []
    for frame_index in range(OUTPUT_FRAME_COUNT):
        if not keep[frame_index].any():
            continue
        mse_values.append(float(np.mean(squared[frame_index][keep[frame_index]])))
        mae_values.append(float(np.mean(absolute[frame_index][keep[frame_index]])))
    mse = float(np.mean(mse_values))
    return {
        "masked_out": "union of selected SAM2 masks from original- and edited-conditioned outputs",
        "mean_absolute_rgb_difference": float(np.mean(mae_values)),
        "mean_squared_rgb_difference": mse,
        "psnr_db": math.inf if mse == 0 else float(10.0 * math.log10(255.0**2 / mse)),
    }


def evaluate_control(
    *,
    original_labels: np.ndarray,
    edited_labels: np.ndarray,
    edit_manifest: Mapping[str, object],
    edited_metadata: Mapping[str, object],
    original_video: Path | None = None,
    edited_video: Path | None = None,
) -> dict[str, object]:
    instance_id = str(edit_manifest["selected_instance_id"])
    label = instance_label(instance_id)
    command = edit_manifest["edited_centroids"]
    original_command = edit_manifest["original_centroids"]
    if len(command) != OUTPUT_FRAME_COUNT or len(original_command) != OUTPUT_FRAME_COUNT:
        raise ValueError("Edit manifest centroid tracks must have exactly 97 points")
    original_track = mask_track(original_labels, label)
    edited_track = mask_track(edited_labels, label)
    edited_actual, edited_target, indices = _paired_points(edited_track["centroids"], command)
    distances = np.linalg.norm(edited_actual - edited_target, axis=1)

    common = [
        index
        for index, (before, after) in enumerate(
            zip(original_track["centroids"], edited_track["centroids"], strict=True)
        )
        if before is not None and after is not None
    ]
    if not common:
        raise ValueError("Original and edited tracks have no jointly visible selected frames")
    original_points = np.asarray([original_track["centroids"][index] for index in common])
    edited_points = np.asarray([edited_track["centroids"][index] for index in common])
    commanded_delta = np.asarray([command[index] for index in common]) - np.asarray(
        [original_command[index] for index in common]
    )
    generated_delta = edited_points - original_points
    command_net = np.asarray(command[-1], dtype=np.float64) - np.asarray(
        command[0], dtype=np.float64
    )
    generated_net = edited_actual[-1] - edited_actual[0]
    endpoint_error = float(np.linalg.norm(edited_actual[-1] - np.asarray(command[indices[-1]])))
    original_endpoint_to_edited_command = float(
        np.linalg.norm(
            np.asarray(original_track["centroids"][common[-1]]) - np.asarray(command[common[-1]])
        )
    )
    direction_cosine = _cosine(generated_net, command_net)
    generated_displacement = float(np.linalg.norm(generated_net))
    commanded_displacement = float(np.linalg.norm(command_net))
    mean_delta_cosines = [
        value
        for value in (
            _cosine(generated, target)
            for generated, target in zip(generated_delta, commanded_delta)
        )
        if value is not None
    ]
    unrelated = unrelated_track_stability(original_labels, edited_labels, selected_label=label)
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "step600_counterfactual_trajectory_control_evaluation",
        "selected_instance_id": instance_id,
        "selected_class": edit_manifest.get("selected_class"),
        "selected_label": label,
        "trajectory_error_pixels": {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
            "endpoint": endpoint_error,
            "observed_frames": len(distances),
            "missing_frames": edited_track["missing_frames"],
        },
        "movement": {
            "commanded_vector_pixels": command_net.tolist(),
            "commanded_displacement_pixels": commanded_displacement,
            "generated_vector_pixels": generated_net.tolist(),
            "generated_displacement_pixels": generated_displacement,
            "direction_cosine": direction_cosine,
            "displacement_ratio": (
                generated_displacement / commanded_displacement if commanded_displacement else None
            ),
            "mean_per_frame_counterfactual_delta_direction_cosine": (
                float(np.mean(mean_delta_cosines)) if mean_delta_cosines else None
            ),
            "original_output_endpoint_error_to_edited_command_pixels": (
                original_endpoint_to_edited_command
            ),
        },
        "ellipse_alignment": ellipse_alignment(
            edited_labels,
            label=label,
            edited_metadata=edited_metadata,
            instance_id=instance_id,
        ),
        "unrelated_entity_stability": unrelated,
        "conventions": {
            "primary_test": "SAM2 selected-entity centroid versus commanded edited centroid",
            "not_primary": "PSNR/SSIM against original ground-truth motion",
            "counterfactual_comparison": (
                "same first RGB frame/checkpoint/prompt/seed/guidance/denoising; conditioning "
                "trajectory is the only intended changed variable"
            ),
        },
    }
    if original_video is not None and edited_video is not None:
        result["unrelated_pixel_stability"] = background_pixel_stability(
            original_video,
            edited_video,
            original_labels,
            edited_labels,
            selected_label=label,
        )
    stable = unrelated["mean_centroid_change_pixels"]
    checks = {
        "direction_cosine_at_least_0.5": direction_cosine is not None and direction_cosine >= 0.5,
        "generated_displacement_at_least_25_percent_of_command": (
            commanded_displacement > 0 and generated_displacement / commanded_displacement >= 0.25
        ),
        "edited_endpoint_closer_than_original_by_20px": (
            endpoint_error + 20.0 <= original_endpoint_to_edited_command
        ),
        "unrelated_mean_centroid_change_at_most_25px": stable is not None and stable <= 25.0,
    }
    result["control_evidence"] = {
        "success": all(checks.values()),
        "checks": checks,
        "thresholds": "predeclared reproduction conventions, not disclosed by the paper",
    }
    return result


def make_contact_sheet(
    *,
    original_conditioning: Path,
    edited_conditioning: Path,
    original_generated: Path,
    edited_generated: Path,
    edit_manifest: Mapping[str, object],
    original_labels: np.ndarray,
    edited_labels: np.ndarray,
    output: Path,
) -> None:
    videos = [
        decode_rgb_video(original_conditioning),
        decode_rgb_video(edited_conditioning),
        decode_rgb_video(original_generated),
        decode_rgb_video(edited_generated),
    ]
    labels = ["original condition", "edited condition", "original output", "edited output"]
    frame_indices = [0, 24, 48, 72, 96]
    tile_size = (384, 288)
    sheet = Image.new("RGB", (tile_size[0] * 4, tile_size[1] * len(frame_indices)), "black")
    draw = ImageDraw.Draw(sheet)
    selected_label = instance_label(str(edit_manifest["selected_instance_id"]))
    command = edit_manifest["edited_centroids"]
    for row, frame_index in enumerate(frame_indices):
        for column, frames in enumerate(videos):
            image = Image.fromarray(frames[frame_index]).resize(tile_size, Image.Resampling.LANCZOS)
            sheet.paste(image, (column * tile_size[0], row * tile_size[1]))
            x0, y0 = column * tile_size[0], row * tile_size[1]
            draw.rectangle((x0, y0, x0 + 170, y0 + 23), fill="black")
            draw.text((x0 + 5, y0 + 4), f"{labels[column]}  f={frame_index}", fill="white")
        scale_x, scale_y = tile_size[0] / TARGET_WIDTH, tile_size[1] / TARGET_HEIGHT
        commanded = command[frame_index]
        cx = 3 * tile_size[0] + commanded[0] * scale_x
        cy = row * tile_size[1] + commanded[1] * scale_y
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline="yellow", width=3)
        for column, tracked in ((2, original_labels), (3, edited_labels)):
            ys, xs = np.nonzero(tracked[frame_index] == selected_label)
            if len(xs):
                px = column * tile_size[0] + float(xs.mean()) * scale_x
                py = row * tile_size[1] + float(ys.mean()) * scale_y
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline="red", width=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def run_gpu_evaluation(
    *,
    original_generated: Path,
    edited_generated: Path,
    first_mask: Path | None,
    source_labels: Path | None,
    sam2_root: Path,
    sam2_checkpoint: Path,
    edit_manifest_path: Path,
    edited_metadata_path: Path,
    original_conditioning: Path,
    edited_conditioning: Path,
    output_dir: Path,
) -> dict[str, object]:
    expected = {"width": 1024, "height": 768, "frames": 97, "fps": 24.0}
    for path in (original_generated, edited_generated, original_conditioning, edited_conditioning):
        if probe_video(path) != expected:
            raise ValueError(f"Video violates trajectory-control contract: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (first_mask is None) == (source_labels is None):
        raise ValueError("Choose exactly one of first_mask or source_labels")
    if source_labels is not None:
        first_mask = output_dir / "source-first-frame-labels.png"
        Image.fromarray(load_labels(source_labels)[0].astype(np.uint8)).save(first_mask)
    assert first_mask is not None
    runner = Sam2VideoRunner(sam2_root, sam2_checkpoint)
    original_masks = output_dir / "original-generated-sam2.npz"
    edited_masks = output_dir / "edited-generated-sam2.npz"
    runner.run(original_generated, first_mask, original_masks)
    runner.run(edited_generated, first_mask, edited_masks)
    original_labels, edited_labels = load_labels(original_masks), load_labels(edited_masks)
    edit_manifest = json.loads(edit_manifest_path.read_text(encoding="utf-8"))
    edited_metadata = json.loads(edited_metadata_path.read_text(encoding="utf-8"))
    result = evaluate_control(
        original_labels=original_labels,
        edited_labels=edited_labels,
        edit_manifest=edit_manifest,
        edited_metadata=edited_metadata,
        original_video=original_generated,
        edited_video=edited_generated,
    )
    result["inputs"] = {
        "original_generated": str(original_generated),
        "edited_generated": str(edited_generated),
        "first_mask": str(first_mask),
        "source_labels": None if source_labels is None else str(source_labels),
        "original_conditioning": str(original_conditioning),
        "edited_conditioning": str(edited_conditioning),
        "edit_manifest": str(edit_manifest_path),
        "edited_metadata": str(edited_metadata_path),
    }
    make_contact_sheet(
        original_conditioning=original_conditioning,
        edited_conditioning=edited_conditioning,
        original_generated=original_generated,
        edited_generated=edited_generated,
        edit_manifest=edit_manifest,
        original_labels=original_labels,
        edited_labels=edited_labels,
        output=output_dir / "synchronized-contact-sheet.png",
    )
    output = output_dir / "control-metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-generated", required=True, type=Path)
    parser.add_argument("--edited-generated", required=True, type=Path)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--first-mask", type=Path)
    prompts.add_argument("--source-labels", type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--edit-manifest", required=True, type=Path)
    parser.add_argument("--edited-metadata", required=True, type=Path)
    parser.add_argument("--original-conditioning", required=True, type=Path)
    parser.add_argument("--edited-conditioning", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_gpu_evaluation(
        original_generated=args.original_generated,
        edited_generated=args.edited_generated,
        first_mask=args.first_mask,
        source_labels=args.source_labels,
        sam2_root=args.sam2_root,
        sam2_checkpoint=args.sam2_checkpoint,
        edit_manifest_path=args.edit_manifest,
        edited_metadata_path=args.edited_metadata,
        original_conditioning=args.original_conditioning,
        edited_conditioning=args.edited_conditioning,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recover ellipse-control geometry and compare it with generated-entity tracks.

Generated RGB is never scored with PSNR/SSIM against the black-canvas ellipse
video.  Controls are parsed through the paper-36 red/green class encoding and
blue relative-depth channel.  Generated entities must be recovered independently
from RGB (detector crops + independent depth).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from or_video_reproduction.geometry.palette import PAPER_36_PALETTE, decode_red_green

from .protocol import (
    IDENTITY_SWITCH_DISTANCE,
    box_iou,
    linear_sum_assignment,
)

PALETTE_TOLERANCE = 8
MIN_COMPONENT_PIXELS = 25


def _palette_lookup() -> dict[tuple[int, int], str]:
    return {color: label for label, color in PAPER_36_PALETTE.items()}


def parse_ellipse_frame(frame: np.ndarray) -> list[dict[str, object]]:
    """Parse class-colored ellipses from one conditioning frame."""

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Ellipse frame must have shape [H,W,3]")
    palette = _palette_lookup()
    occupied = np.any(image != 0, axis=2)
    instances: list[dict[str, object]] = []
    visited = np.zeros(occupied.shape, dtype=bool)
    height, width = occupied.shape
    neighbor = ((1, 0), (-1, 0), (0, 1), (0, -1))
    ys, xs = np.nonzero(occupied)
    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=True):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        pixels: list[tuple[int, int]] = []
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in neighbor:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and occupied[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) < MIN_COMPONENT_PIXELS:
            continue
        coords = np.asarray(pixels)
        colors = image[coords[:, 0], coords[:, 1]]
        red = int(np.median(colors[:, 0]))
        green = int(np.median(colors[:, 1]))
        blue = float(np.mean(colors[:, 2]))
        class_name = None
        for (palette_red, palette_green), label in palette.items():
            if abs(red - palette_red) <= PALETTE_TOLERANCE and abs(green - palette_green) <= PALETTE_TOLERANCE:
                class_name = label
                break
        if class_name is None:
            try:
                class_name = decode_red_green(red, green)
            except KeyError:
                class_name = "unknown"
        ys_c, xs_c = coords[:, 0], coords[:, 1]
        box = [int(xs_c.min()), int(ys_c.min()), int(xs_c.max()) + 1, int(ys_c.max()) + 1]
        instances.append(
            {
                "class_name": class_name,
                "centroid": [float(xs_c.mean()), float(ys_c.mean())],
                "box_xyxy": box,
                "area": int(len(pixels)),
                "relative_depth_blue": blue / 255.0,
                "red": red,
                "green": green,
            }
        )
    instances.sort(key=lambda row: (str(row["class_name"]), row["centroid"][0], row["centroid"][1]))
    return instances


def parse_ellipse_video(frames: np.ndarray) -> list[list[dict[str, object]]]:
    return [parse_ellipse_frame(frame) for frame in np.asarray(frames)]


def track_instances(
    per_frame: Sequence[Sequence[Mapping[str, object]]],
    *,
    switch_distance: float = IDENTITY_SWITCH_DISTANCE,
) -> dict[str, object]:
    """Greedy class-aware centroid tracking with explicit identity-switch flags."""

    tracks: dict[int, dict[str, object]] = {}
    next_id = 0
    switches = 0
    misses = 0
    for frame_index, instances in enumerate(per_frame):
        unused = set(range(len(instances)))
        if frame_index == 0:
            for instance in instances:
                tracks[next_id] = {
                    "id": next_id,
                    "class_name": instance["class_name"],
                    "centroids": [instance["centroid"]],
                    "boxes": [instance["box_xyxy"]],
                    "depths": [instance.get("relative_depth_blue")],
                    "missing": [],
                }
                next_id += 1
            continue
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in tracks.items():
            last = next(
                (point for point in reversed(track["centroids"]) if point is not None),
                None,
            )
            if last is None:
                continue
            for instance_index, instance in enumerate(instances):
                if instance["class_name"] != track["class_name"]:
                    continue
                distance = float(np.linalg.norm(np.asarray(instance["centroid"]) - np.asarray(last)))
                candidates.append((distance, track_id, instance_index))
        candidates.sort()
        used_tracks: set[int] = set()
        for distance, track_id, instance_index in candidates:
            if track_id in used_tracks or instance_index not in unused:
                continue
            if distance > switch_distance:
                switches += 1
                continue
            instance = instances[instance_index]
            track = tracks[track_id]
            track["centroids"].append(instance["centroid"])
            track["boxes"].append(instance["box_xyxy"])
            track["depths"].append(instance.get("relative_depth_blue"))
            used_tracks.add(track_id)
            unused.remove(instance_index)
        for track_id, track in tracks.items():
            if len(track["centroids"]) == frame_index:
                track["centroids"].append(None)
                track["boxes"].append(None)
                track["depths"].append(None)
                track["missing"].append(frame_index)
                misses += 1
        for instance_index in sorted(unused):
            instance = instances[instance_index]
            tracks[next_id] = {
                "id": next_id,
                "class_name": instance["class_name"],
                "centroids": [None] * frame_index + [instance["centroid"]],
                "boxes": [None] * frame_index + [instance["box_xyxy"]],
                "depths": [None] * frame_index + [instance.get("relative_depth_blue")],
                "missing": list(range(frame_index)),
            }
            next_id += 1
            switches += 1
    return {
        "tracks": list(tracks.values()),
        "identity_switches": switches,
        "missed_assignments": misses,
        "switch_distance_pixels": switch_distance,
    }


def _paired_centroids(
    left: Sequence[list[float] | None], right: Sequence[list[float] | None]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    indices = [
        index
        for index, (first, second) in enumerate(zip(left, right, strict=True))
        if first is not None and second is not None
    ]
    if not indices:
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    return (
        np.asarray([left[index] for index in indices], dtype=np.float64),
        np.asarray([right[index] for index in indices], dtype=np.float64),
        indices,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denom == 0 else float(left @ right / denom)


def compare_tracks(
    control: Mapping[str, object],
    generated: Mapping[str, object],
) -> dict[str, object]:
    """Match generated tracks to control tracks and score geometry, not RGB PSNR."""

    control_tracks = list(control["tracks"])
    generated_tracks = list(generated["tracks"])
    per_entity: list[dict[str, object]] = []
    unmatched_control = 0
    unmatched_generated = 0
    if not control_tracks:
        return {
            "status": "unavailable",
            "reason": "control video contained no parsed ellipses",
            "psnr_against_ellipse_rgb": "forbidden",
        }
    if not generated_tracks:
        return {
            "status": "ok",
            "entity_detection_rate": 0.0,
            "unmatched_control_tracks": len(control_tracks),
            "unmatched_generated_tracks": 0,
            "identity_switches_generated": generated.get("identity_switches"),
            "per_entity": [],
            "psnr_against_ellipse_rgb": "forbidden",
        }

    cost = np.zeros((len(control_tracks), len(generated_tracks)), dtype=np.float64)
    for row, control_track in enumerate(control_tracks):
        for col, generated_track in enumerate(generated_tracks):
            penalty = 0.0 if control_track["class_name"] == generated_track["class_name"] else 1e6
            left, right, _ = _paired_centroids(control_track["centroids"], generated_track["centroids"])
            distance = float(np.mean(np.linalg.norm(left - right, axis=1))) if len(left) else 1e5
            cost[row, col] = distance + penalty
    rows, cols = linear_sum_assignment(cost)
    used_generated = set(int(value) for value in cols)
    used_control = set(int(value) for value in rows)
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        control_track = control_tracks[row]
        generated_track = generated_tracks[col]
        left, right, indices = _paired_centroids(control_track["centroids"], generated_track["centroids"])
        class_match = control_track["class_name"] == generated_track["class_name"]
        if not indices:
            unmatched_control += 1
            continue
        errors = np.linalg.norm(left - right, axis=1)
        command = np.asarray(left[-1]) - np.asarray(left[0])
        observed = np.asarray(right[-1]) - np.asarray(right[0])
        commanded_disp = float(np.linalg.norm(command))
        generated_disp = float(np.linalg.norm(observed))
        box_values = []
        for index in indices:
            control_box = control_track["boxes"][index]
            generated_box = generated_track["boxes"][index]
            if control_box is not None and generated_box is not None:
                box_values.append(box_iou(control_box, generated_box))
        control_depths = [control_track["depths"][index] for index in indices]
        generated_depths = [generated_track["depths"][index] for index in indices]
        depth_pairs = [
            (c, g)
            for c, g in zip(control_depths, generated_depths, strict=True)
            if c is not None and g is not None
        ]
        depth_order = None
        if len(depth_pairs) >= 2:
            control_order = np.argsort([pair[0] for pair in depth_pairs])
            generated_order = np.argsort([pair[1] for pair in depth_pairs])
            depth_order = float(np.mean(control_order == generated_order))
        per_entity.append(
            {
                "control_class": control_track["class_name"],
                "generated_class": generated_track["class_name"],
                "class_consistent": class_match,
                "observed_frames": len(indices),
                "control_missing_frames": len(control_track["missing"]),
                "generated_missing_frames": len(generated_track["missing"]),
                "mean_centroid_error_pixels": float(errors.mean()),
                "endpoint_error_pixels": float(errors[-1]),
                "direction_cosine": _cosine(observed, command),
                "displacement_ratio": (
                    None if commanded_disp == 0 else generated_disp / commanded_disp
                ),
                "bbox_iou_to_ellipse": float(np.mean(box_values)) if box_values else None,
                "relative_depth_order_agreement": depth_order,
            }
        )
    unmatched_control += len(control_tracks) - len(used_control)
    unmatched_generated += len(generated_tracks) - len(used_generated)
    detection_rate = 1.0 - (unmatched_control / len(control_tracks))
    class_values = [1.0 if row["class_consistent"] else 0.0 for row in per_entity]
    return {
        "status": "ok",
        "entity_detection_rate": float(detection_rate),
        "unmatched_control_tracks": unmatched_control,
        "unmatched_generated_tracks": unmatched_generated,
        "identity_switches_control": control.get("identity_switches"),
        "identity_switches_generated": generated.get("identity_switches"),
        "class_consistency": float(np.mean(class_values)) if class_values else None,
        "mean_centroid_error_pixels": (
            float(np.mean([row["mean_centroid_error_pixels"] for row in per_entity]))
            if per_entity
            else None
        ),
        "mean_endpoint_error_pixels": (
            float(np.mean([row["endpoint_error_pixels"] for row in per_entity])) if per_entity else None
        ),
        "per_entity": per_entity,
        "psnr_against_ellipse_rgb": "forbidden",
        "ssim_against_ellipse_rgb": "forbidden",
    }


def detections_to_frame_instances(
    detections: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Convert detector outputs {frame, class_name, box, depth?} into track input."""

    if not detections:
        return {"tracks": [], "identity_switches": 0, "missed_assignments": 0}
    max_frame = max(int(row["frame"]) for row in detections)
    per_frame: list[list[dict[str, object]]] = [[] for _ in range(max_frame + 1)]
    for row in detections:
        box = [int(value) for value in row["box_xyxy"]]
        centroid = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
        per_frame[int(row["frame"])].append(
            {
                "class_name": row.get("class_name", "unknown"),
                "centroid": centroid,
                "box_xyxy": box,
                "relative_depth_blue": row.get("relative_depth"),
                "area": max(0, (box[2] - box[0]) * (box[3] - box[1])),
            }
        )
    return track_instances(per_frame)

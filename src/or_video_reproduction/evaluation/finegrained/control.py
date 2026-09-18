"""Recover ellipse-control geometry and compare it with generated-entity tracks.

Generated RGB is never scored with PSNR/SSIM against the black-canvas ellipse
video.  Controls are parsed through the paper-36 red/green class encoding and
blue relative-depth channel.  Generated entities must be recovered independently
from RGB (detector crops + independent depth).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from or_video_reproduction.data.semantics import ENTITY_CLASSES
from or_video_reproduction.geometry.palette import PAPER_36_PALETTE, decode_red_green

from .external import unavailable_metric
from .protocol import (
    IDENTITY_SWITCH_DISTANCE,
    box_iou,
    linear_sum_assignment,
)

RECOVERY_SCOPE = "person_only_plus_independent_depth"
CLASS_CONSISTENCY_UNAVAILABLE_REASON = (
    "Independent RGB recovery uses DETR COCO 'person'; ellipse controls use paper-36 "
    "OR role/entity classes. There is no documented mapping from COCO person onto those "
    "labels, so class consistency is unavailable rather than 0."
)
TAXONOMY_DECISION_REQUIRED = (
    "Provide a source-backed mapping from independently recovered generated classes to "
    "paper-36 entity labels, or recover those labels with a detector that emits paper-36 "
    "classes. Until then, class-filtered matching stays blocked. COCO person is not a "
    "paper-36 class and must not be equated with nurse, patient, head_surgeon, or equipment."
)
INCOMPATIBLE_MATCH_REASON = (
    "Generated people are not matched to unsupported non-person or unmapped OR-role "
    "control entities for geometry or relative-depth scores."
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


def control_class_mapping() -> dict[str, Any]:
    """Documented mapping from independent detector labels onto ellipse classes.

    None exists. Paper-36 entity classes are the ellipse taxonomy. DETR emits COCO
    ``person``, which is not in that taxonomy. This function does not invent a
    person/non-person or person-to-OR-role mapping.
    """

    return {
        "status": "blocked",
        "documented_mapping": None,
        "detector_label_space": ["person"],
        "ellipse_entity_classes": list(ENTITY_CLASSES),
        "reason": CLASS_CONSISTENCY_UNAVAILABLE_REASON,
        "taxonomy_decision_required": TAXONOMY_DECISION_REQUIRED,
        "recovery_scope": RECOVERY_SCOPE,
    }


def _count_by_class(names: Sequence[object]) -> dict[str, int]:
    counts = Counter(str(name) for name in names)
    return dict(sorted(counts.items()))


def _recoverable_classes(
    generated_tracks: Sequence[Mapping[str, object]],
    *,
    detector_label_space: Sequence[str] | None,
    mapping: Mapping[str, Any],
) -> set[str]:
    if mapping.get("documented_mapping"):
        return {str(value) for value in mapping["documented_mapping"].values()}
    if detector_label_space is not None:
        return {str(name) for name in detector_label_space}
    return {str(track["class_name"]) for track in generated_tracks}


def _classes_compatible(
    control_class: str,
    generated_class: str,
    *,
    mapping: Mapping[str, Any],
) -> bool:
    documented = mapping.get("documented_mapping")
    if documented:
        return documented.get(generated_class) == control_class
    return control_class == generated_class


def _class_consistency_payload(recovered: set[str], mapping: Mapping[str, Any]) -> dict[str, Any]:
    paper36 = set(ENTITY_CLASSES)
    if mapping.get("status") == "blocked" and not recovered <= paper36:
        return unavailable_metric(
            "class_consistency",
            str(mapping.get("reason") or CLASS_CONSISTENCY_UNAVAILABLE_REASON),
            taxonomy_decision_required=mapping.get("taxonomy_decision_required"),
            recovered_classes=sorted(recovered),
        )
    if not recovered <= paper36:
        return unavailable_metric(
            "class_consistency",
            "Recovered generated classes are outside the paper-36 ellipse taxonomy; "
            "class consistency is not scored as 0",
            recovered_classes=sorted(recovered),
        )
    return {"status": "ok"}


def _entity_score(
    control_track: Mapping[str, object],
    generated_track: Mapping[str, object],
) -> dict[str, Any] | None:
    left, right, indices = _paired_centroids(control_track["centroids"], generated_track["centroids"])
    if not indices:
        return None
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
    return {
        "control_class": control_track["class_name"],
        "generated_class": generated_track["class_name"],
        "match_status": "matched_class_compatible",
        "class_consistent": control_track["class_name"] == generated_track["class_name"],
        "observed_frames": len(indices),
        "control_missing_frames": len(control_track["missing"]),
        "generated_missing_frames": len(generated_track["missing"]),
        "mean_centroid_error_pixels": float(errors.mean()),
        "endpoint_error_pixels": float(errors[-1]),
        "direction_cosine": _cosine(observed, command),
        "displacement_ratio": None if commanded_disp == 0 else generated_disp / commanded_disp,
        "bbox_iou_to_ellipse": float(np.mean(box_values)) if box_values else None,
        "relative_depth_order_agreement": depth_order,
    }


def _unmatched_control_record(track: Mapping[str, object], *, reason: str, supported: bool) -> dict[str, Any]:
    return {
        "control_class": track["class_name"],
        "generated_class": None,
        "match_status": "supported_unmatched" if supported else "unsupported",
        "class_consistent": None,
        "observed_frames": 0,
        "control_missing_frames": len(track.get("missing") or []),
        "generated_missing_frames": None,
        "mean_centroid_error_pixels": None,
        "endpoint_error_pixels": None,
        "direction_cosine": None,
        "displacement_ratio": None,
        "bbox_iou_to_ellipse": None,
        "relative_depth_order_agreement": None,
        "reason": reason,
    }


def _fidelity_mean(per_entity: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [
        row[key]
        for row in per_entity
        if row.get("match_status") == "matched_class_compatible" and isinstance(row.get(key), (int, float))
    ]
    return float(np.mean(values)) if values else None


def _entity_count_payload(
    *,
    control_names: Sequence[object],
    generated_names: Sequence[object],
    supported_names: Sequence[object],
    unsupported_names: Sequence[object],
    matched_names: Sequence[object],
    unmatched_supported_names: Sequence[object],
    unmatched_generated_names: Sequence[object],
) -> dict[str, Any]:
    return {
        "control_entities_by_class": _count_by_class(control_names),
        "generated_entities_by_class": _count_by_class(generated_names),
        "supported_control_entities_by_class": _count_by_class(supported_names),
        "unsupported_control_entities_by_class": _count_by_class(unsupported_names),
        "matched_control_entities_by_class": _count_by_class(matched_names),
        "unmatched_supported_control_entities_by_class": _count_by_class(unmatched_supported_names),
        "unmatched_generated_entities_by_class": _count_by_class(unmatched_generated_names),
        "control_entity_count": len(control_names),
        "generated_entity_count": len(generated_names),
        "supported_control_entity_count": len(supported_names),
        "unsupported_control_entity_count": len(unsupported_names),
        "matched_control_entity_count": len(matched_names),
        "unmatched_supported_control_entity_count": len(unmatched_supported_names),
        "unmatched_generated_entity_count": len(unmatched_generated_names),
    }


def compare_tracks(
    control: Mapping[str, object],
    generated: Mapping[str, object],
    *,
    detector_label_space: Sequence[str] | None = None,
) -> dict[str, object]:
    """Match generated tracks to control tracks and score geometry, not RGB PSNR.

    Current independent recovery is person-only plus Depth Anything V2. Class-filtered
    matching stays blocked until a documented COCO-person → paper-36 mapping exists.
    Unsupported and unmatched entities affect coverage, not zero-valued fidelity scores.
    """

    mapping = control_class_mapping()
    control_tracks = list(control["tracks"])
    generated_tracks = list(generated["tracks"])
    recovered = _recoverable_classes(
        generated_tracks, detector_label_space=detector_label_space, mapping=mapping
    )
    consistency = _class_consistency_payload(recovered, mapping)
    if not control_tracks:
        return {
            "status": "unavailable",
            "reason": "control video contained no parsed ellipses",
            "recovery_scope": RECOVERY_SCOPE,
            "class_mapping": mapping,
            "class_consistency": consistency if consistency.get("status") != "ok" else unavailable_metric(
                "class_consistency", "control video contained no parsed ellipses"
            ),
            "psnr_against_ellipse_rgb": "forbidden",
            "ssim_against_ellipse_rgb": "forbidden",
        }

    control_names = [track["class_name"] for track in control_tracks]
    generated_names = [track["class_name"] for track in generated_tracks]
    # Ellipse tracks are supported only when the independent detector can emit that
    # paper-36 class. COCO person is not a paper-36 class, so DETR person-only
    # recovery supports none of them.
    supported_idx = [
        index
        for index, track in enumerate(control_tracks)
        if str(track["class_name"]) in recovered and str(track["class_name"]) in set(ENTITY_CLASSES)
    ]
    unsupported_idx = [index for index in range(len(control_tracks)) if index not in set(supported_idx)]
    supported_names = [control_tracks[index]["class_name"] for index in supported_idx]
    unsupported_names = [control_tracks[index]["class_name"] for index in unsupported_idx]

    forbidden = {
        "psnr_against_ellipse_rgb": "forbidden",
        "ssim_against_ellipse_rgb": "forbidden",
        "recovery_scope": RECOVERY_SCOPE,
        "class_mapping": mapping,
        "generated_class_note": (
            "Independent RGB recovery is person-only plus independent Depth Anything V2, "
            "not complete control fidelity. DETR COCO person is not a paper-36 OR role."
        ),
    }

    if not generated_tracks:
        payload = {
            "status": "ok",
            "entity_detection_rate": 0.0,
            "unmatched_control_tracks": len(control_tracks),
            "unmatched_generated_tracks": 0,
            "identity_switches_generated": generated.get("identity_switches"),
            "identity_switches_control": control.get("identity_switches"),
            "class_consistency": consistency if consistency.get("status") != "ok" else unavailable_metric(
                "class_consistency", "no generated entities were independently recovered"
            ),
            "mean_centroid_error_pixels": None,
            "mean_endpoint_error_pixels": None,
            "per_entity": [
                _unmatched_control_record(
                    control_tracks[index],
                    reason="no generated entities recovered" if index in set(supported_idx) else INCOMPATIBLE_MATCH_REASON,
                    supported=index in set(supported_idx),
                )
                for index in range(len(control_tracks))
            ],
            "entity_counts": _entity_count_payload(
                control_names=control_names,
                generated_names=[],
                supported_names=supported_names,
                unsupported_names=unsupported_names,
                matched_names=[],
                unmatched_supported_names=supported_names,
                unmatched_generated_names=[],
            ),
            **forbidden,
        }
        return payload

    cost = np.full((len(control_tracks), len(generated_tracks)), 1e12, dtype=np.float64)
    for row, control_track in enumerate(control_tracks):
        for col, generated_track in enumerate(generated_tracks):
            if not _classes_compatible(
                str(control_track["class_name"]),
                str(generated_track["class_name"]),
                mapping=mapping,
            ):
                continue
            if str(control_track["class_name"]) not in set(ENTITY_CLASSES):
                continue
            left, right, _ = _paired_centroids(control_track["centroids"], generated_track["centroids"])
            cost[row, col] = float(np.mean(np.linalg.norm(left - right, axis=1))) if len(left) else 1e5

    rows, cols = linear_sum_assignment(cost)
    used_generated: set[int] = set()
    used_control: set[int] = set()
    per_entity: list[dict[str, object]] = []
    matched_names: list[object] = []
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if float(cost[row, col]) >= 1e11:
            continue
        scored = _entity_score(control_tracks[row], generated_tracks[col])
        if scored is None:
            continue
        per_entity.append(scored)
        used_control.add(row)
        used_generated.add(col)
        matched_names.append(control_tracks[row]["class_name"])

    unmatched_supported_names: list[object] = []
    for index, track in enumerate(control_tracks):
        if index in used_control:
            continue
        supported = index in set(supported_idx)
        if supported:
            unmatched_supported_names.append(track["class_name"])
            reason = "supported control entity had no class-compatible generated match"
        else:
            reason = INCOMPATIBLE_MATCH_REASON
        per_entity.append(_unmatched_control_record(track, reason=reason, supported=supported))

    unmatched_generated_names = [
        generated_tracks[index]["class_name"]
        for index in range(len(generated_tracks))
        if index not in used_generated
    ]
    unmatched_control = len(control_tracks) - len(used_control)
    unmatched_generated = len(generated_tracks) - len(used_generated)
    detection_rate = len(used_control) / len(control_tracks)
    if consistency.get("status") == "ok":
        matched_flags = [
            1.0 if row.get("class_consistent") else 0.0
            for row in per_entity
            if row.get("match_status") == "matched_class_compatible"
        ]
        class_consistency: object = (
            float(np.mean(matched_flags)) if matched_flags else unavailable_metric(
                "class_consistency", "no class-compatible matches to score"
            )
        )
    else:
        class_consistency = consistency

    return {
        "status": "ok",
        "entity_detection_rate": float(detection_rate),
        "unmatched_control_tracks": unmatched_control,
        "unmatched_generated_tracks": unmatched_generated,
        "identity_switches_control": control.get("identity_switches"),
        "identity_switches_generated": generated.get("identity_switches"),
        "class_consistency": class_consistency,
        "mean_centroid_error_pixels": _fidelity_mean(per_entity, "mean_centroid_error_pixels"),
        "mean_endpoint_error_pixels": _fidelity_mean(per_entity, "endpoint_error_pixels"),
        "per_entity": per_entity,
        "entity_counts": _entity_count_payload(
            control_names=control_names,
            generated_names=generated_names,
            supported_names=supported_names,
            unsupported_names=unsupported_names,
            matched_names=matched_names,
            unmatched_supported_names=unmatched_supported_names,
            unmatched_generated_names=unmatched_generated_names,
        ),
        **forbidden,
    }


def reaggregate_control_clip(record: Mapping[str, object]) -> dict[str, object]:
    """Recompute control summaries from a saved per-clip record. No model loads."""

    if not isinstance(record, Mapping):
        raise ValueError("control per-clip record must be an object")
    status = record.get("status")
    if status in {"unavailable", "blocked", "failed"}:
        return dict(record)
    if status != "ok":
        raise ValueError(f"control per-clip status {status!r} is not reaggregable")
    per_entity = record.get("per_entity")
    if not isinstance(per_entity, list):
        raise ValueError("control per-clip record is missing per_entity")
    mapping = control_class_mapping()
    generated_classes = {
        str(row["generated_class"])
        for row in per_entity
        if row.get("generated_class") not in (None, "")
    }
    if record.get("generated_entities_by_class"):
        generated_classes.update(str(name) for name in record["generated_entities_by_class"])
    counts_in = record.get("entity_counts") if isinstance(record.get("entity_counts"), Mapping) else {}
    if counts_in.get("generated_entities_by_class"):
        generated_classes.update(str(name) for name in counts_in["generated_entities_by_class"])
    detector_space = mapping["detector_label_space"]
    recovered = set(detector_space) | generated_classes
    consistency = _class_consistency_payload(recovered, mapping)

    rewritten: list[dict[str, object]] = []
    matched_names: list[object] = []
    unmatched_supported: list[object] = []
    unsupported_names: list[object] = []
    control_names: list[object] = []
    generated_names: list[object] = []
    unmatched_generated: list[object] = []
    insufficient: list[str] = []

    for row in per_entity:
        if not isinstance(row, Mapping):
            raise ValueError("control per_entity entry is not an object")
        control_class = row.get("control_class")
        generated_class = row.get("generated_class")
        if control_class in (None, ""):
            raise ValueError("control per_entity entry is missing control_class")
        control_names.append(control_class)
        compatible = generated_class not in (None, "") and _classes_compatible(
            str(control_class), str(generated_class), mapping=mapping
        ) and str(control_class) in set(ENTITY_CLASSES)
        if compatible:
            updated = dict(row)
            updated["match_status"] = "matched_class_compatible"
            updated["class_consistent"] = str(control_class) == str(generated_class)
            rewritten.append(updated)
            matched_names.append(control_class)
            generated_names.append(generated_class)
            continue
        if generated_class not in (None, ""):
            generated_names.append(generated_class)
            unmatched_generated.append(generated_class)
        supported = str(control_class) in set(ENTITY_CLASSES) and str(control_class) in recovered
        if supported:
            unmatched_supported.append(control_class)
            reason = "saved match was class-incompatible; geometry discarded"
        else:
            unsupported_names.append(control_class)
            reason = INCOMPATIBLE_MATCH_REASON
        rewritten.append(
            {
                "control_class": control_class,
                "generated_class": None,
                "match_status": "supported_unmatched" if supported else "unsupported",
                "class_consistent": None,
                "observed_frames": 0,
                "control_missing_frames": row.get("control_missing_frames"),
                "generated_missing_frames": None,
                "mean_centroid_error_pixels": None,
                "endpoint_error_pixels": None,
                "direction_cosine": None,
                "displacement_ratio": None,
                "bbox_iou_to_ellipse": None,
                "relative_depth_order_agreement": None,
                "reason": reason,
                "discarded_incompatible_generated_class": generated_class,
            }
        )

    extra_unmatched = record.get("unmatched_control_tracks")
    known_unmatched = len(control_names) - len(matched_names)
    extra_count = 0
    if extra_unmatched is not None:
        extra_count = max(0, int(extra_unmatched) - known_unmatched)
        if extra_count:
            insufficient.append(
                f"{extra_count} unmatched control tracks were stored only as a count, "
                "without class labels"
            )
    extra_gen = record.get("unmatched_generated_tracks")
    extra_gen_count = 0
    if extra_gen is not None:
        extra_gen_count = max(0, int(extra_gen) - len(unmatched_generated))
        if extra_gen_count:
            insufficient.append(
                f"{extra_gen_count} unmatched generated tracks were stored only as a count, "
                "without class labels"
            )

    total_control = len(control_names) + extra_count
    entity_counts = _entity_count_payload(
        control_names=control_names,
        generated_names=generated_names,
        supported_names=matched_names + unmatched_supported,
        unsupported_names=unsupported_names,
        matched_names=matched_names,
        unmatched_supported_names=unmatched_supported,
        unmatched_generated_names=unmatched_generated,
    )
    entity_counts["unmatched_control_tracks_without_saved_class"] = extra_count
    entity_counts["unmatched_generated_tracks_without_saved_class"] = extra_gen_count
    payload = dict(record)
    payload.update(
        {
            "status": "ok",
            "recovery_scope": RECOVERY_SCOPE,
            "class_mapping": mapping,
            "class_consistency": consistency if consistency.get("status") != "ok" else (
                float(np.mean([1.0 if row.get("class_consistent") else 0.0 for row in rewritten if row.get("match_status") == "matched_class_compatible"]))
                if matched_names
                else unavailable_metric("class_consistency", "no class-compatible matches to score")
            ),
            "entity_detection_rate": (len(matched_names) / total_control) if total_control else 0.0,
            "unmatched_control_tracks": total_control - len(matched_names),
            "unmatched_generated_tracks": len(unmatched_generated) + extra_gen_count,
            "mean_centroid_error_pixels": _fidelity_mean(rewritten, "mean_centroid_error_pixels"),
            "mean_endpoint_error_pixels": _fidelity_mean(rewritten, "endpoint_error_pixels"),
            "per_entity": rewritten,
            "entity_counts": entity_counts,
            "psnr_against_ellipse_rgb": "forbidden",
            "ssim_against_ellipse_rgb": "forbidden",
            "generated_class_note": (
                "Independent RGB recovery is person-only plus independent Depth Anything V2, "
                "not complete control fidelity."
            ),
        }
    )
    if insufficient:
        payload["insufficient_for_complete_class_counts"] = insufficient
    return payload


def summarize_control_group(per_clip: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate clip-level person-only control coverage. Fidelity zeros are not invented."""

    mapping = control_class_mapping()
    rates = [
        float(row["entity_detection_rate"])
        for row in per_clip
        if isinstance(row, Mapping) and isinstance(row.get("entity_detection_rate"), (int, float))
    ]
    consistencies = [row.get("class_consistency") for row in per_clip if isinstance(row, Mapping)]
    class_reason = CLASS_CONSISTENCY_UNAVAILABLE_REASON
    numeric_consistency: list[float] = []
    saw_unavailable = False
    for value in consistencies:
        if isinstance(value, Mapping) and value.get("status") in {"unavailable", "blocked", "failed"}:
            saw_unavailable = True
            class_reason = str(value.get("reason") or class_reason)
        elif isinstance(value, (int, float)):
            numeric_consistency.append(float(value))
    if saw_unavailable:
        class_payload: object = unavailable_metric(
            "class_consistency",
            class_reason,
            taxonomy_decision_required=TAXONOMY_DECISION_REQUIRED,
        )
    elif numeric_consistency:
        class_payload = float(np.mean(numeric_consistency))
    else:
        class_payload = unavailable_metric(
            "class_consistency",
            class_reason,
            taxonomy_decision_required=TAXONOMY_DECISION_REQUIRED,
        )

    merged_counts: dict[str, Counter[str]] = {
        "control_entities_by_class": Counter(),
        "generated_entities_by_class": Counter(),
        "supported_control_entities_by_class": Counter(),
        "unsupported_control_entities_by_class": Counter(),
        "matched_control_entities_by_class": Counter(),
        "unmatched_supported_control_entities_by_class": Counter(),
        "unmatched_generated_entities_by_class": Counter(),
    }
    insufficient: list[str] = []
    for row in per_clip:
        if not isinstance(row, Mapping):
            continue
        counts = row.get("entity_counts")
        if isinstance(counts, Mapping):
            for key, counter in merged_counts.items():
                values = counts.get(key)
                if isinstance(values, Mapping):
                    counter.update({str(name): int(count) for name, count in values.items()})
        extra = row.get("insufficient_for_complete_class_counts")
        if extra:
            clip_id = row.get("clip_id")
            insufficient.append(f"{clip_id}: {extra}")
    return {
        "status": "ok",
        "recovery_scope": RECOVERY_SCOPE,
        "class_filtered_matching": mapping["status"],
        "class_mapping": mapping,
        "class_consistency": class_payload,
        "psnr_against_ellipse_rgb": "forbidden",
        "ssim_against_ellipse_rgb": "forbidden",
        "aggregate_entity_detection_rate": float(np.mean(rates)) if rates else None,
        "entity_counts": {key: dict(sorted(counter.items())) for key, counter in merged_counts.items()},
        "insufficient_for_complete_class_counts": insufficient,
        "generated_class_note": (
            "Independent RGB recovery is person-only plus independent Depth Anything V2, "
            "not complete control fidelity."
        ),
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

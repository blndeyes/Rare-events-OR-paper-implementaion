"""Six-case single-ellipse edited-only control suite (3 MMOR + 3 4D-OR)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from or_video_reproduction.data.clips import OUTPUT_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH
from or_video_reproduction.evaluation.corrected_inference import extract_first_frame
from or_video_reproduction.evaluation.trajectory_control import ELLIPSE_PROXY_INTERPRETATION
from or_video_reproduction.geometry.ellipse import Ellipse
from or_video_reproduction.geometry.trajectory import (
    axis_aligned_box,
    edit_trajectory,
    ellipse_in_canvas,
    interpolate_waypoints,
    render_edited_sequence,
)
from or_video_reproduction.evaluation.video import probe_video


HUMAN_ELLIPSE_CLASSES = frozenset(
    {
        "anaesthetist",
        "assistant_surgeon",
        "circulator",
        "head_surgeon",
        "nurse",
        "patient",
        "student",
    }
)

PREFERRED_CASES: tuple[dict[str, str], ...] = (
    {
        "case_id": "01",
        "domain": "MMOR",
        "preferred_clip_id": "mmor-008_PKA-camera01-timestamp000995",
        "style": "horizontal",
    },
    {
        "case_id": "02",
        "domain": "MMOR",
        "preferred_clip_id": "mmor-001_PKA-camera01-timestamp000210",
        "style": "diagonal",
    },
    {
        "case_id": "03",
        "domain": "MMOR",
        "preferred_clip_id": "mmor-037_PKA-camera01-timestamp000585",
        "style": "gentle_curve",
    },
    {
        "case_id": "04",
        "domain": "4D-OR",
        "preferred_clip_id": "4dor-take02-camera01-row000255",
        "style": "horizontal_opposite",
    },
    {
        "case_id": "05",
        "domain": "4D-OR",
        "preferred_clip_id": "4dor-take06-camera01-row000150",
        "style": "diagonal_other",
    },
    {
        "case_id": "06",
        "domain": "4D-OR",
        "preferred_clip_id": "4dor-take02-camera01-row000060",
        "style": "gentle_curve_other",
    },
)

STYLE_DIRECTIONS: dict[str, tuple[float, float]] = {
    "horizontal": (1.0, 0.0),
    "diagonal": (1.0, 1.0),
    "gentle_curve": (1.0, 0.0),
    "horizontal_opposite": (-1.0, 0.0),
    "diagonal_other": (-1.0, -1.0),
    "gentle_curve_other": (0.0, 1.0),
}

TARGET_DISPLACEMENT_MIN = 150.0
TARGET_DISPLACEMENT_MAX = 220.0
MIN_MEANINGFUL_FREE_SPACE = 80.0
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42


class ExperimentDirectoryExistsError(FileExistsError):
    """Raised when a non-empty experiment directory would be overwritten."""


def domain_from_clip_id(clip_id: str) -> str:
    if clip_id.startswith("mmor-"):
        return "MMOR"
    if clip_id.startswith("4dor-"):
        return "4D-OR"
    raise ValueError(f"Cannot infer domain from clip id {clip_id!r}")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_pair_samples(*manifests: Path) -> dict[str, dict[str, object]]:
    samples: dict[str, dict[str, object]] = {}
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
            raise ValueError(f"Invalid pair manifest: {manifest}")
        for row in payload["samples"]:
            clip_id = row.get("id")
            if not isinstance(clip_id, str):
                raise ValueError(f"Sample in {manifest} is missing an id")
            if clip_id in samples:
                raise ValueError(f"Duplicate clip id across pair manifests: {clip_id}")
            samples[clip_id] = {"manifest": str(manifest), **row}
    return samples


def held_out_ids(samples: Mapping[str, Mapping[str, object]], domain: str) -> list[str]:
    allowed_splits = {"heldout", "table1_ood"}
    result = []
    for clip_id, row in samples.items():
        if domain_from_clip_id(clip_id) != domain:
            continue
        split = row.get("split")
        if split not in allowed_splits and split is not None:
            continue
        result.append(clip_id)
    return sorted(result)


def _ellipse_from_instance(row: Mapping[str, object]) -> Ellipse:
    value = row["ellipse"]
    return Ellipse(
        center_x=float(value["center_x"]),
        center_y=float(value["center_y"]),
        major_diameter=float(value["major_diameter"]),
        minor_diameter=float(value["minor_diameter"]),
        angle_degrees=float(value["angle_degrees"]),
        source_pixels=int(value["source_pixels"]),
    )


def _boxes_overlap(left: Sequence[float], right: Sequence[float], *, margin: float = 8.0) -> bool:
    return not (
        left[2] + margin <= right[0]
        or right[2] + margin <= left[0]
        or left[3] + margin <= right[1]
        or right[3] + margin <= left[1]
    )


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0:
        raise ValueError("Direction vector must be non-zero")
    return vector[0] / norm, vector[1] / norm


def max_in_canvas_translation(
    ellipse: Ellipse,
    direction: tuple[float, float],
    others: Sequence[Ellipse],
    *,
    avoid_overlap: bool = True,
) -> float:
    ux, uy = _unit(direction)
    low, high = 0.0, 400.0
    best = 0.0
    for _ in range(24):
        mid = (low + high) / 2.0
        moved = replace(ellipse, center_x=ellipse.center_x + ux * mid, center_y=ellipse.center_y + uy * mid)
        if not ellipse_in_canvas(moved):
            high = mid
            continue
        if avoid_overlap and any(
            _boxes_overlap(axis_aligned_box(moved), axis_aligned_box(other)) for other in others
        ):
            high = mid
            continue
        best = mid
        low = mid
    return best


def plan_single_ellipse_waypoints(
    *,
    style: str,
    ellipse: Ellipse,
    others: Sequence[Ellipse],
    target_min: float = TARGET_DISPLACEMENT_MIN,
    target_max: float = TARGET_DISPLACEMENT_MAX,
) -> dict[str, object]:
    if style not in STYLE_DIRECTIONS:
        raise ValueError(f"Unsupported trajectory style: {style}")
    preferred = STYLE_DIRECTIONS[style]
    flipped = (-preferred[0], -preferred[1])
    with_avoid = max_in_canvas_translation(ellipse, preferred, others, avoid_overlap=True)
    without_avoid = max_in_canvas_translation(ellipse, preferred, others, avoid_overlap=False)
    direction = preferred
    free = with_avoid
    overlap_relaxed = False
    if free < target_min:
        flipped_avoid = max_in_canvas_translation(ellipse, flipped, others, avoid_overlap=True)
        if flipped_avoid > free:
            direction = flipped
            free = flipped_avoid
            without_avoid = max_in_canvas_translation(ellipse, flipped, others, avoid_overlap=False)
    if free < target_min and without_avoid >= min(target_min, MIN_MEANINGFUL_FREE_SPACE):
        free = without_avoid
        overlap_relaxed = True
    displacement = min(target_max, free)
    notes: list[str] = []
    if displacement < target_min:
        notes.append(
            f"scene constraints limited endpoint displacement to {displacement:.1f}px "
            f"(target {target_min:.0f}-{target_max:.0f}px)"
        )
    ux, uy = _unit(direction)
    start = (ellipse.center_x, ellipse.center_y)
    end = (start[0] + ux * displacement, start[1] + uy * displacement)
    if style.startswith("gentle_curve"):
        perp = (-uy, ux) if style == "gentle_curve" else (uy, -ux)
        bulge = min(55.0, max(24.0, displacement * 0.22))
        for scale in (1.0, 0.6, 0.3, 0.0):
            mid = (
                start[0] + 0.5 * (end[0] - start[0]) + perp[0] * bulge * scale,
                start[1] + 0.5 * (end[1] - start[1]) + perp[1] * bulge * scale,
            )
            waypoints = [start, mid, end]
            interpolated = interpolate_waypoints(waypoints, OUTPUT_FRAME_COUNT)
            if all(
                ellipse_in_canvas(
                    replace(ellipse, center_x=point[0], center_y=point[1])
                )
                for point in interpolated
            ):
                break
        else:
            waypoints = [start, end]
            interpolated = interpolate_waypoints(waypoints, OUTPUT_FRAME_COUNT)
    else:
        waypoints = [start, end]
        interpolated = interpolate_waypoints(waypoints, OUTPUT_FRAME_COUNT)
    validate_in_canvas_trajectory(ellipse, interpolated)
    return {
        "style": style,
        "preferred_direction": list(preferred),
        "used_direction": list(direction),
        "overlap_avoidance_relaxed": overlap_relaxed,
        "requested_displacement_pixels": float(
            math.hypot(end[0] - start[0], end[1] - start[1])
        ),
        "below_target_displacement": displacement < target_min,
        "notes": notes,
        "waypoints": [list(point) for point in waypoints],
        "interpolated_centroids": [list(point) for point in interpolated],
        "frame_count": len(interpolated),
        "interpolation": "Ramer-Douglas-Peucker then constant-speed arc-length resampling via edit_trajectory",
    }


def validate_in_canvas_trajectory(
    ellipse: Ellipse, centroids: Sequence[Sequence[float]]
) -> None:
    if len(centroids) != OUTPUT_FRAME_COUNT:
        raise ValueError(f"Trajectory must have {OUTPUT_FRAME_COUNT} centroids")
    for index, point in enumerate(centroids):
        moved = replace(ellipse, center_x=float(point[0]), center_y=float(point[1]))
        if not ellipse_in_canvas(moved):
            raise ValueError(f"Trajectory leaves the 1024x768 canvas at frame {index}")


def human_candidates(metadata: Mapping[str, object]) -> list[dict[str, object]]:
    frame0 = metadata["frames"][0]
    rows = []
    for row in frame0["instances"]:
        class_name = str(row.get("class_name"))
        if class_name not in HUMAN_ELLIPSE_CLASSES:
            continue
        ellipse = _ellipse_from_instance(row)
        if not ellipse_in_canvas(ellipse):
            continue
        if ellipse.major_diameter < 40.0 or ellipse.minor_diameter < 20.0:
            continue
        if ellipse.source_pixels < 200:
            continue
        rest = [
            _ellipse_from_instance(other)
            for other in frame0["instances"]
            if other.get("key") != row.get("key")
        ]
        free_avoid = max(
            max_in_canvas_translation(ellipse, direction, rest, avoid_overlap=True)
            for direction in STYLE_DIRECTIONS.values()
        )
        free_canvas = max(
            max_in_canvas_translation(ellipse, direction, rest, avoid_overlap=False)
            for direction in STYLE_DIRECTIONS.values()
        )
        # Overlap with unselected ellipses is avoided when possible, but a crowded
        # in-canvas person remains eligible if the centroid can still move.
        if free_canvas < MIN_MEANINGFUL_FREE_SPACE:
            continue
        rows.append(
            {
                "instance_id": str(row["key"]),
                "class_name": class_name,
                "ellipse": ellipse,
                "source_pixels": ellipse.source_pixels,
                "free_space_pixels": free_canvas,
                "overlap_free_space_pixels": free_avoid,
            }
        )
    rows.sort(
        key=lambda item: (
            -item["overlap_free_space_pixels"],
            -item["free_space_pixels"],
            -item["source_pixels"],
            item["instance_id"],
        )
    )
    return rows


def select_human_ellipse(metadata: Mapping[str, object], *, style: str) -> dict[str, object]:
    candidates = human_candidates(metadata)
    if not candidates:
        raise ValueError("No fully visible in-canvas human ellipse with meaningful free space")
    others = [
        _ellipse_from_instance(row)
        for row in metadata["frames"][0]["instances"]
        if row.get("key") != candidates[0]["instance_id"]
    ]
    planned = plan_single_ellipse_waypoints(
        style=style, ellipse=candidates[0]["ellipse"], others=others
    )
    selected = candidates[0]
    return {
        "instance_id": selected["instance_id"],
        "class_name": selected["class_name"],
        "source_pixels": selected["source_pixels"],
        "free_space_pixels": selected["free_space_pixels"],
        "trajectory": planned,
    }


def refuse_existing_experiment_dir(path: Path) -> None:
    if not path.exists():
        return
    names = {item.name for item in path.iterdir()}
    blocked = names & {"experiment_manifest.json", "aggregate_metrics.json", "aggregate_metrics.csv"}
    blocked.update(name for name in names if name.startswith("case-"))
    if blocked:
        raise ExperimentDirectoryExistsError(
            f"Refusing to overwrite existing experiment directory {path}: {sorted(blocked)}"
        )


def inventory_experiment_dir(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "path": str(path), "entries": []}
    entries = sorted(str(item.relative_to(path)) for item in path.rglob("*") if item.is_file())
    return {
        "exists": True,
        "empty": not entries,
        "path": str(path),
        "file_count": len(entries),
        "entries": entries,
    }


def validate_six_case_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("Suite manifest must contain exactly six cases")
    domains = [str(row.get("domain")) for row in cases]
    if domains.count("MMOR") != 3 or domains.count("4D-OR") != 3:
        raise ValueError("Suite must contain three MMOR and three 4D-OR cases")
    ids = [str(row.get("case_id")) for row in cases]
    if ids != ["01", "02", "03", "04", "05", "06"]:
        raise ValueError("Case ids must be 01-06 in order")
    styles = [str(row.get("style")) for row in cases]
    expected_styles = [row["style"] for row in PREFERRED_CASES]
    if styles != expected_styles:
        raise ValueError(f"Case styles must be {expected_styles}")
    seen_clips: set[str] = set()
    for row in cases:
        for field in ("clip_id", "instance_id", "class_name", "preferred_clip_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Case {row.get('case_id')} is missing {field}")
        if row["class_name"] not in HUMAN_ELLIPSE_CLASSES:
            raise ValueError(f"Case {row['case_id']} did not select a human ellipse")
        if row["clip_id"] in seen_clips:
            raise ValueError(f"Duplicate clip id in suite: {row['clip_id']}")
        seen_clips.add(str(row["clip_id"]))
        if domain_from_clip_id(str(row["clip_id"])) != row["domain"]:
            raise ValueError(f"Clip {row['clip_id']} does not match domain {row['domain']}")
    return {
        "case_count": 6,
        "mmor_count": 3,
        "four_dor_count": 3,
        "edited_videos": 6,
        "moved_ellipses_per_video": 1,
    }


def _unselected_unchanged(
    original: Mapping[str, object], edited: Mapping[str, object], instance_id: str
) -> bool:
    for before_frame, after_frame in zip(original["frames"], edited["frames"], strict=True):
        before_rows = {row["key"]: row for row in before_frame["instances"]}
        after_rows = {row["key"]: row for row in after_frame["instances"]}
        if before_rows.keys() != after_rows.keys():
            return False
        for key, before in before_rows.items():
            after = after_rows[key]
            if key == instance_id:
                if before["class_name"] != after["class_name"]:
                    return False
                for field in (
                    "major_diameter",
                    "minor_diameter",
                    "angle_degrees",
                    "source_pixels",
                ):
                    if before["ellipse"][field] != after["ellipse"][field]:
                        return False
                if before.get("raw_relative_depth_mean") != after.get("raw_relative_depth_mean"):
                    return False
            elif before != after:
                return False
    return True


def resolve_cases(
    *,
    mmor_manifest: Path,
    four_dor_manifest: Path,
) -> list[dict[str, object]]:
    samples = load_pair_samples(mmor_manifest, four_dor_manifest)
    used: set[str] = set()
    resolved: list[dict[str, object]] = []
    for spec in PREFERRED_CASES:
        domain = spec["domain"]
        preferred = spec["preferred_clip_id"]
        pool = [clip_id for clip_id in held_out_ids(samples, domain) if clip_id not in used]
        reason = None
        clip_id = preferred if preferred in pool else None
        if clip_id is None:
            reason = "preferred clip is not in the held-out pair manifest"
        selected = None
        metadata = None
        sample = None
        candidates = [clip_id] if clip_id else []
        candidates.extend(clip for clip in pool if clip != clip_id)
        for candidate in candidates:
            sample = samples[candidate]
            metadata_path = _resolve(Path(str(sample["manifest"])).parent, str(sample["geometry_metadata"]))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            try:
                selected = select_human_ellipse(metadata, style=spec["style"])
            except ValueError as error:
                if candidate == preferred:
                    reason = str(error)
                continue
            clip_id = candidate
            if candidate != preferred:
                reason = reason or "preferred clip cannot support a valid in-canvas human trajectory"
            break
        if selected is None or clip_id is None or metadata is None or sample is None:
            raise ValueError(f"No replacement clip in {domain} held-out pool for case {spec['case_id']}")
        used.add(clip_id)
        resolved.append(
            {
                "case_id": spec["case_id"],
                "domain": domain,
                "style": spec["style"],
                "preferred_clip_id": preferred,
                "clip_id": clip_id,
                "replaced": clip_id != preferred,
                "replacement_reason": reason if clip_id != preferred else None,
                "instance_id": selected["instance_id"],
                "class_name": selected["class_name"],
                "source_pixels": selected["source_pixels"],
                "trajectory": selected["trajectory"],
                "source_sample": sample,
                "metadata": metadata,
            }
        )
    validate_six_case_manifest({"cases": resolved})
    return resolved


def prepare_case(case: Mapping[str, object], output_dir: Path, *, seed: int = 42) -> dict[str, object]:
    case_dir = output_dir / f"case-{case['case_id']}-{case['clip_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    sample = case["source_sample"]
    manifest_parent = Path(str(sample["manifest"])).parent
    source_conditioning = _resolve(manifest_parent, str(sample["conditioning_video"]))
    source_target = _resolve(manifest_parent, str(sample["target_video"]))
    source_metadata_path = _resolve(manifest_parent, str(sample["geometry_metadata"]))
    extract_first_frame(source_target, case_dir / "initial-rgb.png")
    shutil.copy2(source_conditioning, case_dir / "original-control.mp4")
    original_metadata = case["metadata"]
    edited_metadata, edit_manifest = edit_trajectory(
        original_metadata,
        source_conditioning=source_conditioning,
        source_metadata=source_metadata_path,
        instance_id=str(case["instance_id"]),
        waypoints=[tuple(point) for point in case["trajectory"]["waypoints"]],
        mode="replace",
        simplify_epsilon_pixels=2.0,
        seed=seed,
    )
    if not _unselected_unchanged(original_metadata, edited_metadata, str(case["instance_id"])):
        raise ValueError("Unselected ellipse data changed")
    selected = next(
        row
        for row in original_metadata["frames"][0]["instances"]
        if row["key"] == case["instance_id"]
    )
    if edit_manifest["edited_centroids"][0] != [
        selected["ellipse"]["center_x"],
        selected["ellipse"]["center_y"],
    ]:
        raise ValueError("Frame-0 centroid was not anchored")
    if len(edit_manifest["edited_centroids"]) != OUTPUT_FRAME_COUNT:
        raise ValueError("Edited trajectory is incomplete")
    validate_in_canvas_trajectory(_ellipse_from_instance(selected), edit_manifest["edited_centroids"])
    edit_dir = case_dir / "edit"
    edit_dir.mkdir(exist_ok=True)
    render_edited_sequence(edited_metadata, edit_dir / "conditioning-edited.mp4")
    shutil.copy2(edit_dir / "conditioning-edited.mp4", case_dir / "edited-control.mp4")
    (edit_dir / "metadata.json").write_text(
        json.dumps(edited_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (edit_dir / "trajectory-edit-manifest.json").write_text(
        json.dumps(edit_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (case_dir / "waypoints.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": case["case_id"],
                "style": case["style"],
                "waypoints": case["trajectory"]["waypoints"],
                "requested_displacement_pixels": case["trajectory"]["requested_displacement_pixels"],
                "notes": case["trajectory"]["notes"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "requested-trajectory.json").write_text(
        json.dumps(edit_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source = {
        "case_id": case["case_id"],
        "domain": case["domain"],
        "clip_id": case["clip_id"],
        "preferred_clip_id": case["preferred_clip_id"],
        "replaced": case["replaced"],
        "replacement_reason": case["replacement_reason"],
        "instance_id": case["instance_id"],
        "class_name": case["class_name"],
        "style": case["style"],
        "pair_manifest": sample["manifest"],
        "geometry_metadata": str(source_metadata_path),
        "target_video": str(source_target),
        "conditioning_video": str(source_conditioning),
        "split": sample.get("split"),
    }
    (case_dir / "source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"case_dir": str(case_dir), "source": source, "edit_manifest": str(case_dir / "requested-trajectory.json")}


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def video_level_bootstrap(
    values: Sequence[float | None], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, object]:
    finite = [float(value) for value in values if value is not None]
    if len(finite) < 2:
        return {
            "status": "unavailable",
            "reason": "video-level bootstrap requires at least two videos with this metric",
            "n_videos": len(finite),
            "label": "exploratory: videos are the independent units; do not treat as frame-level CI",
        }
    rng = np.random.default_rng(seed)
    means = []
    array = np.asarray(finite, dtype=np.float64)
    for _ in range(samples):
        indices = rng.integers(0, len(array), size=len(array))
        means.append(float(array[indices].mean()))
    drawn = np.asarray(means, dtype=np.float64)
    return {
        "status": "ok",
        "n_videos": len(finite),
        "samples": samples,
        "seed": seed,
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(drawn, 0.025)), float(np.quantile(drawn, 0.975))],
        "unit": "video",
        "label": "exploratory video-level bootstrap; each domain has only three cases",
        "warning": "do not present frame-level intervals as independent-video uncertainty",
    }


def _video_summary(case_metrics: Mapping[str, object], source: Mapping[str, object]) -> dict[str, object]:
    movement = case_metrics["movement"]
    error = case_metrics["trajectory_error_pixels"]
    alignment = case_metrics["ellipse_alignment"]
    return {
        "case_id": source["case_id"],
        "domain": source["domain"],
        "clip_id": source["clip_id"],
        "instance_id": source["instance_id"],
        "class_name": source["class_name"],
        "valid_track_frames": case_metrics["valid_track_frames"],
        "valid_track_rate": case_metrics["valid_track_rate"],
        "unavailable_lost_frames": case_metrics["unavailable_lost_frames"],
        "mean_trajectory_error_pixels": error["mean"],
        "median_trajectory_error_pixels": error["median"],
        "endpoint_error_pixels": error["endpoint"],
        "requested_displacement_pixels": movement["requested_displacement_pixels"],
        "generated_displacement_pixels": movement["generated_displacement_pixels"],
        "generated_requested_displacement_ratio": movement["generated_requested_displacement_ratio"],
        "direction_cosine": movement["direction_cosine"],
        "segmentation_iou_ellipse_proxy": alignment["segmentation_iou_to_conditioning_ellipse"],
        "bounding_box_iou_ellipse_proxy": alignment["bounding_box_iou_to_conditioning_ellipse"],
        "identity_switching": case_metrics["identity_switching"],
    }


def aggregate_video_rows(rows: Sequence[Mapping[str, object]], *, label: str) -> dict[str, object]:
    keys = [
        "valid_track_rate",
        "unavailable_lost_frames",
        "mean_trajectory_error_pixels",
        "median_trajectory_error_pixels",
        "endpoint_error_pixels",
        "requested_displacement_pixels",
        "generated_displacement_pixels",
        "generated_requested_displacement_ratio",
        "direction_cosine",
        "segmentation_iou_ellipse_proxy",
        "bounding_box_iou_ellipse_proxy",
    ]
    summary = {
        "label": label,
        "n_videos": len(rows),
        "independent_unit": "video",
        "ellipse_iou_interpretation": ELLIPSE_PROXY_INTERPRETATION,
        "identity_switching": "n/a",
        "descriptive": {},
        "exploratory_video_bootstrap": {},
    }
    for key in keys:
        values = [row.get(key) for row in rows]
        summary["descriptive"][key] = {
            "mean": _mean(values),
            "n_available": sum(value is not None for value in values),
        }
        summary["exploratory_video_bootstrap"][key] = video_level_bootstrap(values)
    return summary


def write_results_summary(
    path: Path,
    *,
    videos: Sequence[Mapping[str, object]],
    aggregates: Mapping[str, object],
) -> None:
    lines = [
        "# Single-ellipse edited-only control (step 2000)",
        "",
        ELLIPSE_PROXY_INTERPRETATION,
        "",
        "Identity switching is `n/a` because exactly one object is tracked.",
        "Confidence intervals resample complete videos and are exploratory (three videos per domain).",
        "",
        "| Case | Domain | Clip | Entity | Valid frames | Lost | Mean err | Median err | Endpoint | Req. dx | Gen. dx | Ratio | Cosine | Seg IoU | Box IoU |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in videos:
        def fmt(value: object) -> str:
            if value is None:
                return "unavailable"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        lines.append(
            "| {case_id} | {domain} | `{clip_id}` | `{instance_id}` {class_name} | {valid} | {lost} | {mean} | {median} | {end} | {req} | {gen} | {ratio} | {cos} | {seg} | {box} |".format(
                case_id=row["case_id"],
                domain=row["domain"],
                clip_id=row["clip_id"],
                instance_id=row["instance_id"],
                class_name=row["class_name"],
                valid=row["valid_track_frames"],
                lost=row["unavailable_lost_frames"],
                mean=fmt(row["mean_trajectory_error_pixels"]),
                median=fmt(row["median_trajectory_error_pixels"]),
                end=fmt(row["endpoint_error_pixels"]),
                req=fmt(row["requested_displacement_pixels"]),
                gen=fmt(row["generated_displacement_pixels"]),
                ratio=fmt(row["generated_requested_displacement_ratio"]),
                cos=fmt(row["direction_cosine"]),
                seg=fmt(row["segmentation_iou_ellipse_proxy"]),
                box=fmt(row["bounding_box_iou_ellipse_proxy"]),
            )
        )
    lines.extend(["", "## Domain summaries", ""])
    for key in ("mmor", "four_dor", "overall"):
        block = aggregates[key]
        lines.append(f"### {block['label']} (n={block['n_videos']} videos)")
        desc = block["descriptive"]
        lines.append(
            f"- Mean trajectory error: {desc['mean_trajectory_error_pixels']['mean']}"
        )
        lines.append(
            f"- Mean requested displacement: {desc['requested_displacement_pixels']['mean']}"
        )
        lines.append(
            f"- Mean generated/requested ratio: {desc['generated_requested_displacement_ratio']['mean']}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_suite(output_dir: Path) -> dict[str, object]:
    manifest = json.loads((output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    validate_six_case_manifest(manifest)
    videos = []
    for case in manifest["cases"]:
        case_dir = Path(case["case_dir"])
        metrics = json.loads((case_dir / "metrics" / "case.json").read_text(encoding="utf-8"))
        source = json.loads((case_dir / "source.json").read_text(encoding="utf-8"))
        videos.append(_video_summary(metrics, source))
        generated = case_dir / "edited-generated.mp4"
        if generated.is_file():
            probe = probe_video(generated)
            if probe != {"width": 1024, "height": 768, "frames": 97, "fps": 24.0}:
                raise ValueError(f"Generated video violates contract: {generated}: {probe}")
    mmor = [row for row in videos if row["domain"] == "MMOR"]
    four = [row for row in videos if row["domain"] == "4D-OR"]
    aggregates = {
        "mmor": aggregate_video_rows(mmor, label="MMOR"),
        "four_dor": aggregate_video_rows(four, label="4D-OR"),
        "overall": aggregate_video_rows(videos, label="overall"),
    }
    payload = {
        "schema_version": 1,
        "kind": "edited_only_single_ellipse_six_case_aggregate",
        "n_videos": 6,
        "n_edited_generations": 6,
        "moved_ellipses_per_video": 1,
        "original_conditioned_companions": 0,
        "ellipse_iou_interpretation": ELLIPSE_PROXY_INTERPRETATION,
        "videos": videos,
        "aggregates": aggregates,
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(videos[0].keys())
    with (output_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(videos)
    write_results_summary(output_dir / "results-summary.md", videos=videos, aggregates=aggregates)
    return payload


def write_suite_manifest(output_dir: Path, cases: Sequence[Mapping[str, object]]) -> Path:
    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case["case_id"],
                "domain": case["domain"],
                "style": case["style"],
                "preferred_clip_id": case["preferred_clip_id"],
                "clip_id": case["clip_id"],
                "replaced": case["replaced"],
                "replacement_reason": case["replacement_reason"],
                "instance_id": case["instance_id"],
                "class_name": case["class_name"],
                "case_dir": case.get("case_dir"),
                "moved_ellipses": 1,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "edited_only_single_ellipse_six_case_manifest",
        "checkpoint_step": 2000,
        "edited_only": True,
        "original_conditioned_companions": 0,
        "video_contract": {
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "frames": OUTPUT_FRAME_COUNT,
            "fps": 24,
        },
        "cases": rows,
    }
    validate_six_case_manifest(payload)
    path = output_dir / "experiment_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve-cases")
    resolve.add_argument("--mmor-pair-manifest", required=True, type=Path)
    resolve.add_argument("--four-dor-pair-manifest", required=True, type=Path)
    resolve.add_argument("--output", required=True, type=Path)
    prepare = sub.add_parser("prepare-suite")
    prepare.add_argument("--mmor-pair-manifest", required=True, type=Path)
    prepare.add_argument("--four-dor-pair-manifest", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--seed", type=int, default=42)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--output-dir", required=True, type=Path)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "resolve-cases":
        cases = resolve_cases(
            mmor_manifest=args.mmor_pair_manifest,
            four_dor_manifest=args.four_dor_pair_manifest,
        )
        slim = [{key: value for key, value in case.items() if key not in {"metadata", "source_sample"}} | {
            "source_sample": {
                "id": case["source_sample"]["id"],
                "manifest": case["source_sample"]["manifest"],
                "split": case["source_sample"].get("split"),
            }
        } for case in cases]
        args.output.write_text(json.dumps({"cases": slim}, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "prepare-suite":
        refuse_existing_experiment_dir(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cases = resolve_cases(
            mmor_manifest=args.mmor_pair_manifest,
            four_dor_manifest=args.four_dor_pair_manifest,
        )
        prepared = []
        for case in cases:
            result = prepare_case(case, args.output_dir, seed=args.seed)
            case = dict(case)
            case["case_dir"] = result["case_dir"]
            prepared.append(case)
            (Path(result["case_dir"]) / "prepare.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        write_suite_manifest(args.output_dir, prepared)
        (args.output_dir / "inventory.json").write_text(
            json.dumps(inventory_experiment_dir(args.output_dir), indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    if args.command == "aggregate":
        payload = aggregate_suite(args.output_dir)
        print(json.dumps({"n_videos": payload["n_videos"]}, indent=2))
        return 0
    print(json.dumps(inventory_experiment_dir(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

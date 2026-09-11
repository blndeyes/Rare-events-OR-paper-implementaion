"""Validate target/ellipse training pairs before latent preprocessing."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from or_video_reproduction.data.clips import (
    OUTPUT_FPS,
    OUTPUT_FRAME_COUNT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from or_video_reproduction.data.semantics import (
    ENTITY_CLASSES,
    MMOR_ARTIFACT_LABELS,
    MMOR_SEGMENTATION_LABELS,
)
from or_video_reproduction.geometry.palette import PAPER_36_PALETTE

from .ltx_batch import probe_video


EXPECTED_SHAPE = (OUTPUT_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH)


@dataclass(frozen=True)
class PairJob:
    clip_id: str
    target_video: Path
    conditioning_video: Path
    labels: Path
    depths: Path
    geometry_metadata: Path


def _resolve(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing path field {field}")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_pair_jobs(path: Path) -> list[PairJob]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Pair manifest schema_version must be 1")
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Pair manifest must contain a non-empty samples list")

    jobs: list[PairJob] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each sample must be an object")
        clip_id = row.get("id")
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError("Each sample requires a non-empty id")
        if clip_id in seen:
            raise ValueError(f"Duplicate sample id: {clip_id}")
        jobs.append(
            PairJob(
                clip_id=clip_id,
                target_video=_resolve(path.parent, row.get("target_video"), "target_video"),
                conditioning_video=_resolve(
                    path.parent, row.get("conditioning_video"), "conditioning_video"
                ),
                labels=_resolve(path.parent, row.get("labels"), "labels"),
                depths=_resolve(path.parent, row.get("depths"), "depths"),
                geometry_metadata=_resolve(
                    path.parent, row.get("geometry_metadata"), "geometry_metadata"
                ),
            )
        )
        seen.add(clip_id)
    return jobs


def validate_label_and_depth_arrays(
    labels: np.ndarray, depths: np.ndarray
) -> dict[str, object]:
    errors: list[str] = []
    if labels.shape != EXPECTED_SHAPE:
        errors.append(f"labels shape is {labels.shape}, expected {EXPECTED_SHAPE}")
    if depths.shape != EXPECTED_SHAPE:
        errors.append(f"depths shape is {depths.shape}, expected {EXPECTED_SHAPE}")
    if not np.issubdtype(labels.dtype, np.integer):
        errors.append(f"labels dtype must be integer, got {labels.dtype}")
    if not np.issubdtype(depths.dtype, np.floating):
        errors.append(f"depths dtype must be floating, got {depths.dtype}")
    if np.issubdtype(depths.dtype, np.floating) and not np.isfinite(depths).all():
        errors.append("depths contain NaN or infinite values")

    observed = sorted(int(value) for value in np.unique(labels))
    known_labels = set(MMOR_SEGMENTATION_LABELS) | set(MMOR_ARTIFACT_LABELS)
    unknown = sorted(value for value in observed if value and value not in known_labels)
    if unknown:
        errors.append(f"labels contain unknown MMOR values: {unknown}")
    return {"passed": not errors, "errors": errors, "observed_labels": observed}


def validate_geometry_metadata(metadata: dict[str, object]) -> dict[str, object]:
    errors: list[str] = []
    if metadata.get("kind") != "paper_ellipse_only_geometric_conditioning":
        errors.append("geometry kind is not ellipse-only conditioning")
    if metadata.get("shape") != list(EXPECTED_SHAPE):
        errors.append(f"geometry shape is {metadata.get('shape')}, expected {EXPECTED_SHAPE}")
    if metadata.get("fps") != OUTPUT_FPS:
        errors.append(f"geometry fps is {metadata.get('fps')}, expected {OUTPUT_FPS}")
    if metadata.get("background") != "black":
        errors.append("geometry background is not declared black")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or len(frames) != OUTPUT_FRAME_COUNT:
        errors.append(f"geometry metadata must describe {OUTPUT_FRAME_COUNT} frames")
        frames = []

    instance_count = 0
    for frame in frames:
        for instance in frame.get("instances", []):
            instance_count += 1
            class_name = instance.get("class_name")
            if class_name not in ENTITY_CLASSES:
                errors.append(f"non-entity ellipse class: {class_name!r}")
                continue
            if instance.get("red_green") != list(PAPER_36_PALETTE[class_name]):
                errors.append(f"palette mismatch for {class_name}")
    if frames and instance_count == 0:
        errors.append("geometry contains no ellipse instances")
    return {"passed": not errors, "errors": errors, "instance_count": instance_count}


def _load_array(path: Path, preferred_key: str) -> np.ndarray:
    with np.load(path) as archive:
        if preferred_key in archive:
            return np.asarray(archive[preferred_key])
        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]])
        raise KeyError(f"{path} has no {preferred_key!r} array")


def validate_pair(
    job: PairJob,
    *,
    video_probe: Callable[[Path], dict[str, int]] = probe_video,
) -> dict[str, object]:
    expected_video = {
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "frames": OUTPUT_FRAME_COUNT,
        "fps": OUTPUT_FPS,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "clip_id": job.clip_id,
        "state": "failed",
        "checks": {},
        "paths": {
            "target_video": str(job.target_video),
            "conditioning_video": str(job.conditioning_video),
            "labels": str(job.labels),
            "depths": str(job.depths),
            "geometry_metadata": str(job.geometry_metadata),
        },
    }
    try:
        target_probe = video_probe(job.target_video)
        conditioning_probe = video_probe(job.conditioning_video)
        report["checks"]["target_video"] = {
            "passed": target_probe == expected_video,
            "observed": target_probe,
            "expected": expected_video,
        }
        report["checks"]["conditioning_video"] = {
            "passed": conditioning_probe == expected_video,
            "observed": conditioning_probe,
            "expected": expected_video,
        }
        arrays = validate_label_and_depth_arrays(
            _load_array(job.labels, "labels"), _load_array(job.depths, "depths")
        )
        report["checks"]["arrays"] = arrays
        metadata = json.loads(job.geometry_metadata.read_text(encoding="utf-8"))
        report["checks"]["geometry"] = validate_geometry_metadata(metadata)
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        return report

    passed = all(check["passed"] for check in report["checks"].values())
    report["state"] = "passed" if passed else "failed"
    return report


def validate_batch(jobs: Sequence[PairJob], output_path: Path) -> dict[str, object]:
    samples = [validate_pair(job) for job in jobs]
    counts = Counter(sample["state"] for sample in samples)
    report = {
        "schema_version": 1,
        "kind": "paper_training_pair_validation",
        "state": "passed" if counts.get("failed", 0) == 0 else "failed",
        "counts": {"passed": counts.get("passed", 0), "failed": counts.get("failed", 0)},
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_batch(load_pair_jobs(args.pair_manifest), args.output)
    print(json.dumps({"state": report["state"], "counts": report["counts"]}, indent=2))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

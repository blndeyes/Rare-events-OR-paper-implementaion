"""Run reusable VDA, reusable SAM2, rendering, and validation as one batch."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from or_video_reproduction.data.semantics import (
    MMOR_ARTIFACT_LABELS,
    MMOR_SEGMENTATION_LABELS,
)
from or_video_reproduction.geometry.render_sequence import render_sequence

from .ltx_batch import load_batch_jobs, video_matches_contract
from .sam2_propagation import Sam2VideoRunner, load_point_prompt_manifest
from .validate_pairs import EXPECTED_SHAPE, PairJob, validate_pair
from .vda_depth import VideoDepthRunner, validate_depth_output


@dataclass(frozen=True)
class GeometryJob:
    clip_id: str
    clip_manifest: Path
    target_video: Path
    first_mask: Path | None
    split: str
    take: str | None = None
    prompt_manifest: Path | None = None
    label_classes: dict[int, str] | None = None


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_geometry_jobs(
    batch_manifest: Path,
    *,
    ltx_output_root: Path,
    dataset_root: Path,
    default_seed: int,
) -> list[GeometryJob]:
    """Resolve validated LTX outputs and first-frame masks for downstream work."""

    jobs: list[GeometryJob] = []
    for source in load_batch_jobs(batch_manifest, default_seed=default_seed):
        manifest = _read_json(source.manifest_path)
        status_path = ltx_output_root / source.clip_id / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing LTX status for {source.clip_id}: {status_path}")
        status = _read_json(status_path)
        if status.get("state") not in {
            "completed",
            "skipped_valid_existing",
            "discovered_valid_existing",
        }:
            raise ValueError(f"LTX output for {source.clip_id} is not complete")
        target_video = Path(status["output"])
        if not target_video.is_absolute():
            target_video = (status_path.parent / target_video).resolve()
        if not video_matches_contract(target_video, manifest["paper_contract"]):
            raise ValueError(f"LTX output for {source.clip_id} fails its video contract")

        clip = manifest["clip"]
        first_mask = None
        prompt_manifest = None
        label_classes = None
        if "first_frame_ground_truth_mask" in clip:
            mask_relative = clip["first_frame_ground_truth_mask"]
            first_mask = dataset_root / Path(*Path(mask_relative).parts)
            if not first_mask.is_file():
                raise FileNotFoundError(first_mask)
        elif "first_frame_prompt_manifest" in clip:
            prompt_manifest = Path(clip["first_frame_prompt_manifest"])
            if not prompt_manifest.is_absolute():
                prompt_manifest = (source.manifest_path.parent / prompt_manifest).resolve()
            prompt_rows = load_point_prompt_manifest(prompt_manifest)
            label_classes = {
                int(row["object_id"]): str(row["mmor_class"]) for row in prompt_rows
            }
        else:
            raise ValueError(
                f"{source.clip_id} requires a first-frame mask or point-prompt manifest"
            )
        jobs.append(
            GeometryJob(
                clip_id=source.clip_id,
                clip_manifest=source.manifest_path,
                target_video=target_video,
                first_mask=first_mask,
                split=manifest["clip"]["split"],
                take=manifest["clip"]["take"],
                prompt_manifest=prompt_manifest,
                label_classes=label_classes,
            )
        )
    return jobs


def _load_npz(path: Path, key: str) -> np.ndarray:
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"{path} has no {key!r} array")
        return np.asarray(archive[key])


def depth_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return bool(validate_depth_output(_load_npz(path, "depths"), 24)["passed"])
    except (KeyError, OSError, ValueError):
        return False


def labels_are_valid(path: Path, *, allowed_labels: set[int] | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        labels = _load_npz(path, "labels")
    except (KeyError, OSError, ValueError):
        return False
    if labels.shape != EXPECTED_SHAPE or not np.issubdtype(labels.dtype, np.integer):
        return False
    known = (
        {0, *allowed_labels}
        if allowed_labels is not None
        else {0, *MMOR_SEGMENTATION_LABELS, *MMOR_ARTIFACT_LABELS}
    )
    return all(int(value) in known for value in np.unique(labels))


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_geometry_batch(
    jobs: Sequence[GeometryJob],
    *,
    output_root: Path,
    vda_factory: Callable[[], object],
    sam2_factory: Callable[[], object],
    fail_fast: bool = False,
) -> dict[str, object]:
    """Process missing stages, reuse models, and emit a trainer-pair manifest."""

    output_root.mkdir(parents=True, exist_ok=True)
    vda_runner = None
    sam2_runner = None
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    samples: list[dict[str, object]] = []

    for index, job in enumerate(jobs):
        started = time.time()
        clip_dir = output_root / job.clip_id
        depths_path = clip_dir / "depths.npz"
        labels_path = clip_dir / "labels.npz"
        conditioning_dir = clip_dir / "conditioning"
        conditioning_video = conditioning_dir / "conditioning.mp4"
        geometry_metadata = conditioning_dir / "metadata.json"
        status_path = clip_dir / "status.json"
        pair = PairJob(
            job.clip_id,
            job.target_video,
            conditioning_video,
            labels_path,
            depths_path,
            geometry_metadata,
        )

        existing_report = validate_pair(pair)
        if existing_report["state"] == "passed":
            counts["skipped"] += 1
            samples.append(
                {
                    "id": job.clip_id,
                    "split": job.split,
                    "take": job.take,
                    **(
                        {"prompt_manifest": str(job.prompt_manifest)}
                        if job.prompt_manifest is not None
                        else {}
                    ),
                    **existing_report["paths"],
                }
            )
            if not status_path.is_file():
                _write_json_atomic(
                    status_path,
                    {
                        "schema_version": 1,
                        "state": "discovered_valid_existing",
                        "clip_id": job.clip_id,
                        "job_index": index,
                        "validation": existing_report,
                    },
                )
            continue

        stages: list[str] = []
        try:
            if not depth_is_valid(depths_path):
                if vda_runner is None:
                    vda_runner = vda_factory()
                vda_runner.run(job.target_video, depths_path)
                stages.append("video_depth_anything")
            allowed_labels = set(job.label_classes) if job.label_classes is not None else None
            if not labels_are_valid(labels_path, allowed_labels=allowed_labels):
                if sam2_runner is None:
                    sam2_runner = sam2_factory()
                if job.prompt_manifest is not None:
                    sam2_runner.run_points(job.target_video, job.prompt_manifest, labels_path)
                else:
                    assert job.first_mask is not None
                    sam2_runner.run(job.target_video, job.first_mask, labels_path)
                stages.append("sam2")

            render_sequence(
                labels_path,
                depths_path,
                conditioning_dir,
                label_classes=job.label_classes,
            )
            stages.append("ellipse_rendering")
            validation = validate_pair(pair)
            if validation["state"] != "passed":
                raise RuntimeError("Completed artifacts failed strict pair validation")

            counts["completed"] += 1
            samples.append(
                {
                    "id": job.clip_id,
                    "split": job.split,
                    "take": job.take,
                    **(
                        {"prompt_manifest": str(job.prompt_manifest)}
                        if job.prompt_manifest is not None
                        else {}
                    ),
                    **validation["paths"],
                }
            )
            _write_json_atomic(
                status_path,
                {
                    "schema_version": 1,
                    "state": "completed",
                    "clip_id": job.clip_id,
                    "job_index": index,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "stages_executed": stages,
                    "validation": validation,
                },
            )
        except Exception as error:
            counts["failed"] += 1
            _write_json_atomic(
                status_path,
                {
                    "schema_version": 1,
                    "state": "failed",
                    "clip_id": job.clip_id,
                    "job_index": index,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "stages_executed": stages,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if fail_fast:
                raise

    pair_manifest = {
        "schema_version": 1,
        "kind": "validated_target_ellipse_pairs",
        "counts": counts,
        "samples": samples,
    }
    _write_json_atomic(output_root / "pair-manifest.json", pair_manifest)
    return pair_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--ltx-output-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--vda-root", required=True, type=Path)
    parser.add_argument("--vda-checkpoint", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = load_geometry_jobs(
        args.batch_manifest,
        ltx_output_root=args.ltx_output_root,
        dataset_root=args.dataset_root,
        default_seed=args.seed,
    )
    if args.max_clips is not None:
        if args.max_clips <= 0:
            raise ValueError("max-clips must be positive")
        jobs = jobs[: args.max_clips]
    report = run_geometry_batch(
        jobs,
        output_root=args.output_root,
        vda_factory=lambda: VideoDepthRunner(args.vda_root, args.vda_checkpoint),
        sam2_factory=lambda: Sam2VideoRunner(args.sam2_root, args.sam2_checkpoint),
        fail_fast=args.fail_fast,
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 1 if report["counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

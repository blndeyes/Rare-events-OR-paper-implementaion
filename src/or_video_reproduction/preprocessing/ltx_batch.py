"""Run many LTX interpolation manifests while keeping one pipeline loaded."""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from or_video_reproduction.data.clips import validate_clip_manifest

from .ltx_interpolation import PINNED_LTX_COMMIT, PIPELINE_CONFIGS, _git_head

CLIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class BatchJob:
    clip_id: str
    manifest_path: Path
    seed: int


def load_batch_jobs(path: Path, *, default_seed: int) -> list[BatchJob]:
    """Load the small versioned batch manifest without touching dataset media."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Batch manifest schema_version must be 1")
    rows = payload.get("clips")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Batch manifest must contain a non-empty clips list")

    jobs: list[BatchJob] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Each batch clip must be an object")
        clip_id = row.get("id")
        manifest_value = row.get("manifest")
        if not isinstance(clip_id, str) or not CLIP_ID_PATTERN.fullmatch(clip_id):
            raise ValueError(f"Unsafe or missing clip id: {clip_id!r}")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip id: {clip_id}")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError(f"Clip {clip_id} has no manifest path")
        seed = row.get("seed", default_seed)
        if not isinstance(seed, int):
            raise TypeError(f"Clip {clip_id} seed must be an integer")
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = path.parent / manifest_path
        jobs.append(BatchJob(clip_id, manifest_path.resolve(), seed))
        seen.add(clip_id)
    return jobs


def probe_video(path: Path) -> dict[str, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": numerator // denominator,
    }


def video_matches_contract(path: Path, contract: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        probe = probe_video(path)
    except (FileNotFoundError, KeyError, ValueError, subprocess.SubprocessError):
        return False
    width, height = contract["resolution"]
    return probe == {
        "width": width,
        "height": height,
        "frames": contract["output_frames"],
        "fps": contract["output_fps"],
    }


def _write_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _release_pipeline_memory() -> None:
    """Drop unreachable modules and return cached CUDA blocks to the allocator."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_reusable_inference(ltx_root: Path):
    if _git_head(ltx_root) != PINNED_LTX_COMMIT:
        raise ValueError(f"LTX checkout must be pinned to {PINNED_LTX_COMMIT}")
    sys.path.insert(0, str(ltx_root))
    module = importlib.import_module("ltx_video.inference")
    if "pipeline" not in inspect.signature(module.infer).parameters:
        raise RuntimeError(
            "Pinned LTX checkout lacks the reusable-pipeline patch: "
            "patches/ltx-video-0.9.7-reusable-pipeline.patch"
        )
    return module


def _inference_config(
    module,
    manifest: dict[str, object],
    *,
    dataset_root: Path,
    output_dir: Path,
    prompt: str,
    seed: int,
    pipeline_config: Path,
):
    validate_clip_manifest(manifest)
    contract = manifest["paper_contract"]
    keyframes = manifest["clip"]["source_keyframes"]
    width, height = contract["resolution"]
    return module.InferenceConfig(
        prompt=prompt,
        output_path=str(output_dir),
        pipeline_config=str(pipeline_config),
        seed=seed,
        height=height,
        width=width,
        num_frames=contract["output_frames"],
        frame_rate=contract["output_fps"],
        offload_to_cpu=True,
        conditioning_media_paths=[str(dataset_root / row["rgb_path"]) for row in keyframes],
        conditioning_strengths=[1.0] * len(keyframes),
        conditioning_start_frames=[row["output_frame_index"] for row in keyframes],
    )


def run_batch(
    jobs: Sequence[BatchJob],
    *,
    dataset_root: Path,
    ltx_root: Path,
    output_root: Path,
    prompt: str,
    precision: str,
    fail_fast: bool,
    reload_pipeline_after_each_clip: bool = False,
) -> dict[str, int]:
    if not prompt.strip():
        raise ValueError("An explicit interpolation prompt is required")
    if precision not in PIPELINE_CONFIGS:
        raise ValueError(f"Unsupported precision: {precision}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    output_root.mkdir(parents=True, exist_ok=True)
    module = None
    pipeline = None
    counts = {"completed": 0, "skipped": 0, "failed": 0}

    for job_index, job in enumerate(jobs):
        manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
        validate_clip_manifest(manifest)
        clip_dir = output_root / job.clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        status_path = clip_dir / "status.json"

        existing_videos = sorted(clip_dir.glob("*.mp4"))
        valid_existing = [
            video
            for video in existing_videos
            if video_matches_contract(video, manifest["paper_contract"])
        ]
        if valid_existing:
            counts["skipped"] += 1
            if not status_path.is_file():
                _write_status(
                    status_path,
                    {
                        "schema_version": 1,
                        "state": "discovered_valid_existing",
                        "clip_id": job.clip_id,
                        "output": str(valid_existing[-1]),
                        "job_index": job_index,
                    },
                )
            continue

        started = time.time()
        try:
            if module is None:
                module = _load_reusable_inference(ltx_root)
            config = _inference_config(
                module,
                manifest,
                dataset_root=dataset_root,
                output_dir=clip_dir,
                prompt=prompt,
                seed=job.seed,
                pipeline_config=ltx_root / PIPELINE_CONFIGS[precision],
            )
            pipeline_was_reused = pipeline is not None
            pipeline, outputs = module.infer(config=config, pipeline=pipeline)
            output = Path(outputs[-1])
            if not video_matches_contract(output, manifest["paper_contract"]):
                raise RuntimeError(f"Generated video fails the paper contract: {output}")
            counts["completed"] += 1
            _write_status(
                status_path,
                {
                    "schema_version": 1,
                    "state": "completed",
                    "clip_id": job.clip_id,
                    "manifest": str(job.manifest_path),
                    "output": str(output),
                    "seed": job.seed,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "pipeline_reused": pipeline_was_reused,
                    "validation": probe_video(output),
                },
            )
            if reload_pipeline_after_each_clip:
                pipeline = None
                _release_pipeline_memory()
        except Exception as error:
            counts["failed"] += 1
            pipeline = None
            _release_pipeline_memory()
            _write_status(
                status_path,
                {
                    "schema_version": 1,
                    "state": "failed",
                    "clip_id": job.clip_id,
                    "manifest": str(job.manifest_path),
                    "seed": job.seed,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if fail_fast:
                raise
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--ltx-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--precision", choices=sorted(PIPELINE_CONFIGS), default="fp8")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--reload-pipeline-after-each-clip",
        action="store_true",
        help=(
            "Release the pipeline and CUDA cache after every completed clip. "
            "Use for BF16 batches when the pinned reusable pipeline retains GPU memory."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = load_batch_jobs(args.batch_manifest, default_seed=args.seed)
    if args.max_clips is not None:
        if args.max_clips <= 0:
            raise ValueError("max-clips must be positive")
        jobs = jobs[: args.max_clips]
    counts = run_batch(
        jobs,
        dataset_root=args.dataset_root,
        ltx_root=args.ltx_root,
        output_root=args.output_root,
        prompt=args.prompt,
        precision=args.precision,
        fail_fast=args.fail_fast,
        reload_pipeline_after_each_clip=args.reload_pipeline_after_each_clip,
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run held-out inference with both the real first frame and ellipse video."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from or_video_reproduction.data.clips import (
    OUTPUT_FPS,
    OUTPUT_FRAME_COUNT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from or_video_reproduction.training.ic_lora_smoke import PINNED_TRAINER_COMMIT, _git_head

from .video import probe_video


@dataclass(frozen=True)
class InferenceJob:
    clip_id: str
    target_video: Path
    ellipse_video: Path


def _resolve(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing path field {field}")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_inference_jobs(pair_manifest: Path, *, split: str = "heldout") -> list[InferenceJob]:
    payload = json.loads(pair_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
        raise ValueError("Pair manifest must have schema_version 1 and a samples list")
    rows = [row for row in payload["samples"] if row.get("split") == split]
    if not rows:
        raise ValueError(f"Pair manifest contains no {split!r} samples")
    jobs: list[InferenceJob] = []
    seen: set[str] = set()
    for row in rows:
        clip_id = row.get("id")
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError("Each inference sample requires a non-empty id")
        if clip_id in seen:
            raise ValueError(f"Duplicate inference sample id: {clip_id}")
        jobs.append(
            InferenceJob(
                clip_id=clip_id,
                target_video=_resolve(
                    pair_manifest.parent, row.get("target_video"), "target_video"
                ),
                ellipse_video=_resolve(
                    pair_manifest.parent, row.get("conditioning_video"), "conditioning_video"
                ),
            )
        )
        seen.add(clip_id)
    return jobs


def validate_conditioning_videos(
    jobs: Sequence[InferenceJob],
    *,
    probe: Callable[[Path], dict[str, int | float]] = probe_video,
) -> None:
    expected = {
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "frames": OUTPUT_FRAME_COUNT,
        "fps": float(OUTPUT_FPS),
    }
    for job in jobs:
        for role, path in (("target", job.target_video), ("ellipse", job.ellipse_video)):
            observed = probe(path)
            if observed != expected:
                raise ValueError(
                    f"{job.clip_id} {role} video violates the 97-frame protocol: "
                    f"{observed} != {expected}"
                )


def extract_first_frame(video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(temporary),
        ],
        check=True,
    )
    with Image.open(temporary) as image:
        if image.size != (TARGET_WIDTH, TARGET_HEIGHT):
            raise ValueError(f"Extracted first frame has unexpected size {image.size}")
    temporary.replace(output)


def build_inference_config(
    official: dict[str, object],
    *,
    checkpoint: Path,
    output_dir: Path,
    jobs: Sequence[InferenceJob],
    first_frames: Sequence[Path],
    prompt: str,
    seed: int,
) -> dict[str, object]:
    if len(jobs) != len(first_frames):
        raise ValueError("Each inference job must have exactly one first frame")
    effective = copy.deepcopy(official)
    effective["model"].update(
        {
            "model_source": "LTXV_13B_097_DEV",
            "training_mode": "lora",
            "load_checkpoint": str(checkpoint),
        }
    )
    effective["acceleration"].update(
        {
            "mixed_precision_mode": "bf16",
            "quantization": None,
            "load_text_encoder_in_8bit": False,
        }
    )
    effective["validation"].update(
        {
            "prompts": [prompt] * len(jobs),
            "images": [str(path) for path in first_frames],
            "reference_videos": [str(job.ellipse_video) for job in jobs],
            "video_dims": [TARGET_WIDTH, TARGET_HEIGHT, OUTPUT_FRAME_COUNT],
            "seed": seed,
            "inference_steps": 50,
            "guidance_scale": 3.5,
            "videos_per_prompt": 1,
            "interval": None,
            "skip_initial_validation": True,
        }
    )
    effective["wandb"]["enabled"] = False
    effective["hub"]["push_to_hub"] = False
    effective["seed"] = seed
    effective["output_dir"] = str(output_dir)
    return effective


def assert_corrected_upstream(trainer_class: type) -> None:
    source = inspect.getsource(trainer_class._sample_videos)
    required = (
        '"output_reference_comparison": False',
        "image.size != (width, height)",
        "output_size=(height, width)",
    )
    missing = [value for value in required if value not in source]
    if missing:
        raise RuntimeError(
            "Pinned trainer is missing patches/ltx-trainer-correct-validation-conditioning.patch; "
            f"missing source markers: {missing}"
        )


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot infer checkpoint step from {path.name}")
    return int(match.group(1))


def run_corrected_inference(
    *,
    trainer_root: Path,
    pair_manifest: Path,
    checkpoint: Path,
    output_dir: Path,
    prompt: str,
    seed: int,
    split: str = "heldout",
) -> dict[str, object]:
    if _git_head(trainer_root) != PINNED_TRAINER_COMMIT:
        raise ValueError(f"Trainer checkout must be pinned to {PINNED_TRAINER_COMMIT}")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000:
        raise ValueError(f"Checkpoint is missing or implausibly small: {checkpoint}")
    from safetensors import safe_open

    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise ValueError(f"Checkpoint has no tensors: {checkpoint}")

    jobs = load_inference_jobs(pair_manifest, split=split)
    validate_conditioning_videos(jobs)
    first_frame_dir = output_dir / "first-frames"
    first_frames = [first_frame_dir / f"{job.clip_id}.png" for job in jobs]
    for job, first_frame in zip(jobs, first_frames, strict=True):
        extract_first_frame(job.target_video, first_frame)

    import yaml

    official = yaml.safe_load(
        (trainer_root / "configs/ltxv_13b_ic_lora.yaml").read_text(encoding="utf-8")
    )
    effective = build_inference_config(
        official,
        checkpoint=checkpoint,
        output_dir=output_dir,
        jobs=jobs,
        first_frames=first_frames,
        prompt=prompt,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "effective-inference-config.yaml").write_text(
        yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"
    )

    sys.path.insert(0, str(trainer_root / "src"))
    from ltxv_trainer.config import LtxvTrainerConfig
    from ltxv_trainer.trainer import LtxvTrainer

    assert_corrected_upstream(LtxvTrainer)
    trainer = LtxvTrainer(LtxvTrainerConfig(**effective))
    trainer._global_step = _checkpoint_step(checkpoint)
    generated_paths = trainer._sample_videos(MagicMock())
    if generated_paths is None or len(generated_paths) != len(jobs):
        raise RuntimeError(f"Expected {len(jobs)} generated videos, got {generated_paths}")

    rows = []
    expected_probe = {
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "frames": OUTPUT_FRAME_COUNT,
        "fps": float(OUTPUT_FPS),
    }
    for job, first_frame, generated in zip(jobs, first_frames, generated_paths, strict=True):
        generated = Path(generated)
        observed = probe_video(generated)
        if observed != expected_probe:
            raise ValueError(f"Generated video violates protocol: {generated}: {observed}")
        rows.append(
            {
                "id": job.clip_id,
                "reference_video": str(job.target_video),
                "first_frame": str(first_frame),
                "ellipse_video": str(job.ellipse_video),
                "generated_video": str(generated),
            }
        )

    report = {
        "schema_version": 1,
        "kind": "corrected_first_frame_plus_ellipse_inference",
        "classification": "reduced_scale_reproduction_hypothesis",
        "checkpoint": str(checkpoint),
        "checkpoint_step": _checkpoint_step(checkpoint),
        "prompt": prompt,
        "seed": seed,
        "inference_steps": 50,
        "guidance_scale": 3.5,
        "video_contract": expected_probe,
        "samples": rows,
    }
    report_path = output_dir / "inference-manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-root", required=True, type=Path)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split", default="heldout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_corrected_inference(
        trainer_root=args.trainer_root,
        pair_manifest=args.pair_manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        prompt=args.prompt,
        seed=args.seed,
        split=args.split,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

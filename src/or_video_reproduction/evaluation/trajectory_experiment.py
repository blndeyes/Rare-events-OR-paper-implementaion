"""Prepare and run a strict original-versus-edited step-600 control pair."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from .corrected_inference import run_corrected_inference


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def prepare_control_pair(
    *,
    source_pair_manifest: Path,
    clip_id: str,
    edited_conditioning: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, object]]:
    payload = json.loads(source_pair_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
        raise ValueError("Source pair manifest must have schema_version 1 and samples")
    matches = [row for row in payload["samples"] if row.get("id") == clip_id]
    if len(matches) != 1:
        raise ValueError(f"Source pair manifest must contain clip {clip_id!r} exactly once")
    source = matches[0]
    required = ("target_video", "conditioning_video", "geometry_metadata", "labels")
    missing = [field for field in required if not isinstance(source.get(field), str)]
    if missing:
        raise ValueError(f"Source sample is missing fields: {', '.join(missing)}")
    if not edited_conditioning.is_file():
        raise FileNotFoundError(edited_conditioning)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_dir = output_dir / "conditioning"
    conditioning_dir.mkdir(exist_ok=True)
    original_copy = conditioning_dir / "original.mp4"
    edited_copy = conditioning_dir / "edited.mp4"
    shutil.copy2(_resolve(source_pair_manifest.parent, source["conditioning_video"]), original_copy)
    shutil.copy2(edited_conditioning, edited_copy)
    rows = []
    for role, conditioning in (("original", original_copy), ("edited", edited_copy)):
        rows.append(
            {
                "id": f"{clip_id}-{role}",
                "split": "heldout",
                "take": source.get("take"),
                "target_video": str(_resolve(source_pair_manifest.parent, source["target_video"])),
                "conditioning_video": str(conditioning.resolve()),
            }
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "step600_trajectory_control_inference_pair",
        "classification": "reduced_scale_reproduction_hypothesis",
        "source_pair_manifest": str(source_pair_manifest),
        "source_clip_id": clip_id,
        "source_geometry_metadata": str(
            _resolve(source_pair_manifest.parent, source["geometry_metadata"])
        ),
        "source_labels": str(_resolve(source_pair_manifest.parent, source["labels"])),
        "controlled_variable": "conditioning_video only",
        "samples": rows,
    }
    path = output_dir / "control-pair-manifest.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, result


def audit_control_inference(report: dict[str, object]) -> dict[str, object]:
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != 2:
        raise ValueError("Control inference must return exactly two samples")
    original, edited = samples
    if not original["id"].endswith("-original") or not edited["id"].endswith("-edited"):
        raise ValueError("Control inference sample order changed")
    invariant_fields = ("reference_video", "first_frame")
    invariants = {field: original[field] == edited[field] for field in invariant_fields}
    # Two separately extracted first-frame paths are expected, so verify bytes below.
    invariants["first_frame"] = (
        Path(original["first_frame"]).read_bytes() == Path(edited["first_frame"]).read_bytes()
    )
    invariants.update(
        {
            "checkpoint_step_600": report.get("checkpoint_step") == 600,
            "same_prompt": isinstance(report.get("prompt"), str),
            "same_seed": isinstance(report.get("seed"), int),
            "inference_steps_50": report.get("inference_steps") == 50,
            "guidance_scale_3.5": report.get("guidance_scale") == 3.5,
            "conditioning_is_different": original["ellipse_video"] != edited["ellipse_video"],
        }
    )
    if not all(invariants.values()):
        raise ValueError(f"Control inference invariant failed: {invariants}")
    return invariants


def run_experiment(
    *,
    trainer_root: Path,
    source_pair_manifest: Path,
    clip_id: str,
    edited_conditioning: Path,
    checkpoint: Path,
    output_dir: Path,
    prompt: str,
    seed: int,
    argv: Sequence[str],
) -> dict[str, object]:
    pair_path, pair = prepare_control_pair(
        source_pair_manifest=source_pair_manifest,
        clip_id=clip_id,
        edited_conditioning=edited_conditioning,
        output_dir=output_dir,
    )
    inference = run_corrected_inference(
        trainer_root=trainer_root,
        pair_manifest=pair_path,
        checkpoint=checkpoint,
        output_dir=output_dir / "inference",
        prompt=prompt,
        seed=seed,
    )
    invariants = audit_control_inference(inference)
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "step600_trajectory_control_experiment",
        "classification": "reduced_scale_reproduction_hypothesis",
        "source_clip_id": clip_id,
        "checkpoint": str(checkpoint),
        "checkpoint_step": inference["checkpoint_step"],
        "prompt": prompt,
        "seed": seed,
        "inference_steps": inference["inference_steps"],
        "guidance_scale": inference["guidance_scale"],
        "control_pair": pair,
        "inference_manifest": str(output_dir / "inference" / "inference-manifest.json"),
        "invariants": invariants,
        "reproduction_command_argv": list(argv),
        "reproduction_command_shell": shlex.join(argv),
    }
    (output_dir / "experiment-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(argv) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-root", required=True, type=Path)
    parser.add_argument("--source-pair-manifest", required=True, type=Path)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--edited-conditioning", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(parsed_argv)
    command = [
        sys.executable,
        "-m",
        "or_video_reproduction.evaluation.trajectory_experiment",
        *parsed_argv,
    ]
    report = run_experiment(
        trainer_root=args.trainer_root,
        source_pair_manifest=args.source_pair_manifest,
        clip_id=args.clip_id,
        edited_conditioning=args.edited_conditioning,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        prompt=args.prompt,
        seed=args.seed,
        argv=command,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare and run original-versus-edited or edited-only control inference."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from .checkpoint_audit import audited_checkpoint_step
from .corrected_inference import run_corrected_inference


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def infer_inference_split(clip_id: str, source: dict[str, object]) -> str:
    raw = source.get("split")
    if isinstance(raw, str) and raw:
        return raw
    if clip_id.startswith("4dor-"):
        return "table1_ood"
    return "heldout"


def prepare_control_pair(
    *,
    source_pair_manifest: Path,
    clip_id: str,
    edited_conditioning: Path,
    output_dir: Path,
    edited_only: bool = False,
) -> tuple[Path, dict[str, object]]:
    payload = json.loads(source_pair_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
        raise ValueError("Source pair manifest must have schema_version 1 and samples")
    matches = [row for row in payload["samples"] if row.get("id") == clip_id]
    if len(matches) != 1:
        raise ValueError(f"Source pair manifest must contain clip {clip_id!r} exactly once")
    source = matches[0]
    required = ("target_video", "conditioning_video", "geometry_metadata")
    if not edited_only:
        required = (*required, "labels")
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
    split = "heldout" if not edited_only else infer_inference_split(clip_id, source)
    rows = []
    roles = (("edited", edited_copy),) if edited_only else (("original", original_copy), ("edited", edited_copy))
    for role, conditioning in roles:
        rows.append(
            {
                "id": f"{clip_id}-{role}",
                "split": split,
                "take": source.get("take"),
                "target_video": str(_resolve(source_pair_manifest.parent, source["target_video"])),
                "conditioning_video": str(conditioning.resolve()),
            }
        )
    labels = source.get("labels")
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": (
            "edited_only_trajectory_control_inference"
            if edited_only
            else "step600_trajectory_control_inference_pair"
        ),
        "classification": "reduced_scale_reproduction_hypothesis",
        "source_pair_manifest": str(source_pair_manifest),
        "source_clip_id": clip_id,
        "source_geometry_metadata": str(
            _resolve(source_pair_manifest.parent, source["geometry_metadata"])
        ),
        "source_labels": (
            None
            if not isinstance(labels, str)
            else str(_resolve(source_pair_manifest.parent, labels))
        ),
        "controlled_variable": "conditioning_video only",
        "edited_only": edited_only,
        "inference_split": split,
        "samples": rows,
    }
    path = output_dir / "control-pair-manifest.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, result


def audit_control_inference(
    report: dict[str, object], *, expected_checkpoint_step: int = 600
) -> dict[str, object]:
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
            f"checkpoint_step_{expected_checkpoint_step}": (
                report.get("checkpoint_step") == expected_checkpoint_step
            ),
            "same_prompt": isinstance(report.get("prompt"), str),
            "same_seed": isinstance(report.get("seed"), int),
            "inference_steps_50": report.get("inference_steps") == 50,
            "guidance_scale_3.5": report.get("guidance_scale") == 3.5,
            "conditioning_is_different": original["ellipse_video"] != edited["ellipse_video"],
            "sample_count_2": True,
        }
    )
    if not all(invariants.values()):
        raise ValueError(f"Control inference invariant failed: {invariants}")
    return invariants


def audit_edited_only_inference(
    report: dict[str, object], *, expected_checkpoint_step: int
) -> dict[str, object]:
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError("Edited-only inference must return exactly one sample")
    edited = samples[0]
    if not str(edited["id"]).endswith("-edited"):
        raise ValueError("Edited-only inference sample id must end with -edited")
    invariants = {
        f"checkpoint_step_{expected_checkpoint_step}": (
            report.get("checkpoint_step") == expected_checkpoint_step
        ),
        "sample_count_1": True,
        "inference_steps_50": report.get("inference_steps") == 50,
        "guidance_scale_3.5": report.get("guidance_scale") == 3.5,
        "no_original_companion": not any(
            str(row.get("id", "")).endswith("-original") for row in samples
        ),
    }
    if not all(invariants.values()):
        raise ValueError(f"Edited-only inference invariant failed: {invariants}")
    return invariants


def _require_checkpoint_step(
    *,
    checkpoint: Path,
    expected_checkpoint_step: int,
    training_report: Path | None,
    expected_sha256: str | None,
) -> dict[str, object] | None:
    if expected_checkpoint_step != 600 and training_report is None:
        raise ValueError(
            f"Checkpoint step {expected_checkpoint_step} is unverifiable without a training report"
        )
    if training_report is None:
        return None
    return audited_checkpoint_step(
        training_report=training_report,
        checkpoint=checkpoint,
        expected_step=expected_checkpoint_step,
        expected_sha256=expected_sha256,
    )


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
    edited_only: bool = False,
    expected_checkpoint_step: int = 600,
    training_report: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    checkpoint_audit = _require_checkpoint_step(
        checkpoint=checkpoint,
        expected_checkpoint_step=expected_checkpoint_step,
        training_report=training_report,
        expected_sha256=expected_sha256,
    )
    pair_path, pair = prepare_control_pair(
        source_pair_manifest=source_pair_manifest,
        clip_id=clip_id,
        edited_conditioning=edited_conditioning,
        output_dir=output_dir,
        edited_only=edited_only,
    )
    inference = run_corrected_inference(
        trainer_root=trainer_root,
        pair_manifest=pair_path,
        checkpoint=checkpoint,
        output_dir=output_dir / "inference",
        prompt=prompt,
        seed=seed,
        split=str(pair["inference_split"]),
    )
    if inference.get("checkpoint_step") != expected_checkpoint_step:
        raise ValueError(
            f"Inference checkpoint step {inference.get('checkpoint_step')} != "
            f"expected {expected_checkpoint_step}"
        )
    if edited_only:
        invariants = audit_edited_only_inference(
            inference, expected_checkpoint_step=expected_checkpoint_step
        )
        kind = "edited_only_trajectory_control_experiment"
    else:
        invariants = audit_control_inference(
            inference, expected_checkpoint_step=expected_checkpoint_step
        )
        kind = "step600_trajectory_control_experiment"
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "classification": "reduced_scale_reproduction_hypothesis",
        "source_clip_id": clip_id,
        "checkpoint": str(checkpoint),
        "checkpoint_step": inference["checkpoint_step"],
        "checkpoint_audit": checkpoint_audit,
        "edited_only": edited_only,
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
    parser.add_argument("--edited-only", action="store_true")
    parser.add_argument("--expected-checkpoint-step", type=int, default=600)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--expected-sha256")
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
        edited_only=args.edited_only,
        expected_checkpoint_step=args.expected_checkpoint_step,
        training_report=args.training_report,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

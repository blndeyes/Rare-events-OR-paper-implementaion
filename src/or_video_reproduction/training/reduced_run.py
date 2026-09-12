"""Launch the fresh 30-video/600-step BF16 reproduction hypothesis."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .ic_lora_smoke import PINNED_TRAINER_COMMIT, _git_head
from .profiles import load_training_profile

EXPECTED_CHECKPOINT_STEPS = (100, 200, 300, 400, 500, 600)
CHECKPOINT_PATTERN = re.compile(r"^lora_weights_step_(\d+)\.safetensors$")


def build_reduced_run_config(
    official: dict[str, object],
    *,
    precomputed_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Return a fresh, unquantized 600-step config derived from the pinned template."""

    effective = copy.deepcopy(official)
    effective["model"].update(
        {"model_source": "LTXV_13B_097_DEV", "training_mode": "lora", "load_checkpoint": None}
    )
    effective["lora"].update({"rank": 128, "alpha": 128, "dropout": 0.0})
    effective["conditioning"].update(
        {"mode": "reference_video", "first_frame_conditioning_p": 0.2}
    )
    effective["optimization"].update(
        {
            "learning_rate": 2e-4,
            "steps": 600,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw",
            "scheduler_type": "linear",
            "scheduler_params": {"start_factor": 1.0, "end_factor": 0.1},
            "enable_gradient_checkpointing": True,
        }
    )
    effective["acceleration"].update(
        {
            "mixed_precision_mode": "bf16",
            "quantization": None,
            "load_text_encoder_in_8bit": False,
        }
    )
    effective["data"].update(
        {"preprocessed_data_root": str(precomputed_root), "num_dataloader_workers": 2}
    )
    effective["validation"].update(
        {
            "prompts": [],
            "images": None,
            "reference_videos": None,
            "video_dims": [1024, 768, 97],
            "seed": 42,
            "inference_steps": 50,
            "guidance_scale": 3.5,
            "interval": None,
            "skip_initial_validation": True,
        }
    )
    effective["checkpoints"].update({"interval": 100, "keep_last_n": -1})
    effective["wandb"].update({"enabled": False, "log_validation_videos": False})
    effective["hub"].update({"push_to_hub": False, "hub_model_id": None})
    effective["seed"] = 42
    effective["output_dir"] = str(output_dir)
    return effective


class PersistentCsvMetrics:
    """Append and fsync each optimizer-step metric emitted by the official trainer."""

    fieldnames = (
        "recorded_at_utc",
        "global_step",
        "loss",
        "learning_rate",
        "step_time_seconds",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"Refusing to append to existing loss history: {self.path}")
        self._handle = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._sync()

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def record(self, metrics: Mapping[str, object]) -> None:
        required = {
            "train/global_step",
            "train/loss",
            "train/learning_rate",
            "train/step_time",
        }
        if not required.issubset(metrics):
            return
        self._writer.writerow(
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": int(metrics["train/global_step"]),
                "loss": float(metrics["train/loss"]),
                "learning_rate": float(metrics["train/learning_rate"]),
                "step_time_seconds": float(metrics["train/step_time"]),
            }
        )
        self._sync()

    def close(self) -> None:
        if not self._handle.closed:
            self._sync()
            self._handle.close()

    def wrap(
        self, original: Callable[[dict[str, object]], object]
    ) -> Callable[[dict[str, object]], object]:
        def wrapped(metrics: dict[str, object]) -> object:
            self.record(metrics)
            return original(metrics)

        return wrapped

    def __enter__(self) -> PersistentCsvMetrics:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def validate_loss_csv(path: Path, *, expected_steps: int = 600) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed_steps = [int(row["global_step"]) for row in rows]
    expected = list(range(1, expected_steps + 1))
    errors: list[str] = []
    if observed_steps != expected:
        errors.append(
            f"loss history steps are not exactly 1..{expected_steps}: "
            f"observed {len(observed_steps)} rows"
        )
    for row in rows:
        for field in ("loss", "learning_rate", "step_time_seconds"):
            if not math.isfinite(float(row[field])):
                errors.append(f"non-finite {field} at step {row['global_step']}")
    return {
        "passed": not errors,
        "path": str(path),
        "row_count": len(rows),
        "errors": errors,
    }


def audit_checkpoints(
    checkpoints_dir: Path,
    *,
    expected_steps: Sequence[int] = EXPECTED_CHECKPOINT_STEPS,
    minimum_bytes: int = 1_000_000_000,
    opener: Callable[[str], object] | None = None,
) -> dict[str, object]:
    """Check checkpoint presence, plausible size, and safetensors readability."""

    if opener is None:
        from safetensors import safe_open

        opener = lambda path: safe_open(path, framework="pt", device="cpu")

    observed: dict[int, Path] = {}
    if checkpoints_dir.is_dir():
        for path in checkpoints_dir.iterdir():
            match = CHECKPOINT_PATTERN.match(path.name)
            if match:
                observed[int(match.group(1))] = path

    samples: list[dict[str, object]] = []
    for step in expected_steps:
        path = observed.get(step)
        sample: dict[str, object] = {"step": step, "passed": False}
        if path is None:
            sample["errors"] = ["missing checkpoint"]
            samples.append(sample)
            continue
        size = path.stat().st_size
        sample.update({"path": str(path), "size_bytes": size})
        errors = [] if size >= minimum_bytes else [f"size is below {minimum_bytes} bytes"]
        try:
            with opener(str(path)) as handle:
                keys = list(handle.keys())
            sample["tensor_count"] = len(keys)
            if not keys:
                errors.append("checkpoint contains no tensors")
        except Exception as error:  # noqa: BLE001 - report every safetensors reader failure.
            errors.append(f"unreadable safetensors: {type(error).__name__}: {error}")
        sample["errors"] = errors
        sample["passed"] = not errors
        samples.append(sample)

    return {
        "passed": all(bool(sample["passed"]) for sample in samples),
        "expected_steps": list(expected_steps),
        "samples": samples,
    }


def ensure_disk_headroom(path: Path, *, minimum_free_gib: float) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.2f} GiB free at {path}; require {minimum_free_gib:.2f} GiB"
        )
    return {"path": str(path), "free_gib": free_gib, "minimum_free_gib": minimum_free_gib}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_reduced_experiment(
    *,
    trainer_root: Path,
    precomputed_root: Path,
    output_dir: Path,
    report_path: Path,
    profiles_config: Path,
    minimum_free_gib: float,
) -> dict[str, object]:
    """Execute the pinned trainer with strict fresh-run and persistence guards."""

    if _git_head(trainer_root) != PINNED_TRAINER_COMMIT:
        raise ValueError(f"Trainer checkout must be pinned to {PINNED_TRAINER_COMMIT}")
    profile = load_training_profile(profiles_config, "faithful_bf16")
    if not profile.paper_faithful:
        raise ValueError("Reduced run requires the faithful BF16 profile")
    if not precomputed_root.is_dir():
        raise FileNotFoundError(precomputed_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Fresh run output directory is not empty: {output_dir}")

    disk = ensure_disk_headroom(output_dir, minimum_free_gib=minimum_free_gib)
    import yaml

    official = yaml.safe_load(
        (trainer_root / "configs/ltxv_13b_ic_lora.yaml").read_text(encoding="utf-8")
    )
    effective = build_reduced_run_config(
        official, precomputed_root=precomputed_root, output_dir=output_dir
    )
    effective_path = output_dir / "effective-config.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")

    sys.path.insert(0, str(trainer_root / "src"))
    import ltxv_trainer.trainer as trainer_module
    from ltxv_trainer.config import LtxvTrainerConfig

    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "reduced_30train_600step_reproduction_hypothesis",
        "state": "running",
        "trainer_revision": PINNED_TRAINER_COMMIT,
        "fresh_lora": True,
        "patchgan_enabled": False,
        "disk_preflight": disk,
        "effective_config": str(effective_path),
    }
    _write_json(report_path, report)
    metrics_path = output_dir / "loss-history.csv"
    try:
        config = LtxvTrainerConfig(**effective)
        trainer = trainer_module.LtxvTrainer(config)
        with PersistentCsvMetrics(metrics_path) as metrics:
            trainer._log_metrics = metrics.wrap(trainer._log_metrics)
            output, stats = trainer.train(disable_progress_bars=True)
        loss_audit = validate_loss_csv(metrics_path)
        checkpoint_audit = audit_checkpoints(output_dir / "checkpoints")
        report.update(
            {
                "state": (
                    "passed"
                    if loss_audit["passed"] and checkpoint_audit["passed"]
                    else "failed"
                ),
                "output": str(output),
                "stats": stats.__dict__,
                "loss_audit": loss_audit,
                "checkpoint_audit": checkpoint_audit,
            }
        )
    except Exception as error:  # noqa: BLE001 - persist every expensive training failure.
        report.update(
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        if metrics_path.is_file():
            report["loss_audit"] = validate_loss_csv(metrics_path)
        report["checkpoint_audit"] = audit_checkpoints(output_dir / "checkpoints")
    _write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-root", required=True, type=Path)
    parser.add_argument("--precomputed-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--profiles-config", type=Path, default=Path("configs/training_profiles.yaml")
    )
    parser.add_argument("--minimum-free-gib", type=float, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_reduced_experiment(
        trainer_root=args.trainer_root,
        precomputed_root=args.precomputed_root,
        output_dir=args.output_dir,
        report_path=args.report,
        profiles_config=args.profiles_config,
        minimum_free_gib=args.minimum_free_gib,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

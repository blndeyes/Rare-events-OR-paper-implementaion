"""Run one official IC-LoRA optimization step as a hardware integration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

from .ic_lora_smoke import PINNED_TRAINER_COMMIT, _git_head


def build_one_step_config(
    official: dict[str, object],
    paper: dict[str, object],
    *,
    precomputed_root: Path,
    output_dir: Path,
    quantization: str,
    mixed_precision: str,
) -> dict[str, object]:
    """Apply only integration-gate overrides to an official trainer configuration."""

    official["optimization"]["steps"] = 1
    official["acceleration"]["quantization"] = quantization
    official["acceleration"]["mixed_precision_mode"] = mixed_precision
    official["data"]["preprocessed_data_root"] = str(precomputed_root)
    official["data"]["num_dataloader_workers"] = 0
    official["validation"]["interval"] = None
    official["validation"]["video_dims"] = [
        *paper["experiment"]["target_resolution"],
        paper["experiment"]["frames"],
    ]
    official["checkpoints"]["interval"] = None
    official["output_dir"] = str(output_dir)
    official["wandb"]["enabled"] = False
    return official


def run_one_step(
    paper_config_path: Path,
    trainer_root: Path,
    precomputed_root: Path,
    output_dir: Path,
    report_path: Path,
    *,
    quantization: str,
    mixed_precision: str,
    transformer_load_dtype: str,
) -> dict[str, object]:
    if _git_head(trainer_root) != PINNED_TRAINER_COMMIT:
        raise ValueError(f"Trainer checkout must be pinned to {PINNED_TRAINER_COMMIT}")

    import torch
    import yaml

    paper = yaml.safe_load(paper_config_path.read_text(encoding="utf-8"))
    official = yaml.safe_load(
        (trainer_root / "configs/ltxv_13b_ic_lora.yaml").read_text(encoding="utf-8")
    )
    effective = build_one_step_config(
        official,
        paper,
        precomputed_root=precomputed_root,
        output_dir=output_dir,
        quantization=quantization,
        mixed_precision=mixed_precision,
    )

    sys.path.insert(0, str(trainer_root / "src"))
    from ltxv_trainer.config import LtxvTrainerConfig
    import ltxv_trainer.trainer as trainer_module

    if transformer_load_dtype == "fp16":
        official_loader = trainer_module.load_ltxv_components

        def load_fp16_components(*args, **kwargs):
            kwargs["transformer_dtype"] = torch.float16
            return official_loader(*args, **kwargs)

        trainer_module.load_ltxv_components = load_fp16_components

    config = LtxvTrainerConfig(**effective)
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer = trainer_module.LtxvTrainer(config)
        output, stats = trainer.train(disable_progress_bars=True)
        report: dict[str, object] = {
            "state": "passed",
            "output": str(output),
            "stats": stats.__dict__,
        }
    except Exception as error:
        report = {
            "state": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    report.update(
        {
            "schema_version": 1,
            "kind": "official_ic_lora_one_step_hardware_gate",
            "trainer_revision": PINNED_TRAINER_COMMIT,
            "precomputed_root": str(precomputed_root),
            "paper_video_dims": effective["validation"]["video_dims"],
            "quantization": quantization,
            "mixed_precision": mixed_precision,
            "transformer_load_dtype": transformer_load_dtype,
            "paper_deviation": quantization != "no_change"
            or mixed_precision != paper["training"]["mixed_precision"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-config", required=True, type=Path)
    parser.add_argument("--trainer-root", required=True, type=Path)
    parser.add_argument("--precomputed-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--quantization",
        required=True,
        choices=("no_change", "int8-quanto", "int4-quanto", "int2-quanto"),
    )
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--transformer-load-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_one_step(
        args.paper_config,
        args.trainer_root,
        args.precomputed_root,
        args.output_dir,
        args.report,
        quantization=args.quantization,
        mixed_precision=args.mixed_precision,
        transformer_load_dtype=args.transformer_load_dtype,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

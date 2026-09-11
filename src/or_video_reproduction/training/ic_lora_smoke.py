"""Validate and optionally construct the pinned official 13B IC-LoRA model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


PINNED_TRAINER_COMMIT = "e055182fa36dba6f48eb0919aef09d277da30fbd"


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def alignment_report(paper: dict[str, object], official: dict[str, object]) -> dict[str, object]:
    """Compare disclosed paper fields with the selected official trainer template."""

    model = paper["model"]
    training = paper["training"]
    inference = paper["inference"]
    checks = {
        "checkpoint": official["model"]["model_source"] == model["checkpoint"],
        "training_mode": official["model"]["training_mode"] == model["training_mode"],
        "conditioning_mode": official["conditioning"]["mode"] == model["conditioning_mode"],
        "lora_rank": official["lora"]["rank"] == model["lora"]["rank"],
        "lora_alpha": official["lora"]["alpha"] == model["lora"]["alpha"],
        "lora_targets": official["lora"]["target_modules"] == model["lora"]["target_modules"],
        "learning_rate": official["optimization"]["learning_rate"] == training["learning_rate"],
        "batch_size": official["optimization"]["batch_size"] == training["batch_size"],
        "first_frame_probability": official["conditioning"]["first_frame_conditioning_p"]
        == training["first_frame_conditioning_probability"],
        "precision": official["acceleration"]["mixed_precision_mode"]
        == training["mixed_precision"],
        "inference_steps": official["validation"]["inference_steps"]
        == inference["denoising_steps"],
        "guidance_scale": official["validation"]["guidance_scale"]
        == inference["guidance_scale"],
    }
    return {"checks": checks, "all_match": all(checks.values())}


def run_smoke(
    paper_config_path: Path,
    trainer_root: Path,
    output_path: Path,
    *,
    load_transformer: bool,
) -> dict[str, object]:
    if _git_head(trainer_root) != PINNED_TRAINER_COMMIT:
        raise ValueError(f"Trainer checkout must be pinned to {PINNED_TRAINER_COMMIT}")

    import yaml

    paper = yaml.safe_load(paper_config_path.read_text(encoding="utf-8"))
    official_path = trainer_root / "configs/ltxv_13b_ic_lora.yaml"
    official = yaml.safe_load(official_path.read_text(encoding="utf-8"))
    report = alignment_report(paper, official)
    if not report["all_match"]:
        failed = [key for key, matched in report["checks"].items() if not matched]
        raise ValueError(f"Paper and official trainer configuration disagree: {failed}")

    official["optimization"]["steps"] = paper["training"]["steps"]
    official["validation"]["video_dims"] = [
        *paper["experiment"]["target_resolution"],
        paper["experiment"]["frames"],
    ]
    report.update(
        {
            "schema_version": 1,
            "trainer_revision": PINNED_TRAINER_COMMIT,
            "official_config": str(official_path),
            "paper_config": str(paper_config_path),
            "effective_steps": official["optimization"]["steps"],
            "effective_video_dims": official["validation"]["video_dims"],
            "model_constructed": False,
        }
    )

    if load_transformer:
        sys.path.insert(0, str(trainer_root / "src"))
        import torch
        from peft import LoraConfig
        from ltxv_trainer.config import LtxvTrainerConfig
        from ltxv_trainer.model_loader import load_transformer as load_official_transformer

        config = LtxvTrainerConfig(**official)
        started = time.time()
        transformer = load_official_transformer(
            config.model.model_source,
            dtype=torch.bfloat16,
        )
        base_parameters = sum(parameter.numel() for parameter in transformer.parameters())
        transformer.requires_grad_(False)
        transformer.add_adapter(
            LoraConfig(
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.dropout,
                init_lora_weights=True,
            )
        )
        trainable_parameters = sum(
            parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad
        )
        report.update(
            {
                "model_constructed": True,
                "base_parameters": base_parameters,
                "trainable_lora_parameters": trainable_parameters,
                "trainable_percent": 100.0 * trainable_parameters / base_parameters,
                "base_dtype": str(next(transformer.parameters()).dtype),
                "minimum_base_weight_gib": base_parameters * 2 / 1024**3,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-config", required=True, type=Path)
    parser.add_argument("--trainer-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--load-transformer", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_smoke(
        args.paper_config,
        args.trainer_root,
        args.output,
        load_transformer=args.load_transformer,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

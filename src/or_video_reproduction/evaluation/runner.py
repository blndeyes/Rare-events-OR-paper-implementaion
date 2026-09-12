"""Manifest-driven evaluator for generated operating-room videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .backends import dover_score, fvd_i3d_embeddings, inception_logits, lpips_video
from .core import (
    binary_classification_metrics,
    bounding_box_iou,
    peak_signal_to_noise_ratio,
    segmentation_iou,
    structural_similarity,
)
from .distribution import frechet_distance, inception_score
from .video import decode_rgb_video, validate_pair


ALL_METRICS = {"psnr", "ssim", "lpips", "fvd", "is", "dover", "box_iou", "mask_iou", "classification"}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _load_masks(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "labels" not in payload:
            raise ValueError(f"Mask archive must contain a 'labels' array: {path}")
        return np.asarray(payload["labels"])


def evaluate_manifest(
    manifest_path: Path,
    *,
    metrics: set[str],
    device: str,
    batch_size: int,
    google_research_root: Path | None = None,
    dover_root: Path | None = None,
    dover_python: str = "python",
) -> dict[str, object]:
    unknown = metrics - ALL_METRICS
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
        raise ValueError("Evaluation manifest must have schema_version 1 and a samples list")
    root = manifest_path.parent
    samples = payload["samples"]
    if not samples:
        raise ValueError("Evaluation manifest has no samples")
    reference_paths = [_resolve(root, row["reference_video"]) for row in samples]
    generated_paths = [_resolve(root, row["generated_video"]) for row in samples]
    results: dict[str, object] = {
        "schema_version": 1,
        "sample_count": len(samples),
        "metrics": {},
        "conventions": {
            "paired": "strict native-resolution frame mean",
            "iou": "macro over foreground frame/class unions",
            "is": "torchvision Inception-v3 ImageNet V1, generated frames",
            "fvd": "official Google Kinetics-400 I3D, 224x224",
        },
    }
    metric_results = results["metrics"]
    paired = metrics & {"psnr", "ssim", "lpips"}
    if paired:
        collected: dict[str, list[float]] = {name: [] for name in paired}
        for reference_path, generated_path in zip(reference_paths, generated_paths, strict=True):
            validate_pair(reference_path, generated_path)
            reference = decode_rgb_video(reference_path)
            generated = decode_rgb_video(generated_path)
            if "psnr" in paired:
                collected["psnr"].append(peak_signal_to_noise_ratio(reference, generated))
            if "ssim" in paired:
                collected["ssim"].append(structural_similarity(reference, generated))
            if "lpips" in paired:
                collected["lpips"].append(
                    lpips_video(reference, generated, device=device, batch_size=batch_size)
                )
        for name, values in collected.items():
            metric_results[name] = {"mean": float(np.mean(values)), "per_sample": values}
    structural = metrics & {"box_iou", "mask_iou"}
    if structural:
        collected_structural: dict[str, list[float]] = {name: [] for name in structural}
        for row in samples:
            if "reference_masks" not in row or "generated_masks" not in row:
                raise ValueError("IoU metrics require reference_masks and generated_masks on every sample")
            reference = _load_masks(_resolve(root, row["reference_masks"]))
            generated = _load_masks(_resolve(root, row["generated_masks"]))
            if "box_iou" in structural:
                collected_structural["box_iou"].append(float(bounding_box_iou(reference, generated)["mean"]))
            if "mask_iou" in structural:
                collected_structural["mask_iou"].append(float(segmentation_iou(reference, generated)["mean"]))
        for name, values in collected_structural.items():
            metric_results[name] = {"mean": float(np.mean(values)), "per_sample": values}
    if "dover" in metrics:
        if dover_root is None:
            raise ValueError("DOVER requires --dover-root")
        values = [dover_score(path, dover_root=dover_root, python=dover_python) for path in generated_paths]
        metric_results["dover"] = {"mean": float(np.mean(values)), "per_sample": values}
    if "is" in metrics:
        logits = inception_logits(generated_paths, device=device, batch_size=batch_size)
        metric_results["is"] = inception_score(logits, splits=min(10, len(logits)))
    if "fvd" in metrics:
        if google_research_root is None:
            raise ValueError("FVD requires --google-research-root")
        reference_embeddings = fvd_i3d_embeddings(reference_paths, google_research_root=google_research_root)
        generated_embeddings = fvd_i3d_embeddings(generated_paths, google_research_root=google_research_root)
        metric_results["fvd"] = frechet_distance(reference_embeddings, generated_embeddings)
    if "classification" in metrics:
        classification = payload.get("classification")
        if not isinstance(classification, dict):
            raise ValueError("Classification requires manifest.classification")
        metric_results["classification"] = binary_classification_metrics(
            classification["targets"], classification["predictions"],
            positive_label=int(classification.get("positive_label", 1)),
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(sorted(ALL_METRICS)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--google-research-root", type=Path)
    parser.add_argument("--dover-root", type=Path)
    parser.add_argument("--dover-python", default="python")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    metrics = {item.strip().lower() for item in arguments.metrics.split(",") if item.strip()}
    results = evaluate_manifest(
        arguments.manifest,
        metrics=metrics,
        device=arguments.device,
        batch_size=arguments.batch_size,
        google_research_root=arguments.google_research_root,
        dover_root=arguments.dover_root,
        dover_python=arguments.dover_python,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

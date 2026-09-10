"""Create a visual geometry-conditioning audit from one real MMOR frame."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from or_video_reproduction.data.semantics import (
    ENTITY_CLASSES,
    MMOR_ARTIFACT_LABELS,
    MMOR_SEGMENTATION_LABELS,
)

from .depth import mean_valid_depth, normalize_depths
from .ellipse import fit_ellipse
from .palette import PAPER_36_PALETTE
from .render import RenderInstance, render_conditioning


TARGET_SIZE = (1024, 768)


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)


def _load_labels(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize(TARGET_SIZE, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def _load_depth(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.size != TARGET_SIZE:
        image = image.resize(TARGET_SIZE, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float64)


def create_preview(
    rgb_path: Path,
    mask_path: Path,
    depth_path: Path,
    output_dir: Path,
    *,
    smaller_is_nearer: bool,
) -> dict[str, object]:
    rgb = _load_rgb(rgb_path)
    labels = _load_labels(mask_path)
    depth = _load_depth(depth_path)
    if labels.shape != depth.shape:
        raise ValueError(f"Mask {labels.shape} and depth {depth.shape} do not align")

    fitted: dict[str, tuple[int, object, float]] = {}
    skipped: list[dict[str, object]] = []
    for raw_label in (int(value) for value in np.unique(labels) if value != 0):
        class_name = MMOR_SEGMENTATION_LABELS.get(raw_label)
        if raw_label in MMOR_ARTIFACT_LABELS:
            skipped.append({"raw_label": raw_label, "reason": "official_artifact_label"})
            continue
        if class_name is None:
            skipped.append({"raw_label": raw_label, "reason": "unknown_label"})
            continue
        if class_name not in ENTITY_CLASSES:
            skipped.append(
                {"raw_label": raw_label, "class_name": class_name, "reason": "outside_paper_36"}
            )
            continue
        binary = labels == raw_label
        key = f"{raw_label}:{class_name}"
        try:
            ellipse = fit_ellipse(binary)
            raw_depth = mean_valid_depth(depth, binary)
        except ValueError as error:
            skipped.append(
                {"raw_label": raw_label, "class_name": class_name, "reason": str(error)}
            )
            continue
        fitted[key] = (raw_label, ellipse, raw_depth)

    normalized = normalize_depths(
        {key: value[2] for key, value in fitted.items()},
        smaller_is_nearer=smaller_is_nearer,
    )
    instances = [
        RenderInstance(
            key=key,
            class_name=MMOR_SEGMENTATION_LABELS[raw_label],
            ellipse=ellipse,
            normalized_depth=normalized[key],
            raw_depth=raw_depth,
        )
        for key, (raw_label, ellipse, raw_depth) in fitted.items()
    ]
    conditioning = render_conditioning(
        instances,
        (TARGET_SIZE[1], TARGET_SIZE[0]),
        smaller_is_nearer=smaller_is_nearer,
    )

    label_visual = np.zeros_like(conditioning)
    for raw_label, class_name in MMOR_SEGMENTATION_LABELS.items():
        if class_name in PAPER_36_PALETTE:
            red, green = PAPER_36_PALETTE[class_name]
            label_visual[labels == raw_label] = (red, green, 128)

    condition_image = Image.fromarray(conditioning)
    label_image = Image.fromarray(label_visual)
    foreground = Image.fromarray((conditioning.any(axis=2) * 255).astype(np.uint8))
    blended = Image.blend(rgb, condition_image, alpha=0.55)
    overlay = Image.composite(blended, rgb, foreground)

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb.save(output_dir / "source.png")
    label_image.save(output_dir / "semantic_mask.png")
    condition_image.save(output_dir / "conditioning.png")
    overlay.save(output_dir / "overlay.png")
    sheet = Image.new("RGB", (TARGET_SIZE[0] * 2, TARGET_SIZE[1] * 2))
    for index, image in enumerate((rgb, label_image, condition_image, overlay)):
        sheet.paste(image, ((index % 2) * TARGET_SIZE[0], (index // 2) * TARGET_SIZE[1]))
    sheet.save(output_dir / "comparison.png")

    payload: dict[str, object] = {
        "schema_version": 1,
        "method": "filled-mask second-moment matched ellipse",
        "angle_convention": "degrees clockwise from +x in image coordinates, major axis",
        "target_size": list(TARGET_SIZE),
        "depth_source": "provided image; MMOR sensor depth is preview-only, not Video Depth Anything",
        "depth_normalization": "min-max over visible supported instances in this frame",
        "smaller_depth_is_nearer": smaller_is_nearer,
        "instances": [
            {
                "key": instance.key,
                "class_name": instance.class_name,
                "raw_depth_mean": instance.raw_depth,
                "normalized_depth": instance.normalized_depth,
                "red_green": list(PAPER_36_PALETTE[instance.class_name]),
                "ellipse": asdict(instance.ellipse),
            }
            for instance in instances
        ],
        "skipped_labels": skipped,
        "outputs": [
            "source.png",
            "semantic_mask.png",
            "conditioning.png",
            "overlay.png",
            "comparison.png",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--depth-direction",
        choices=("smaller-is-nearer", "larger-is-nearer"),
        default="smaller-is-nearer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = create_preview(
        args.rgb,
        args.mask,
        args.depth,
        args.output_dir,
        smaller_is_nearer=args.depth_direction == "smaller-is-nearer",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render SAM2 labels and Video Depth Anything output as ellipse-only frames."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from or_video_reproduction.data.semantics import ENTITY_CLASSES, MMOR_SEGMENTATION_LABELS

from .depth import mean_valid_depth, normalize_depths
from .ellipse import fit_ellipse
from .palette import PAPER_36_PALETTE
from .render import RenderInstance, render_conditioning


def instances_from_frame(
    labels: NDArray[np.integer],
    depth: NDArray[np.floating],
    *,
    smaller_is_nearer: bool,
    label_classes: Mapping[int, str] | None = None,
) -> tuple[list[RenderInstance], list[dict[str, object]]]:
    """Fit one ellipse to each supported semantic entity in a frame."""

    label_array = np.asarray(labels)
    depth_array = np.asarray(depth)
    if label_array.ndim != 2 or depth_array.ndim != 2:
        raise ValueError("Labels and depth must both be two-dimensional")
    if label_array.shape != depth_array.shape:
        raise ValueError(f"Labels {label_array.shape} and depth {depth_array.shape} must align")

    fitted: dict[str, tuple[str, object, float]] = {}
    skipped: list[dict[str, object]] = []
    for raw_value in np.unique(label_array):
        raw_label = int(raw_value)
        if raw_label == 0:
            continue
        class_name = (
            label_classes.get(raw_label)
            if label_classes is not None
            else MMOR_SEGMENTATION_LABELS.get(raw_label)
        )
        if class_name not in ENTITY_CLASSES:
            skipped.append(
                {
                    "raw_label": raw_label,
                    "class_name": class_name,
                    "reason": "not_a_paper_entity",
                }
            )
            continue
        binary = label_array == raw_label
        key = f"{raw_label}:{class_name}"
        try:
            fitted[key] = (class_name, fit_ellipse(binary), mean_valid_depth(depth_array, binary))
        except ValueError as error:
            skipped.append(
                {"raw_label": raw_label, "class_name": class_name, "reason": str(error)}
            )

    normalized = normalize_depths(
        {key: item[2] for key, item in fitted.items()},
        smaller_is_nearer=smaller_is_nearer,
    )
    instances = [
        RenderInstance(
            key=key,
            class_name=class_name,
            ellipse=ellipse,
            normalized_depth=normalized[key],
            raw_depth=raw_depth,
        )
        for key, (class_name, ellipse, raw_depth) in fitted.items()
    ]
    return instances, skipped


def render_frame(
    labels: NDArray[np.integer],
    depth: NDArray[np.floating],
    *,
    smaller_is_nearer: bool,
    label_classes: Mapping[int, str] | None = None,
) -> tuple[NDArray[np.uint8], list[RenderInstance], list[dict[str, object]]]:
    instances, skipped = instances_from_frame(
        labels,
        depth,
        smaller_is_nearer=smaller_is_nearer,
        label_classes=label_classes,
    )
    image = render_conditioning(
        instances,
        tuple(int(value) for value in labels.shape),
        smaller_is_nearer=smaller_is_nearer,
    )
    return image, instances, skipped


def _load_array(path: Path, preferred_key: str) -> NDArray[np.generic]:
    with np.load(path) as archive:
        if preferred_key in archive:
            return np.asarray(archive[preferred_key])
        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]])
        raise KeyError(f"{path} has no {preferred_key!r} array; keys are {archive.files}")


def render_sequence(
    labels_path: Path,
    depth_path: Path,
    output_dir: Path,
    *,
    fps: int = 24,
    smaller_is_nearer: bool = False,
    label_classes: Mapping[int, str] | None = None,
) -> dict[str, object]:
    labels = _load_array(labels_path, "labels")
    depths = _load_array(depth_path, "depths")
    if labels.ndim != 3 or depths.ndim != 3:
        raise ValueError(
            f"Expected frame x height x width arrays, got {labels.shape}, {depths.shape}"
        )
    if labels.shape != depths.shape:
        raise ValueError(f"Label shape {labels.shape} and depth shape {depths.shape} must match")
    if fps <= 0:
        raise ValueError("fps must be positive")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[dict[str, object]] = []
    contact_indices = sorted({0, *(min(index, labels.shape[0] - 1) for index in (24, 48, 72, 96))})
    contact_images: dict[int, Image.Image] = {}

    for frame_index in range(labels.shape[0]):
        image_array, instances, skipped = render_frame(
            labels[frame_index], depths[frame_index],
            smaller_is_nearer=smaller_is_nearer, label_classes=label_classes,
        )
        image = Image.fromarray(image_array)
        image.save(frames_dir / f"{frame_index:06d}.png")
        if frame_index in contact_indices:
            contact_images[frame_index] = image.copy()
        frame_records.append(
            {
                "frame_index": frame_index,
                "instances": [
                    {
                        "key": instance.key,
                        "class_name": instance.class_name,
                        "ellipse": asdict(instance.ellipse),
                        "raw_relative_depth_mean": instance.raw_depth,
                        "normalized_nearness": instance.normalized_depth,
                        "red_green": list(PAPER_36_PALETTE[instance.class_name]),
                    }
                    for instance in instances
                ],
                "skipped": skipped,
            }
        )

    width, height = labels.shape[2], labels.shape[1]
    sheet = Image.new("RGB", (width * len(contact_indices), height))
    for column, frame_index in enumerate(contact_indices):
        sheet.paste(contact_images[frame_index], (column * width, 0))
    sheet.save(output_dir / "conditioning_contact_sheet.png")

    video_name: str | None = None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        video_name = "conditioning.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_dir / video_name),
            ],
            check=True,
        )

    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "paper_ellipse_only_geometric_conditioning",
        "labels_path": str(labels_path),
        "depth_path": str(depth_path),
        "shape": list(labels.shape),
        "fps": fps,
        "background": "black",
        "depth_source": "Video Depth Anything relative ViT-L",
        "depth_direction": "smaller_is_nearer" if smaller_is_nearer else "larger_is_nearer",
        "depth_normalization": "per-frame min-max over visible entity instance means",
        "ellipse_fit": "filled-mask second-moment match",
        "label_classes": (
            {str(key): value for key, value in sorted(label_classes.items())}
            if label_classes is not None
            else None
        ),
        "contact_frames": contact_indices,
        "video_output": video_name,
        "frames": frame_records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--depths", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--depth-direction",
        choices=("larger-is-nearer", "smaller-is-nearer"),
        default="larger-is-nearer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = render_sequence(
        args.labels,
        args.depths,
        args.output_dir,
        fps=args.fps,
        smaller_is_nearer=args.depth_direction == "smaller-is-nearer",
    )
    print(
        json.dumps(
            {key: metadata[key] for key in ("kind", "shape", "fps", "video_output")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

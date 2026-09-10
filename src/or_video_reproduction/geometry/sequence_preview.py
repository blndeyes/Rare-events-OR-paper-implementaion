"""Render and audit a short sequence of real MMOR geometry frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from .preview import TARGET_SIZE, create_preview


def _angle_delta(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _temporal_deltas(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    deltas: list[dict[str, object]] = []
    for previous, current in zip(frames, frames[1:]):
        previous_instances = {
            row["class_name"]: row for row in previous["instances"]  # type: ignore[index]
        }
        current_instances = {
            row["class_name"]: row for row in current["instances"]  # type: ignore[index]
        }
        for class_name in sorted(previous_instances.keys() & current_instances.keys()):
            before = previous_instances[class_name]["ellipse"]
            after = current_instances[class_name]["ellipse"]
            dx = float(after["center_x"]) - float(before["center_x"])
            dy = float(after["center_y"]) - float(before["center_y"])
            major_before = float(before["major_diameter"])
            minor_before = float(before["minor_diameter"])
            deltas.append(
                {
                    "from_frame": previous["frame"],
                    "to_frame": current["frame"],
                    "class_name": class_name,
                    "center_displacement_pixels": math.hypot(dx, dy),
                    "major_diameter_ratio": float(after["major_diameter"]) / major_before,
                    "minor_diameter_ratio": float(after["minor_diameter"]) / minor_before,
                    "angle_delta_degrees": _angle_delta(
                        float(before["angle_degrees"]), float(after["angle_degrees"])
                    ),
                }
            )
    return deltas


def create_sequence_preview(
    root: Path,
    procedure: str,
    camera: int,
    frame_numbers: list[int],
    output_dir: Path,
    *,
    smaller_is_nearer: bool,
) -> dict[str, object]:
    if not frame_numbers:
        raise ValueError("At least one frame number is required")
    camera_name = f"{camera:02d}"
    procedure_root = root / procedure
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_payloads: list[dict[str, object]] = []
    overlay_images: list[Image.Image] = []
    conditioning_images: list[Image.Image] = []

    for frame_number in frame_numbers:
        stem = f"camera{camera_name}_colorimage-{frame_number:06d}"
        frame_dir = output_dir / f"frame-{frame_number:06d}"
        payload = create_preview(
            procedure_root / "colorimage" / f"{stem}.jpg",
            procedure_root / f"segmentation_export_{camera}" / f"{stem}.png",
            procedure_root / "depthimage" / f"camera{camera_name}_depthimage-{frame_number:06d}.tiff",
            frame_dir,
            smaller_is_nearer=smaller_is_nearer,
        )
        payload["frame"] = frame_number
        frame_payloads.append(payload)
        with Image.open(frame_dir / "overlay.png") as image:
            overlay_images.append(image.copy())
        with Image.open(frame_dir / "conditioning.png") as image:
            conditioning_images.append(image.copy())

    duration_ms = 1000
    overlay_images[0].save(
        output_dir / "overlay.gif",
        save_all=True,
        append_images=overlay_images[1:],
        duration=duration_ms,
        loop=0,
    )
    conditioning_images[0].save(
        output_dir / "conditioning.gif",
        save_all=True,
        append_images=conditioning_images[1:],
        duration=duration_ms,
        loop=0,
    )

    thumbnail_size = (512, 384)

    def save_contact_sheet(images: list[Image.Image], name: str) -> None:
        sheet = Image.new(
            "RGB", (thumbnail_size[0] * len(images), thumbnail_size[1] + 28), color=(0, 0, 0)
        )
        draw = ImageDraw.Draw(sheet)
        for index, (frame_number, image) in enumerate(zip(frame_numbers, images)):
            thumbnail = image.resize(thumbnail_size, Image.Resampling.NEAREST)
            x = index * thumbnail_size[0]
            sheet.paste(thumbnail, (x, 28))
            draw.text((x + 8, 7), f"frame {frame_number:06d}", fill=(255, 255, 255))
        sheet.save(output_dir / name)

    save_contact_sheet(conditioning_images, "conditioning_contact_sheet.png")
    save_contact_sheet(overlay_images, "diagnostic_overlay_contact_sheet.png")

    payload = {
        "schema_version": 1,
        "procedure": procedure,
        "camera": camera,
        "frames": frame_payloads,
        "temporal_deltas": _temporal_deltas(frame_payloads),
        "model_conditioning_output": "conditioning.gif",
        "model_conditioning_frames": [
            f"frame-{frame_number:06d}/conditioning.png" for frame_number in frame_numbers
        ],
        "diagnostic_outputs": [
            "overlay.gif",
            "conditioning_contact_sheet.png",
            "diagnostic_overlay_contact_sheet.png",
        ],
    }
    (output_dir / "sequence_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--procedure", required=True)
    parser.add_argument("--camera", required=True, type=int)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--count", default=5, type=int)
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--depth-direction",
        choices=("smaller-is-nearer", "larger-is-nearer"),
        default="smaller-is-nearer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count <= 0 or args.step <= 0:
        raise SystemExit("--count and --step must be positive")
    frames = [args.start_frame + index * args.step for index in range(args.count)]
    payload = create_sequence_preview(
        args.root,
        args.procedure,
        args.camera,
        frames,
        args.output_dir,
        smaller_is_nearer=args.depth_direction == "smaller-is-nearer",
    )
    print(
        json.dumps(
            {
                "procedure": payload["procedure"],
                "camera": payload["camera"],
                "frame_count": len(payload["frames"]),
                "temporal_comparisons": len(payload["temporal_deltas"]),
                "model_conditioning_output": payload["model_conditioning_output"],
                "diagnostic_outputs": payload["diagnostic_outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

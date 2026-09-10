"""Read-only dataset inventory commands.

The first inventory deliberately depends only on the Python standard library so it can
run before the heavyweight training environment is installed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


COLOR_FRAME_PATTERN = re.compile(
    r"^camera(?P<camera>\d+)_colorimage-(?P<frame>\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
PROCEDURE_PATTERN = re.compile(r"(?:^|[-_])(?:\d+)(?:[-_]\d+)?_(?:PKA|TKA)$", re.IGNORECASE)


@dataclass(frozen=True)
class CameraFrames:
    camera: str
    count: int
    minimum_frame: int
    maximum_frame: int
    missing_inside_range: int


def _summarize_color_frames(paths: Iterable[Path]) -> list[CameraFrames]:
    frames_by_camera: dict[str, set[int]] = defaultdict(set)
    for path in paths:
        match = COLOR_FRAME_PATTERN.match(path.name)
        if match:
            frames_by_camera[match.group("camera")].add(int(match.group("frame")))

    summaries = []
    for camera, frames in sorted(frames_by_camera.items()):
        minimum = min(frames)
        maximum = max(frames)
        summaries.append(
            CameraFrames(
                camera=camera,
                count=len(frames),
                minimum_frame=minimum,
                maximum_frame=maximum,
                missing_inside_range=(maximum - minimum + 1) - len(frames),
            )
        )
    return summaries


def _count_files(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.is_file()) if directory.is_dir() else 0


def inventory_mmor(root: Path) -> dict[str, object]:
    """Return a filename-level inventory of an extracted MMOR dataset."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MMOR root is not a directory: {root}")

    procedure_rows: list[dict[str, object]] = []
    for procedure in sorted(path for path in root.iterdir() if path.is_dir()):
        if not PROCEDURE_PATTERN.search(procedure.name):
            continue

        color_dir = procedure / "colorimage"
        color_frames = _summarize_color_frames(color_dir.iterdir() if color_dir.is_dir() else ())
        annotation_counts = {
            child.name: _count_files(child)
            for child in sorted(procedure.iterdir())
            if child.is_dir()
            and (child.name.startswith("panoptic_seg_") or child.name.startswith("segmentation_export_"))
        }
        procedure_rows.append(
            {
                "procedure": procedure.name,
                "color_frames": [asdict(summary) for summary in color_frames],
                "annotation_file_counts": annotation_counts,
            }
        )

    camera_totals: Counter[str] = Counter()
    for row in procedure_rows:
        for camera in row["color_frames"]:  # type: ignore[union-attr]
            camera_totals[camera["camera"]] += camera["count"]

    return {
        "schema_version": 1,
        "dataset": "MMOR",
        "root": str(root),
        "procedure_count": len(procedure_rows),
        "camera_frame_totals": dict(sorted(camera_totals.items())),
        "procedures": procedure_rows,
    }


def write_inventory(payload: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    mmor = subparsers.add_parser("mmor", help="inventory an extracted MMOR dataset")
    mmor.add_argument("--root", required=True, type=Path)
    mmor.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "mmor":
        write_inventory(inventory_mmor(args.root), args.output)
        return 0
    raise AssertionError(f"Unhandled dataset: {args.dataset}")


if __name__ == "__main__":
    raise SystemExit(main())


"""Build and validate the paper's five-keyframe/97-frame MMOR clip contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Sequence

from .inventory import MMOR_TAKE_TO_FOLDER


SOURCE_FPS = 1
OUTPUT_FPS = 24
SOURCE_KEYFRAME_COUNT = 5
OUTPUT_FRAME_COUNT = 97
KEYFRAME_OUTPUT_INDICES = (0, 24, 48, 72, 96)
TARGET_WIDTH = 1024
TARGET_HEIGHT = 768
SUPPORTED_AZURE_CAMERAS = (1, 4, 5)


def _load_timestamp_rows(path: Path) -> list[tuple[int, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    timestamps = payload.get("timestamps") if isinstance(payload, dict) else None
    if not isinstance(timestamps, dict):
        raise ValueError(f"Expected a timestamps object in {path}")
    rows: list[tuple[int, dict[str, object]]] = []
    for key, value in timestamps.items():
        if not str(key).isdigit() or not isinstance(value, dict):
            raise ValueError(f"Invalid timestamp row {key!r} in {path}")
        rows.append((int(key), value))
    return sorted(rows)


def build_mmor_clip(
    root: Path,
    *,
    take: str,
    camera: int,
    start_timestamp: int,
    split: str = "smoke",
) -> dict[str, object]:
    """Build one clip without modifying or copying source data."""

    root = root.expanduser().resolve()
    if camera not in SUPPORTED_AZURE_CAMERAS:
        raise ValueError(
            f"Camera {camera} has no official MMOR panoptic export; expected one of "
            f"{SUPPORTED_AZURE_CAMERAS}"
        )
    timestamp_path = root / "take_jsons" / f"{take}.json"
    rows = _load_timestamp_rows(timestamp_path)
    index_by_timestamp = {timestamp: index for index, (timestamp, _) in enumerate(rows)}
    if start_timestamp not in index_by_timestamp:
        raise ValueError(f"Timestamp {start_timestamp} is absent from {timestamp_path}")
    start_index = index_by_timestamp[start_timestamp]
    selected = rows[start_index : start_index + SOURCE_KEYFRAME_COUNT]
    if len(selected) != SOURCE_KEYFRAME_COUNT:
        raise ValueError("Clip would extend beyond the end of the logical take")
    expected_timestamps = list(range(start_timestamp, start_timestamp + SOURCE_KEYFRAME_COUNT))
    actual_timestamps = [timestamp for timestamp, _ in selected]
    if actual_timestamps != expected_timestamps:
        raise ValueError(
            f"Source timestamps must be consecutive: expected {expected_timestamps}, "
            f"got {actual_timestamps}"
        )

    procedure = MMOR_TAKE_TO_FOLDER.get(take, take)
    source_rows = []
    for (timestamp, row), output_index in zip(selected, KEYFRAME_OUTPUT_INDICES):
        azure_id = row.get("azure")
        if azure_id is None or not str(azure_id).isdigit():
            raise ValueError(f"Timestamp {timestamp} has no numeric Azure frame ID")
        frame_id = str(azure_id).zfill(6)
        rgb_relative = PurePosixPath(
            procedure, "colorimage", f"camera{camera:02d}_colorimage-{frame_id}.jpg"
        )
        rgb_path = root / Path(*rgb_relative.parts)
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB keyframe: {rgb_path}")
        source_rows.append(
            {
                "timestamp_index": timestamp,
                "azure_frame_id": frame_id,
                "output_frame_index": output_index,
                "rgb_path": rgb_relative.as_posix(),
            }
        )

    first_id = source_rows[0]["azure_frame_id"]
    mask_relative = PurePosixPath(
        procedure,
        f"segmentation_export_{camera}",
        f"camera{camera:02d}_colorimage-{first_id}.png",
    )
    first_mask = root / Path(*mask_relative.parts)
    if not first_mask.is_file():
        raise FileNotFoundError(f"Missing first-frame ground-truth mask: {first_mask}")

    clip_id = f"mmor-{take}-camera{camera:02d}-timestamp{start_timestamp:06d}"
    return {
        "schema_version": 1,
        "status": "smoke" if split == "smoke" else "selection_required",
        "paper_contract": {
            "source_fps": SOURCE_FPS,
            "source_keyframes": SOURCE_KEYFRAME_COUNT,
            "output_fps": OUTPUT_FPS,
            "output_frames": OUTPUT_FRAME_COUNT,
            "duration_seconds": (OUTPUT_FRAME_COUNT - 1) / OUTPUT_FPS,
            "resolution": [TARGET_WIDTH, TARGET_HEIGHT],
            "keyframe_output_indices": list(KEYFRAME_OUTPUT_INDICES),
        },
        "clip": {
            "clip_id": clip_id,
            "dataset": "MMOR",
            "split": split,
            "take": take,
            "procedure_directory": procedure,
            "camera": camera,
            "source_keyframes": source_rows,
            "first_frame_ground_truth_mask": mask_relative.as_posix(),
        },
        "unresolved": [
            "authors' exact clip identity",
            "interpolation prompt",
            "interpolation seed and sampling settings",
        ],
    }


def validate_clip_manifest(payload: dict[str, object]) -> None:
    contract = payload["paper_contract"]
    clip = payload["clip"]
    if contract["output_frames"] != OUTPUT_FRAME_COUNT:
        raise ValueError("Paper contract must contain 97 output frames")
    if contract["output_fps"] != OUTPUT_FPS:
        raise ValueError("Paper contract must use 24 fps")
    keyframes = clip["source_keyframes"]
    indices = [row["output_frame_index"] for row in keyframes]
    if indices != list(KEYFRAME_OUTPUT_INDICES):
        raise ValueError(f"Invalid keyframe output indices: {indices}")
    if len({row["rgb_path"] for row in keyframes}) != SOURCE_KEYFRAME_COUNT:
        raise ValueError("The five source keyframes must be distinct")


def write_manifest(payload: dict[str, object], output: Path) -> None:
    validate_clip_manifest(payload)
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--take", required=True)
    parser.add_argument("--camera", required=True, type=int)
    parser.add_argument("--start-timestamp", required=True, type=int)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_mmor_clip(
        args.root,
        take=args.take,
        camera=args.camera,
        start_timestamp=args.start_timestamp,
        split=args.split,
    )
    write_manifest(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

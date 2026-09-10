"""Read-only MMOR and 4DOR dataset inventory commands.

This module intentionally uses only the Python standard library. It can therefore
run on the training host before the heavyweight model environment is installed.
The inventory never writes below a dataset root.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


FRAME_PATTERN = re.compile(
    r"^camera(?P<camera>\d+)_(?P<modality>[a-z0-9_]+)-(?P<frame>\d+)\.(?P<ext>[a-z0-9]+)$",
    re.IGNORECASE,
)
COLOR_FRAME_PATTERN = re.compile(
    r"^camera(?P<camera>\d+)_colorimage-(?P<frame>\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
PROCEDURE_PATTERN = re.compile(r"(?:^|[-_])(?:\d+)(?:[-_]\d+)?_(?:PKA|TKA)$", re.IGNORECASE)
TAKE_PATTERN = re.compile(r"^export_holistic_take(?P<take>\d+)_processed$")
MMOR_LOGICAL_TAKES = (
    "001_PKA",
    "002_PKA",
    "003_TKA",
    "004_PKA",
    "005_TKA",
    "006_PKA",
    "007_TKA",
    "008_PKA",
    "009_TKA",
    "010_PKA",
    "011_TKA",
    "012_1_PKA",
    "012_2_PKA",
    "013_PKA",
    "014_PKA",
    "015_PKA",
    "016_PKA",
    "017_PKA",
    "018_1_PKA",
    "018_2_PKA",
    "019_PKA",
    "020_PKA",
    "021_PKA",
    "022_PKA",
    "023_PKA",
    "024_PKA",
    "025_PKA",
    "026_PKA",
    "027_PKA",
    "028_PKA",
    "029_PKA",
    "030_PKA",
    "031_PKA",
    "032_PKA",
    "033_PKA",
    "035_PKA",
    "036_PKA",
    "037_TKA",
    "038_TKA",
)
MMOR_TAKE_TO_FOLDER = {
    "012_1_PKA": "012_PKA",
    "012_2_PKA": "012_PKA",
    "015_PKA": "015-018_PKA",
    "016_PKA": "015-018_PKA",
    "017_PKA": "015-018_PKA",
    "018_1_PKA": "015-018_PKA",
    "018_2_PKA": "015-018_PKA",
    "019_PKA": "019-022_PKA",
    "020_PKA": "019-022_PKA",
    "021_PKA": "019-022_PKA",
    "022_PKA": "019-022_PKA",
    "023_PKA": "023-032_PKA",
    "024_PKA": "023-032_PKA",
    "025_PKA": "023-032_PKA",
    "026_PKA": "023-032_PKA",
    "027_PKA": "023-032_PKA",
    "028_PKA": "023-032_PKA",
    "029_PKA": "023-032_PKA",
    "030_PKA": "023-032_PKA",
    "031_PKA": "023-032_PKA",
    "032_PKA": "023-032_PKA",
}
MMOR_OFFICIAL_PANOPTIC_SPLIT = {
    "train": [
        "001_PKA",
        "003_TKA",
        "005_TKA",
        "006_PKA",
        "008_PKA",
        "010_PKA",
        "012_1_PKA",
        "012_2_PKA",
        "035_PKA",
        "037_TKA",
    ],
    "val": ["002_PKA", "007_TKA", "009_TKA"],
    "test": ["004_PKA", "011_TKA", "036_PKA", "038_TKA"],
    "short_clips": list(MMOR_LOGICAL_TAKES[13:35]),
}


@dataclass(frozen=True)
class CameraFrames:
    camera: str
    count: int
    minimum_frame: int
    maximum_frame: int
    missing_inside_range: int


def _frame_sets(paths: Iterable[Path]) -> tuple[dict[str, set[int]], Counter[str]]:
    frames_by_camera: dict[str, set[int]] = defaultdict(set)
    extensions: Counter[str] = Counter()
    for path in paths:
        match = FRAME_PATTERN.match(path.name)
        if match:
            frames_by_camera[match.group("camera")].add(int(match.group("frame")))
            extensions[match.group("ext").lower()] += 1
    return frames_by_camera, extensions


def _summarize_frame_sets(frames_by_camera: dict[str, set[int]]) -> list[CameraFrames]:
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


def _summarize_color_frames(paths: Iterable[Path]) -> list[CameraFrames]:
    frames_by_camera: dict[str, set[int]] = defaultdict(set)
    for path in paths:
        match = COLOR_FRAME_PATTERN.match(path.name)
        if match:
            frames_by_camera[match.group("camera")].add(int(match.group("frame")))
    return _summarize_frame_sets(frames_by_camera)


def _files(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.is_file()] if directory.is_dir() else []


def _count_files(directory: Path) -> int:
    return len(_files(directory))


def _sequence_inventory(directory: Path) -> tuple[dict[str, object], dict[str, set[int]]]:
    files = _files(directory)
    frame_sets, extensions = _frame_sets(files)
    return (
        {
            "file_count": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "extensions": dict(sorted(extensions.items())),
            "camera_frames": [asdict(row) for row in _summarize_frame_sets(frame_sets)],
        },
        frame_sets,
    )


def _coverage(reference: set[int], candidate: set[int]) -> dict[str, int | float]:
    matched = len(reference & candidate)
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "matched": matched,
        "missing_from_candidate": len(reference - candidate),
        "extra_without_reference": len(candidate - reference),
        "coverage": round(matched / len(reference), 6) if reference else 0.0,
    }


def _numeric_values(rows: list[tuple[object, dict[str, object]]], key: str) -> list[int]:
    return [int(row[key]) for _, row in rows if key in row and str(row[key]).isdigit()]


def _json_summary(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {"path": path.name, "exists": path.is_file()}
    if not path.is_file():
        return summary
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        summary["valid"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
        return summary
    summary["valid"] = True
    summary["type"] = type(payload).__name__
    summary["count"] = len(payload) if isinstance(payload, (dict, list)) else None
    if isinstance(payload, dict):
        summary["keys"] = sorted(str(key) for key in payload)
    return summary


def _timestamp_rows(path: Path) -> tuple[list[tuple[object, dict[str, object]]], dict[str, object]]:
    summary = _json_summary(path)
    if not summary.get("valid"):
        return [], summary
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows: list[tuple[object, dict[str, object]]] = []
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, list) and len(row) == 2 and isinstance(row[1], dict):
                rows.append((row[0], row[1]))
    elif isinstance(payload, dict):
        source = payload.get("timestamps", payload)
        if isinstance(source, dict):
            rows.extend((key, value) for key, value in source.items() if isinstance(value, dict))
    summary["timestamp_rows"] = len(rows)
    summary["channel_keys"] = sorted({str(key) for _, row in rows for key in row})
    return rows, summary


def _discover_mmor_modalities(procedure: Path) -> list[Path]:
    prefixes = (
        "colorimage",
        "depthimage",
        "panoptic_seg_",
        "segmentation_export_",
        "simstation_segmentation_export_",
    )
    return sorted(
        child for child in procedure.iterdir() if child.is_dir() and child.name.startswith(prefixes)
    )


def inventory_mmor(root: Path) -> dict[str, object]:
    """Return a filename- and metadata-level inventory of extracted MMOR data."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MMOR root is not a directory: {root}")

    procedure_rows: list[dict[str, object]] = []
    for procedure in sorted(path for path in root.iterdir() if path.is_dir()):
        if not PROCEDURE_PATTERN.search(procedure.name):
            continue

        modalities: dict[str, object] = {}
        modality_frames: dict[str, dict[str, set[int]]] = {}
        for directory in _discover_mmor_modalities(procedure):
            summary, frames = _sequence_inventory(directory)
            modalities[directory.name] = summary
            modality_frames[directory.name] = frames

        color_frames = modality_frames.get("colorimage", {})
        correspondence: dict[str, dict[str, object]] = {}
        for name, frames_by_camera in modality_frames.items():
            if name == "colorimage":
                continue
            for camera, frames in frames_by_camera.items():
                correspondence[f"{name}:camera{camera}"] = _coverage(
                    color_frames.get(camera, set()), frames
                )

        annotation_counts = {
            name: value["file_count"]
            for name, value in modalities.items()
            if name.startswith(("panoptic_seg_", "segmentation_export_"))
            and isinstance(value, dict)
        }
        metadata_files = [
            "timestamps.json",
            "timestamp_to_pcd_and_frames_list.json",
            "trackercam_separation_indices.json",
        ]
        metadata = {name: _json_summary(procedure / name) for name in metadata_files}
        logical_takes = [
            take
            for take in MMOR_LOGICAL_TAKES
            if MMOR_TAKE_TO_FOLDER.get(take, take) == procedure.name
        ]
        take_rows: list[tuple[object, dict[str, object]]] = []
        take_json_summaries = {}
        for take in logical_takes:
            rows, summary = _timestamp_rows(root / "take_jsons" / f"{take}.json")
            take_rows.extend(rows)
            take_json_summaries[take] = summary
        expected_azure_values = _numeric_values(take_rows, "azure")
        expected_azure = set(expected_azure_values)
        timestamp_correspondence = {
            camera: {
                **_coverage(expected_azure, frames),
                "timestamp_rows": len(expected_azure_values),
                "duplicate_timestamp_references": (
                    len(expected_azure_values) - len(expected_azure)
                ),
            }
            for camera, frames in sorted(color_frames.items())
        }

        procedure_rows.append(
            {
                "procedure": procedure.name,
                "color_frames": [asdict(row) for row in _summarize_frame_sets(color_frames)],
                "annotation_file_counts": annotation_counts,
                "modalities": modalities,
                "correspondence": correspondence,
                "metadata": metadata,
                "logical_takes": logical_takes,
                "take_jsons": take_json_summaries,
                "timestamp_correspondence": timestamp_correspondence,
            }
        )

    camera_totals: Counter[str] = Counter()
    for row in procedure_rows:
        for camera in row["color_frames"]:  # type: ignore[union-attr]
            camera_totals[camera["camera"]] += camera["count"]

    return {
        "schema_version": 2,
        "dataset": "MMOR",
        "root": str(root),
        "procedure_count": len(procedure_rows),
        "camera_frame_totals": dict(sorted(camera_totals.items())),
        "logical_take_count": len(MMOR_LOGICAL_TAKES),
        "official_panoptic_split": MMOR_OFFICIAL_PANOPTIC_SPLIT,
        "procedures": procedure_rows,
    }


def _resolve_4dor_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"4DOR root is not a directory: {root}")
    if any(TAKE_PATTERN.match(child.name) for child in root.iterdir() if child.is_dir()):
        return root
    nested = root / "4D-OR"
    if nested.is_dir() and any(
        TAKE_PATTERN.match(child.name) for child in nested.iterdir() if child.is_dir()
    ):
        return nested
    raise FileNotFoundError(f"No export_holistic_take*_processed directories below: {root}")


def _expected_4dor_coverage(
    rows: list[tuple[object, dict[str, object]]],
    color_frames: dict[str, set[int]],
    depth_frames: dict[str, set[int]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for camera in sorted(color_frames):
        number = int(camera)
        color_values = _numeric_values(rows, f"color_{number}")
        depth_values = _numeric_values(rows, f"depth_{number}")
        expected_color = set(color_values)
        expected_depth = set(depth_values)
        result[camera] = {
            "color": {
                **_coverage(expected_color, color_frames.get(camera, set())),
                "timestamp_rows": len(color_values),
                "duplicate_timestamp_references": len(color_values) - len(expected_color),
            },
            "depth": {
                **_coverage(expected_depth, depth_frames.get(camera, set())),
                "timestamp_rows": len(depth_values),
                "duplicate_timestamp_references": len(depth_values) - len(expected_depth),
            },
        }
    return result


def inventory_4dor(root: Path) -> dict[str, object]:
    """Return a filename- and timestamp-level inventory of extracted 4DOR data."""

    requested_root = root.expanduser().resolve()
    resolved_root = _resolve_4dor_root(root)
    takes: list[dict[str, object]] = []
    camera_totals: Counter[str] = Counter()
    candidates = (
        child
        for child in resolved_root.iterdir()
        if child.is_dir() and TAKE_PATTERN.match(child.name)
    )

    def take_number(path: Path) -> int:
        match = TAKE_PATTERN.match(path.name)
        assert match is not None
        return int(match.group("take"))

    for take in sorted(candidates, key=take_number):
        match = TAKE_PATTERN.match(take.name)
        assert match is not None
        color_summary, color_frames = _sequence_inventory(take / "colorimage")
        depth_summary, depth_frames = _sequence_inventory(take / "depthimage")
        for camera, frames in color_frames.items():
            camera_totals[camera] += len(frames)

        timestamp_rows, timestamp_summary = _timestamp_rows(
            take / "timestamp_to_pcd_and_frames_list.json"
        )
        takes.append(
            {
                "take": int(match.group("take")),
                "directory": take.name,
                "modalities": {"colorimage": color_summary, "depthimage": depth_summary},
                "pcd_count": _count_files(take / "pcds"),
                "annotation_count": _count_files(take / "annotations"),
                "metadata": {
                    "timestamp_to_pcd_and_frames_list.json": timestamp_summary,
                    "timestamp_to_pcd_and_frames_dict.json": _json_summary(
                        take / "timestamp_to_pcd_and_frames_dict.json"
                    ),
                    "2D_keypoint_annotations.json": _json_summary(
                        take / "2D_keypoint_annotations.json"
                    ),
                },
                "timestamp_correspondence": _expected_4dor_coverage(
                    timestamp_rows, color_frames, depth_frames
                ),
            }
        )

    return {
        "schema_version": 2,
        "dataset": "4DOR",
        "requested_root": str(requested_root),
        "root": str(resolved_root),
        "take_count": len(takes),
        "camera_frame_totals": dict(sorted(camera_totals.items())),
        "official_split": {
            "train": [1, 3, 5, 7, 9, 10],
            "val": [4, 8],
            "test": [2, 6],
        },
        "takes": takes,
    }


def render_report(payload: dict[str, object]) -> str:
    """Render a compact Markdown summary from an inventory payload."""

    dataset = payload["dataset"]
    lines = [f"# {dataset} dataset inventory", "", f"- Schema: {payload['schema_version']}"]
    lines.append(f"- Resolved root: `{payload['root']}`")
    label = "Physical procedure directories" if dataset == "MMOR" else "Takes"
    count = payload["procedure_count"] if dataset == "MMOR" else payload["take_count"]
    lines.append(f"- {label}: {count}")
    totals = payload["camera_frame_totals"]
    lines.extend(
        ["", "## RGB frame totals", "", "| Camera | Frames |", "| --- | ---: |"]
    )
    lines.extend(  # type: ignore[union-attr]
        f"| {camera} | {value} |" for camera, value in totals.items()
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Counts describe extracted source frames, not the paper's reconstructed "
                "97-frame clips."
            ),
            (
                "The numeric index range is not a missing-frame test for sampled streams. "
                "Use timestamp correspondence to test required-file completeness."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


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
    for name, help_text in (
        ("mmor", "inventory an extracted MMOR dataset"),
        ("4dor", "inventory an extracted 4DOR dataset"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--output", type=Path)
        command.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = inventory_mmor(args.root) if args.dataset == "mmor" else inventory_4dor(args.root)
    write_inventory(payload, args.output)
    if args.report:
        report = args.report.expanduser()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

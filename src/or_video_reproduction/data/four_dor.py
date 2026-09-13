"""Build deterministic, explicitly hypothetical 4D-OR evaluation clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Sequence

from PIL import Image

from .clips import (
    KEYFRAME_OUTPUT_INDICES,
    OUTPUT_FPS,
    OUTPUT_FRAME_COUNT,
    SOURCE_FPS,
    SOURCE_KEYFRAME_COUNT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    validate_clip_manifest,
)
from .inventory import _resolve_4dor_root


OFFICIAL_TEST_TAKES = (2, 6)
SUPPORTED_CAMERAS = (1, 2, 3, 4, 5, 6)
DEFAULT_EVALUATION_COUNT = 6
FOUR_DOR_TO_MMOR_MAPPING = {
    "anaesthetist": "anaesthetist",
    "anesthetist": "anaesthetist",
    "anesthesiologist": "anaesthetist",
    "anesthesia_equipment": "anesthesia_equipment",
    "assistant_surgeon": "assistant_surgeon",
    "circulating_nurse": "circulator",
    "circulator": "circulator",
    "head_surgeon": "head_surgeon",
    "instrument": "instrument",
    "instrument_table": "instrument_table",
    "nurse": "nurse",
    "operating_table": "operating_table",
    "patient": "patient",
    "secondary_table": "instrument_table",
}


def _timestamp_rows(path: Path) -> list[tuple[str, dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of timestamp pairs in {path}")
    rows: list[tuple[str, dict[str, object]]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], dict):
            raise ValueError(f"Invalid 4D-OR timestamp row in {path}: {row!r}")
        rows.append((str(row[0]), row[1]))
    return rows


def build_four_dor_clip(
    root: Path,
    *,
    take: int,
    camera: int,
    start_index: int,
    split: str = "table1_ood",
    prompt_manifest: str | None = None,
) -> dict[str, object]:
    """Build one five-keyframe 4D-OR manifest without copying source data."""

    if take <= 0:
        raise ValueError("take must be positive")
    if camera not in SUPPORTED_CAMERAS:
        raise ValueError(f"camera must be one of {SUPPORTED_CAMERAS}")
    if start_index < 0:
        raise ValueError("start_index cannot be negative")
    requested_root = root.expanduser().resolve()
    resolved_root = _resolve_4dor_root(requested_root)
    root_prefix = resolved_root.relative_to(requested_root)
    directory = f"export_holistic_take{take}_processed"
    take_root = resolved_root / directory
    rows = _timestamp_rows(take_root / "timestamp_to_pcd_and_frames_list.json")
    selected = rows[start_index : start_index + SOURCE_KEYFRAME_COUNT]
    if len(selected) != SOURCE_KEYFRAME_COUNT:
        raise ValueError("Clip would extend beyond the end of the 4D-OR take")

    source_rows: list[dict[str, object]] = []
    for row_index, ((timestamp, row), output_index) in enumerate(
        zip(selected, KEYFRAME_OUTPUT_INDICES, strict=True), start=start_index
    ):
        value = row.get(f"color_{camera}")
        if value is None or not str(value).isdigit():
            raise ValueError(f"Timestamp row {row_index} has no numeric color_{camera}")
        frame_id = str(value).zfill(6)
        relative = PurePosixPath(
            *root_prefix.parts,
            directory,
            "colorimage",
            f"camera{camera:02d}_colorimage-{frame_id}.jpg",
        )
        if not (requested_root / Path(*relative.parts)).is_file():
            raise FileNotFoundError(requested_root / Path(*relative.parts))
        source_rows.append(
            {
                "timestamp_row_index": row_index,
                "timestamp": timestamp,
                "source_frame_id": frame_id,
                "output_frame_index": output_index,
                "rgb_path": relative.as_posix(),
            }
        )

    clip_id = f"4dor-take{take:02d}-camera{camera:02d}-row{start_index:06d}"
    clip: dict[str, object] = {
        "clip_id": clip_id,
        "dataset": "4DOR",
        "split": split,
        "take": take,
        "procedure_directory": directory,
        "camera": camera,
        "start_timestamp_row": start_index,
        "source_keyframes": source_rows,
    }
    if prompt_manifest is not None:
        clip["first_frame_prompt_manifest"] = prompt_manifest
    result = {
        "schema_version": 1,
        "status": "reproduction_hypothesis",
        "paper_contract": {
            "source_fps": SOURCE_FPS,
            "source_keyframes": SOURCE_KEYFRAME_COUNT,
            "output_fps": OUTPUT_FPS,
            "output_frames": OUTPUT_FRAME_COUNT,
            "duration_seconds": (OUTPUT_FRAME_COUNT - 1) / OUTPUT_FPS,
            "resolution": [TARGET_WIDTH, TARGET_HEIGHT],
            "keyframe_output_indices": list(KEYFRAME_OUTPUT_INDICES),
        },
        "clip": clip,
        "unresolved": [
            "authors' exact six clip identities",
            "authors' camera selection",
            "five-keyframe construction is inferred from 1 fps and 97 frames at 24 fps",
            "manual SAM2 prompts",
            "interpolation prompt and seed",
        ],
    }
    validate_clip_manifest(result)
    return result


def enumerate_candidates(
    root: Path, *, takes: Sequence[int], cameras: Sequence[int], stride_rows: int
) -> list[dict[str, object]]:
    if stride_rows <= 0:
        raise ValueError("stride_rows must be positive")
    candidates: list[dict[str, object]] = []
    resolved = _resolve_4dor_root(root)
    for take in sorted(set(takes)):
        path = (
            resolved
            / f"export_holistic_take{take}_processed"
            / "timestamp_to_pcd_and_frames_list.json"
        )
        rows = _timestamp_rows(path)
        for start_index in range(0, len(rows) - SOURCE_KEYFRAME_COUNT + 1, stride_rows):
            for camera in sorted(set(cameras)):
                try:
                    candidates.append(
                        build_four_dor_clip(
                            root, take=take, camera=camera, start_index=start_index
                        )
                    )
                except (FileNotFoundError, ValueError):
                    continue
    return sorted(candidates, key=lambda item: item["clip"]["clip_id"])


def deterministic_selection(
    candidates: Sequence[dict[str, object]], *, count: int, seed: int
) -> list[dict[str, object]]:
    if count <= 0 or len(candidates) < count:
        raise ValueError(f"Need {count} candidates, found {len(candidates)}")

    def rank(item: dict[str, object]) -> tuple[str, str]:
        clip_id = str(item["clip"]["clip_id"])
        return hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest(), clip_id

    return sorted(candidates, key=rank)[:count]


def _prompt_template(source_image: Path, clip_id: str) -> dict[str, object]:
    with Image.open(source_image) as image:
        width, height = image.size
    return {
        "schema_version": 1,
        "status": "manual_annotation_required",
        "dataset": "4DOR",
        "clip_id": clip_id,
        "source_image": str(source_image),
        "source_image_size": [width, height],
        "coordinate_system": "normalized_xy",
        "instances": [],
        "instructions": (
            "Add one record per visible entity with unique object_id, a source 4D-OR "
            "class, its mapped MMOR class, normalized [x,y] points, and 1/0 point labels."
        ),
        "classification": "manual_reproduction_hypothesis",
    }


def write_selection(
    output_dir: Path,
    selected: Sequence[dict[str, object]],
    *,
    dataset_root: Path,
    seed: int,
    takes: Sequence[int],
    cameras: Sequence[int],
    stride_rows: int,
    eligible_count: int,
) -> dict[str, object]:
    clips_dir = output_dir / "clips"
    prompts_dir = output_dir / "prompts"
    clips_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    requested_root = dataset_root.expanduser().resolve()
    resolved_root = _resolve_4dor_root(requested_root)
    rows = []
    for manifest in selected:
        clip_id = str(manifest["clip"]["clip_id"])
        prompt_relative = Path("..") / "prompts" / f"{clip_id}.json"
        manifest["clip"]["first_frame_prompt_manifest"] = prompt_relative.as_posix()
        first_relative = Path(*Path(manifest["clip"]["source_keyframes"][0]["rgb_path"]).parts)
        prompt = _prompt_template(requested_root / first_relative, clip_id)
        (prompts_dir / f"{clip_id}.json").write_text(
            json.dumps(prompt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        clip_relative = Path("clips") / f"{clip_id}.json"
        (output_dir / clip_relative).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append({"id": clip_id, "manifest": clip_relative.as_posix(), "split": "table1_ood"})
    batch = {
        "schema_version": 1,
        "kind": "4dor_table1_six_video_reproduction_hypothesis",
        "dataset_root_at_creation": str(requested_root),
        "resolved_4dor_root_at_creation": str(resolved_root),
        "eligible_count": eligible_count,
        "selection": {
            "takes": sorted(set(takes)),
            "cameras": sorted(set(cameras)),
            "stride_rows": stride_rows,
            "seed": seed,
            "ranking": "ascending SHA-256 of '<seed>:<clip_id>'",
            "paper_disclosed_identities": False,
        },
        "clips": rows,
    }
    (output_dir / "batch.json").write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return batch


def _csv_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--takes", type=_csv_ints, default=list(OFFICIAL_TEST_TAKES))
    parser.add_argument("--cameras", type=_csv_ints, default=[1])
    parser.add_argument("--stride-rows", type=int, default=5)
    parser.add_argument("--count", type=int, default=DEFAULT_EVALUATION_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-undisclosed-selection-hypothesis", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = enumerate_candidates(
        args.root, takes=args.takes, cameras=args.cameras, stride_rows=args.stride_rows
    )
    selected = deterministic_selection(candidates, count=args.count, seed=args.seed)
    summary: dict[str, object] = {
        "eligible": len(candidates), "selected": len(selected), "written": False
    }
    if args.accept_undisclosed_selection_hypothesis:
        write_selection(
            args.output_dir,
            selected,
            dataset_root=args.root,
            seed=args.seed,
            takes=args.takes,
            cameras=args.cameras,
            stride_rows=args.stride_rows,
            eligible_count=len(candidates),
        )
        summary.update({"written": True, "batch_manifest": str(args.output_dir / "batch.json")})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

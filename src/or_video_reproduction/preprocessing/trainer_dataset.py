"""Convert validated pairs into the pinned trainer's IC-LoRA dataset format."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path


CLIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _resolve(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {field}")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def build_trainer_dataset(
    pair_manifest: Path,
    *,
    caption: str,
    expected_train_count: int = 30,
    staging_root: Path | None = None,
) -> list[dict[str, str]]:
    payload = json.loads(pair_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
        raise ValueError("Pair manifest must have schema_version 1 and a samples list")
    train = [row for row in payload["samples"] if row.get("split") == "train"]
    heldout = [row for row in payload["samples"] if row.get("split") == "heldout"]
    if len(train) != expected_train_count:
        raise ValueError(f"Expected {expected_train_count} training pairs, found {len(train)}")
    if not heldout:
        raise ValueError("Expected at least one held-out pair")
    train_takes = {row.get("take") for row in train}
    heldout_takes = {row.get("take") for row in heldout}
    if None in train_takes or None in heldout_takes:
        raise ValueError("Every train and held-out pair must record its logical take")
    overlap = sorted(train_takes & heldout_takes)
    if overlap:
        raise ValueError(f"Training and held-out takes overlap: {overlap}")

    if staging_root is not None:
        staging_root = staging_root.resolve()

    result = []
    for row in train:
        clip_id = row.get("id")
        if not isinstance(clip_id, str) or not CLIP_ID_PATTERN.fullmatch(clip_id):
            raise ValueError(f"Unsafe or missing clip id: {clip_id!r}")
        target = _resolve(pair_manifest.parent, row.get("target_video"), "target_video")
        reference = _resolve(
            pair_manifest.parent,
            row.get("conditioning_video"),
            "conditioning_video",
        )
        if staging_root is not None:
            target = _stage_hardlink(target, staging_root / "videos" / f"{clip_id}.mp4")
            reference = _stage_hardlink(
                reference,
                staging_root / "reference_videos" / f"{clip_id}.mp4",
            )
            target_value = target.relative_to(staging_root).as_posix()
            reference_value = reference.relative_to(staging_root).as_posix()
        else:
            target_value = str(target)
            reference_value = str(reference)
        result.append(
            {
                "caption": caption,
                "media_path": target_value,
                "reference_path": reference_value,
            }
        )
    return result


def _stage_hardlink(source: Path, destination: Path) -> Path:
    """Expose media beneath the dataset root without duplicating large videos."""

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return destination
        raise FileExistsError(f"Refusing to replace staged media: {destination}")
    destination.hardlink_to(source)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--expected-train-count", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_trainer_dataset(
        args.pair_manifest,
        caption=args.caption,
        expected_train_count=args.expected_train_count,
        staging_root=args.output.parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"training_pairs": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

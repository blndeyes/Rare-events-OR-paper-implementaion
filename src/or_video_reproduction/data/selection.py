"""Build deterministic, explicitly hypothetical MMOR train/ablation clip splits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .clips import build_mmor_clip, write_manifest
from .inventory import MMOR_LOGICAL_TAKES


def enumerate_eligible_clips(
    root: Path,
    *,
    takes: Sequence[str],
    cameras: Sequence[int],
    stride_seconds: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Enumerate non-overlapping-capable windows which pass the real clip contract."""

    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if not takes:
        raise ValueError("At least one logical take is required")
    if not cameras:
        raise ValueError("At least one camera is required")

    eligible: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    for take in sorted(set(takes)):
        timestamp_path = root / "take_jsons" / f"{take}.json"
        if not timestamp_path.is_file():
            raise FileNotFoundError(timestamp_path)
        payload = json.loads(timestamp_path.read_text(encoding="utf-8"))
        timestamps = payload.get("timestamps", {})
        numeric = sorted(int(value) for value in timestamps if str(value).isdigit())
        if not numeric:
            rejected["take_without_numeric_timestamps"] += 1
            continue

        for start_timestamp in range(numeric[0], numeric[-1] + 1, stride_seconds):
            for camera in sorted(set(cameras)):
                try:
                    manifest = build_mmor_clip(
                        root,
                        take=take,
                        camera=camera,
                        start_timestamp=start_timestamp,
                        split="candidate",
                    )
                except (FileNotFoundError, ValueError) as error:
                    rejected[type(error).__name__] += 1
                    continue
                eligible.append(manifest)

    eligible.sort(key=lambda item: item["clip"]["clip_id"])
    return eligible, dict(sorted(rejected.items()))


def deterministic_split(
    eligible: Sequence[dict[str, object]],
    *,
    train_count: int,
    ablation_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Rank candidates by a stable hash and return the requested video-wise split."""

    if train_count <= 0 or ablation_count <= 0:
        raise ValueError("train_count and ablation_count must be positive")
    required = train_count + ablation_count
    if len(eligible) < required:
        raise ValueError(f"Need {required} eligible clips, found {len(eligible)}")

    def rank(manifest: dict[str, object]) -> tuple[str, str]:
        clip_id = manifest["clip"]["clip_id"]
        digest = hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()
        return digest, clip_id

    selected = sorted(eligible, key=rank)[:required]
    return selected[:train_count], selected[train_count:]


def write_hypothesis_split(
    output_dir: Path,
    train: Sequence[dict[str, object]],
    ablation: Sequence[dict[str, object]],
    *,
    dataset_root: Path,
    takes: Sequence[str],
    cameras: Sequence[int],
    stride_seconds: int,
    seed: int,
    eligible_count: int,
    rejected: dict[str, int],
) -> dict[str, object]:
    """Write individual manifests and a provenance-rich batch manifest."""

    manifests_dir = output_dir / "clips"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for split_name, manifests in (("train", train), ("ablation", ablation)):
        for manifest in manifests:
            manifest["status"] = "frozen_reproduction_hypothesis"
            manifest["clip"]["split"] = split_name
            manifest["selection_policy"] = {
                "classification": "reproduction_hypothesis",
                "reason": "The paper discloses counts and video-wise splitting only.",
                "seed": seed,
                "stride_seconds": stride_seconds,
                "source_takes": sorted(set(takes)),
                "camera_ids": sorted(set(cameras)),
                "ranking": "ascending SHA-256 of '<seed>:<clip_id>'",
            }
            clip_id = manifest["clip"]["clip_id"]
            relative = Path("clips") / f"{clip_id}.json"
            write_manifest(manifest, output_dir / relative)
            rows.append({"id": clip_id, "manifest": relative.as_posix(), "split": split_name})

    batch = {
        "schema_version": 1,
        "kind": "mmor_video_wise_reproduction_hypothesis",
        "paper_counts": {"train": len(train), "ablation": len(ablation)},
        "dataset_root_at_creation": str(dataset_root.resolve()),
        "eligible_count": eligible_count,
        "rejected_candidate_counts": rejected,
        "policy": {
            "source_takes": sorted(set(takes)),
            "camera_ids": sorted(set(cameras)),
            "stride_seconds": stride_seconds,
            "seed": seed,
            "ranking": "ascending SHA-256 of '<seed>:<clip_id>'",
            "classification": "reproduction_hypothesis_not_author_ground_truth",
        },
        "clips": rows,
    }
    (output_dir / "batch.json").write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return batch


def _csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated non-empty list")
    return result


def resolve_take_names(values: Sequence[str]) -> list[str]:
    if list(values) == ["all"]:
        return list(MMOR_LOGICAL_TAKES)
    if "all" in values:
        raise ValueError("Use --takes all by itself")
    return list(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--takes", required=True, type=_csv_strings)
    parser.add_argument("--cameras", required=True, type=_csv_strings)
    parser.add_argument("--stride-seconds", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-count", type=int, default=338)
    parser.add_argument("--ablation-count", type=int, default=50)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--accept-undisclosed-selection-hypothesis",
        action="store_true",
        help="Required acknowledgement before writing a split the paper does not disclose.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cameras = [int(value) for value in args.cameras]
    takes = resolve_take_names(args.takes)
    eligible, rejected = enumerate_eligible_clips(
        args.root,
        takes=takes,
        cameras=cameras,
        stride_seconds=args.stride_seconds,
    )
    train, ablation = deterministic_split(
        eligible,
        train_count=args.train_count,
        ablation_count=args.ablation_count,
        seed=args.seed,
    )
    summary = {
        "eligible": len(eligible),
        "selected_train": len(train),
        "selected_ablation": len(ablation),
        "rejected": rejected,
        "written": False,
    }
    if args.accept_undisclosed_selection_hypothesis:
        write_hypothesis_split(
            args.output_dir,
            train,
            ablation,
            dataset_root=args.root,
            takes=takes,
            cameras=cameras,
            stride_seconds=args.stride_seconds,
            seed=args.seed,
            eligible_count=len(eligible),
            rejected=rejected,
        )
        summary["written"] = True
        summary["output"] = str(args.output_dir / "batch.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

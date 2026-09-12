"""Build deterministic, explicitly hypothetical MMOR train/ablation clip splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

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


def deterministic_take_disjoint_split(
    eligible: Sequence[dict[str, object]],
    *,
    train_count: int,
    validation_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Select diverse clips while keeping logical takes exclusive to one split.

    The reduced experiment uses one clip per take whenever enough eligible takes
    exist.  If a split requests more clips than assigned takes, deterministic
    additional clips are drawn only from that split's already assigned takes.
    """

    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train_count and validation_count must be positive")

    by_take: dict[str, list[dict[str, object]]] = {}
    for manifest in eligible:
        try:
            take = str(manifest["clip"]["take"])
            clip_id = str(manifest["clip"]["clip_id"])
        except (KeyError, TypeError) as error:
            raise ValueError("Eligible clip is missing clip.take or clip.clip_id") from error
        if not take or not clip_id:
            raise ValueError("Eligible clip has an empty take or clip_id")
        by_take.setdefault(take, []).append(manifest)

    required_distinct_takes = validation_count + 1
    if len(by_take) < required_distinct_takes:
        raise ValueError(
            "Take-disjoint selection needs at least "
            f"{required_distinct_takes} eligible takes, found {len(by_take)}"
        )

    def digest(namespace: str, value: str) -> str:
        return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()

    ranked_takes = sorted(by_take, key=lambda take: (digest("take", take), take))
    validation_takes = set(ranked_takes[:validation_count])
    remaining_takes = ranked_takes[validation_count:]
    train_take_count = min(train_count, len(remaining_takes))
    train_takes = set(remaining_takes[:train_take_count])

    def select(assigned_takes: set[str], count: int, namespace: str) -> list[dict[str, object]]:
        candidates = [row for take in assigned_takes for row in by_take[take]]
        if len(candidates) < count:
            raise ValueError(
                f"Split {namespace} needs {count} clips but its assigned takes contain "
                f"only {len(candidates)} eligible clips"
            )
        per_take_first = [
            min(
                by_take[take],
                key=lambda row: (
                    digest(namespace, str(row["clip"]["clip_id"])),
                    str(row["clip"]["clip_id"]),
                ),
            )
            for take in assigned_takes
        ]
        chosen = sorted(
            per_take_first,
            key=lambda row: (
                digest(namespace, str(row["clip"]["clip_id"])),
                str(row["clip"]["clip_id"]),
            ),
        )[:count]
        chosen_ids = {str(row["clip"]["clip_id"]) for row in chosen}
        if len(chosen) < count:
            extras = sorted(
                (
                    row
                    for row in candidates
                    if str(row["clip"]["clip_id"]) not in chosen_ids
                ),
                key=lambda row: (
                    digest(namespace, str(row["clip"]["clip_id"])),
                    str(row["clip"]["clip_id"]),
                ),
            )
            chosen.extend(extras[: count - len(chosen)])
        return chosen

    train = select(train_takes, train_count, "train")
    validation = select(validation_takes, validation_count, "heldout")
    observed_train_takes = {str(row["clip"]["take"]) for row in train}
    observed_validation_takes = {str(row["clip"]["take"]) for row in validation}
    if observed_train_takes & observed_validation_takes:
        raise RuntimeError("Internal error: take-disjoint split contains overlapping takes")
    return train, validation


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
    validation_name: str = "ablation",
    split_mode: str = "video_wise",
) -> dict[str, object]:
    """Write individual manifests and a provenance-rich batch manifest."""

    manifests_dir = output_dir / "clips"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    ranking = (
        "takes by SHA-256 of '<seed>:take:<take>'; then clips by "
        "'<seed>:<split>:<clip_id>'"
        if split_mode == "take_disjoint"
        else "ascending SHA-256 of '<seed>:<clip_id>'"
    )
    rows: list[dict[str, object]] = []
    for split_name, manifests in (("train", train), (validation_name, ablation)):
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
                "ranking": ranking,
                "split_mode": split_mode,
            }
            clip_id = manifest["clip"]["clip_id"]
            relative = Path("clips") / f"{clip_id}.json"
            write_manifest(manifest, output_dir / relative)
            rows.append({"id": clip_id, "manifest": relative.as_posix(), "split": split_name})

    batch = {
        "schema_version": 1,
        "kind": f"mmor_{split_mode}_reproduction_hypothesis",
        "paper_counts": {"train": len(train), "ablation": len(ablation)},
        "selected_counts": {"train": len(train), validation_name: len(ablation)},
        "dataset_root_at_creation": str(dataset_root.resolve()),
        "eligible_count": eligible_count,
        "rejected_candidate_counts": rejected,
        "policy": {
            "source_takes": sorted(set(takes)),
            "camera_ids": sorted(set(cameras)),
            "stride_seconds": stride_seconds,
            "seed": seed,
            "ranking": ranking,
            "classification": "reproduction_hypothesis_not_author_ground_truth",
            "split_mode": split_mode,
            "take_overlap": sorted(
                {str(row["clip"]["take"]) for row in train}
                & {str(row["clip"]["take"]) for row in ablation}
            ),
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
    parser.add_argument(
        "--split-mode",
        choices=("video-wise", "take-disjoint"),
        default="video-wise",
        help="Use take-disjoint for held-out generalization experiments.",
    )
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
    if args.split_mode == "take-disjoint":
        train, ablation = deterministic_take_disjoint_split(
            eligible,
            train_count=args.train_count,
            validation_count=args.ablation_count,
            seed=args.seed,
        )
        validation_name = "heldout"
    else:
        train, ablation = deterministic_split(
            eligible,
            train_count=args.train_count,
            ablation_count=args.ablation_count,
            seed=args.seed,
        )
        validation_name = "ablation"
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
            validation_name=validation_name,
            split_mode=args.split_mode.replace("-", "_"),
        )
        summary["written"] = True
        summary["output"] = str(args.output_dir / "batch.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

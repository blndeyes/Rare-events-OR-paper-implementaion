"""Build a frozen evaluation manifest from inference identities, not filenames."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re

from or_video_reproduction.data.clips import OUTPUT_FPS, OUTPUT_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH
from or_video_reproduction.evaluation.video import probe_video, sha256_file

from .protocol import SCHEMA_VERSION, VIDEO_CONTRACT, WAN_DIAGNOSTIC_WARNING, write_json

ProbeFn = Callable[[Path], dict[str, int | float]]
HashFn = Callable[[Path], str]

FOUR_DOR_SPLIT_TAKES = (2, 6)
STEP_SAMPLE_NAME = re.compile(r"step_000600_(\d+)\.mp4$")
TAKE_PATTERN = re.compile(r"take0*([0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class BundleLayout:
    group_id: str
    domain: str
    comparison_class: str
    inference_manifest: str | None
    generated_dir: str | None
    reference_dir: str | None
    control_dir: str | None
    generated_path: str | None = None
    reference_path: str | None = None


PRIMARY_GROUPS = (
    BundleLayout(
        group_id="4dor_step600",
        domain="4DOR",
        comparison_class="six_video_distributional",
        inference_manifest="4dor/run/inference-manifest.json",
        generated_dir="4dor/run/samples",
        reference_dir="4dor/references",
        control_dir="4dor/controls",
    ),
    BundleLayout(
        group_id="mmor_step600",
        domain="MMOR",
        comparison_class="six_video_distributional",
        inference_manifest="mmor/run/inference-manifest.json",
        generated_dir="mmor/run/samples",
        reference_dir="mmor/references",
        control_dir="mmor/controls",
    ),
    BundleLayout(
        group_id="wan_latest_diagnostic",
        domain="4DOR",
        comparison_class="diagnostic_single_video",
        inference_manifest=None,
        generated_dir=None,
        reference_dir=None,
        control_dir=None,
        generated_path="wan-latest/4dor_export_holistic_take9_processed_flf2v.mp4",
        reference_path="wan-latest/reference.mp4",
    ),
)


def _resolve(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def inspect_video(
    path: Path,
    *,
    probe: ProbeFn = probe_video,
    hasher: HashFn = sha256_file,
) -> dict[str, object]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    record: dict[str, object] = {
        "path": str(path),
        "exists": exists,
        "size": size,
    }
    if not exists or size <= 0:
        record["ok"] = False
        record["errors"] = ["missing_or_empty"]
        return record
    try:
        metadata = probe(path)
        digest = hasher(path)
    except Exception as error:  # noqa: BLE001
        record["ok"] = False
        record["errors"] = [f"decode_or_hash_failed: {error}"]
        return record
    errors = []
    for field, expected in (
        ("width", TARGET_WIDTH),
        ("height", TARGET_HEIGHT),
        ("frames", OUTPUT_FRAME_COUNT),
    ):
        if int(metadata[field]) != expected:
            errors.append(f"{field}={metadata[field]} expected {expected}")
    if abs(float(metadata["fps"]) - float(OUTPUT_FPS)) > 1e-6:
        errors.append(f"fps={metadata['fps']} expected {OUTPUT_FPS}")
    record.update(
        {
            "ok": not errors,
            "errors": errors,
            "sha256": digest,
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "frames": int(metadata["frames"]),
            "fps": float(metadata["fps"]),
        }
    )
    return record


def _clip_take(clip_id: str) -> int | None:
    match = TAKE_PATTERN.search(clip_id)
    return int(match.group(1)) if match else None


def _load_inference_samples(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
        raise ValueError(f"{path} must have schema_version 1 and a samples list")
    samples = payload["samples"]
    if not samples:
        raise ValueError(f"{path} has no samples")
    return samples


def _pair_from_inference(
    sample: Mapping[str, object],
    index: int,
    layout: BundleLayout,
    input_root: Path,
) -> dict[str, object]:
    clip_id = sample.get("id")
    if not isinstance(clip_id, str) or not clip_id:
        raise ValueError("Inference sample is missing a clip id")
    generated_name = Path(str(sample.get("generated_video", ""))).name
    match = STEP_SAMPLE_NAME.fullmatch(generated_name)
    if match is None:
        raise ValueError(
            f"Generated video for {clip_id} is not a step-600 sample: {generated_name}"
        )
    sample_index = int(match.group(1))
    if sample_index != index:
        raise ValueError(
            f"Manifest order/index mismatch for {clip_id}: sample {index} maps to {generated_name}"
        )
    assert layout.generated_dir and layout.reference_dir and layout.control_dir
    generated = _resolve(input_root, f"{layout.generated_dir}/{generated_name}")
    reference = _resolve(input_root, f"{layout.reference_dir}/{clip_id}.mp4")
    control = _resolve(input_root, f"{layout.control_dir}/{clip_id}.mp4")
    return {
        "id": clip_id,
        "manifest_index": index,
        "generated_video": str(generated),
        "reference_video": str(reference),
        "control_video": str(control),
        "source_generated_field": sample.get("generated_video"),
        "source_reference_field": sample.get("reference_video"),
        "source_ellipse_field": sample.get("ellipse_video"),
    }


def _pair_wan(layout: BundleLayout, input_root: Path, split_clip_ids: set[str]) -> dict[str, object]:
    assert layout.generated_path and layout.reference_path
    generated = _resolve(input_root, layout.generated_path)
    reference = _resolve(input_root, layout.reference_path)
    clip_id = generated.stem
    take = _clip_take(clip_id)
    in_split = clip_id in split_clip_ids
    return {
        "id": clip_id,
        "manifest_index": 0,
        "generated_video": str(generated),
        "reference_video": str(reference),
        "control_video": None,
        "wan_take": take,
        "in_frozen_4dor_split": in_split,
        "diagnostic": True,
        "warning": WAN_DIAGNOSTIC_WARNING,
    }


def build_evaluation_manifest(
    input_root: Path,
    *,
    probe: ProbeFn = probe_video,
    hasher: HashFn = sha256_file,
) -> dict[str, object]:
    """Map each generation to its reference and ellipse control by clip identity."""

    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input bundle does not exist: {input_root}")

    four_dor_ids: list[str] = []
    groups: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = {}
    errors: list[str] = []

    for layout in PRIMARY_GROUPS:
        if layout.inference_manifest:
            manifest_path = _resolve(input_root, layout.inference_manifest)
            samples = _load_inference_samples(manifest_path)
            if layout.comparison_class == "six_video_distributional" and len(samples) != 6:
                errors.append(f"{layout.group_id} expected 6 clips, found {len(samples)}")
            pairs = [
                _pair_from_inference(sample, index, layout, input_root)
                for index, sample in enumerate(samples)
            ]
            if layout.group_id == "4dor_step600":
                four_dor_ids = [str(row["id"]) for row in pairs]
        else:
            pairs = [_pair_wan(layout, input_root, set(four_dor_ids))]

        seen_ids: set[str] = set()
        pair_records: list[dict[str, object]] = []
        for pair in pairs:
            clip_id = str(pair["id"])
            if clip_id in seen_ids:
                errors.append(f"duplicate clip identity {clip_id} in {layout.group_id}")
            seen_ids.add(clip_id)
            roles = {
                "generated": Path(str(pair["generated_video"])),
                "reference": Path(str(pair["reference_video"])),
            }
            if pair.get("control_video"):
                roles["control"] = Path(str(pair["control_video"]))
            inspected: dict[str, object] = {}
            for role, path in roles.items():
                record = inspect_video(path, probe=probe, hasher=hasher)
                inspected[role] = record
                inventory.append(
                    {
                        "group": layout.group_id,
                        "clip_id": clip_id,
                        "role": role,
                        **record,
                    }
                )
                digest = record.get("sha256")
                if isinstance(digest, str):
                    hashes.setdefault(digest, []).append(f"{layout.group_id}:{clip_id}:{role}")
                if not record["ok"]:
                    errors.append(f"{layout.group_id}:{clip_id}:{role}: {record.get('errors')}")
            generated_hash = inspected["generated"].get("sha256")
            reference_hash = inspected["reference"].get("sha256")
            if generated_hash and generated_hash == reference_hash:
                errors.append(f"{clip_id} generated and reference files are identical")
            pair_records.append({**pair, "videos": inspected})
        groups.append(
            {
                "id": layout.group_id,
                "domain": layout.domain,
                "comparison_class": layout.comparison_class,
                "sample_count": len(pair_records),
                "clip_ids": [row["id"] for row in pair_records],
                "pairs": pair_records,
                "merge_with_other_domains": False,
            }
        )

    split_takes = sorted(
        {take for clip_id in four_dor_ids if (take := _clip_take(clip_id)) is not None}
    )
    wan_group = next(group for group in groups if group["id"] == "wan_latest_diagnostic")
    wan_pair = wan_group["pairs"][0]
    wan_take = wan_pair.get("wan_take")
    wan_in_split = bool(wan_pair.get("in_frozen_4dor_split"))
    if wan_take in FOUR_DOR_SPLIT_TAKES:
        wan_note = "WAN take overlaps the 4D-OR take list; identity still depends on exact clip rows"
    else:
        wan_note = (
            f"WAN take {wan_take} is outside frozen 4D-OR takes {list(FOUR_DOR_SPLIT_TAKES)}"
        )
    if split_takes != list(FOUR_DOR_SPLIT_TAKES):
        errors.append(f"Unexpected 4D-OR takes {split_takes}; expected {list(FOUR_DOR_SPLIT_TAKES)}")

    duplicate_files = [
        {"sha256": digest, "uses": uses} for digest, uses in hashes.items() if len(uses) > 1
    ]
    # Identical hashes across unrelated generated clips in the same group are errors.
    for duplicate in duplicate_files:
        generated_uses = [use for use in duplicate["uses"] if use.endswith(":generated")]
        groups_seen = {use.split(":", 1)[0] for use in generated_uses}
        if len(generated_uses) > 1 and len(groups_seen) == 1:
            errors.append(f"duplicate generated video content: {generated_uses}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "finegrained_frozen_evaluation_manifest",
        "input_root": str(input_root),
        "video_contract": VIDEO_CONTRACT,
        "pairing_rule": (
            "Clip identities and sample order come from each inference-manifest.json. "
            "Generated files are resolved by the step_000600_N basename; references and "
            "ellipse controls are resolved as {clip_id}.mp4. Alphabetical directory order "
            "is never used for pairing."
        ),
        "groups": groups,
        "four_dor_split_clip_ids": four_dor_ids,
        "four_dor_split_takes": split_takes,
        "wan_in_frozen_4dor_split": wan_in_split,
        "wan_note": wan_note,
        "duplicate_files": duplicate_files,
        "inventory": inventory,
        "errors": errors,
        "ok": not errors,
        "do_not_compare_across_unequal_splits": True,
        "do_not_merge_mmor_and_4dor": True,
    }
    return report


def save_evaluation_manifest(report: Mapping[str, object], output_root: Path) -> Path:
    path = output_root / "evaluation-manifest.json"
    write_json(path, report)
    return path

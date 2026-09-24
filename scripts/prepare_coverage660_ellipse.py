"""Audit and stage the frozen 90-key coverage660 ellipse inference bundle.

Run on irtazapc after training, before copying the immutable input bundle directly
to Stone. This script never starts a GPU job or overwrites an output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath

REQUESTS_SHA256 = "7dff636e03e90b4ca0c058b7dbdec1bfbcc45b154e309a146f190cd1d4574582"
FREEZE_SHA256 = "33a67d0cfad53dcb163ba1118a51ddf8e944a7b67cb4e48a50ac96c049e3d939"
SPEC_SHA256 = "807e9f0fd2699b3aaa5c67c9732ba6226071671afd80a9f2375263f3c093e0fe"
CAPTION = "Fixed overhead surveillance view of an operating room."
EVENTS = {
    ("006_PKA", 2407), ("006_PKA", 2440),
    ("013_PKA", 228), ("013_PKA", 520),
    ("033_PKA", 36), ("033_PKA", 75),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_source(record: dict, hashes: dict[str, str]) -> Path:
    path = Path(record["path"])
    expected = record["sha256"]
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = hashes.get(str(path))
    if observed is None:
        observed = file_sha256(path)
        hashes[str(path)] = observed
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {observed} != {expected}")
    return path


def prepare(
    requests_path: Path, freeze_path: Path, output: Path, stone_root: PurePosixPath, *, stage: bool
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if file_sha256(requests_path) != REQUESTS_SHA256:
        raise ValueError("Wrong or changed requests_v1.json")
    if file_sha256(freeze_path) != FREEZE_SHA256:
        raise ValueError("Wrong or changed challenge/FREEZE.json")
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("spec_sha256") != SPEC_SHA256 or freeze.get("generation_started") is not False:
        raise ValueError("Challenge freeze is not the audited pre-generation freeze")
    if requests.get("schema") != "eval_coverage660_requests/v1":
        raise ValueError("Unexpected request schema")
    contract = requests["generation_contract"]
    for field, expected in {
        "frames": 97,
        "width": 1024,
        "height": 768,
        "fps": 24,
        "inference_steps": 50,
        "guidance_scale": 3.5,
        "prompt": CAPTION,
    }.items():
        if contract.get(field) != expected:
            raise ValueError(f"Unexpected generation contract {field}: {contract.get(field)}")

    hashes: dict[str, str] = {}
    media: dict[str, dict] = {}
    jobs: dict[int, list[dict]] = {42: [], 43: []}
    counts: Counter[str] = Counter()
    events: set[str] = set()
    challenge_combinations: set[tuple] = set()
    dev_clips: set[str] = set()
    keys: set[str] = set()

    def add_media(source: dict, relative: str) -> str:
        path = _checked_source(source, hashes)
        prior = media.get(relative)
        entry = {"source": str(path), "sha256": source["sha256"]}
        if prior is not None and prior != entry:
            raise ValueError(f"Two sources map to {relative}")
        media[relative] = entry
        return str(stone_root / "inputs" / relative)

    for row in requests["requests"]:
        split = row["set"]
        if split == "base":
            continue
        if split not in ("challenge", "dev"):
            raise ValueError(f"Unexpected split {split}")
        key = row["key"]
        if key in keys:
            raise ValueError(f"Duplicate request key {key}")
        keys.add(key)
        if row["take"] == "038_TKA" and row["timestamp"] == 1956:
            raise ValueError("Excluded 038_TKA@1956 entered the request set")
        name = key.replace("/", "__")
        if not re.fullmatch(r"[A-Za-z0-9@_.-]+", name):
            raise ValueError(f"Unsafe request key {key}")
        seed = row["gen_seed"]
        if seed not in jobs:
            raise ValueError(f"Unexpected seed {seed}")
        _checked_source(row["first_frame"], hashes)
        clip_id = row["clip_id"]
        if split == "challenge":
            if (row["take"], row["timestamp"]) not in EVENTS or row["camera"] not in (4, 5):
                raise ValueError(f"Unexpected challenge clip {clip_id}")
            if row["control_id"] not in ("noedit", "command", "alt"):
                raise ValueError(f"Unexpected challenge control {key}")
            control_key = f"{row['event_id']}/cam{row['camera']:02d}/{row['control_id']}"
            frozen_control = freeze["controls"].get(control_key)
            if not frozen_control or frozen_control.get("status") != "built":
                raise ValueError(f"Missing built ellipse control: {control_key}")
            control = frozen_control["ellipse_depth"]
            target = freeze["targets"][clip_id]
            events.add(row["event_id"])
            challenge_combinations.add(
                (row["event_id"], row["camera"], row["control_id"], seed)
            )
        else:
            if row["take"] != "011_TKA" or row["control_id"] != "recon" or seed != 42:
                raise ValueError(f"Unexpected development request {key}")
            control = row["control"]["ellipse_depth"]
            target = row["target"]
            dev_clips.add(clip_id)
        reference = add_media(control, f"controls/{clip_id}__{row['control_id']}.mp4")
        target_path = add_media(target, f"targets/{clip_id}.mp4")
        jobs[seed].append(
            {
                "name": name,
                "key": key,
                "reference": reference,
                "target": target_path,
                "start": 0,
                "frames": 97,
                "caption": CAPTION,
            }
        )
        counts[split] += 1

    if counts != {"challenge": 72, "dev": 18} or len(events) != 6 or len(dev_clips) != 18:
        raise ValueError(f"Wrong request coverage: {dict(counts)}, {len(events)} events")
    expected_combinations = {
        (f"ch_{take[:3]}_{timestamp}", camera, control, seed)
        for take, timestamp in EVENTS
        for camera in (4, 5)
        for control in ("noedit", "command", "alt")
        for seed in (42, 43)
    }
    if challenge_combinations != expected_combinations or len(jobs[42]) != 54 or len(jobs[43]) != 36:
        raise ValueError("Challenge combinations or per-seed counts are incomplete")
    output.mkdir(parents=True)
    if stage:
        for relative, entry in media.items():
            destination = output / "inputs" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry["source"], destination)
            if file_sha256(destination) != entry["sha256"]:
                raise ValueError(f"Staged file hash mismatch: {destination}")
    jobs_dir = output / "jobs"
    jobs_dir.mkdir()
    for seed, rows in jobs.items():
        (jobs_dir / f"g{seed}.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
    (output / "sha256sums.txt").write_text(
        "".join(f"{entry['sha256']}  inputs/{relative}\n" for relative, entry in sorted(media.items())),
        encoding="utf-8",
    )
    audit = {
        "kind": "coverage660_competitor_ellipse_step3000_preparation",
        "requests_sha256": REQUESTS_SHA256,
        "freeze_sha256": FREEZE_SHA256,
        "spec_sha256": SPEC_SHA256,
        "counts": dict(counts),
        "per_seed": {str(seed): len(rows) for seed, rows in jobs.items()},
        "events": sorted(events),
        "staged": stage,
        "stone_root": str(stone_root),
        "media": media,
    }
    (output / "input-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stone-root", required=True, type=PurePosixPath)
    parser.add_argument("--stage", action="store_true", help="Copy and rehash input videos")
    args = parser.parse_args()
    audit = prepare(args.requests, args.freeze, args.output, args.stone_root, stage=args.stage)
    print(json.dumps({key: audit[key] for key in ("counts", "per_seed", "events", "staged")}, indent=2))


if __name__ == "__main__":
    main()

"""Audit and stage the frozen 90-key testset_v1 ellipse reconstruction bundle.

Run on irtazapc. The output is immutable input data for the pinned Stone
infer_validate.py; this script neither starts generation nor overwrites files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath

CAPTION = "Fixed overhead surveillance view of an operating room."
SELECTION_SHA256 = "b01ca617b785742618a4738d6abe36c8416a7f8b62c21deff551298361484b58"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(requests_path: Path, selection_path: Path, output: Path, stone_root: PurePosixPath) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing existing bundle: {output}")
    if not str(stone_root).startswith("/scratch/irtaza/"):
        raise ValueError("Stone bundle must be under /scratch/irtaza")
    expected_manifest = requests_path.with_suffix(".sha256").read_text().split()[0]
    observed_manifest = sha256(requests_path)
    if observed_manifest != expected_manifest:
        raise ValueError("Request manifest SHA-256 mismatch")
    if sha256(selection_path) != SELECTION_SHA256:
        raise ValueError("Wrong frozen testset_v1 selection")
    manifest = json.loads(requests_path.read_text())
    if manifest.get("selection_sha256") != SELECTION_SHA256 or manifest.get("TEST_STANDIN"):
        raise ValueError("Request manifest is not the frozen testset_v1")
    if manifest.get("counts") != {"test": 90, "testswap": 20}:
        raise ValueError("Wrong reconstruction/role-swap counts")
    contract = manifest["generation_contract"]
    expected_contract = {
        "frames": 97, "width": 1024, "height": 768, "fps": 24,
        "inference_steps": 50, "guidance_scale": 3.5, "prompt": CAPTION,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"Wrong generation contract {field}: {contract.get(field)}")

    rows = [row for row in manifest["requests"] if row["set"] == "test"]
    if len(rows) != 90 or len({row["key"] for row in rows}) != 90:
        raise ValueError("Expected 90 distinct reconstruction keys")
    if any(row["set"] not in ("test", "testswap") for row in manifest["requests"]):
        raise ValueError("Unexpected request split")
    if len(manifest["requests"]) != 110:
        raise ValueError("Expected exactly 90 reconstruction and 20 role-swap requests")

    verified: dict[str, str] = {}
    media: dict[str, dict[str, str]] = {}
    jobs: list[dict] = []

    def verify(source: dict) -> Path:
        path = Path(source["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = verified.setdefault(str(path), sha256(path))
        if digest != source["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
        return path

    def stage(source: dict, relative: str) -> str:
        path = verify(source)
        entry = {"source": str(path), "sha256": source["sha256"]}
        if relative in media and media[relative] != entry:
            raise ValueError(f"Conflicting source for {relative}")
        media[relative] = entry
        return str(stone_root / "inputs" / relative)

    for row in rows:
        key = row["key"]
        name = key.replace("/", "__")
        if not re.fullmatch(r"[A-Za-z0-9@_.-]+", name):
            raise ValueError(f"Unsafe key: {key}")
        if row["control_id"] != "recon" or row["gen_seed"] != 42 or not key.endswith("/recon/g42"):
            raise ValueError(f"Not a seed-42 reconstruction request: {key}")
        if row["control"]["status"] != "ready" or row["target"]["status"] != "ready":
            raise ValueError(f"Source not ready: {key}")
        verify(row["first_frame"])
        clip_id = row["clip_id"]
        guide = stage(row["control"]["ellipse_depth"], f"controls/{clip_id}__ellipse_depth.mp4")
        target = stage(row["target"], f"targets/{clip_id}.mp4")
        jobs.append({
            "name": name, "key": key, "reference": guide, "target": target,
            "start": 0, "frames": 97, "caption": CAPTION,
        })

    output.mkdir(parents=True)
    for relative, entry in sorted(media.items()):
        destination = output / "inputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["source"], destination)
        if sha256(destination) != entry["sha256"]:
            raise ValueError(f"Staged SHA-256 mismatch: {destination}")
    jobs_dir = output / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "g42.json").write_text(json.dumps(jobs, indent=2) + "\n")
    (output / "sha256sums.txt").write_text(
        "".join(f"{entry['sha256']}  inputs/{relative}\n" for relative, entry in sorted(media.items()))
    )
    audit = {
        "kind": "coverage660_testset_v1_competitor_ellipse_s3000",
        "selection_sha256": SELECTION_SHA256,
        "requests_sha256": observed_manifest,
        "counts": {"reconstruction": len(jobs), "role_swap_excluded": 20},
        "generation_contract": expected_contract,
        "stone_root": str(stone_root),
        "staged": True,
        "keys": [job["key"] for job in jobs],
        "media": media,
    }
    (output / "input-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stone-root", type=PurePosixPath, required=True)
    args = parser.parse_args()
    audit = prepare(args.requests, args.selection, args.output, args.stone_root)
    print(json.dumps({key: audit[key] for key in ("counts", "requests_sha256", "stone_root")}, indent=2))


if __name__ == "__main__":
    main()

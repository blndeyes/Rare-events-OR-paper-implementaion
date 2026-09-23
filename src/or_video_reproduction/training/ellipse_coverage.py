"""Prepare an audited ellipse-control replacement for the coverage run.

This changes only the conditioning arm: IDs, roles, targets, caption, and VAE
recipe remain frozen.  The existing encoder is then run against the new
prepared manifest; no previously encoded mask controls are reused.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_artifact(artifact: dict) -> None:
    path = Path(artifact["path"])
    if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
        raise ValueError(f"Artifact failed size/SHA-256 audit: {path}")


def prepare_encoding(
    prepared_path: Path, original_encoded_path: Path, encoder_config_path: Path,
    encoder_script: Path, output: Path,
) -> dict:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    original_encoded = json.loads(original_encoded_path.read_text(encoding="utf-8"))
    rows = prepared["samples"]
    encoded_by_id = {row["id"]: row for row in original_encoded["samples"]}
    if prepared.get("status") != "complete" or original_encoded.get("status") != "complete":
        raise ValueError("Both source manifests must be complete")
    if len(rows) != 663 or len(encoded_by_id) != 663 or {row["id"] for row in rows} != set(encoded_by_id):
        raise ValueError("Expected the same 663 unique IDs in both manifests")
    if sum(row["role"] == "train" for row in rows) != 645 or sum(row["role"] == "development" for row in rows) != 18:
        raise ValueError("Train/development split changed")
    if original_encoded["prepared_manifest_sha256"] != sha256(prepared_path):
        raise ValueError("Original encoding is not bound to this prepared manifest")
    if original_encoded["selection_sha256"] != prepared["selection_sha256"]:
        raise ValueError("Selection differs between prepared and encoded manifests")

    converted = copy.deepcopy(prepared)
    audit = []
    for row in converted["samples"]:
        clip_id = row["id"]
        original = encoded_by_id[clip_id]
        if original["role"] != row["role"] or original["source"]["target"] != row["target"]:
            raise ValueError(f"Role or target changed for {clip_id}")
        if Path(row["control"]["path"]).name != "mask_depth.mp4":
            raise ValueError(f"Expected mask source for {clip_id}")
        verify_artifact(row["control"])
        verify_artifact(row["target"])
        ellipse = Path(row["control"]["path"]).with_name("ellipse_depth.mp4")
        provenance_path = ellipse.parent / "manifest.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        declared = provenance["outputs"]["ellipse_depth"]
        if provenance["id"] != clip_id or declared["frames"] != 97:
            raise ValueError(f"Ellipse provenance differs for {clip_id}")
        control = {"path": str(ellipse), "bytes": ellipse.stat().st_size, "sha256": sha256(ellipse)}
        if control["sha256"] != declared["sha256"]:
            raise ValueError(f"Ellipse SHA-256 differs from the frozen recipe for {clip_id}")
        row["control"] = control
        audit.append({"id": clip_id, "role": row["role"], "mask_control": original["source"]["control"],
                      "ellipse_control": control, "target": row["target"], "ellipse_provenance": str(provenance_path)})

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest_path = output / "prepared_ellipse_manifest.json"
    write_json(manifest_path, converted)
    write_json(output / "pairing_audit.json", {"kind": "mask_to_ellipse_same_target", "pairs": audit,
                                                  "original_prepared_sha256": sha256(prepared_path)})
    config = json.loads(encoder_config_path.read_text(encoding="utf-8"))
    config["root"] = str(output / "encoded")
    config["approval"] = str(output / "encoding_approval.json")
    config["prepared_manifest"] = str(manifest_path)
    canonical = original_encoded["samples"][0]["encoded"]["conditions"]
    verify_artifact(canonical)
    config["canonical_condition"] = canonical
    config_path = output / "encode_ellipse.json"
    write_json(config_path, config)
    write_json(output / "encoding_approval.json", {"approved": True, "config_sha256": sha256(config_path),
                                                    "script_sha256": sha256(encoder_script),
                                                    "review": "User-requested ellipse-control replacement; same 645/18 split and targets"})
    return {"manifest": str(manifest_path), "encoder_config": str(config_path),
            "train": 645, "development": 18, "pairs": len(audit)}


def prepare_stone_run(source_run_path: Path, encoded_path: Path, output: Path) -> dict:
    run = json.loads(source_run_path.read_text(encoding="utf-8"))
    encoded = json.loads(encoded_path.read_text(encoding="utf-8"))
    rows = encoded["samples"]
    if encoded.get("status") != "complete" or len(rows) != 663:
        raise ValueError("Ellipse encoding is not complete for 663 pairs")
    if sum(row["role"] == "train" for row in rows) != 645 or sum(row["role"] == "development" for row in rows) != 18:
        raise ValueError("Ellipse encoding split differs")
    for row in rows:
        if Path(row["source"]["control"]["path"]).name != "ellipse_depth.mp4":
            raise ValueError(f"Non-ellipse control in {row['id']}")
        for artifact in row["encoded"].values():
            verify_artifact(artifact)
    old_manifest = run["encoded_manifest"]
    matching = [record for record in run["bound_files"] if record["path"] == old_manifest]
    if len(matching) != 1:
        raise ValueError("Expected one bound encoded manifest")
    matching[0].update(path=str(encoded_path), sha256=sha256(encoded_path), bytes=encoded_path.stat().st_size)
    run["encoded_manifest"] = str(encoded_path)
    run["trainer_config"]["data"]["preprocessed_data_root"] = str(encoded_path.parent / "encoded")
    run["training_entrypoint"] = str(source_run_path.parent / "train_runner.py")
    run["control_representation"] = "ellipse_depth"
    run["experiment_contract"]["name"] = "coverage660 ellipse-depth PatchGAN, LTX-Video 13B 0.9.7 IC-LoRA"
    write_json(output, run)
    return {"run": str(output), "encoded_manifest_sha256": sha256(encoded_path), "pairs": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    pc = sub.add_parser("prepare-encoding")
    pc.add_argument("--prepared", type=Path, required=True)
    pc.add_argument("--original-encoded", type=Path, required=True)
    pc.add_argument("--encoder-config", type=Path, required=True)
    pc.add_argument("--encoder-script", type=Path, required=True)
    pc.add_argument("--output", type=Path, required=True)
    stone = sub.add_parser("prepare-stone-run")
    stone.add_argument("--source-run", type=Path, required=True)
    stone.add_argument("--encoded-manifest", type=Path, required=True)
    stone.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (prepare_encoding(args.prepared, args.original_encoded, args.encoder_config, args.encoder_script, args.output)
              if args.mode == "prepare-encoding" else prepare_stone_run(args.source_run, args.encoded_manifest, args.output))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

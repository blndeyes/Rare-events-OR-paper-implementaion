import hashlib
import json
from pathlib import Path

from or_video_reproduction.training.ellipse_coverage import prepare_encoding, prepare_stone_run


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def _artifact(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_prepare_encoding_switches_only_control_and_preserves_split(tmp_path: Path) -> None:
    rows, old_rows = [], []
    condition = tmp_path / "condition.pt"
    _write(condition, b"caption")
    for index in range(663):
        clip_id = f"clip-{index:03d}"
        controls = tmp_path / "clips" / clip_id / "controls"
        mask, ellipse = controls / "mask_depth.mp4", controls / "ellipse_depth.mp4"
        target = tmp_path / "clips" / clip_id / "target.mp4"
        _write(mask, b"mask")
        _write(ellipse, b"ellipse")
        _write(target, b"target")
        _write(controls / "manifest.json", {"id": clip_id, "outputs": {
            "ellipse_depth": {"sha256": _artifact(ellipse)["sha256"], "frames": 97}}})
        row = {"id": clip_id, "role": "train" if index < 645 else "development",
               "control": _artifact(mask), "target": _artifact(target)}
        rows.append(row)
        old_rows.append({"id": clip_id, "role": row["role"], "source": row,
                         "encoded": {"conditions": _artifact(condition)}})
    prepared_path = tmp_path / "prepared.json"
    _write(prepared_path, {"status": "complete", "selection_sha256": "selection", "samples": rows})
    original_encoded_path = tmp_path / "original_encoded.json"
    _write(original_encoded_path, {"status": "complete", "selection_sha256": "selection",
                                   "prepared_manifest_sha256": _artifact(prepared_path)["sha256"], "samples": old_rows})
    config_path, script_path = tmp_path / "encoder_config.json", tmp_path / "encoder.py"
    _write(config_path, {"root": "original", "approval": "original", "prepared_manifest": str(prepared_path)})
    _write(script_path, b"encoder")
    result = prepare_encoding(prepared_path, original_encoded_path, config_path, script_path, tmp_path / "new")
    assert result["train"] == 645 and result["development"] == 18
    new_rows = json.loads(Path(result["manifest"]).read_text())["samples"]
    assert all(Path(row["control"]["path"]).name == "ellipse_depth.mp4" for row in new_rows)
    assert [row["target"] for row in new_rows] == [row["target"] for row in rows]


def test_prepare_stone_run_rejects_mask_controls(tmp_path: Path) -> None:
    source_run = tmp_path / "run.json"
    old_manifest = tmp_path / "old.json"
    _write(source_run, {"encoded_manifest": str(old_manifest), "bound_files": [{"path": str(old_manifest)}],
                        "trainer_config": {"data": {"preprocessed_data_root": "old"}},
                        "experiment_contract": {"name": "old"}})
    encoded = tmp_path / "new.json"
    _write(encoded, {"status": "complete", "samples": [
        {"role": "train" if index < 645 else "development", "id": str(index),
         "source": {"control": {"path": "mask_depth.mp4"}}, "encoded": {}}
        for index in range(663)]})
    try:
        prepare_stone_run(source_run, encoded, tmp_path / "new_run.json")
    except ValueError as error:
        assert "Non-ellipse control" in str(error)
    else:
        raise AssertionError("Mask controls were accepted")

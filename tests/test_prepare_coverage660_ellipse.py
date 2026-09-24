import json
import unittest
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import prepare_coverage660_ellipse as bundle


def _fixture(root: Path) -> tuple[Path, Path]:
    target = root / "target.mp4"
    control = root / "control.mp4"
    photo = root / "photo.jpg"
    for path in (target, control, photo):
        path.write_bytes(path.name.encode())
    target_record = {"path": str(target), "sha256": bundle.file_sha256(target)}
    control_record = {"path": str(control), "sha256": bundle.file_sha256(control)}
    photo_record = {"path": str(photo), "sha256": bundle.file_sha256(photo)}
    rows = []
    frozen_targets = {}
    frozen_controls = {}
    for take, timestamp in sorted(bundle.EVENTS):
        event = f"ch_{take[:3]}_{timestamp}"
        for camera in (4, 5):
            clip = f"mmor-{take}-camera{camera:02d}-timestamp{timestamp:06d}"
            frozen_targets[clip] = target_record
            for control_id in ("noedit", "command", "alt"):
                frozen_controls[f"{event}/cam{camera:02d}/{control_id}"] = {
                    "status": "built", "ellipse_depth": control_record
                }
                for seed in (42, 43):
                    rows.append({
                        "key": f"ch/{take}@{timestamp:06d}/cam{camera:02d}/{control_id}/g{seed}",
                        "set": "challenge", "event_id": event, "take": take,
                        "timestamp": timestamp, "camera": camera, "clip_id": clip,
                        "control_id": control_id, "gen_seed": seed,
                        "first_frame": photo_record,
                    })
    for interval in range(6):
        for camera in (1, 4, 5):
            clip = f"mmor-011_TKA-camera{camera:02d}-timestamp{interval:06d}"
            rows.append({
                "key": f"dev/011_TKA@{interval:06d}/cam{camera:02d}/recon/g42",
                "set": "dev", "take": "011_TKA", "timestamp": interval,
                "camera": camera, "clip_id": clip, "control_id": "recon",
                "gen_seed": 42, "first_frame": photo_record,
                "control": {"ellipse_depth": control_record}, "target": target_record,
            })
    rows.append({"key": "base/skipped", "set": "base"})
    requests = root / "requests.json"
    requests.write_text(json.dumps({
        "schema": "eval_coverage660_requests/v1",
        "generation_contract": {
            "frames": 97, "width": 1024, "height": 768, "fps": 24,
            "inference_steps": 50, "guidance_scale": 3.5, "prompt": bundle.CAPTION,
        },
        "requests": rows,
    }))
    freeze = root / "FREEZE.json"
    freeze.write_text(json.dumps({
        "spec_sha256": bundle.SPEC_SHA256,
        "generation_started": False,
        "targets": frozen_targets,
        "controls": frozen_controls,
    }))
    return requests, freeze


class Coverage660PreparationTests(unittest.TestCase):
    def test_prepares_exact_90_jobs_without_base(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests, freeze = _fixture(root)
            with patch.object(bundle, "REQUESTS_SHA256", bundle.file_sha256(requests)), patch.object(
                bundle, "FREEZE_SHA256", bundle.file_sha256(freeze)
            ):
                output = root / "bundle"
                audit = bundle.prepare(requests, freeze, output, PurePosixPath("/scratch/test"), stage=True)
                self.assertEqual(audit["counts"], {"challenge": 72, "dev": 18})
                self.assertEqual(audit["per_seed"], {"42": 54, "43": 36})
                rows_42 = json.loads((output / "jobs/g42.json").read_text())
                rows_43 = json.loads((output / "jobs/g43.json").read_text())
                self.assertEqual(len(rows_42), 54)
                self.assertEqual(len(rows_43), 36)
                self.assertTrue(all("base" not in row["key"] for row in rows_42))
                self.assertTrue(all(row["reference"].startswith("/scratch/test/inputs/") for row in rows_42))
                self.assertEqual(len(list((output / "inputs/controls").glob("*.mp4"))), 54)
                with self.assertRaises(FileExistsError):
                    bundle.prepare(requests, freeze, output, PurePosixPath("/scratch/test"), stage=False)


    def test_rejects_changed_freeze(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests, freeze = _fixture(root)
            with patch.object(bundle, "REQUESTS_SHA256", bundle.file_sha256(requests)):
                with self.assertRaisesRegex(ValueError, "Wrong or changed challenge"):
                    bundle.prepare(requests, freeze, root / "bundle", PurePosixPath("/scratch/test"), stage=False)

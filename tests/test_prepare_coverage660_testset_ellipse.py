import json
import unittest
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import prepare_coverage660_testset_ellipse as bundle


def record(path: Path) -> dict:
    path.write_bytes(path.name.encode())
    return {"path": str(path), "sha256": bundle.sha256(path), "status": "ready"}


class TestsetEllipsePreparationTests(unittest.TestCase):
    def test_exact_reconstruction_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo = record(root / "frame.jpg")
            target = record(root / "target.mp4")
            control = record(root / "ellipse.mp4")
            rows = []
            for index in range(90):
                clip = f"mmor-006_PKA-camera01-timestamp{index:06d}"
                rows.append({
                    "key": f"test/006_PKA@{index:06d}/cam01/recon/g42",
                    "set": "test", "control_id": "recon", "gen_seed": 42,
                    "clip_id": clip, "first_frame": photo, "target": target,
                    "control": {"status": "ready", "ellipse_depth": control},
                })
            rows.extend({"key": f"testswap/{i}", "set": "testswap"} for i in range(20))
            selection = root / "SELECTION.json"
            selection.write_text("{}")
            requests = root / "requests.json"
            requests.write_text(json.dumps({
                "selection_sha256": bundle.sha256(selection), "TEST_STANDIN": False,
                "counts": {"test": 90, "testswap": 20},
                "generation_contract": {
                    "frames": 97, "width": 1024, "height": 768, "fps": 24,
                    "inference_steps": 50, "guidance_scale": 3.5, "prompt": bundle.CAPTION,
                },
                "requests": rows,
            }))
            requests.with_suffix(".sha256").write_text(bundle.sha256(requests) + "  requests.json\n")
            stone = PurePosixPath("/scratch/irtaza/testset-ellipse-test")
            with patch.object(bundle, "SELECTION_SHA256", bundle.sha256(selection)):
                out = root / "bundle"
                audit = bundle.prepare(requests, selection, out, stone)
                self.assertEqual(audit["counts"], {"reconstruction": 90, "role_swap_excluded": 20})
                jobs = json.loads((out / "jobs/g42.json").read_text())
                self.assertEqual(len(jobs), 90)
                self.assertTrue(all(job["key"].startswith("test/") for job in jobs))
                self.assertTrue(all(job["reference"].startswith(str(stone)) for job in jobs))
                self.assertEqual(len(list((out / "inputs/controls").glob("*.mp4"))), 90)
                with self.assertRaises(FileExistsError):
                    bundle.prepare(requests, selection, out, stone)

    def test_rejects_manifest_hash_change(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "requests.json"
            requests.write_text("{}")
            requests.with_suffix(".sha256").write_text("0" * 64)
            with self.assertRaisesRegex(ValueError, "Request manifest SHA-256 mismatch"):
                bundle.prepare(requests, root / "SELECTION.json", root / "bundle", PurePosixPath("/scratch/irtaza/test"))


if __name__ == "__main__":
    unittest.main()

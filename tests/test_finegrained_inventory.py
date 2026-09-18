import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from or_video_reproduction.evaluation.finegrained.inventory import build_evaluation_manifest
from or_video_reproduction.evaluation.finegrained.runner import build_preflight, merge_group_results
from or_video_reproduction.evaluation.finegrained.protocol import frame_sampling_manifest


def _touch(path: Path, payload: bytes = b"video") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fake_probe(_path: Path) -> dict[str, int | float]:
    return {"width": 1024, "height": 768, "frames": 97, "fps": 24.0}


class InventoryPairingTests(unittest.TestCase):
    def _bundle(self, root: Path) -> None:
        four = {
            "schema_version": 1,
            "samples": [
                {
                    "id": "4dor-take02-camera01-row000255",
                    "generated_video": "/scratch/hidden/step_000600_0.mp4",
                    "reference_video": "/scratch/hidden/ref0.mp4",
                    "ellipse_video": "/scratch/hidden/ell0.mp4",
                },
                {
                    "id": "4dor-take06-camera01-row000150",
                    "generated_video": "/scratch/hidden/step_000600_1.mp4",
                    "reference_video": "/scratch/hidden/ref1.mp4",
                    "ellipse_video": "/scratch/hidden/ell1.mp4",
                },
                {
                    "id": "4dor-take02-camera01-row000060",
                    "generated_video": "/scratch/hidden/step_000600_2.mp4",
                    "reference_video": "/scratch/hidden/ref2.mp4",
                    "ellipse_video": "/scratch/hidden/ell2.mp4",
                },
                {
                    "id": "4dor-take02-camera01-row000555",
                    "generated_video": "/scratch/hidden/step_000600_3.mp4",
                    "reference_video": "/scratch/hidden/ref3.mp4",
                    "ellipse_video": "/scratch/hidden/ell3.mp4",
                },
                {
                    "id": "4dor-take06-camera01-row000105",
                    "generated_video": "/scratch/hidden/step_000600_4.mp4",
                    "reference_video": "/scratch/hidden/ref4.mp4",
                    "ellipse_video": "/scratch/hidden/ell4.mp4",
                },
                {
                    "id": "4dor-take02-camera01-row000630",
                    "generated_video": "/scratch/hidden/step_000600_5.mp4",
                    "reference_video": "/scratch/hidden/ref5.mp4",
                    "ellipse_video": "/scratch/hidden/ell5.mp4",
                },
            ],
        }
        mmor_ids = [
            "mmor-008_PKA-camera01-timestamp000995",
            "mmor-001_PKA-camera01-timestamp000210",
            "mmor-037_TKA-camera01-timestamp003020",
            "mmor-022_PKA-camera01-timestamp000100",
            "mmor-026_PKA-camera01-timestamp000040",
            "mmor-019_PKA-camera01-timestamp000050",
        ]
        mmor = {
            "schema_version": 1,
            "samples": [
                {
                    "id": clip_id,
                    "generated_video": f"/scratch/hidden/step_000600_{index}.mp4",
                    "reference_video": f"/scratch/hidden/{clip_id}.mp4",
                    "ellipse_video": f"/scratch/hidden/{clip_id}-c.mp4",
                }
                for index, clip_id in enumerate(mmor_ids)
            ],
        }
        (root / "4dor/run/inference-manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (root / "mmor/run/inference-manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (root / "4dor/run/inference-manifest.json").write_text(json.dumps(four), encoding="utf-8")
        (root / "mmor/run/inference-manifest.json").write_text(json.dumps(mmor), encoding="utf-8")
        for index, sample in enumerate(four["samples"]):
            _touch(root / "4dor/run/samples" / f"step_000600_{index}.mp4", f"gen4-{index}".encode())
            _touch(root / "4dor/references" / f"{sample['id']}.mp4", f"ref4-{index}".encode())
            _touch(root / "4dor/controls" / f"{sample['id']}.mp4", f"ctl4-{index}".encode())
        for index, clip_id in enumerate(mmor_ids):
            _touch(root / "mmor/run/samples" / f"step_000600_{index}.mp4", f"genm-{index}".encode())
            _touch(root / "mmor/references" / f"{clip_id}.mp4", f"refm-{index}".encode())
            _touch(root / "mmor/controls" / f"{clip_id}.mp4", f"ctlm-{index}".encode())
        _touch(root / "wan-latest/4dor_export_holistic_take9_processed_flf2v.mp4", b"wan-gen")
        _touch(root / "wan-latest/reference.mp4", b"wan-ref")

    def test_pairs_by_manifest_identity_not_alphabetical_references(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            hashes = {}

            def hasher(path: Path) -> str:
                digest = path.read_bytes().hex()
                hashes[str(path)] = digest
                return digest

            report = build_evaluation_manifest(root, probe=_fake_probe, hasher=hasher)
            four = next(group for group in report["groups"] if group["id"] == "4dor_step600")
            self.assertEqual(
                four["clip_ids"],
                [
                    "4dor-take02-camera01-row000255",
                    "4dor-take06-camera01-row000150",
                    "4dor-take02-camera01-row000060",
                    "4dor-take02-camera01-row000555",
                    "4dor-take06-camera01-row000105",
                    "4dor-take02-camera01-row000630",
                ],
            )
            self.assertTrue(four["clip_ids"] != sorted(four["clip_ids"]))
            first = four["pairs"][0]
            self.assertTrue(first["generated_video"].endswith("step_000600_0.mp4"))
            self.assertTrue(first["reference_video"].endswith("4dor-take02-camera01-row000255.mp4"))
            self.assertTrue(first["control_video"].endswith("4dor-take02-camera01-row000255.mp4"))
            wan = next(group for group in report["groups"] if group["id"] == "wan_latest_diagnostic")
            self.assertEqual(wan["comparison_class"], "diagnostic_single_video")
            self.assertFalse(report["wan_in_frozen_4dor_split"])
            self.assertTrue(report["ok"])
            self.assertTrue(report["do_not_merge_mmor_and_4dor"])

    def test_duplicate_clip_identity_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            payload = json.loads((root / "4dor/run/inference-manifest.json").read_text(encoding="utf-8"))
            payload["samples"][1]["id"] = payload["samples"][0]["id"]
            (root / "4dor/run/inference-manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            report = build_evaluation_manifest(root, probe=_fake_probe, hasher=lambda path: path.name)
            self.assertFalse(report["ok"])
            self.assertTrue(any("duplicate clip identity" in error for error in report["errors"]))

    def test_preflight_marks_wan_and_rank_deficient_frechet(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            manifest = build_evaluation_manifest(
                root, probe=_fake_probe, hasher=lambda path: path.read_bytes().hex()
            )
            preflight = build_preflight(
                manifest,
                frame_sampling_manifest(),
                output_root=root / "out",
                vbench_root=None,
                hf_cache=None,
                torch_cache=None,
            )
            self.assertTrue(any("diagnostic" in item.lower() for item in preflight["statistically_invalid_or_weak"]))
            self.assertTrue(any("rank-deficient" in item for item in preflight["statistically_invalid_or_weak"]))
            self.assertFalse(preflight["wan_in_frozen_4dor_split"])


class ResultMergeTests(unittest.TestCase):
    def test_later_stage_keeps_earlier_metrics(self) -> None:
        previous = {
            "group_id": "4dor_step600",
            "metrics": {"clip_cmmd": {"status": "ok", "clip_cmmd_unbiased_rbf_x1000": 1.0}},
        }
        current = {
            "group_id": "4dor_step600",
            "metrics": {"dinov2": {"status": "ok", "frechet": 2.0}},
        }
        merged = merge_group_results(previous, current)
        self.assertEqual(merged["metrics"]["clip_cmmd"]["clip_cmmd_unbiased_rbf_x1000"], 1.0)
        self.assertEqual(merged["metrics"]["dinov2"]["frechet"], 2.0)


if __name__ == "__main__":
    unittest.main()

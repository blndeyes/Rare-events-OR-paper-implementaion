import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from or_video_reproduction.evaluation.trajectory_experiment import prepare_control_pair


class TrajectoryExperimentTests(unittest.TestCase):
    def test_prepares_two_rows_with_only_conditioning_changed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("target.mp4", "original.mp4", "edited.mp4", "metadata.json", "labels.npz"):
                (root / name).write_bytes(name.encode())
            source = root / "pairs.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "id": "clip",
                                "take": "take1",
                                "target_video": "target.mp4",
                                "conditioning_video": "original.mp4",
                                "geometry_metadata": "metadata.json",
                                "labels": "labels.npz",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            path, payload = prepare_control_pair(
                source_pair_manifest=source,
                clip_id="clip",
                edited_conditioning=root / "edited.mp4",
                output_dir=root / "output",
            )

            original, edited = payload["samples"]
            self.assertTrue(path.is_file())
            self.assertEqual(original["target_video"], edited["target_video"])
            self.assertNotEqual(original["conditioning_video"], edited["conditioning_video"])
            self.assertEqual(Path(original["conditioning_video"]).read_bytes(), b"original.mp4")
            self.assertEqual(Path(edited["conditioning_video"]).read_bytes(), b"edited.mp4")
            self.assertEqual(payload["controlled_variable"], "conditioning_video only")
            self.assertEqual(len(payload["samples"]), 2)
            self.assertFalse(payload["edited_only"])

    def test_edited_only_prepares_one_inference_row(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("target.mp4", "original.mp4", "edited.mp4", "metadata.json"):
                (root / name).write_bytes(name.encode())
            source = root / "pairs.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "id": "4dor-take02-camera01-row000255",
                                "split": "table1_ood",
                                "take": "take2",
                                "target_video": "target.mp4",
                                "conditioning_video": "original.mp4",
                                "geometry_metadata": "metadata.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            path, payload = prepare_control_pair(
                source_pair_manifest=source,
                clip_id="4dor-take02-camera01-row000255",
                edited_conditioning=root / "edited.mp4",
                output_dir=root / "output",
                edited_only=True,
            )

            self.assertTrue(path.is_file())
            self.assertTrue(payload["edited_only"])
            self.assertEqual(len(payload["samples"]), 1)
            self.assertEqual(payload["samples"][0]["id"], "4dor-take02-camera01-row000255-edited")
            self.assertEqual(payload["samples"][0]["split"], "table1_ood")
            self.assertIsNone(payload["source_labels"])
            self.assertTrue((root / "output" / "conditioning" / "original.mp4").is_file())

    def test_legacy_pair_mode_still_requires_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "edited.mp4").write_bytes(b"edited")
            source = root / "pairs.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "id": "clip",
                                "target_video": "target.mp4",
                                "conditioning_video": "original.mp4",
                                "geometry_metadata": "metadata.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "labels"):
                prepare_control_pair(
                    source_pair_manifest=source,
                    clip_id="clip",
                    edited_conditioning=root / "edited.mp4",
                    output_dir=root / "output",
                )

    def test_rejects_missing_or_duplicate_clip(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            edited = root / "edited.mp4"
            edited.write_bytes(b"edited")
            source = root / "pairs.json"
            source.write_text(json.dumps({"schema_version": 1, "samples": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly once"):
                prepare_control_pair(
                    source_pair_manifest=source,
                    clip_id="missing",
                    edited_conditioning=edited,
                    output_dir=root / "output",
                )


if __name__ == "__main__":
    unittest.main()

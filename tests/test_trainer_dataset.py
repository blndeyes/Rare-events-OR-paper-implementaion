import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from or_video_reproduction.preprocessing.trainer_dataset import build_trainer_dataset


class TrainerDatasetTests(unittest.TestCase):
    def test_emits_train_only_and_rejects_take_leakage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pairs.json"
            samples = [
                {
                    "id": "train",
                    "split": "train",
                    "take": "take-a",
                    "target_video": "target.mp4",
                    "conditioning_video": "ellipse.mp4",
                },
                {
                    "id": "heldout",
                    "split": "heldout",
                    "take": "take-b",
                    "target_video": "heldout.mp4",
                    "conditioning_video": "heldout-ellipse.mp4",
                },
            ]
            manifest.write_text(
                json.dumps({"schema_version": 1, "samples": samples}), encoding="utf-8"
            )
            result = build_trainer_dataset(
                manifest, caption="fixed caption", expected_train_count=1
            )
            samples[1]["take"] = "take-a"
            manifest.write_text(
                json.dumps({"schema_version": 1, "samples": samples}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                build_trainer_dataset(manifest, caption="fixed", expected_train_count=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["caption"], "fixed caption")
        self.assertEqual(result[0]["media_path"], str(root / "target.mp4"))
        self.assertEqual(result[0]["reference_path"], str(root / "ellipse.mp4"))

    def test_stages_relative_hardlinks_for_official_preprocessor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            target = source / "target.mp4"
            reference = source / "ellipse.mp4"
            target.write_bytes(b"target")
            reference.write_bytes(b"reference")
            manifest = source / "pairs.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "id": "clip-001",
                                "split": "train",
                                "take": "take-a",
                                "target_video": str(target),
                                "conditioning_video": str(reference),
                            },
                            {
                                "id": "heldout",
                                "split": "heldout",
                                "take": "take-b",
                                "target_video": "unused.mp4",
                                "conditioning_video": "unused-reference.mp4",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            staging = root / "trainer-data"

            rows = build_trainer_dataset(
                manifest,
                caption="fixed caption",
                expected_train_count=1,
                staging_root=staging,
            )

            staged_target = staging / rows[0]["media_path"]
            staged_reference = staging / rows[0]["reference_path"]
            self.assertEqual(rows[0]["media_path"], "videos/clip-001.mp4")
            self.assertEqual(
                rows[0]["reference_path"], "reference_videos/clip-001.mp4"
            )
            self.assertTrue(os.path.samefile(target, staged_target))
            self.assertTrue(os.path.samefile(reference, staged_reference))


if __name__ == "__main__":
    unittest.main()

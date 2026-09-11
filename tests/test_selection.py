import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from or_video_reproduction.data.selection import (
    deterministic_split,
    enumerate_eligible_clips,
    write_hypothesis_split,
)


class SelectionTests(unittest.TestCase):
    def _dataset(self, root: Path) -> None:
        timestamps = {
            str(index): {"azure": f"{100 + index:06d}"} for index in range(20)
        }
        (root / "take_jsons").mkdir()
        (root / "take_jsons/001_PKA.json").write_text(
            json.dumps({"timestamps": timestamps}), encoding="utf-8"
        )
        for camera in (1, 4, 5):
            (root / f"001_PKA/segmentation_export_{camera}").mkdir(parents=True)
            for index in range(20):
                (root / "001_PKA/colorimage").mkdir(parents=True, exist_ok=True)
                filename = f"camera{camera:02d}_colorimage-{100 + index:06d}.jpg"
                (root / "001_PKA/colorimage" / filename).touch()
            for index in (0, 5, 10, 15):
                filename = f"camera{camera:02d}_colorimage-{100 + index:06d}.png"
                (root / f"001_PKA/segmentation_export_{camera}" / filename).touch()

    def test_enumerates_real_windows_and_splits_deterministically(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            eligible, rejected = enumerate_eligible_clips(
                root, takes=["001_PKA"], cameras=[1, 4, 5], stride_seconds=5
            )
            first = deterministic_split(eligible, train_count=5, ablation_count=2, seed=42)
            second = deterministic_split(eligible, train_count=5, ablation_count=2, seed=42)

        self.assertEqual(len(eligible), 12)
        self.assertEqual(rejected, {})
        self.assertEqual(
            [[item["clip"]["clip_id"] for item in split] for split in first],
            [[item["clip"]["clip_id"] for item in split] for split in second],
        )

    def test_written_split_is_explicitly_a_hypothesis(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            self._dataset(root)
            eligible, rejected = enumerate_eligible_clips(
                root, takes=["001_PKA"], cameras=[1], stride_seconds=5
            )
            train, ablation = deterministic_split(
                eligible, train_count=2, ablation_count=1, seed=7
            )
            output = Path(directory) / "output"
            batch = write_hypothesis_split(
                output,
                train,
                ablation,
                dataset_root=root,
                takes=["001_PKA"],
                cameras=[1],
                stride_seconds=5,
                seed=7,
                eligible_count=len(eligible),
                rejected=rejected,
            )
            clip = json.loads((output / batch["clips"][0]["manifest"]).read_text())

        self.assertEqual(
            batch["policy"]["classification"],
            "reproduction_hypothesis_not_author_ground_truth",
        )
        self.assertEqual(clip["status"], "frozen_reproduction_hypothesis")

    def test_rejects_insufficient_pool(self) -> None:
        with self.assertRaisesRegex(ValueError, "Need 388 eligible clips"):
            deterministic_split([], train_count=338, ablation_count=50, seed=42)


if __name__ == "__main__":
    unittest.main()

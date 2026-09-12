import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from or_video_reproduction.data.selection import (
    deterministic_split,
    deterministic_take_disjoint_split,
    enumerate_eligible_clips,
    resolve_take_names,
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

    def test_take_disjoint_split_is_diverse_and_deterministic(self) -> None:
        eligible = []
        for take_index in range(8):
            for clip_index in range(3):
                take = f"take-{take_index:02d}"
                eligible.append(
                    {
                        "clip": {
                            "take": take,
                            "clip_id": f"{take}-clip-{clip_index:02d}",
                        }
                    }
                )

        first = deterministic_take_disjoint_split(
            eligible, train_count=5, validation_count=2, seed=42
        )
        second = deterministic_take_disjoint_split(
            eligible, train_count=5, validation_count=2, seed=42
        )
        train, validation = first
        train_takes = {row["clip"]["take"] for row in train}
        validation_takes = {row["clip"]["take"] for row in validation}

        self.assertEqual(first, second)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(train_takes), 5)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(validation_takes), 2)
        self.assertFalse(train_takes & validation_takes)

    def test_take_disjoint_split_fails_without_enough_takes(self) -> None:
        eligible = [
            {"clip": {"take": "only", "clip_id": f"clip-{index}"}}
            for index in range(20)
        ]
        with self.assertRaisesRegex(ValueError, "eligible takes"):
            deterministic_take_disjoint_split(
                eligible, train_count=5, validation_count=1, seed=42
            )

    def test_all_take_alias_is_explicit_and_not_mixable(self) -> None:
        self.assertGreater(len(resolve_take_names(["all"])), 30)
        with self.assertRaisesRegex(ValueError, "by itself"):
            resolve_take_names(["all", "001_PKA"])


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from or_video_reproduction.preprocessing.validate_pairs import (
    EXPECTED_SHAPE,
    PairJob,
    load_pair_jobs,
    validate_geometry_metadata,
    validate_label_and_depth_arrays,
    validate_pair,
)
from or_video_reproduction.geometry.palette import PAPER_36_PALETTE
import or_video_reproduction.preprocessing.validate_pairs as pair_validation


class PairValidationTests(unittest.TestCase):
    def test_rejects_nonfinite_or_misaligned_arrays(self) -> None:
        labels = np.zeros((1, 2, 3), dtype=np.uint8)
        depths = np.zeros((1, 2, 3), dtype=np.float32)
        depths[0, 0, 0] = np.nan
        result = validate_label_and_depth_arrays(labels, depths)
        self.assertFalse(result["passed"])
        self.assertTrue(any("expected" in error for error in result["errors"]))
        self.assertTrue(any("NaN" in error for error in result["errors"]))

    def test_rejects_predicate_as_ellipse(self) -> None:
        frames = [{"instances": [], "frame_index": index} for index in range(97)]
        frames[0]["instances"] = [
            {"class_name": "holding", "red_green": [1, 2]}
        ]
        result = validate_geometry_metadata(
            {
                "kind": "paper_ellipse_only_geometric_conditioning",
                "shape": list(EXPECTED_SHAPE),
                "fps": 24,
                "background": "black",
                "frames": frames,
            }
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("non-entity" in error for error in result["errors"]))

    def test_loads_relative_pair_paths_and_rejects_duplicates(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            sample = {
                "id": "clip-1",
                "target_video": "target.mp4",
                "conditioning_video": "conditioning.mp4",
                "labels": "labels.npz",
                "depths": "depths.npz",
                "geometry_metadata": "metadata.json",
            }
            path.write_text(
                json.dumps({"schema_version": 1, "samples": [sample]}), encoding="utf-8"
            )
            jobs = load_pair_jobs(path)
            self.assertEqual(jobs[0].target_video, Path(directory) / "target.mp4")
            path.write_text(
                json.dumps({"schema_version": 1, "samples": [sample, sample]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_pair_jobs(path)

    def test_valid_pair_passes_with_video_probe(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            small_shape = (2, 8, 8)
            labels = np.zeros(small_shape, dtype=np.uint8)
            labels[:, 2:6, 2:6] = 1
            depths = np.ones(small_shape, dtype=np.float32)
            np.savez_compressed(root / "labels.npz", labels=labels)
            np.savez_compressed(root / "depths.npz", depths=depths)
            frames = []
            for index in range(97):
                frames.append(
                    {
                        "frame_index": index,
                        "instances": [
                            {
                                "class_name": "instrument_table",
                                "red_green": list(PAPER_36_PALETTE["instrument_table"]),
                            }
                        ],
                    }
                )
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "kind": "paper_ellipse_only_geometric_conditioning",
                        "shape": list(small_shape),
                        "fps": 24,
                        "background": "black",
                        "frames": frames,
                    }
                ),
                encoding="utf-8",
            )
            job = PairJob(
                "clip",
                root / "target.mp4",
                root / "conditioning.mp4",
                root / "labels.npz",
                root / "depths.npz",
                root / "metadata.json",
            )
            expected = {"width": 1024, "height": 768, "frames": 97, "fps": 24}
            with patch.object(pair_validation, "EXPECTED_SHAPE", small_shape):
                result = validate_pair(job, video_probe=lambda _: expected)
        self.assertEqual(result["state"], "passed")


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from or_video_reproduction.preprocessing.geometry_batch import (
    GeometryJob,
    run_geometry_batch,
)
import or_video_reproduction.preprocessing.geometry_batch as geometry_batch


class FakeRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *args) -> None:
        self.calls += 1


class GeometryBatchTests(unittest.TestCase):
    def test_reuses_models_and_records_completed_pairs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                GeometryJob(
                    f"clip-{index}",
                    root / f"manifest-{index}.json",
                    root / f"target-{index}.mp4",
                    root / f"mask-{index}.png",
                    "train",
                )
                for index in range(2)
            ]
            vda = FakeRunner()
            sam2 = FakeRunner()
            failed_validation = {"state": "failed"}

            def validation(pair):
                if pair.conditioning_video.is_file():
                    return {
                        "state": "passed",
                        "paths": {
                            "target_video": str(pair.target_video),
                            "conditioning_video": str(pair.conditioning_video),
                            "labels": str(pair.labels),
                            "depths": str(pair.depths),
                            "geometry_metadata": str(pair.geometry_metadata),
                        },
                    }
                return failed_validation

            def render(_labels, _depths, output):
                output.mkdir(parents=True)
                (output / "conditioning.mp4").touch()
                (output / "metadata.json").write_text("{}", encoding="utf-8")

            with (
                patch.object(geometry_batch, "depth_is_valid", return_value=False),
                patch.object(geometry_batch, "labels_are_valid", return_value=False),
                patch.object(geometry_batch, "render_sequence", side_effect=render),
                patch.object(geometry_batch, "validate_pair", side_effect=validation),
            ):
                report = run_geometry_batch(
                    jobs,
                    output_root=root / "output",
                    vda_factory=lambda: vda,
                    sam2_factory=lambda: sam2,
                )

            saved = json.loads((root / "output/pair-manifest.json").read_text())

        self.assertEqual(vda.calls, 2)
        self.assertEqual(sam2.calls, 2)
        self.assertEqual(report["counts"], {"completed": 2, "skipped": 0, "failed": 0})
        self.assertEqual(len(saved["samples"]), 2)

    def test_records_failure_and_continues(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                GeometryJob("broken", root / "m", root / "v", root / "mask", "train"),
                GeometryJob("next", root / "m2", root / "v2", root / "mask2", "train"),
            ]
            runner = FakeRunner()
            with (
                patch.object(geometry_batch, "validate_pair", return_value={"state": "failed"}),
                patch.object(geometry_batch, "depth_is_valid", return_value=False),
                patch.object(runner, "run", side_effect=RuntimeError("test failure")),
            ):
                report = run_geometry_batch(
                    jobs,
                    output_root=root / "output",
                    vda_factory=lambda: runner,
                    sam2_factory=lambda: runner,
                )

            status = json.loads((root / "output/broken/status.json").read_text())

        self.assertEqual(report["counts"]["failed"], 2)
        self.assertEqual(status["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()


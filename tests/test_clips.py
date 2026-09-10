import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from or_video_reproduction.data.clips import (
    KEYFRAME_OUTPUT_INDICES,
    build_mmor_clip,
    validate_clip_manifest,
)


class MmorClipTests(unittest.TestCase):
    def _dataset(self, root: Path) -> None:
        timestamps = {
            str(index): {"azure": f"{329 + index:06d}", "original_timestamp": 329 + index}
            for index in range(7)
        }
        take_jsons = root / "take_jsons"
        take_jsons.mkdir(parents=True)
        (take_jsons / "001_PKA.json").write_text(
            json.dumps({"folder": "001_PKA", "timestamps": timestamps}), encoding="utf-8"
        )
        procedure = root / "001_PKA"
        (procedure / "colorimage").mkdir(parents=True)
        (procedure / "segmentation_export_1").mkdir()
        for index in range(7):
            (procedure / "colorimage" / f"camera01_colorimage-{329 + index:06d}.jpg").touch()
        (procedure / "segmentation_export_1" / "camera01_colorimage-000329.png").touch()

    def test_builds_exact_five_to_97_frame_contract(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)

            payload = build_mmor_clip(
                root, take="001_PKA", camera=1, start_timestamp=0, split="smoke"
            )
            validate_clip_manifest(payload)

            self.assertEqual(payload["paper_contract"]["output_frames"], 97)
            self.assertEqual(payload["paper_contract"]["output_fps"], 24)
            self.assertEqual(payload["paper_contract"]["duration_seconds"], 4.0)
            self.assertEqual(
                [row["output_frame_index"] for row in payload["clip"]["source_keyframes"]],
                list(KEYFRAME_OUTPUT_INDICES),
            )

    def test_rejects_camera_without_ground_truth_exports(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            with self.assertRaisesRegex(ValueError, "no official MMOR panoptic export"):
                build_mmor_clip(root, take="001_PKA", camera=2, start_timestamp=0)

    def test_requires_first_frame_ground_truth_mask(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            (root / "001_PKA/segmentation_export_1/camera01_colorimage-000329.png").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "ground-truth mask"):
                build_mmor_clip(root, take="001_PKA", camera=1, start_timestamp=0)


if __name__ == "__main__":
    unittest.main()

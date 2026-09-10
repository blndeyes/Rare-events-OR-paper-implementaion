from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from or_video_reproduction.data.inventory import inventory_mmor


def touch_many(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).touch()


class MmorInventoryTests(unittest.TestCase):
    def test_groups_cameras_and_detects_gaps(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            procedure = root / "007_TKA"
            touch_many(
                procedure / "colorimage",
                [
                    "camera01_colorimage-000000.jpg",
                    "camera01_colorimage-000002.jpg",
                    "camera02_colorimage-000010.png",
                    "unrelated.txt",
                ],
            )
            touch_many(
                procedure / "panoptic_seg_1",
                ["camera01_colorimage-000000.png", "camera01_colorimage-000002.png"],
            )

            inventory = inventory_mmor(root)

            self.assertEqual(inventory["procedure_count"], 1)
            self.assertEqual(inventory["camera_frame_totals"], {"01": 2, "02": 1})
            cameras = inventory["procedures"][0]["color_frames"]
            self.assertEqual(
                cameras[0],
                {
                    "camera": "01",
                    "count": 2,
                    "minimum_frame": 0,
                    "maximum_frame": 2,
                    "missing_inside_range": 1,
                },
            )
            self.assertEqual(
                inventory["procedures"][0]["annotation_file_counts"]["panoptic_seg_1"], 2
            )

    def test_ignores_nonprocedure_directories(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "take_jsons").mkdir()
            (root / "001_PKA").mkdir()

            inventory = inventory_mmor(root)

            self.assertEqual(
                [row["procedure"] for row in inventory["procedures"]], ["001_PKA"]
            )

    def test_rejects_missing_root(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "MMOR root is not a directory"):
                inventory_mmor(Path(directory) / "missing")


if __name__ == "__main__":
    unittest.main()

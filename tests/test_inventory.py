from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import json

from or_video_reproduction.data.inventory import inventory_4dor, inventory_mmor


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
            take_jsons = root / "take_jsons"
            take_jsons.mkdir()
            (take_jsons / "007_TKA.json").write_text(
                json.dumps({"timestamps": {"0": {"azure": 0}, "1": {"azure": 1}}}),
                encoding="utf-8",
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
            procedure_row = inventory["procedures"][0]
            self.assertEqual(
                procedure_row["modalities"]["colorimage"]["extensions"],
                {"jpg": 2, "png": 1},
            )
            self.assertEqual(
                procedure_row["correspondence"]["panoptic_seg_1:camera01"]["coverage"],
                1.0,
            )
            self.assertEqual(procedure_row["logical_takes"], ["007_TKA"])
            self.assertEqual(
                procedure_row["timestamp_correspondence"]["01"]["coverage"], 0.5
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


class FourDorInventoryTests(unittest.TestCase):
    def test_resolves_nested_root_and_checks_timestamp_correspondence(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            take = root / "4D-OR" / "export_holistic_take1_processed"
            touch_many(
                take / "colorimage",
                [
                    "camera01_colorimage-000010.jpg",
                    "camera01_colorimage-000011.jpg",
                    "camera02_colorimage-000020.jpg",
                ],
            )
            touch_many(
                take / "depthimage",
                [
                    "camera01_depthimage-000030.tiff",
                    "camera02_depthimage-000040.tiff",
                ],
            )
            touch_many(take / "pcds", ["000000.pcd", "000001.pcd"])
            touch_many(take / "annotations", ["000000.json"])
            timestamps = [
                [
                    "1.0",
                    {
                        "color_1": "000010",
                        "depth_1": "000030",
                        "color_2": "000020",
                        "depth_2": "000040",
                        "pcd": "000000",
                    },
                ],
                [
                    "2.0",
                    {
                        "color_1": "000099",
                        "depth_1": "000098",
                        "pcd": "000001",
                    },
                ],
            ]
            (take / "timestamp_to_pcd_and_frames_list.json").write_text(
                json.dumps(timestamps), encoding="utf-8"
            )

            inventory = inventory_4dor(root)

            self.assertEqual(inventory["take_count"], 1)
            self.assertEqual(inventory["camera_frame_totals"], {"01": 2, "02": 1})
            take_row = inventory["takes"][0]
            self.assertEqual(take_row["pcd_count"], 2)
            self.assertEqual(take_row["annotation_count"], 1)
            self.assertEqual(
                take_row["timestamp_correspondence"]["01"]["color"]["matched"], 1
            )
            self.assertEqual(
                take_row["timestamp_correspondence"]["01"]["color"][
                    "missing_from_candidate"
                ],
                1,
            )
            self.assertEqual(
                take_row["timestamp_correspondence"]["01"]["color"][
                    "duplicate_timestamp_references"
                ],
                0,
            )

    def test_rejects_root_without_takes(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No export_holistic"):
                inventory_4dor(Path(directory))


if __name__ == "__main__":
    unittest.main()

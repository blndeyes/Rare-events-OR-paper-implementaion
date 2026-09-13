import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from or_video_reproduction.data.four_dor import (
    build_four_dor_clip,
    deterministic_selection,
    enumerate_candidates,
    write_selection,
)
from or_video_reproduction.preprocessing.sam2_propagation import (
    load_point_prompt_manifest,
    point_prompt_output_is_current,
    point_prompt_sha256,
)


def build_take(root: Path, *, row_count: int = 10) -> Path:
    take = root / "export_holistic_take2_processed"
    images = take / "colorimage"
    images.mkdir(parents=True)
    rows = []
    for index in range(row_count):
        frame = f"{index + 100:06d}"
        Image.new("RGB", (20, 12), (index, 0, 0)).save(
            images / f"camera01_colorimage-{frame}.jpg"
        )
        rows.append([str(index), {"color_1": frame}])
    (take / "timestamp_to_pcd_and_frames_list.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    return take


class FourDorClipTests(unittest.TestCase):
    def test_builds_five_keyframes_at_paper_output_indices(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_take(root)

            manifest = build_four_dor_clip(root, take=2, camera=1, start_index=2)

        self.assertEqual(manifest["clip"]["dataset"], "4DOR")
        self.assertEqual(
            [row["output_frame_index"] for row in manifest["clip"]["source_keyframes"]],
            [0, 24, 48, 72, 96],
        )
        self.assertEqual(manifest["clip"]["source_keyframes"][0]["source_frame_id"], "000102")

    def test_selection_is_deterministic_and_writes_prompt_templates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_take(root, row_count=15)
            candidates = enumerate_candidates(root, takes=[2], cameras=[1], stride_rows=1)
            first = deterministic_selection(candidates, count=2, seed=42)
            second = deterministic_selection(candidates, count=2, seed=42)
            output = root / "selection"

            batch = write_selection(
                output,
                first,
                dataset_root=root,
                seed=42,
                takes=[2],
                cameras=[1],
                stride_rows=1,
                eligible_count=len(candidates),
            )

            prompt = json.loads(
                (output / "prompts" / f"{batch['clips'][0]['id']}.json").read_text()
            )
        self.assertEqual(
            [item["clip"]["clip_id"] for item in first],
            [item["clip"]["clip_id"] for item in second],
        )
        self.assertEqual(prompt["source_image_size"], [20, 12])
        self.assertEqual(prompt["instances"], [])


class FourDorPromptTests(unittest.TestCase):
    def test_stale_prompt_outputs_are_detected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.json"
            output = root / "labels.npz"
            prompt.write_text("first", encoding="utf-8")
            output.touch()
            output.with_suffix(".json").write_text(
                json.dumps({"prompt_manifest_sha256": point_prompt_sha256(prompt)}),
                encoding="utf-8",
            )
            self.assertTrue(point_prompt_output_is_current(output, prompt))
            prompt.write_text("changed", encoding="utf-8")
            self.assertFalse(point_prompt_output_is_current(output, prompt))

    def test_validates_unique_instances_and_mapping(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "4DOR",
                        "coordinate_system": "normalized_xy",
                        "instances": [
                            {
                                "object_id": 41,
                                "four_dor_class": "head_surgeon",
                                "mmor_class": "head_surgeon",
                                "points": [[0.25, 0.5], [0.1, 0.1]],
                                "point_labels": [1, 0],
                            },
                            {
                                "object_id": 42,
                                "four_dor_class": "head_surgeon",
                                "mmor_class": "head_surgeon",
                                "points": [[0.75, 0.5]],
                                "point_labels": [1],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = load_point_prompt_manifest(path)

        self.assertEqual([row["object_id"] for row in rows], [41, 42])

    def test_rejects_ambiguous_generic_human_class(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "4DOR",
                        "coordinate_system": "normalized_xy",
                        "instances": [
                            {
                                "object_id": 1,
                                "four_dor_class": "human",
                                "mmor_class": "nurse",
                                "points": [[0.5, 0.5]],
                                "point_labels": [1],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                load_point_prompt_manifest(path)


if __name__ == "__main__":
    unittest.main()

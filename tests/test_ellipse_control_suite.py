import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from or_video_reproduction.evaluation.ellipse_control_suite import (
    ExperimentDirectoryExistsError,
    PREFERRED_CASES,
    aggregate_video_rows,
    human_candidates,
    plan_single_ellipse_waypoints,
    refuse_existing_experiment_dir,
    select_human_ellipse,
    validate_in_canvas_trajectory,
    validate_six_case_manifest,
    video_level_bootstrap,
)
from or_video_reproduction.geometry.ellipse import Ellipse
from or_video_reproduction.geometry.trajectory import edit_trajectory, ellipse_in_canvas


def instance(
    key: str,
    class_name: str,
    x: float,
    y: float,
    *,
    depth: float,
    major: float = 80.0,
    minor: float = 40.0,
    angle: float = 20.0,
    source_pixels: int = 4000,
) -> dict[str, object]:
    return {
        "key": key,
        "class_name": class_name,
        "ellipse": {
            "center_x": x,
            "center_y": y,
            "major_diameter": major,
            "minor_diameter": minor,
            "angle_degrees": angle,
            "source_pixels": source_pixels,
        },
        "raw_relative_depth_mean": depth,
        "normalized_nearness": depth / 10.0,
        "red_green": [1, 2],
    }


def human_sequence() -> dict[str, object]:
    frames = []
    for frame_index in range(97):
        frames.append(
            {
                "frame_index": frame_index,
                "instances": [
                    instance("10:head_surgeon", "head_surgeon", 200.0 + frame_index, 300.0, depth=8.0),
                    instance(
                        "20:instrument_table",
                        "instrument_table",
                        600.0,
                        350.0,
                        depth=4.0,
                        major=120.0,
                        minor=70.0,
                        angle=0.0,
                    ),
                ],
                "skipped": [],
            }
        )
    return {
        "schema_version": 1,
        "kind": "paper_ellipse_only_geometric_conditioning",
        "shape": [97, 768, 1024],
        "fps": 24,
        "depth_direction": "larger_is_nearer",
        "frames": frames,
    }


class EllipseControlSuiteTests(unittest.TestCase):
    def test_six_case_manifest_requires_three_domains_each(self) -> None:
        cases = []
        for spec in PREFERRED_CASES:
            cases.append(
                {
                    "case_id": spec["case_id"],
                    "domain": spec["domain"],
                    "style": spec["style"],
                    "preferred_clip_id": spec["preferred_clip_id"],
                    "clip_id": spec["preferred_clip_id"],
                    "instance_id": "12:nurse",
                    "class_name": "nurse",
                }
            )
        result = validate_six_case_manifest({"cases": cases})
        self.assertEqual(result["case_count"], 6)
        self.assertEqual(result["mmor_count"], 3)
        self.assertEqual(result["four_dor_count"], 3)
        with self.assertRaisesRegex(ValueError, "three MMOR"):
            broken = copy.deepcopy(cases)
            broken[3]["domain"] = "MMOR"
            broken[3]["clip_id"] = "mmor-extra_PKA-camera01-timestamp000001"
            validate_six_case_manifest({"cases": broken})

    def test_refuses_to_overwrite_nonempty_experiment_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            root.mkdir()
            (root / "experiment_manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ExperimentDirectoryExistsError):
                refuse_existing_experiment_dir(root)
            empty = Path(directory) / "empty"
            empty.mkdir()
            refuse_existing_experiment_dir(empty)

    def test_one_ellipse_centroid_edit_preserves_unselected_fields(self) -> None:
        metadata = human_sequence()
        original = copy.deepcopy(metadata)
        start = metadata["frames"][0]["instances"][0]["ellipse"]
        edited, manifest = edit_trajectory(
            metadata,
            source_conditioning=Path("conditioning.mp4"),
            source_metadata=Path("metadata.json"),
            instance_id="10:head_surgeon",
            waypoints=[(start["center_x"], start["center_y"]), (start["center_x"] + 180.0, start["center_y"])],
            mode="replace",
            simplify_epsilon_pixels=2.0,
        )
        self.assertEqual(manifest["edited_centroids"][0], [start["center_x"], start["center_y"]])
        self.assertEqual(len(manifest["edited_centroids"]), 97)
        self.assertEqual(len(manifest["interpolated_waypoints"]), 97)
        for frame_index in range(97):
            self.assertEqual(
                edited["frames"][frame_index]["instances"][1],
                original["frames"][frame_index]["instances"][1],
            )
            before = original["frames"][frame_index]["instances"][0]
            after = edited["frames"][frame_index]["instances"][0]
            for field in ("major_diameter", "minor_diameter", "angle_degrees", "source_pixels"):
                self.assertEqual(before["ellipse"][field], after["ellipse"][field])
            self.assertEqual(before["raw_relative_depth_mean"], after["raw_relative_depth_mean"])
            self.assertEqual(before["class_name"], after["class_name"])

    def test_complete_97_frame_interpolation_and_canvas_validation(self) -> None:
        ellipse = Ellipse(400.0, 300.0, 80.0, 40.0, 15.0, 2000)
        planned = plan_single_ellipse_waypoints(style="horizontal", ellipse=ellipse, others=())
        self.assertEqual(planned["frame_count"], 97)
        self.assertGreaterEqual(planned["requested_displacement_pixels"], 150.0)
        self.assertLessEqual(planned["requested_displacement_pixels"], 220.0)
        validate_in_canvas_trajectory(ellipse, planned["interpolated_centroids"])
        self.assertEqual(planned["interpolated_centroids"][0], [400.0, 300.0])
        cramped = Ellipse(30.0, 40.0, 40.0, 24.0, 0.0, 800)
        with self.assertRaisesRegex(ValueError, "leaves the 1024x768 canvas"):
            validate_in_canvas_trajectory(
                cramped,
                [(30.0, 40.0)] + [(1010.0, 40.0)] * 96,
            )

    def test_selects_human_and_rejects_equipment(self) -> None:
        metadata = human_sequence()
        selected = select_human_ellipse(metadata, style="diagonal")
        self.assertEqual(selected["instance_id"], "10:head_surgeon")
        self.assertEqual(selected["class_name"], "head_surgeon")
        metadata["frames"][0]["instances"][0]["class_name"] = "instrument_table"
        for frame in metadata["frames"]:
            frame["instances"][0]["class_name"] = "instrument_table"
        with self.assertRaisesRegex(ValueError, "human ellipse"):
            select_human_ellipse(metadata, style="diagonal")

    def test_video_level_bootstrap_resamples_videos_not_frames(self) -> None:
        values = [10.0, 20.0, 30.0]
        result = video_level_bootstrap(values, samples=200, seed=0)
        self.assertEqual(result["n_videos"], 3)
        self.assertEqual(result["unit"], "video")
        self.assertIn("exploratory", result["label"])
        rows = [
            {"mean_trajectory_error_pixels": 10.0, "direction_cosine": 1.0},
            {"mean_trajectory_error_pixels": 20.0, "direction_cosine": 0.5},
            {"mean_trajectory_error_pixels": None, "direction_cosine": 0.0},
        ]
        summary = aggregate_video_rows(rows, label="MMOR")
        self.assertEqual(summary["n_videos"], 3)
        self.assertEqual(summary["independent_unit"], "video")
        self.assertEqual(summary["descriptive"]["mean_trajectory_error_pixels"]["n_available"], 2)
        self.assertEqual(summary["identity_switching"], "n/a")

    def test_truncated_or_tiny_person_is_not_selected(self) -> None:
        metadata = human_sequence()
        metadata["frames"][0]["instances"][0]["ellipse"].update(
            {"center_x": 10.0, "center_y": 10.0, "major_diameter": 80.0, "minor_diameter": 40.0}
        )
        metadata["frames"][0]["instances"][0]["class_name"] = "nurse"
        metadata["frames"][0]["instances"][0]["key"] = "7:nurse"
        candidates = human_candidates(metadata)
        self.assertTrue(all(row["instance_id"] != "7:nurse" for row in candidates))
        self.assertFalse(ellipse_in_canvas(Ellipse(10.0, 10.0, 80.0, 40.0, 0.0, 4000)))

    def test_crowded_in_canvas_human_remains_eligible(self) -> None:
        metadata = human_sequence()
        metadata["frames"][0]["instances"][1]["ellipse"].update(
            {
                "center_x": 210.0,
                "center_y": 310.0,
                "major_diameter": 80.0,
                "minor_diameter": 40.0,
            }
        )
        candidates = human_candidates(metadata)
        self.assertIn("10:head_surgeon", [row["instance_id"] for row in candidates])
        selected = select_human_ellipse(metadata, style="horizontal")
        self.assertEqual(selected["instance_id"], "10:head_surgeon")
        self.assertGreaterEqual(selected["trajectory"]["requested_displacement_pixels"], 80.0)


if __name__ == "__main__":
    unittest.main()

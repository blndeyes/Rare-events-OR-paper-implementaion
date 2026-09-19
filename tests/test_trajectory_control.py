import unittest

import numpy as np

from or_video_reproduction.evaluation.trajectory_control import (
    build_parser,
    evaluate_control,
    evaluate_edited_only,
    instance_label,
    mask_track,
    overlay_frame,
)


def draw_square(frame: np.ndarray, label: int, center_x: int, center_y: int) -> None:
    frame[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3] = label


def fixtures() -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    original = np.zeros((97, 64, 64), dtype=np.uint8)
    edited = np.zeros_like(original)
    original_centroids, edited_centroids, frames = [], [], []
    for index in range(97):
        x = 10 + round(30 * index / 96)
        draw_square(original[index], 10, 10, 20)
        draw_square(edited[index], 10, x, 20)
        draw_square(original[index], 20, 50, 50)
        draw_square(edited[index], 20, 50, 50)
        original_centroids.append([10.0, 20.0])
        edited_centroids.append([float(x), 20.0])
        frames.append(
            {
                "frame_index": index,
                "instances": [
                    {
                        "key": "10:head_surgeon",
                        "class_name": "head_surgeon",
                        "ellipse": {
                            "center_x": float(x),
                            "center_y": 20.0,
                            "major_diameter": 5.0,
                            "minor_diameter": 5.0,
                            "angle_degrees": 0.0,
                            "source_pixels": 25,
                        },
                    }
                ],
            }
        )
    manifest = {
        "selected_instance_id": "10:head_surgeon",
        "selected_class": "head_surgeon",
        "original_centroids": original_centroids,
        "edited_centroids": edited_centroids,
    }
    return original, edited, manifest, {"frames": frames}


class TrajectoryControlTests(unittest.TestCase):
    def test_tracks_centroids_and_reports_control_evidence(self) -> None:
        original, edited, manifest, metadata = fixtures()
        result = evaluate_control(
            original_labels=original,
            edited_labels=edited,
            edit_manifest=manifest,
            edited_metadata=metadata,
        )

        self.assertEqual(result["trajectory_error_pixels"]["mean"], 0.0)
        self.assertEqual(result["trajectory_error_pixels"]["endpoint"], 0.0)
        self.assertEqual(result["movement"]["generated_vector_pixels"], [30.0, 0.0])
        self.assertEqual(result["movement"]["direction_cosine"], 1.0)
        self.assertEqual(result["unrelated_entity_stability"]["mean_centroid_change_pixels"], 0.0)
        self.assertTrue(result["control_evidence"]["success"])
        self.assertGreater(
            result["ellipse_alignment"]["bounding_box_iou_to_conditioning_ellipse"], 0.5
        )

    def test_ignored_edit_fails_control_evidence(self) -> None:
        original, _, manifest, metadata = fixtures()
        result = evaluate_control(
            original_labels=original,
            edited_labels=original.copy(),
            edit_manifest=manifest,
            edited_metadata=metadata,
        )
        self.assertFalse(result["control_evidence"]["success"])
        self.assertEqual(result["movement"]["generated_displacement_pixels"], 0.0)

    def test_missing_frames_are_reported(self) -> None:
        original, _, _, _ = fixtures()
        original[10] = 0
        track = mask_track(original, 10)
        self.assertIn(10, track["missing_frames"])
        self.assertIsNone(track["centroids"][10])

    def test_instance_label_validation(self) -> None:
        self.assertEqual(instance_label("10:head_surgeon"), 10)
        with self.assertRaises(ValueError):
            instance_label("head_surgeon")

    def test_lost_sam2_frames_are_unavailable_not_zero(self) -> None:
        original, edited, manifest, metadata = fixtures()
        edited_only = np.where(edited == 10, 1, 0).astype(np.uint8)
        edited_only[10] = 0
        edited_only[96] = 0
        manifest = dict(manifest)
        manifest["edited_centroids"] = [
            [point[0] + 10.0, point[1]] for point in manifest["edited_centroids"]
        ]
        result = evaluate_edited_only(
            edited_labels=edited_only,
            edit_manifest=manifest,
            edited_metadata=metadata,
            sam2_object_id=1,
        )
        self.assertEqual(result["identity_switching"], "n/a")
        self.assertIn(10, result["unavailable_frame_indices"])
        self.assertIn(96, result["unavailable_frame_indices"])
        self.assertIsNone(result["per_frame"][10]["trajectory_error_pixels"])
        self.assertIsNone(result["per_frame"][10]["segmentation_iou"])
        self.assertIsNone(result["per_frame"][96]["bounding_box_iou"])
        self.assertNotEqual(result["per_frame"][10]["trajectory_error_pixels"], 0.0)
        self.assertIsNone(result["trajectory_error_pixels"]["endpoint"])
        self.assertFalse(result["trajectory_error_pixels"]["zero_filled"])
        self.assertEqual(result["valid_track_frames"], 95)
        self.assertAlmostEqual(result["trajectory_error_pixels"]["mean"], 10.0, places=5)

    def test_all_lost_frames_keep_aggregate_metrics_unavailable(self) -> None:
        original, _, manifest, metadata = fixtures()
        empty = np.zeros_like(original)
        result = evaluate_edited_only(
            edited_labels=empty,
            edit_manifest=manifest,
            edited_metadata=metadata,
            sam2_object_id=1,
        )
        self.assertIsNone(result["trajectory_error_pixels"]["mean"])
        self.assertIsNone(result["movement"]["generated_displacement_pixels"])
        self.assertIsNone(result["ellipse_alignment"]["segmentation_iou_to_conditioning_ellipse"])
        self.assertEqual(result["valid_track_frames"], 0)

    def test_edited_only_parser_makes_original_generated_optional(self) -> None:
        args = build_parser().parse_args(
            [
                "--edited-only",
                "--edited-generated",
                "edited.mp4",
                "--sam2-root",
                "sam2",
                "--sam2-checkpoint",
                "sam2.pt",
                "--edit-manifest",
                "edit.json",
                "--edited-metadata",
                "meta.json",
                "--original-conditioning",
                "orig.mp4",
                "--edited-conditioning",
                "cond.mp4",
                "--output-dir",
                "out",
            ]
        )
        self.assertTrue(args.edited_only)
        self.assertIsNone(args.original_generated)

    def test_legacy_mode_still_requires_original_generated(self) -> None:
        from or_video_reproduction.evaluation import trajectory_control

        with self.assertRaises(SystemExit):
            trajectory_control.main(
                [
                    "--edited-generated",
                    "edited.mp4",
                    "--sam2-root",
                    "sam2",
                    "--sam2-checkpoint",
                    "sam2.pt",
                    "--edit-manifest",
                    "edit.json",
                    "--edited-metadata",
                    "meta.json",
                    "--original-conditioning",
                    "orig.mp4",
                    "--edited-conditioning",
                    "cond.mp4",
                    "--output-dir",
                    "out",
                    "--source-labels",
                    "labels.npz",
                ]
            )

    def test_overlay_marks_lost_frames_without_inventing_a_centroid(self) -> None:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        frame = overlay_frame(rgb, mask, commanded=[10.0, 10.0], observed=None)
        self.assertEqual(frame.shape, (32, 32, 3))
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()

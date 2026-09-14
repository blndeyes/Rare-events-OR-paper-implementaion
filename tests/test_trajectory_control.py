import unittest

import numpy as np

from or_video_reproduction.evaluation.trajectory_control import (
    evaluate_control,
    instance_label,
    mask_track,
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


if __name__ == "__main__":
    unittest.main()

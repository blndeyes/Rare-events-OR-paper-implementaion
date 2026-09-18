import unittest

import numpy as np

from or_video_reproduction.geometry.palette import PAPER_36_PALETTE
from or_video_reproduction.geometry.render import RenderInstance, render_conditioning
from or_video_reproduction.geometry.ellipse import Ellipse
from or_video_reproduction.evaluation.finegrained.control import (
    compare_tracks,
    parse_ellipse_frame,
    track_instances,
)
from or_video_reproduction.evaluation.finegrained.scoring import (
    detection_rate,
    match_people_by_iou,
    mpjpe_from_matches,
    nearest_anchor_scores,
)


class EllipseControlTests(unittest.TestCase):
    def test_parses_class_and_forbids_rgb_canvas_scores(self) -> None:
        frame = render_conditioning(
            [
                RenderInstance(
                    key="1:nurse",
                    class_name="nurse",
                    ellipse=Ellipse(40, 40, 30, 16, 0.0, 10),
                    normalized_depth=0.8,
                    raw_depth=1.0,
                ),
                RenderInstance(
                    key="2:patient",
                    class_name="patient",
                    ellipse=Ellipse(120, 80, 40, 18, 20.0, 10),
                    normalized_depth=0.2,
                    raw_depth=4.0,
                ),
            ],
            (160, 200),
            smaller_is_nearer=True,
        )
        instances = parse_ellipse_frame(frame)
        classes = {row["class_name"] for row in instances}
        self.assertEqual(classes, {"nurse", "patient"})
        tracked = track_instances([instances, instances])
        generated = track_instances(
            [
                [
                    {
                        "class_name": "nurse",
                        "centroid": [42.0, 41.0],
                        "box_xyxy": [20, 20, 60, 60],
                        "relative_depth_blue": 0.75,
                    }
                ],
                [
                    {
                        "class_name": "nurse",
                        "centroid": [43.0, 41.0],
                        "box_xyxy": [21, 20, 61, 60],
                        "relative_depth_blue": 0.74,
                    }
                ],
            ]
        )
        scored = compare_tracks(tracked, generated)
        self.assertEqual(scored["psnr_against_ellipse_rgb"], "forbidden")
        self.assertEqual(scored["ssim_against_ellipse_rgb"], "forbidden")
        self.assertLess(scored["unmatched_control_tracks"], 2)
        self.assertIn("nurse", PAPER_36_PALETTE)

    def test_missing_generated_entities_are_not_perfect_scores(self) -> None:
        control = track_instances(
            [
                [
                    {
                        "class_name": "nurse",
                        "centroid": [10.0, 10.0],
                        "box_xyxy": [0, 0, 20, 20],
                        "relative_depth_blue": 0.5,
                    }
                ]
            ]
        )
        generated = track_instances([[]])
        scored = compare_tracks(control, generated)
        self.assertEqual(scored["entity_detection_rate"], 0.0)
        self.assertEqual(scored["unmatched_control_tracks"], 1)


class AnatomyAndHandScoringTests(unittest.TestCase):
    def test_detection_failures_are_counted(self) -> None:
        rate = detection_rate([{"frame": 0}], frame_count=4)
        self.assertEqual(rate["clips_or_frames_without_detection"], 3)
        self.assertEqual(rate["frame_detection_rate"], 0.25)

    def test_hand_score_unavailable_without_both_sides(self) -> None:
        generated = np.ones((2, 4))
        empty = np.zeros((0, 4))
        result = nearest_anchor_scores(generated, empty)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["generated_crops"], 2)

    def test_unmatched_people_do_not_get_zero_mpjpe(self) -> None:
        real = [
            {
                "frame": 0,
                "box_xyxy": [0, 0, 10, 20],
                "keypoints": np.zeros((17, 2)),
                "keypoint_scores": np.ones(17),
            }
        ]
        generated = [
            {
                "frame": 0,
                "box_xyxy": [80, 80, 100, 120],
                "keypoints": np.ones((17, 2)),
                "keypoint_scores": np.ones(17),
            }
        ]
        matched = match_people_by_iou(real, generated, min_iou=0.3)
        self.assertEqual(matched["unmatched_real"], 1)
        mpjpe = mpjpe_from_matches(matched, width=1024, height=768)
        self.assertEqual(mpjpe["status"], "unavailable")
        self.assertNotIn("mpjpe_pixels", mpjpe)
        self.assertFalse(mpjpe.get("zero_mpjpe_for_misses", False))


if __name__ == "__main__":
    unittest.main()

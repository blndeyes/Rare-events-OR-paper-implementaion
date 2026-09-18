import unittest

import numpy as np

from or_video_reproduction.geometry.palette import PAPER_36_PALETTE
from or_video_reproduction.geometry.render import RenderInstance, render_conditioning
from or_video_reproduction.geometry.ellipse import Ellipse
from or_video_reproduction.evaluation.finegrained.backends import (
    frames_to_jedi_video_tensor,
    xyxy_to_xywh,
)
from or_video_reproduction.evaluation.finegrained.control import (
    compare_tracks,
    parse_ellipse_frame,
    track_instances,
)
from or_video_reproduction.evaluation.finegrained.scoring import (
    aggregate_clip_detection_rates,
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
        self.assertEqual(scored["unmatched_control_tracks"], 1)
        self.assertEqual(scored["entity_detection_rate"], 0.5)
        self.assertEqual(scored["class_consistency"], 1.0)
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
        self.assertIsInstance(scored["class_consistency"], dict)
        self.assertEqual(scored["class_consistency"]["status"], "unavailable")
        self.assertIsNone(scored["mean_centroid_error_pixels"])


    def test_person_is_not_matched_to_or_roles_or_equipment(self) -> None:
        control = track_instances(
            [
                [
                    {
                        "class_name": "nurse",
                        "centroid": [10.0, 10.0],
                        "box_xyxy": [0, 0, 20, 20],
                        "relative_depth_blue": 0.4,
                    },
                    {
                        "class_name": "operating_table",
                        "centroid": [80.0, 80.0],
                        "box_xyxy": [60, 60, 100, 100],
                        "relative_depth_blue": 0.8,
                    },
                ]
            ]
        )
        generated = track_instances(
            [
                [
                    {
                        "class_name": "person",
                        "centroid": [12.0, 11.0],
                        "box_xyxy": [1, 1, 21, 21],
                        "relative_depth_blue": 0.41,
                    }
                ]
            ]
        )
        scored = compare_tracks(control, generated, detector_label_space=("person",))
        self.assertEqual(scored["recovery_scope"], "person_only_plus_independent_depth")
        self.assertEqual(scored["class_mapping"]["status"], "blocked")
        self.assertEqual(scored["class_mapping"]["documented_mapping"], None)
        self.assertEqual(scored["class_consistency"]["status"], "unavailable")
        self.assertIn("COCO", scored["class_consistency"]["reason"])
        self.assertEqual(scored["entity_detection_rate"], 0.0)
        self.assertIsNone(scored["mean_centroid_error_pixels"])
        self.assertIsNone(scored["mean_endpoint_error_pixels"])
        counts = scored["entity_counts"]
        self.assertEqual(counts["unsupported_control_entities_by_class"], {"nurse": 1, "operating_table": 1})
        self.assertEqual(counts["matched_control_entity_count"], 0)
        self.assertEqual(counts["generated_entities_by_class"], {"person": 1})
        for row in scored["per_entity"]:
            self.assertNotEqual(row.get("match_status"), "matched_class_compatible")
            self.assertIsNone(row.get("mean_centroid_error_pixels"))
            self.assertIsNone(row.get("relative_depth_order_agreement"))


class AnatomyAndHandScoringTests(unittest.TestCase):
    def test_detection_failures_are_counted(self) -> None:
        rate = detection_rate([{"frame": 0}], frame_count=4)
        self.assertEqual(rate["clips_or_frames_without_detection"], 3)
        self.assertEqual(rate["frame_detection_rate"], 0.25)

    def test_group_detection_rate_does_not_collapse_shared_frame_indices(self) -> None:
        sampled = list(range(16))
        clip_a = [{"id": "clip-a", "frame": index} for index in sampled]
        clip_b = [{"id": "clip-b", "frame": index} for index in sampled[:4]]
        per_clip = [
            {"id": "clip-a", "real": detection_rate(clip_a, 16, clip_id="clip-a"), "generated": detection_rate(clip_a, 16, clip_id="clip-a")},
            {"id": "clip-b", "real": detection_rate(clip_b, 16, clip_id="clip-b"), "generated": detection_rate([], 16, clip_id="clip-b")},
        ]
        concatenated = clip_a + clip_b
        collapsed = detection_rate(concatenated, 32)
        aggregated = aggregate_clip_detection_rates(per_clip, "real")
        self.assertEqual(collapsed["frames_with_detection"], 16)
        self.assertEqual(collapsed["frame_detection_rate"], 0.5)
        self.assertEqual(aggregated["frames_with_detection"], 20)
        self.assertEqual(aggregated["frame_count"], 32)
        self.assertEqual(aggregated["detection_count"], 20)
        self.assertAlmostEqual(aggregated["frame_detection_rate"], 20 / 32)
        generated = aggregate_clip_detection_rates(per_clip, "generated")
        self.assertEqual(generated["frames_with_detection"], 16)
        self.assertEqual(generated["detection_count"], 16)
        self.assertAlmostEqual(generated["frame_detection_rate"], 0.5)

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


class JediAndPoseHelperTests(unittest.TestCase):
    def test_xyxy_to_xywh(self) -> None:
        self.assertEqual(xyxy_to_xywh([10, 20, 40, 80]), [10.0, 20.0, 30.0, 60.0])

    def test_jedi_tensor_is_tchw_unit_interval(self) -> None:
        frames = np.full((3, 8, 10, 3), 255, dtype=np.uint8)
        video = frames_to_jedi_video_tensor(frames)
        self.assertEqual(video.shape, (3, 3, 8, 10))
        self.assertEqual(video.dtype, np.float32)
        np.testing.assert_allclose(video, 1.0)


if __name__ == "__main__":
    unittest.main()

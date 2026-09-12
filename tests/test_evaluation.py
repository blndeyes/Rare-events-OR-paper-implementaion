import math
import unittest

import numpy as np

from or_video_reproduction.evaluation.backends import parse_dover_fused_score
from or_video_reproduction.evaluation.core import (
    binary_classification_metrics,
    bounding_box_iou,
    peak_signal_to_noise_ratio,
    segmentation_iou,
    structural_similarity,
)
from or_video_reproduction.evaluation.distribution import frechet_distance, inception_score


class PairedMetricTests(unittest.TestCase):
    def test_identical_video_has_perfect_scores(self) -> None:
        video = np.arange(2 * 16 * 16 * 3, dtype=np.uint8).reshape(2, 16, 16, 3)

        self.assertTrue(math.isinf(peak_signal_to_noise_ratio(video, video)))
        self.assertAlmostEqual(structural_similarity(video, video), 1.0, places=12)

    def test_psnr_matches_known_constant_error(self) -> None:
        reference = np.zeros((1, 12, 12, 3), dtype=np.uint8)
        generated = np.full_like(reference, 10)

        expected = 20.0 * math.log10(255.0 / 10.0)
        self.assertAlmostEqual(peak_signal_to_noise_ratio(reference, generated), expected)

    def test_mismatched_video_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ in shape"):
            structural_similarity(np.zeros((12, 12, 3)), np.zeros((13, 12, 3)))


class StructuralMetricTests(unittest.TestCase):
    def test_mask_and_box_iou_are_calculated_per_present_class(self) -> None:
        truth = np.zeros((1, 8, 8), dtype=np.uint8)
        prediction = np.zeros_like(truth)
        truth[0, 1:5, 1:5] = 2
        prediction[0, 1:5, 3:7] = 2

        self.assertAlmostEqual(segmentation_iou(truth, prediction)["mean"], 1.0 / 3.0)
        self.assertAlmostEqual(bounding_box_iou(truth, prediction)["mean"], 1.0 / 3.0)

    def test_missing_prediction_scores_zero(self) -> None:
        truth = np.zeros((4, 4), dtype=np.uint8)
        truth[1:3, 1:3] = 1

        self.assertEqual(segmentation_iou(truth, np.zeros_like(truth))["mean"], 0.0)
        self.assertEqual(bounding_box_iou(truth, np.zeros_like(truth))["mean"], 0.0)


class DistributionMetricTests(unittest.TestCase):
    def test_frechet_is_zero_for_identical_embeddings(self) -> None:
        embeddings = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]])

        self.assertAlmostEqual(frechet_distance(embeddings, embeddings), 0.0, places=10)

    def test_frechet_captures_mean_shift(self) -> None:
        reference = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])

        self.assertAlmostEqual(frechet_distance(reference, reference + 1.0), 2.0, places=10)

    def test_uniform_predictions_have_inception_score_one(self) -> None:
        logits = np.zeros((20, 3))

        score = inception_score(logits, splits=10)
        self.assertAlmostEqual(score["mean"], 1.0)
        self.assertAlmostEqual(score["std"], 0.0)

    def test_confident_diverse_predictions_raise_inception_score(self) -> None:
        logits = np.tile(np.array([[10.0, -10.0], [-10.0, 10.0]]), (10, 1))

        self.assertGreater(inception_score(logits, splits=1)["mean"], 1.9)


class DownstreamAndAdapterTests(unittest.TestCase):
    def test_accuracy_and_positive_recall(self) -> None:
        result = binary_classification_metrics([1, 1, 0, 0], [1, 0, 0, 0])

        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["recall"], 0.5)

    def test_no_positive_targets_report_undefined_recall(self) -> None:
        self.assertIsNone(binary_classification_metrics([0, 0], [0, 1])["recall"])

    def test_parses_official_dover_fused_output(self) -> None:
        output = "Normalized fused overall score (scale in [0,1]): 0.73125"

        self.assertEqual(parse_dover_fused_score(output), 0.73125)


if __name__ == "__main__":
    unittest.main()

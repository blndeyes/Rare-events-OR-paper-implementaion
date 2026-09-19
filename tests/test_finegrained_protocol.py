import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from or_video_reproduction.evaluation.finegrained.backends import (
    ModelUnavailable,
    coerce_clip_image_features,
    unwrap_clip_features,
)
from or_video_reproduction.evaluation.finegrained.cache import ArtifactCache
from or_video_reproduction.evaluation.finegrained.protocol import (
    DEFAULT_SAMPLE_COUNT,
    clip_preprocess_numpy,
    density_coverage,
    frame_sampling_manifest,
    greedy_kcenter_coreset,
    kid_polynomial,
    linear_sum_assignment,
    polynomial_kernel,
    rbf_kernel,
    scaled_unbiased_rbf_mmd,
    select_frame_indices,
    spatial_average_pool_3x3,
    statistical_flags,
    unbiased_mmd2,
    video_bootstrap,
)


class FrameSamplingTests(unittest.TestCase):
    def test_shared_sampler_is_deterministic_and_keeps_frame_zero(self) -> None:
        first = select_frame_indices(97, 16, include_frame_zero=True)
        second = select_frame_indices(97, 16, include_frame_zero=True)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 96)
        self.assertEqual(len(set(first)), 16)
        manifest = frame_sampling_manifest()
        self.assertEqual(manifest["indices"], first)
        self.assertEqual(manifest["sample_count"], DEFAULT_SAMPLE_COUNT)

    def test_excluding_frame_zero_is_consistent(self) -> None:
        indices = select_frame_indices(97, 16, include_frame_zero=False)
        self.assertNotEqual(indices[0], 0)
        self.assertEqual(indices[-1], 96)


class KernelTests(unittest.TestCase):
    def test_unbiased_rbf_mmd_is_smaller_for_identical_than_shifted_sets(self) -> None:
        values = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
        identical = scaled_unbiased_rbf_mmd(values, values)
        shifted = scaled_unbiased_rbf_mmd(values, values + 3.0)
        self.assertLess(identical, shifted)

    def test_polynomial_kernel_matches_definition(self) -> None:
        left = np.array([[1.0, 0.0], [0.0, 2.0]])
        right = np.array([[1.0, 1.0]])
        expected = np.array(
            [
                [(1.0 / 2 + 1.0) ** 3],
                [(2.0 / 2 + 1.0) ** 3],
            ]
        )
        np.testing.assert_allclose(polynomial_kernel(left, right), expected)

    def test_kid_uses_equal_n_and_recorded_seed(self) -> None:
        rng = np.random.default_rng(0)
        real = rng.normal(size=(20, 8))
        fake = rng.normal(size=(20, 8))
        first = kid_polynomial(real, fake, subsets=10, seed=42)
        second = kid_polynomial(real, fake, subsets=10, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["n"], 20)
        self.assertEqual(first["subset_size"], 10)
        self.assertIn("embeddings", first["n_definition"])

    def test_mmd_rejects_unequal_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "equally many"):
            unbiased_mmd2(np.ones((3, 2)), np.ones((2, 2)), rbf_kernel)


class DensityCoresetBootstrapTests(unittest.TestCase):
    def test_density_and_coverage_on_identical_points(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [2.0, 2.0]])
        result = density_coverage(points, points, nearest_k=2)
        self.assertAlmostEqual(result["coverage"], 1.0)
        self.assertGreater(result["density"], 0.5)

    def test_coreset_is_seed_deterministic(self) -> None:
        features = np.arange(40, dtype=np.float64).reshape(10, 4)
        first = greedy_kcenter_coreset(features, 4, seed=42)
        second = greedy_kcenter_coreset(features, 4, seed=42)
        np.testing.assert_array_equal(first["indices"], second["indices"])
        self.assertEqual(len(set(first["indices"].tolist())), 4)

    def test_pool_preserves_grid_shape(self) -> None:
        grid = np.arange(5 * 5 * 2, dtype=np.float64).reshape(5, 5, 2)
        pooled = spatial_average_pool_3x3(grid)
        self.assertEqual(pooled.shape, (5, 5, 2))

    def test_video_bootstrap_resamples_clips_not_frames(self) -> None:
        real = [np.ones((3, 2)), np.zeros((3, 2)), np.full((3, 2), 2.0)]
        fake = [np.ones((3, 2)), np.zeros((3, 2)), np.full((3, 2), 2.0)]
        result = video_bootstrap(real, fake, scaled_unbiased_rbf_mmd, samples=20, seed=42)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["unit"], "video")
        self.assertEqual(result["n_videos"], 3)
        self.assertEqual(len(result["ci95"]), 2)

    def test_single_video_bootstrap_is_unavailable(self) -> None:
        result = video_bootstrap([np.ones((2, 2))], [np.zeros((2, 2))], scaled_unbiased_rbf_mmd)
        self.assertEqual(result["status"], "unavailable")

    def test_rank_deficient_frechet_is_flagged(self) -> None:
        flags = statistical_flags(n_videos=6, n_embeddings=96, feature_dim=1024)
        self.assertIn("frechet_covariance_rank_deficient", flags)
        self.assertIn("weak_video_level_distribution", flags)

    def test_assignment_is_optimal_on_small_costs(self) -> None:
        cost = np.array([[4.0, 1.0], [2.0, 5.0]])
        rows, cols = linear_sum_assignment(cost)
        self.assertEqual(list(rows), [0, 1])
        self.assertEqual(list(cols), [1, 0])

    def test_clip_preprocess_shape(self) -> None:
        frames = np.zeros((2, 32, 48, 3), dtype=np.uint8)
        processed = clip_preprocess_numpy(frames, size=16)
        self.assertEqual(processed.shape, (2, 16, 16, 3))


class CacheResumeTests(unittest.TestCase):
    def test_array_roundtrip_and_config_isolation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = ArtifactCache(root, namespace="clip", config={"size": 336})
            second = ArtifactCache(root, namespace="clip", config={"size": 224})
            self.assertNotEqual(first.key, second.key)
            first.save_array("demo", x=np.arange(6).reshape(2, 3))
            loaded = first.load_array("demo")
            np.testing.assert_array_equal(loaded["x"], np.arange(6).reshape(2, 3))
            self.assertFalse(second.has_array("demo"))


class ClipOutputUnwrapTests(unittest.TestCase):
    def test_prefers_image_embeds_over_pooler_output(self) -> None:
        payload = SimpleNamespace(
            image_embeds=np.ones((2, 3)),
            pooler_output=np.zeros((2, 8)),
            last_hidden_state=np.zeros((2, 5, 8)),
        )
        np.testing.assert_array_equal(unwrap_clip_features(payload), np.ones((2, 3)))

    def test_accepts_a_plain_tensor_like_object(self) -> None:
        class TensorLike:
            def float(self) -> "TensorLike":
                return self

            def cpu(self) -> "TensorLike":
                return self

        value = TensorLike()
        self.assertIs(unwrap_clip_features(value), value)

    def test_does_not_treat_pooled_hidden_states_as_clip_features(self) -> None:
        payload = SimpleNamespace(pooler_output=np.zeros((2, 8)), last_hidden_state=np.zeros((2, 5, 8)))
        self.assertIsNone(unwrap_clip_features(payload))


class FakeProjection:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.calls: list[object] = []

    def __call__(self, value: np.ndarray) -> np.ndarray:
        self.calls.append(value)
        return np.full((len(value), self.out_features), 7.0)


class ClipFeatureCoerceTests(unittest.TestCase):
    def test_uses_already_projected_768d_pooler(self) -> None:
        projection = FakeProjection(1024, 768)
        pooler = np.ones((2, 768))
        payload = SimpleNamespace(pooler_output=pooler, last_hidden_state=np.zeros((2, 5, 1024)))
        model = SimpleNamespace(visual_projection=projection)
        result = coerce_clip_image_features(model, payload)
        np.testing.assert_array_equal(result, pooler)
        self.assertEqual(projection.calls, [])

    def test_projects_1024d_pooler(self) -> None:
        projection = FakeProjection(1024, 768)
        pooler = np.ones((2, 1024))
        payload = SimpleNamespace(pooler_output=pooler, last_hidden_state=np.zeros((2, 5, 1024)))
        model = SimpleNamespace(visual_projection=projection)
        result = coerce_clip_image_features(model, payload)
        np.testing.assert_array_equal(result, np.full((2, 768), 7.0))
        self.assertEqual(len(projection.calls), 1)

    def test_projects_cls_token_when_pooler_is_missing(self) -> None:
        projection = FakeProjection(1024, 768)
        hidden = np.concatenate([np.ones((2, 1, 1024)), np.zeros((2, 4, 1024))], axis=1)
        payload = SimpleNamespace(last_hidden_state=hidden)
        model = SimpleNamespace(visual_projection=projection)
        result = coerce_clip_image_features(model, payload)
        np.testing.assert_array_equal(result, np.full((2, 768), 7.0))

    def test_rejects_incompatible_pooling_shapes(self) -> None:
        projection = FakeProjection(1024, 768)
        payload = SimpleNamespace(pooler_output=np.ones((2, 512)), last_hidden_state=np.ones((2, 5, 512)))
        model = SimpleNamespace(visual_projection=projection)
        with self.assertRaises(ModelUnavailable):
            coerce_clip_image_features(model, payload)


if __name__ == "__main__":
    unittest.main()

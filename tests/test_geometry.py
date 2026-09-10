import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from or_video_reproduction.data.semantics import vocabulary
from or_video_reproduction.geometry.depth import (
    depth_to_blue,
    mean_valid_depth,
    normalize_depths,
)
from or_video_reproduction.geometry.ellipse import Ellipse, fit_ellipse, rasterize_ellipse
from or_video_reproduction.geometry.palette import PAPER_36_PALETTE, decode_red_green
from or_video_reproduction.geometry.preview import create_preview
from or_video_reproduction.geometry.render import RenderInstance, render_conditioning
from or_video_reproduction.geometry.sequence_preview import _temporal_deltas


def angular_error_mod_180(actual: float, expected: float) -> float:
    difference = abs(actual - expected) % 180.0
    return min(difference, 180.0 - difference)


class EllipseTests(unittest.TestCase):
    def test_recovers_synthetic_filled_ellipse(self) -> None:
        expected = Ellipse(61.0, 43.0, 42.0, 18.0, 27.0, 0)
        mask = rasterize_ellipse(expected, (96, 128))

        actual = fit_ellipse(mask)

        self.assertAlmostEqual(actual.center_x, expected.center_x, delta=0.2)
        self.assertAlmostEqual(actual.center_y, expected.center_y, delta=0.2)
        self.assertAlmostEqual(actual.major_diameter, expected.major_diameter, delta=0.5)
        self.assertAlmostEqual(actual.minor_diameter, expected.minor_diameter, delta=0.5)
        self.assertLess(angular_error_mod_180(actual.angle_degrees, expected.angle_degrees), 0.6)

    def test_rejects_too_small_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 5"):
            fit_ellipse(np.eye(2, dtype=bool))


class PaletteTests(unittest.TestCase):
    def test_palette_has_unique_round_trippable_pair_for_every_label(self) -> None:
        self.assertEqual(set(PAPER_36_PALETTE), set(vocabulary()))
        self.assertEqual(len(set(PAPER_36_PALETTE.values())), 36)
        for label, pair in PAPER_36_PALETTE.items():
            self.assertEqual(decode_red_green(*pair), label)


class DepthTests(unittest.TestCase):
    def test_aggregates_only_nonzero_finite_masked_depth(self) -> None:
        depth = np.array([[0.0, 2.0], [4.0, np.nan]])
        mask = np.ones((2, 2), dtype=bool)
        self.assertEqual(mean_valid_depth(depth, mask), 3.0)

    def test_normalization_always_encodes_nearer_as_brighter(self) -> None:
        values = {"near": 1.0, "middle": 2.0, "far": 3.0}
        normalized = normalize_depths(values, smaller_is_nearer=True)
        self.assertEqual(normalized, {"near": 1.0, "middle": 0.5, "far": 0.0})
        self.assertEqual(depth_to_blue(normalized["near"]), 255)


class RenderTests(unittest.TestCase):
    def test_near_instance_occludes_far_instance(self) -> None:
        ellipse = Ellipse(10.0, 10.0, 10.0, 8.0, 0.0, 30)
        instances = [
            RenderInstance("near", "patient", ellipse, 1.0, 1.0),
            RenderInstance("far", "nurse", ellipse, 0.0, 3.0),
        ]

        image = render_conditioning(instances, (24, 24), smaller_is_nearer=True)

        self.assertEqual(tuple(image[10, 10]), (*PAPER_36_PALETTE["patient"], 255))

    def test_predicate_cannot_be_rendered_as_an_ellipse(self) -> None:
        ellipse = Ellipse(10.0, 10.0, 10.0, 8.0, 0.0, 30)
        predicate = RenderInstance("edge", "holding", ellipse, 0.5, 2.0)

        with self.assertRaisesRegex(ValueError, "Only segmented entity nodes"):
            render_conditioning([predicate], (24, 24), smaller_is_nearer=True)


class PreviewTests(unittest.TestCase):
    def test_writes_auditable_preview_bundle(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.full((24, 32, 3), 100, dtype=np.uint8)).save(root / "rgb.png")
            labels = np.zeros((24, 32), dtype=np.uint8)
            labels[5:19, 8:25] = 5  # official raw label for patient
            Image.fromarray(labels).save(root / "mask.png")
            Image.fromarray(np.full((24, 32), 1000, dtype=np.uint16)).save(root / "depth.tiff")

            payload = create_preview(
                root / "rgb.png",
                root / "mask.png",
                root / "depth.tiff",
                root / "output",
                smaller_is_nearer=True,
            )

            self.assertEqual(len(payload["instances"]), 1)
            self.assertEqual(payload["instances"][0]["class_name"], "patient")
            expected_files = (
                payload["model_conditioning_output"],
                *payload["diagnostic_outputs"],
                "metadata.json",
            )
            for name in expected_files:
                self.assertTrue((root / "output" / name).is_file(), name)

            conditioning = np.asarray(
                Image.open(root / "output" / payload["model_conditioning_output"])
            )
            self.assertTrue(np.all(conditioning[0, 0] == 0))
            self.assertGreater(np.count_nonzero(conditioning), 0)

    def test_temporal_deltas_handle_axis_angle_wraparound(self) -> None:
        frames = [
            {
                "frame": 1,
                "instances": [
                    {
                        "class_name": "patient",
                        "ellipse": {
                            "center_x": 10.0,
                            "center_y": 20.0,
                            "major_diameter": 40.0,
                            "minor_diameter": 20.0,
                            "angle_degrees": 179.0,
                        },
                    }
                ],
            },
            {
                "frame": 2,
                "instances": [
                    {
                        "class_name": "patient",
                        "ellipse": {
                            "center_x": 13.0,
                            "center_y": 24.0,
                            "major_diameter": 44.0,
                            "minor_diameter": 18.0,
                            "angle_degrees": 1.0,
                        },
                    }
                ],
            },
        ]

        delta = _temporal_deltas(frames)[0]

        self.assertEqual(delta["center_displacement_pixels"], 5.0)
        self.assertAlmostEqual(delta["major_diameter_ratio"], 1.1)
        self.assertAlmostEqual(delta["minor_diameter_ratio"], 0.9)
        self.assertEqual(delta["angle_delta_degrees"], 2.0)


if __name__ == "__main__":
    unittest.main()

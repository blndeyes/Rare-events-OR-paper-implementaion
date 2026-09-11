import unittest
from unittest.mock import patch

import numpy as np

import or_video_reproduction.preprocessing.vda_depth as vda_depth
from or_video_reproduction.preprocessing.vda_depth import validate_depth_output


class VideoDepthTests(unittest.TestCase):
    def test_validates_paper_shape_and_finite_values(self) -> None:
        depths = np.zeros((2, 3, 4), dtype=np.float32)
        with (
            patch.object(vda_depth, "OUTPUT_FRAME_COUNT", 2),
            patch.object(vda_depth, "TARGET_HEIGHT", 3),
            patch.object(vda_depth, "TARGET_WIDTH", 4),
        ):
            result = validate_depth_output(depths, 24)
        self.assertTrue(result["passed"])

    def test_rejects_wrong_shape_fps_and_nonfinite_values(self) -> None:
        depths = np.zeros((2, 3, 4), dtype=np.float32)
        depths[0, 0, 0] = np.inf
        result = validate_depth_output(depths, 23.976)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["errors"]), 3)


if __name__ == "__main__":
    unittest.main()

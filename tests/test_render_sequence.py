import unittest

import numpy as np

from or_video_reproduction.geometry.render_sequence import render_frame


class RenderSequenceTests(unittest.TestCase):
    def test_renders_only_ellipses_on_black(self) -> None:
        labels = np.zeros((12, 16), dtype=np.uint8)
        labels[2:7, 2:7] = 1
        labels[5:10, 10:15] = 4
        depth = np.full(labels.shape, 5.0, dtype=np.float32)
        depth[labels == 1] = 10.0
        depth[labels == 4] = 2.0

        image, instances, skipped = render_frame(labels, depth, smaller_is_nearer=False)

        self.assertEqual(image.shape, (12, 16, 3))
        self.assertEqual(len(instances), 2)
        self.assertEqual(skipped, [])
        self.assertTrue(np.all(image[0, 0] == 0))
        blue_by_class = {item.class_name: item.normalized_depth for item in instances}
        self.assertEqual(blue_by_class["instrument_table"], 1.0)
        self.assertEqual(blue_by_class["mps_station"], 0.0)

    def test_rejects_misaligned_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "must align"):
            render_frame(
                np.zeros((8, 8), dtype=np.uint8),
                np.zeros((7, 8), dtype=np.float32),
                smaller_is_nearer=False,
            )


if __name__ == "__main__":
    unittest.main()

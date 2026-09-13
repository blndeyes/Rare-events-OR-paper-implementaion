import unittest

import numpy as np

from or_video_reproduction.geometry.render_sequence import render_frame


class RenderSequenceTests(unittest.TestCase):
    def test_external_instance_ids_use_explicit_class_mapping(self) -> None:
        labels = np.zeros((32, 32), dtype=np.uint8)
        labels[4:16, 3:12] = 41
        labels[16:29, 19:30] = 42
        depth = np.ones((32, 32), dtype=np.float32)

        _, instances, skipped = render_frame(
            labels,
            depth,
            smaller_is_nearer=False,
            label_classes={41: "head_surgeon", 42: "head_surgeon"},
        )

        self.assertEqual(len(instances), 2)
        self.assertEqual({item.key for item in instances}, {"41:head_surgeon", "42:head_surgeon"})
        self.assertEqual(skipped, [])

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

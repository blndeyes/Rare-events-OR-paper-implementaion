import unittest

import numpy as np

from or_video_reproduction.data.semantics import MMOR_ARTIFACT_LABELS
from or_video_reproduction.preprocessing.sam2_propagation import (
    compose_label_frame,
    tracked_labels,
)


class Sam2PropagationTests(unittest.TestCase):
    def test_tracks_only_known_entity_labels(self) -> None:
        artifact = next(iter(MMOR_ARTIFACT_LABELS))
        labels = np.array([[0, 1, artifact, 255]], dtype=np.uint8)
        self.assertEqual(tracked_labels(labels), [1])

    def test_composes_positive_logits_and_resolves_overlap(self) -> None:
        logits = np.array(
            [
                [[1.0, 0.1], [-0.2, 0.4]],
                [[0.5, 0.8], [-0.1, 0.4]],
            ],
            dtype=np.float32,
        )
        labels = compose_label_frame([3, 7], logits)
        np.testing.assert_array_equal(labels, [[3, 7], [0, 3]])

    def test_rejects_mismatched_object_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "Object id count"):
            compose_label_frame([1], np.zeros((2, 3, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

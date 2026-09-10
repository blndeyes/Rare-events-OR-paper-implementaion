import json
import unittest

from or_video_reproduction.data.semantics import (
    ENTITY_CLASSES,
    EXCLUDED_SPATIAL_PREDICATES,
    PREDICATE_CLASSES,
    summary,
    validate_profile,
    vocabulary,
)


class Paper36SemanticsTests(unittest.TestCase):
    def test_profile_is_exactly_21_entities_plus_15_predicates(self) -> None:
        validate_profile()

        self.assertEqual(len(ENTITY_CLASSES), 21)
        self.assertEqual(len(PREDICATE_CLASSES), 15)
        self.assertEqual(len(vocabulary()), 36)
        self.assertEqual(len(set(vocabulary())), 36)

    def test_close_to_is_explicitly_excluded(self) -> None:
        self.assertEqual(EXCLUDED_SPATIAL_PREDICATES, ("close_to",))
        self.assertNotIn("close_to", vocabulary())
        self.assertIn("lying_on", PREDICATE_CLASSES)

    def test_summary_is_json_serializable(self) -> None:
        payload = json.loads(json.dumps(summary()))

        self.assertEqual(payload["profile"], "paper_36")
        self.assertEqual(payload["semantic_label_count"], 36)


if __name__ == "__main__":
    unittest.main()

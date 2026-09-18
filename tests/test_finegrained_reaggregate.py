import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from or_video_reproduction.evaluation.finegrained.reaggregate import (
    ReaggregationError,
    assert_preserved_metrics,
    reaggregate_group,
    run_reaggregation,
)
from or_video_reproduction.evaluation.finegrained.control import reaggregate_control_clip


CLIP_METRIC = {
    "status": "ok",
    "clip_cmmd_unbiased_rbf_x1000": 66.2110943037062,
    "model": {"identifier": "openai/clip-vit-large-patch14-336", "revision": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"},
}
KID_METRIC = {"status": "ok", "mean": 0.058175140581767835, "std": 0.004728916150153844}
DINO2_METRIC = {"status": "ok", "frechet": 96.58930895040808, "density": 0.14375, "coverage": 0.15625}
DINO3_METRIC = {"status": "ok", "mmd_unbiased_rbf_x1000": 3.2942399994615457}


def _group() -> dict:
    per_clip = [
        {
            "id": "clip-a",
            "real": {
                "frames_with_detection": 16,
                "frame_count": 16,
                "frame_detection_rate": 1.0,
                "detection_count": 50,
                "clips_or_frames_without_detection": 0,
            },
            "generated": {
                "frames_with_detection": 16,
                "frame_count": 16,
                "frame_detection_rate": 1.0,
                "detection_count": 48,
                "clips_or_frames_without_detection": 0,
            },
            "mpjpe": {"status": "ok", "mpjpe_pixels": 16.5},
        },
        {
            "id": "clip-b",
            "real": {
                "frames_with_detection": 16,
                "frame_count": 16,
                "frame_detection_rate": 1.0,
                "detection_count": 40,
                "clips_or_frames_without_detection": 0,
            },
            "generated": {
                "frames_with_detection": 8,
                "frame_count": 16,
                "frame_detection_rate": 0.5,
                "detection_count": 10,
                "clips_or_frames_without_detection": 8,
            },
            "mpjpe": {"status": "unavailable"},
        },
    ]
    return {
        "comparison_class": "six_video_distributional",
        "metrics": {
            "clip_cmmd": copy.deepcopy(CLIP_METRIC),
            "clip_kid": copy.deepcopy(KID_METRIC),
            "dinov2": copy.deepcopy(DINO2_METRIC),
            "dinov3": copy.deepcopy(DINO3_METRIC),
            "anatomy": {
                "status": "ok",
                "parts": {
                    "person_detection": {
                        "status": "ok",
                        "detector": "facebook/detr-resnet-50",
                        "real": {
                            "frames_with_detection": 16,
                            "frame_count": 32,
                            "frame_detection_rate": 0.5,
                            "detection_count": 90,
                            "clips_or_frames_without_detection": 16,
                        },
                        "generated": {
                            "frames_with_detection": 16,
                            "frame_count": 32,
                            "frame_detection_rate": 0.5,
                            "detection_count": 58,
                        },
                        "per_clip": per_clip,
                    },
                    "mpjpe": {"status": "ok", "per_clip": per_clip},
                    "vbench2_human_anatomy": {"status": "unavailable", "reason": "--vbench-root was not provided"},
                },
            },
            "control": {
                "status": "ok",
                "class_consistency": 0.0,
                "aggregate_entity_detection_rate": 0.4,
                "mean_centroid_error_pixels": 12.0,
                "per_clip": [
                    {
                        "status": "ok",
                        "clip_id": "clip-a",
                        "class_consistency": 0.0,
                        "entity_detection_rate": 1.0,
                        "unmatched_control_tracks": 2,
                        "unmatched_generated_tracks": 0,
                        "mean_centroid_error_pixels": 12.0,
                        "per_entity": [
                            {
                                "control_class": "nurse",
                                "generated_class": "person",
                                "class_consistent": False,
                                "mean_centroid_error_pixels": 12.0,
                                "endpoint_error_pixels": 15.0,
                                "relative_depth_order_agreement": 0.0,
                            }
                        ],
                    }
                ],
            },
        },
    }


class ReaggregationTests(unittest.TestCase):
    def test_anatomy_rates_are_clip_scoped_and_clip_metrics_are_untouched(self) -> None:
        original = {"4dor_step600": _group()}
        before = copy.deepcopy(original)
        updated = reaggregate_group(original["4dor_step600"])
        person = updated["metrics"]["anatomy"]["parts"]["person_detection"]
        self.assertEqual(person["real"]["frames_with_detection"], 32)
        self.assertEqual(person["real"]["frame_count"], 32)
        self.assertAlmostEqual(person["real"]["frame_detection_rate"], 1.0)
        self.assertEqual(person["generated"]["frames_with_detection"], 24)
        self.assertAlmostEqual(person["generated"]["frame_detection_rate"], 0.75)
        self.assertEqual(updated["metrics"]["clip_cmmd"], CLIP_METRIC)
        self.assertEqual(updated["metrics"]["clip_kid"], KID_METRIC)
        self.assertEqual(updated["metrics"]["dinov2"], DINO2_METRIC)
        self.assertEqual(updated["metrics"]["dinov3"], DINO3_METRIC)
        assert_preserved_metrics(before, {"4dor_step600": updated})

    def test_control_reaggregation_drops_incompatible_geometry(self) -> None:
        updated = reaggregate_control_clip(_group()["metrics"]["control"]["per_clip"][0])
        self.assertEqual(updated["class_consistency"]["status"], "unavailable")
        self.assertEqual(updated["entity_detection_rate"], 0.0)
        self.assertIsNone(updated["mean_centroid_error_pixels"])
        self.assertEqual(updated["entity_counts"]["unsupported_control_entities_by_class"], {"nurse": 1})
        self.assertEqual(updated["entity_counts"]["matched_control_entity_count"], 0)
        self.assertIn("unmatched control tracks were stored only as a count", updated["insufficient_for_complete_class_counts"][0])
        self.assertIsNone(updated["per_entity"][0]["mean_centroid_error_pixels"])

    def test_malformed_anatomy_is_refused(self) -> None:
        group = _group()
        del group["metrics"]["anatomy"]["parts"]["person_detection"]["per_clip"]
        with self.assertRaises(ReaggregationError):
            reaggregate_group(group)

    def test_run_does_not_erase_earlier_metrics_or_overwrite_source(self) -> None:
        with TemporaryDirectory() as raw:
            source = Path(raw) / "source"
            output = Path(raw) / "reagg"
            source.mkdir()
            payload = {"4dor_step600": _group()}
            (source / "per-clip-results.json").write_text(json.dumps(payload), encoding="utf-8")
            (source / "evaluation-manifest.json").write_text(
                json.dumps(
                    {
                        "pairing_rule": "identity pairing",
                        "wan_in_frozen_4dor_split": False,
                        "wan_note": "diagnostic",
                        "input_root": "/tmp/inputs",
                        "groups": [{"id": "4dor_step600", "comparison_class": "six_video_distributional", "clip_ids": ["clip-a", "clip-b"], "sample_count": 2}],
                    }
                ),
                encoding="utf-8",
            )
            run_reaggregation(source_root=source, output_root=output, reaggregation_commit="test")
            written = json.loads((output / "per-clip-results.json").read_text(encoding="utf-8"))
            self.assertEqual(written["4dor_step600"]["metrics"]["clip_cmmd"], CLIP_METRIC)
            self.assertEqual(written["4dor_step600"]["metrics"]["dinov3"], DINO3_METRIC)
            person = written["4dor_step600"]["metrics"]["anatomy"]["parts"]["person_detection"]
            self.assertAlmostEqual(person["real"]["frame_detection_rate"], 1.0)
            control = written["4dor_step600"]["metrics"]["control"]
            self.assertEqual(control["class_consistency"]["status"], "unavailable")
            self.assertEqual(control["recovery_scope"], "person_only_plus_independent_depth")
            source_again = json.loads((source / "per-clip-results.json").read_text(encoding="utf-8"))
            self.assertEqual(source_again["4dor_step600"]["metrics"]["anatomy"]["parts"]["person_detection"]["real"]["frame_detection_rate"], 0.5)
            with self.assertRaises(ReaggregationError):
                run_reaggregation(source_root=source, output_root=source)


if __name__ == "__main__":
    unittest.main()

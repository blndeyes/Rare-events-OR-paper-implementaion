import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from or_video_reproduction.evaluation.corrected_inference import (
    InferenceJob,
    build_inference_config,
    load_inference_jobs,
    validate_conditioning_videos,
)


class CorrectedInferenceTests(unittest.TestCase):
    def test_manifest_filters_heldout_and_preserves_pair_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pairs.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "id": "train",
                                "split": "train",
                                "target_video": "train-target.mp4",
                                "conditioning_video": "train-ellipse.mp4",
                            },
                            {
                                "id": "heldout-a",
                                "split": "heldout",
                                "target_video": "target-a.mp4",
                                "conditioning_video": "ellipse-a.mp4",
                            },
                            {
                                "id": "heldout-b",
                                "split": "heldout",
                                "target_video": "target-b.mp4",
                                "conditioning_video": "ellipse-b.mp4",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            jobs = load_inference_jobs(manifest)

        self.assertEqual([job.clip_id for job in jobs], ["heldout-a", "heldout-b"])
        self.assertEqual(jobs[0].target_video, root / "target-a.mp4")

    def test_config_includes_both_conditioning_sources_with_fixed_settings(self) -> None:
        jobs = [InferenceJob("a", Path("target.mp4"), Path("ellipse.mp4"))]
        official = {
            "model": {},
            "acceleration": {},
            "validation": {},
            "wandb": {},
            "hub": {},
        }
        result = build_inference_config(
            official,
            checkpoint=Path("step_000100.safetensors"),
            output_dir=Path("output"),
            jobs=jobs,
            first_frames=[Path("first.png")],
            prompt="fixed prompt",
            seed=42,
        )
        validation = result["validation"]

        self.assertEqual(validation["images"], ["first.png"])
        self.assertEqual(validation["reference_videos"], ["ellipse.mp4"])
        self.assertEqual(validation["inference_steps"], 50)
        self.assertEqual(validation["guidance_scale"], 3.5)
        self.assertEqual(validation["video_dims"], [1024, 768, 97])
        self.assertIsNone(result["acceleration"]["quantization"])

    def test_video_contract_rejects_side_by_side_output(self) -> None:
        jobs = [InferenceJob("a", Path("target.mp4"), Path("ellipse.mp4"))]
        with self.assertRaisesRegex(ValueError, "violates"):
            validate_conditioning_videos(
                jobs,
                probe=lambda _: {"width": 2048, "height": 768, "frames": 97, "fps": 24.0},
            )


if __name__ == "__main__":
    unittest.main()

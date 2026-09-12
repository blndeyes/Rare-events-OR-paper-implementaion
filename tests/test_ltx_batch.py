import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from or_video_reproduction.preprocessing import ltx_batch
from or_video_reproduction.preprocessing.ltx_batch import BatchJob, load_batch_jobs


class LtxBatchTests(unittest.TestCase):
    def test_can_reload_pipeline_between_bf16_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for index in range(2):
                manifest = root / f"clip-{index}.json"
                manifest.write_text(json.dumps({"paper_contract": {}}), encoding="utf-8")
                jobs.append(BatchJob(f"clip-{index}", manifest, 42))

            received_pipelines = []

            class FakeModule:
                @staticmethod
                def infer(*, config, pipeline):
                    received_pipelines.append(pipeline)
                    return object(), [str(config)]

            with (
                patch.object(ltx_batch, "validate_clip_manifest"),
                patch.object(ltx_batch, "_load_reusable_inference", return_value=FakeModule),
                patch.object(
                    ltx_batch,
                    "_inference_config",
                    side_effect=[root / "output-0.mp4", root / "output-1.mp4"],
                ),
                patch.object(ltx_batch, "video_matches_contract", return_value=True),
                patch.object(
                    ltx_batch,
                    "probe_video",
                    return_value={"width": 1024, "height": 768, "frames": 97, "fps": 24},
                ),
                patch.object(ltx_batch, "_release_pipeline_memory") as release,
            ):
                counts = ltx_batch.run_batch(
                    jobs,
                    dataset_root=root,
                    ltx_root=root,
                    output_root=root / "outputs",
                    prompt="fixed",
                    precision="bf16",
                    fail_fast=True,
                    reload_pipeline_after_each_clip=True,
                )

        self.assertEqual(counts, {"completed": 2, "skipped": 0, "failed": 0})
        self.assertEqual(received_pipelines, [None, None])
        self.assertEqual(release.call_count, 2)

    def test_loads_relative_manifests_and_seed_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "clips": [
                            {"id": "clip-001", "manifest": "clips/001.json"},
                            {"id": "clip-002", "manifest": "clips/002.json", "seed": 9},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            jobs = load_batch_jobs(batch, default_seed=7)
            self.assertEqual([job.clip_id for job in jobs], ["clip-001", "clip-002"])
            self.assertEqual([job.seed for job in jobs], [7, 9])
            self.assertEqual(jobs[0].manifest_path, (root / "clips/001.json").resolve())

    def test_rejects_unsafe_or_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / "batch.json"
            for clip_ids, message in [(["../bad"], "Unsafe"), (["same", "same"], "Duplicate")]:
                batch.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "clips": [
                                {"id": clip_id, "manifest": f"{index}.json"}
                                for index, clip_id in enumerate(clip_ids)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_batch_jobs(batch, default_seed=7)


if __name__ == "__main__":
    unittest.main()

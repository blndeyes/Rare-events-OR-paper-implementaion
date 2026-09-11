import json
from pathlib import Path
import tempfile
import unittest

from or_video_reproduction.preprocessing.ltx_batch import load_batch_jobs


class LtxBatchTests(unittest.TestCase):
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

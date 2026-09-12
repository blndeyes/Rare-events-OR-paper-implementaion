import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from or_video_reproduction.training.reduced_run import (
    PersistentCsvMetrics,
    audit_checkpoints,
    build_reduced_run_config,
    validate_loss_csv,
)


class _ReadableCheckpoint:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def keys(self):
        return ["transformer.block.lora_A.weight"]


class ReducedRunTests(unittest.TestCase):
    def test_config_is_fresh_bf16_and_keeps_all_six_checkpoints(self) -> None:
        official = {
            "model": {"load_checkpoint": "/old"},
            "lora": {"rank": 8, "alpha": 8},
            "conditioning": {},
            "optimization": {},
            "acceleration": {"quantization": "int2-quanto"},
            "data": {},
            "validation": {},
            "checkpoints": {},
            "wandb": {},
            "hub": {},
        }
        result = build_reduced_run_config(
            official, precomputed_root=Path("/data"), output_dir=Path("/output")
        )

        self.assertIsNone(result["model"]["load_checkpoint"])
        self.assertEqual(result["optimization"]["steps"], 600)
        self.assertEqual(result["optimization"]["learning_rate"], 2e-4)
        self.assertEqual(result["lora"]["rank"], 128)
        self.assertEqual(result["lora"]["alpha"], 128)
        self.assertEqual(result["acceleration"]["mixed_precision_mode"], "bf16")
        self.assertIsNone(result["acceleration"]["quantization"])
        self.assertEqual(result["checkpoints"], {"interval": 100, "keep_last_n": -1})
        self.assertIsNone(result["validation"]["interval"])

    def test_metrics_are_fsynced_per_step_and_validate_exact_sequence(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "loss.csv"
            forwarded = []
            with PersistentCsvMetrics(path) as logger:
                wrapped = logger.wrap(forwarded.append)
                for step in range(1, 4):
                    wrapped(
                        {
                            "train/global_step": step,
                            "train/loss": 1.0 / step,
                            "train/learning_rate": 2e-4,
                            "train/step_time": 0.5,
                        }
                    )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            audit = validate_loss_csv(path, expected_steps=3)

        self.assertEqual(len(forwarded), 3)
        self.assertEqual([int(row["global_step"]) for row in rows], [1, 2, 3])
        self.assertTrue(audit["passed"])

    def test_checkpoint_audit_rejects_truncation_and_missing_steps(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lora_weights_step_00100.safetensors").write_bytes(b"x" * 10)
            report = audit_checkpoints(
                root,
                expected_steps=(100, 200),
                minimum_bytes=20,
                opener=lambda _: _ReadableCheckpoint(),
            )

        self.assertFalse(report["passed"])
        self.assertIn("below", report["samples"][0]["errors"][0])
        self.assertEqual(report["samples"][1]["errors"], ["missing checkpoint"])


if __name__ == "__main__":
    unittest.main()

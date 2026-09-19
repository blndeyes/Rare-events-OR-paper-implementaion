import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from or_video_reproduction.evaluation.checkpoint_audit import audited_checkpoint_step
from or_video_reproduction.evaluation.trajectory_experiment import _require_checkpoint_step
from or_video_reproduction.evaluation.video import sha256_file


def write_report(path: Path, checkpoint: Path, *, step: int, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "steps": step,
                "checkpoint_audit": {
                    "passed": passed,
                    "samples": [
                        {
                            "path": str(checkpoint),
                            "step": step,
                            "passed": passed,
                            "size_bytes": checkpoint.stat().st_size,
                            "tensor_count": 960,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


class CheckpointAuditTests(unittest.TestCase):
    def test_reads_step_2000_from_training_report(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "lora_weights_step_02000.safetensors"
            checkpoint.write_bytes(b"lora-2000")
            report = root / "training-report.json"
            write_report(report, checkpoint, step=2000)

            result = audited_checkpoint_step(
                training_report=report,
                checkpoint=checkpoint,
                expected_step=2000,
                expected_sha256=sha256_file(checkpoint),
            )

            self.assertEqual(result["step"], 2000)
            self.assertEqual(result["sha256"], sha256_file(checkpoint))

    def test_rejects_mismatched_expected_step(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "lora_weights_step_02000.safetensors"
            checkpoint.write_bytes(b"lora-2000")
            report = root / "training-report.json"
            write_report(report, checkpoint, step=2000)
            with self.assertRaisesRegex(ValueError, "expected 600"):
                audited_checkpoint_step(
                    training_report=report,
                    checkpoint=checkpoint,
                    expected_step=600,
                )

    def test_rejects_filename_step_disagreement(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "lora_weights_step_00600.safetensors"
            checkpoint.write_bytes(b"wrong-name")
            report = root / "training-report.json"
            write_report(report, checkpoint, step=2000)
            with self.assertRaisesRegex(ValueError, "Filename step"):
                audited_checkpoint_step(
                    training_report=report,
                    checkpoint=checkpoint,
                    expected_step=2000,
                )

    def test_rejects_unverifiable_step_without_training_report(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "lora_weights_step_02000.safetensors"
            checkpoint.write_bytes(b"lora")
            with self.assertRaisesRegex(ValueError, "unverifiable"):
                _require_checkpoint_step(
                    checkpoint=checkpoint,
                    expected_checkpoint_step=2000,
                    training_report=None,
                    expected_sha256=None,
                )

    def test_step_600_does_not_require_a_training_report(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "lora_weights_step_00600.safetensors"
            checkpoint.write_bytes(b"lora")
            self.assertIsNone(
                _require_checkpoint_step(
                    checkpoint=checkpoint,
                    expected_checkpoint_step=600,
                    training_report=None,
                    expected_sha256=None,
                )
            )


if __name__ == "__main__":
    unittest.main()

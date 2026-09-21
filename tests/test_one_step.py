import os
import tempfile
import unittest
from pathlib import Path

from or_video_reproduction.training.ic_lora_smoke import (
    ensure_trainer_venv_bin_on_path,
    trainer_venv_bin_dir,
)
from or_video_reproduction.training.one_step import build_one_step_config, build_parser
from or_video_reproduction.training.profiles import load_training_profile


class OneStepTests(unittest.TestCase):
    def test_applies_gate_overrides_without_changing_lora(self) -> None:
        official = {
            "optimization": {"steps": 2000},
            "acceleration": {"quantization": None, "mixed_precision_mode": "bf16"},
            "data": {"preprocessed_data_root": "/old", "num_dataloader_workers": 2},
            "validation": {"interval": 250, "video_dims": [512, 512, 81]},
            "checkpoints": {"interval": 250},
            "wandb": {"enabled": True},
            "lora": {"rank": 128},
            "output_dir": "/old-output",
        }
        paper = {"experiment": {"target_resolution": [1024, 768], "frames": 97}}
        result = build_one_step_config(
            official,
            paper,
            precomputed_root="/data",
            output_dir="/output",
            quantization="int2-quanto",
            mixed_precision="bf16",
        )
        self.assertEqual(result["optimization"]["steps"], 1)
        self.assertEqual(result["validation"]["video_dims"], [1024, 768, 97])
        self.assertEqual(result["lora"]["rank"], 128)
        self.assertEqual(result["acceleration"]["quantization"], "int2-quanto")
        self.assertIsNone(result["validation"]["interval"])

    def test_profiles_switch_between_faithful_and_integration_modes(self) -> None:
        source = """\
schema_version: 1
profiles:
  faithful_bf16:
    description: faithful
    quantization: no_change
    mixed_precision: bf16
    transformer_load_dtype: bf16
    paper_faithful: true
  integration:
    description: fallback
    quantization: int2-quanto
    mixed_precision: bf16
    transformer_load_dtype: bf16
    paper_faithful: false
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.yaml"
            path.write_text(source, encoding="utf-8")
            faithful = load_training_profile(path, "faithful_bf16")
            integration = load_training_profile(path, "integration")

        self.assertTrue(faithful.paper_faithful)
        self.assertEqual(faithful.quantization, "no_change")
        self.assertFalse(integration.paper_faithful)
        self.assertEqual(integration.quantization, "int2-quanto")

    def test_faithful_profile_maps_no_change_to_official_null(self) -> None:
        official = {
            "optimization": {"steps": 2000},
            "acceleration": {"quantization": "int2-quanto", "mixed_precision_mode": "bf16"},
            "data": {"preprocessed_data_root": "/old", "num_dataloader_workers": 2},
            "validation": {"interval": 250, "video_dims": [512, 512, 81]},
            "checkpoints": {"interval": 250},
            "wandb": {"enabled": True},
            "lora": {"rank": 128},
            "output_dir": "/old-output",
        }
        paper = {"experiment": {"target_resolution": [1024, 768], "frames": 97}}

        result = build_one_step_config(
            official,
            paper,
            precomputed_root="/data",
            output_dir="/output",
            quantization="no_change",
            mixed_precision="bf16",
        )

        self.assertIsNone(result["acceleration"]["quantization"])

    def test_rejects_quantized_profile_marked_faithful(self) -> None:
        source = """\
schema_version: 1
profiles:
  invalid:
    description: invalid
    quantization: int8-quanto
    mixed_precision: bf16
    transformer_load_dtype: bf16
    paper_faithful: true
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unquantized BF16"):
                load_training_profile(path, "invalid")

    def test_patchgan_hardware_gate_is_explicitly_opt_in(self) -> None:
        args = build_parser().parse_args(
            [
                "--paper-config",
                "paper.yaml",
                "--trainer-root",
                "trainer",
                "--precomputed-root",
                "data",
                "--output-dir",
                "output",
                "--report",
                "report.json",
                "--profile",
                "integration",
                "--patchgan-config",
                "patchgan.yaml",
            ]
        )

        self.assertEqual(args.patchgan_config, Path("patchgan.yaml"))

    def test_venv_ninja_is_prepended_to_path_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer_root = Path(directory)
            venv_bin = trainer_venv_bin_dir(trainer_root)
            venv_bin.mkdir(parents=True)
            ninja = venv_bin / ("ninja.exe" if os.name == "nt" else "ninja")
            ninja.write_bytes(b"")
            original = os.environ.get("PATH", "")
            try:
                located = ensure_trainer_venv_bin_on_path(trainer_root)
                self.assertEqual(located, venv_bin.resolve())
                self.assertTrue(
                    os.environ["PATH"].startswith(str(venv_bin.resolve()) + os.pathsep)
                    or os.environ["PATH"] == str(venv_bin.resolve())
                )
            finally:
                os.environ["PATH"] = original

    def test_missing_venv_ninja_does_not_change_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = os.environ.get("PATH", "")
            try:
                located = ensure_trainer_venv_bin_on_path(Path(directory))
                self.assertIsNone(located)
                self.assertEqual(os.environ.get("PATH", ""), original)
            finally:
                os.environ["PATH"] = original


if __name__ == "__main__":
    unittest.main()

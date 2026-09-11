import unittest
from pathlib import Path
import tempfile

from or_video_reproduction.training.one_step import build_one_step_config
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


if __name__ == "__main__":
    unittest.main()

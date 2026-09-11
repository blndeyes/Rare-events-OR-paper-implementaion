import unittest

from or_video_reproduction.training.one_step import build_one_step_config


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


if __name__ == "__main__":
    unittest.main()

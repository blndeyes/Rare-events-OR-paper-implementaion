import copy
import unittest

from or_video_reproduction.training.ic_lora_smoke import alignment_report


PAPER = {
    "model": {
        "checkpoint": "LTXV_13B_097_DEV",
        "training_mode": "lora",
        "conditioning_mode": "reference_video",
        "lora": {"rank": 128, "alpha": 128, "target_modules": ["to_q"]},
    },
    "training": {
        "learning_rate": 2e-4,
        "batch_size": 1,
        "first_frame_conditioning_probability": 0.2,
        "mixed_precision": "bf16",
    },
    "inference": {"denoising_steps": 50, "guidance_scale": 3.5},
}
OFFICIAL = {
    "model": {"model_source": "LTXV_13B_097_DEV", "training_mode": "lora"},
    "conditioning": {"mode": "reference_video", "first_frame_conditioning_p": 0.2},
    "lora": {"rank": 128, "alpha": 128, "target_modules": ["to_q"]},
    "optimization": {"learning_rate": "2e-4", "batch_size": 1},
    "acceleration": {"mixed_precision_mode": "bf16"},
    "validation": {"inference_steps": 50, "guidance_scale": 3.5},
}


class IcLoraSmokeTests(unittest.TestCase):
    def test_matching_configuration_passes(self) -> None:
        report = alignment_report(PAPER, OFFICIAL)
        self.assertTrue(report["all_match"])

    def test_mismatch_is_named(self) -> None:
        changed = copy.deepcopy(OFFICIAL)
        changed["lora"]["rank"] = 64
        report = alignment_report(PAPER, changed)
        self.assertFalse(report["all_match"])
        self.assertFalse(report["checks"]["lora_rank"])


if __name__ == "__main__":
    unittest.main()

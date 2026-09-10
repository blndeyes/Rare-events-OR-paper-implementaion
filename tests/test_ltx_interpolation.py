import unittest
from pathlib import Path

from or_video_reproduction.preprocessing.ltx_interpolation import (
    PIPELINE_CONFIG,
    build_interpolation_command,
)


class LtxInterpolationPlanTests(unittest.TestCase):
    def _manifest(self) -> dict[str, object]:
        return {
            "paper_contract": {
                "output_frames": 97,
                "output_fps": 24,
                "resolution": [1024, 768],
            },
            "clip": {
                "source_keyframes": [
                    {"rgb_path": f"take/frame-{index}.jpg", "output_frame_index": frame}
                    for index, frame in enumerate((0, 24, 48, 72, 96))
                ]
            },
        }

    def test_builds_official_multi_keyframe_shape(self) -> None:
        command = build_interpolation_command(
            self._manifest(),
            dataset_root=Path("/data/mmor"),
            ltx_root=Path("/upstreams/ltx-video"),
            python_executable=Path("/env/bin/python"),
            output_dir=Path("/outputs/smoke"),
            prompt="Fixed overhead view of an operating room.",
            seed=42,
            verify_revision=False,
        )

        starts_at = command.index("--conditioning_start_frames") + 1
        strengths_at = command.index("--conditioning_strengths")
        self.assertEqual(command[starts_at:strengths_at], ["0", "24", "48", "72", "96"])
        self.assertEqual(command[command.index("--num_frames") + 1], "97")
        self.assertEqual(command[command.index("--frame_rate") + 1], "24")
        self.assertEqual(command[command.index("--height") + 1], "768")
        self.assertEqual(command[command.index("--width") + 1], "1024")
        self.assertIn("--offload_to_cpu", command)
        pipeline_path = Path(command[command.index("--pipeline_config") + 1])
        self.assertEqual(pipeline_path.name, Path(PIPELINE_CONFIG).name)

    def test_requires_explicit_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit interpolation prompt"):
            build_interpolation_command(
                self._manifest(),
                dataset_root=Path("/data/mmor"),
                ltx_root=Path("/upstreams/ltx-video"),
                python_executable=Path("/env/bin/python"),
                output_dir=Path("/outputs/smoke"),
                prompt=" ",
                seed=42,
                verify_revision=False,
            )


if __name__ == "__main__":
    unittest.main()

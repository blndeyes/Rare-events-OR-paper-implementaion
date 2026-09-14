import copy
import hashlib
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from or_video_reproduction.geometry.ellipse import Ellipse
from or_video_reproduction.geometry.trajectory import (
    clip_ellipse_to_frame,
    edit_trajectory,
    interpolate_waypoints,
    list_instances,
    load_trajectory,
    render_edited_sequence,
    rendered_frames,
    select_instance,
    simplify_waypoints,
)


def find_ffmpeg() -> Path:
    if executable := shutil.which("ffmpeg"):
        return Path(executable)
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        return Path(r"C:\Program Files\Softdeluxe\Free Download Manager\ffmpeg.exe")


FFMPEG = find_ffmpeg()


def supports_h264_encoder() -> bool:
    if not FFMPEG.is_file():
        return False
    result = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and "libx264" in result.stdout


def instance(
    key: str,
    class_name: str,
    x: float,
    y: float,
    *,
    depth: float,
    major: float = 80.0,
    minor: float = 40.0,
    angle: float = 20.0,
) -> dict[str, object]:
    return {
        "key": key,
        "class_name": class_name,
        "ellipse": {
            "center_x": x,
            "center_y": y,
            "major_diameter": major,
            "minor_diameter": minor,
            "angle_degrees": angle,
            "source_pixels": 1234,
        },
        "raw_relative_depth_mean": depth,
        "normalized_nearness": depth / 10.0,
        "red_green": [1, 2],
    }


def sequence() -> dict[str, object]:
    frames = []
    for frame_index in range(97):
        frames.append(
            {
                "frame_index": frame_index,
                "instances": [
                    instance(
                        "10:head_surgeon",
                        "head_surgeon",
                        200.0 + frame_index,
                        300.0,
                        depth=8.0,
                    ),
                    instance(
                        "20:instrument_table",
                        "instrument_table",
                        600.0,
                        350.0,
                        depth=4.0,
                        major=120.0,
                        minor=70.0,
                        angle=0.0,
                    ),
                ],
                "skipped": [],
            }
        )
    return {
        "schema_version": 1,
        "kind": "paper_ellipse_only_geometric_conditioning",
        "shape": [97, 768, 1024],
        "fps": 24,
        "depth_direction": "larger_is_nearer",
        "frames": frames,
    }


class TrajectoryCoreTests(unittest.TestCase):
    def test_instance_listing_exposes_exact_selection_geometry(self) -> None:
        rows = list_instances(sequence())

        self.assertEqual(
            [row["instance_id"] for row in rows],
            ["10:head_surgeon", "20:instrument_table"],
        )
        self.assertEqual(rows[0]["class_name"], "head_surgeon")
        self.assertEqual(rows[0]["centroid"], [200.0, 300.0])
        self.assertEqual(rows[0]["major_diameter"], 80.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            list_instances(sequence(), frame_index=97)

    def test_selection_by_id_and_click_uses_topmost_ellipse(self) -> None:
        metadata = sequence()
        metadata["frames"][0]["instances"][1]["ellipse"].update(
            {"center_x": 200.0, "center_y": 300.0}
        )

        self.assertEqual(
            select_instance(metadata, instance_id="10:head_surgeon"),
            ("10:head_surgeon", "head_surgeon"),
        )
        # Larger raw depth is drawn last for this metadata and is therefore selected.
        self.assertEqual(
            select_instance(metadata, click=(200.0, 300.0)),
            ("10:head_surgeon", "head_surgeon"),
        )
        with self.assertRaisesRegex(ValueError, "No ellipse"):
            select_instance(metadata, click=(0.0, 0.0))

    def test_simplification_and_arc_length_interpolation_are_exact(self) -> None:
        simplified = simplify_waypoints([(0.0, 0.0), (1.0, 0.01), (2.0, 0.0), (2.0, 2.0)], 0.1)
        self.assertEqual(simplified, [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)])
        result = interpolate_waypoints(simplified)
        self.assertEqual(len(result), 97)
        self.assertEqual(result[0], (0.0, 0.0))
        self.assertEqual(result[48], (2.0, 0.0))
        self.assertEqual(result[-1], (2.0, 2.0))
        distances = np.linalg.norm(np.diff(np.asarray(result), axis=0), axis=1)
        np.testing.assert_allclose(distances, np.full(96, 4.0 / 96.0), atol=1e-12)

    def test_replace_moves_only_selected_centroid_and_preserves_attributes(self) -> None:
        metadata = sequence()
        original = copy.deepcopy(metadata)
        edited, manifest = edit_trajectory(
            metadata,
            source_conditioning=Path("conditioning.mp4"),
            source_metadata=Path("metadata.json"),
            instance_id="10:head_surgeon",
            waypoints=[(200.0, 300.0), (300.0, 320.0), (420.0, 350.0)],
            mode="replace",
            simplify_epsilon_pixels=0.0,
            seed=17,
        )

        self.assertEqual(len(manifest["edited_centroids"]), 97)
        self.assertEqual(manifest["mode"], "replace")
        self.assertEqual(manifest["random_seed"], 17)
        self.assertEqual(manifest["edited_centroids"][0], [200.0, 300.0])
        self.assertEqual(manifest["edited_centroids"][-1], [420.0, 350.0])
        for frame_index in range(97):
            self.assertEqual(
                edited["frames"][frame_index]["instances"][1],
                original["frames"][frame_index]["instances"][1],
            )
            before = original["frames"][frame_index]["instances"][0]
            after = edited["frames"][frame_index]["instances"][0]
            self.assertEqual(before["key"], after["key"])
            self.assertEqual(before["class_name"], after["class_name"])
            self.assertEqual(before["raw_relative_depth_mean"], after["raw_relative_depth_mean"])
            self.assertEqual(before["normalized_nearness"], after["normalized_nearness"])
            for field in (
                "major_diameter",
                "minor_diameter",
                "angle_degrees",
                "source_pixels",
            ):
                self.assertEqual(before["ellipse"][field], after["ellipse"][field])

    def test_offset_preserves_source_motion(self) -> None:
        metadata = sequence()
        edited, _ = edit_trajectory(
            metadata,
            source_conditioning=Path("conditioning.mp4"),
            source_metadata=Path("metadata.json"),
            instance_id="10:head_surgeon",
            waypoints=[(200.0, 300.0), (200.0, 400.0)],
            mode="offset",
            simplify_epsilon_pixels=0.0,
        )
        last = edited["frames"][-1]["instances"][0]["ellipse"]
        self.assertEqual(last["center_x"], 296.0)
        self.assertEqual(last["center_y"], 400.0)

    def test_boundary_clipping_keeps_rotated_ellipse_inside(self) -> None:
        ellipse = Ellipse(-50.0, 900.0, 100.0, 40.0, 45.0, 100)
        clipped, event = clip_ellipse_to_frame(ellipse)
        self.assertIsNotNone(event)
        theta = np.radians(clipped.angle_degrees)
        a, b = clipped.major_diameter / 2.0, clipped.minor_diameter / 2.0
        extent_x = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
        extent_y = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
        self.assertGreaterEqual(clipped.center_x - extent_x, -1e-10)
        self.assertLessEqual(clipped.center_x + extent_x, 1024 + 1e-10)
        self.assertGreaterEqual(clipped.center_y - extent_y, -1e-10)
        self.assertLessEqual(clipped.center_y + extent_y, 768 + 1e-10)

    def test_rendered_frames_are_deterministic_black_background(self) -> None:
        metadata = sequence()
        first = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in rendered_frames(metadata)]
        second = [
            hashlib.sha256(frame.tobytes()).hexdigest() for frame in rendered_frames(metadata)
        ]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 97)
        frame = next(iter(rendered_frames(metadata)))
        self.assertEqual(frame.shape, (768, 1024, 3))
        np.testing.assert_array_equal(frame[0, 0], np.zeros(3, dtype=np.uint8))

    @unittest.skipUnless(supports_h264_encoder(), "local ffmpeg lacks libx264")
    def test_mp4_render_is_byte_deterministic_and_valid_97_frame_output(self) -> None:
        metadata = sequence()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.mp4", root / "second.mp4"
            render_edited_sequence(metadata, first, ffmpeg=str(FFMPEG))
            render_edited_sequence(metadata, second, ffmpeg=str(FFMPEG))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            result = subprocess.run(
                [str(FFMPEG), "-hide_banner", "-i", str(first), "-f", "null", "-"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("1024x768", result.stderr)
            self.assertIn("24 fps", result.stderr)
            matches = re.findall(r"frame=\s*(\d+)", result.stderr)
            self.assertTrue(matches)
            self.assertEqual(int(matches[-1]), 97)

    def test_malformed_trajectory_input_is_rejected(self) -> None:
        cases = [
            "not-json",
            json.dumps({"schema_version": 2, "waypoints": [[1, 2], [3, 4]]}),
            json.dumps({"schema_version": 1, "waypoints": [[1, 2]]}),
            json.dumps({"schema_version": 1, "waypoints": [[1, 2], [float("nan"), 4]]}),
            json.dumps({"schema_version": 1, "waypoints": [[1, 2], [3, 4]], "mode": "bad"}),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            for payload in cases:
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_trajectory(path)

    def test_first_waypoint_must_begin_inside_selected_ellipse(self) -> None:
        with self.assertRaisesRegex(ValueError, "first waypoint"):
            edit_trajectory(
                sequence(),
                source_conditioning=Path("conditioning.mp4"),
                source_metadata=Path("metadata.json"),
                instance_id="10:head_surgeon",
                waypoints=[(0.0, 0.0), (400.0, 300.0)],
            )


if __name__ == "__main__":
    unittest.main()

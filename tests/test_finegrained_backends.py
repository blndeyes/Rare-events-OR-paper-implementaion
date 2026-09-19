from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

from or_video_reproduction.evaluation.finegrained.backends import (
    ModelUnavailable,
    detect_hands_mediapipe,
    select_hand_landmarker_api,
)
from or_video_reproduction.evaluation.finegrained.external import CommandResult
from or_video_reproduction.evaluation.finegrained.official import (
    inspect_fvmd_python,
    inspect_jedi,
    inspect_vbench_root,
    parse_fvmd_score,
    parse_jedi_driver_output,
    parse_vbench_results,
    run_fvmd,
    run_jedi,
    run_vbench_anatomy,
    write_fvmd_npy,
)
from or_video_reproduction.evaluation.finegrained.protocol import HAND_DETECTOR
from or_video_reproduction.evaluation.finegrained.scoring import (
    nearest_anchor_scores,
    summarize_hands_metric,
)


class FakeLandmark(SimpleNamespace):
    pass


class FakeLandmarker:
    def __init__(self, per_frame: list[list[list[FakeLandmark]]]) -> None:
        self.per_frame = per_frame
        self.index = 0
        self.closed = False

    def detect(self, _image: object) -> SimpleNamespace:
        hands = self.per_frame[self.index]
        self.index += 1
        return SimpleNamespace(hand_landmarks=hands, handedness=[[SimpleNamespace(score=0.9)] for _ in hands])

    def close(self) -> None:
        self.closed = True


class HandsBackendTests(unittest.TestCase):
    def test_tasks_detector_reports_boxes_and_failed_frames(self) -> None:
        frames = np.zeros((3, 40, 40, 3), dtype=np.uint8)
        hand = [FakeLandmark(x=0.25, y=0.25), FakeLandmark(x=0.5, y=0.5)]
        landmarker = FakeLandmarker([[hand], [], [hand]])
        with TemporaryDirectory() as directory:
            asset = Path(directory) / "hand_landmarker.task"
            asset.write_bytes(b"fake-model")
            payload = detect_hands_mediapipe(
                frames,
                model_asset_path=asset,
                landmarker_factory=lambda _path: landmarker,
            )
        self.assertEqual(payload["detector"], HAND_DETECTOR)
        self.assertEqual(payload["frames_with_hands"], 2)
        self.assertEqual(payload["failed_frame_count"], 1)
        self.assertEqual(len(payload["detections"]), 2)
        self.assertTrue(landmarker.closed)

    def test_missing_model_asset_is_blocked(self) -> None:
        frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ModelUnavailable) as caught:
            detect_hands_mediapipe(frames, model_asset_path=None)
        self.assertEqual(caught.exception.status, "blocked")
        self.assertIn("--hand-landmarker-model", caught.exception.reason)

    def test_missing_model_file_is_blocked(self) -> None:
        frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ModelUnavailable) as caught:
            detect_hands_mediapipe(frames, model_asset_path=Path("missing-hand_landmarker.task"))
        self.assertEqual(caught.exception.status, "blocked")

    def test_solutions_hands_is_not_a_silent_fallback(self) -> None:
        fake_mp = SimpleNamespace(
            __version__="1.0.1",
            solutions=SimpleNamespace(hands=object()),
        )
        with patch.dict(
            sys.modules,
            {
                "mediapipe": fake_mp,
                "mediapipe.tasks": None,
                "mediapipe.tasks.python": None,
                "mediapipe.tasks.python.vision": None,
            },
        ):
            with self.assertRaises(ModelUnavailable) as caught:
                select_hand_landmarker_api()
        self.assertIn("not the selected backend", caught.exception.reason)
        self.assertIn("HandLandmarker", caught.exception.reason)

    def test_missing_mediapipe_is_blocked(self) -> None:
        with patch.dict(sys.modules, {"mediapipe": None}):
            with self.assertRaises(ModelUnavailable) as caught:
                select_hand_landmarker_api()
        self.assertIn("not installed", caught.exception.reason)

    def test_no_detection_does_not_become_zero_or_one(self) -> None:
        empty = np.zeros((0, 4))
        scores = nearest_anchor_scores(empty, empty)
        self.assertEqual(scores["status"], "unavailable")
        self.assertIsNone(scores["mean_max_cosine"])
        self.assertIsNone(scores["mean_cosine"])
        summary = summarize_hands_metric(
            detector_identity=HAND_DETECTOR,
            detector_version="0.10.14",
            model_asset="/tmp/hand_landmarker.task",
            embedding_identity="openai/clip-vit-large-patch14-336",
            embedding_version=None,
            real_rate={"frame_detection_rate": 0.0, "detection_count": 0, "frame_count": 16},
            generated_rate={"frame_detection_rate": 0.0, "detection_count": 0, "frame_count": 16},
            real_crop_count=0,
            generated_crop_count=0,
            real_failed_frames=16,
            generated_failed_frames=16,
            per_clip=[],
            embedding_scores=scores,
        )
        self.assertEqual(summary["status"], "ok")
        self.assertIsNone(summary["score"])
        self.assertEqual(summary["score_status"], "unavailable")
        self.assertNotEqual(summary["score"], 0)
        self.assertNotEqual(summary["score"], 1)


class FvmdAdapterTests(unittest.TestCase):
    def test_missing_interpreter_is_blocked(self) -> None:
        payload = inspect_fvmd_python(None)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("--fvmd-python", payload["reason"])

    def test_probe_failure_is_blocked(self) -> None:
        def fake_probe(python: str, module: str) -> dict[str, object]:
            return {"available": False, "error": "module not found", "executable": python, "module": module}

        with patch(
            "or_video_reproduction.evaluation.finegrained.official.python_module_probe",
            fake_probe,
        ):
            payload = inspect_fvmd_python("/missing/python")
        self.assertEqual(payload["status"], "blocked")

    def test_successful_official_command_parses_json_score(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gen = write_fvmd_npy(root / "gen.npy", [np.zeros((2, 4, 4, 3), dtype=np.uint8)])
            ref = write_fvmd_npy(root / "ref.npy", [np.ones((2, 4, 4, 3), dtype=np.uint8)])
            log_dir = root / "logs"

            def fake_runner(command, **_kwargs):
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "fvmd.json").write_text("12.5\n", encoding="utf-8")
                return CommandResult(list(command), 0, "FVMD: 12.5\n", "", str(log_dir))

            with patch(
                "or_video_reproduction.evaluation.finegrained.official.python_module_probe",
                lambda python, module: {
                    "available": True,
                    "version": "1.0.0",
                    "path": "/opt/fvmd",
                    "python": "3.10",
                    "executable": python,
                    "module": module,
                },
            ):
                payload = run_fvmd(gen, ref, log_dir, python="/opt/fvmd-venv/bin/python", runner=fake_runner)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["score"], 12.5)
        self.assertEqual(payload["package"], "1.0.0")
        self.assertIn("-m", payload["command"])
        self.assertIn("fvmd", payload["command"])

    def test_subprocess_failure_is_failed_not_a_score(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_runner(command, **_kwargs):
                return CommandResult(list(command), 2, "", "sklearn mismatch", None)

            with patch(
                "or_video_reproduction.evaluation.finegrained.official.python_module_probe",
                lambda python, module: {"available": True, "version": "1.0.0", "executable": python, "module": module},
            ):
                payload = run_fvmd(
                    root / "g.npy",
                    root / "r.npy",
                    root / "logs",
                    python="/opt/fvmd-venv/bin/python",
                    runner=fake_runner,
                )
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("score", payload)
        self.assertIn("exited 2", payload["reason"])

    def test_malformed_output_is_failed(self) -> None:
        with TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with self.assertRaises(ModelUnavailable) as caught:
                parse_fvmd_score(log_dir=log_dir, stdout="no score here", stderr="")
        self.assertEqual(caught.exception.status, "failed")

    def test_non_finite_score_is_failed(self) -> None:
        with TemporaryDirectory() as directory:
            log_dir = Path(directory)
            (log_dir / "fvmd.json").write_text("NaN\n", encoding="utf-8")
            with self.assertRaises(ModelUnavailable) as caught:
                parse_fvmd_score(log_dir=log_dir, stdout="", stderr="")
        self.assertEqual(caught.exception.status, "failed")


class JediAdapterTests(unittest.TestCase):
    def _ok_probe(self, python: str, module: str) -> dict[str, object]:
        return {
            "available": True,
            "version": "1.1.0",
            "path": "/opt/videojedi",
            "executable": python,
            "module": module,
            "python": "3.10",
        }

    def test_missing_checkpoints_are_blocked(self) -> None:
        with TemporaryDirectory() as directory:
            payload = inspect_jedi(python=sys.executable, model_dir=Path(directory), config_path=None)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("vith16.pth.tar", payload["reason"])
        self.assertEqual(payload["role"], "relative_ranking_only")

    def test_missing_package_is_blocked(self) -> None:
        with patch(
            "or_video_reproduction.evaluation.finegrained.official.python_module_probe",
            lambda python, module: {"available": False, "error": "missing", "executable": python, "module": module},
        ):
            payload = inspect_jedi(python="/opt/jedi/bin/python", model_dir=None, config_path=None)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("videojedi", payload["reason"])

    def test_successful_driver_json(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "ckpts"
            model_dir.mkdir()
            (model_dir / "vith16.pth.tar").write_bytes(b"enc")
            (model_dir / "ssv2-probe.pth.tar").write_bytes(b"probe")
            config = root / "vith16_ssv2_16x2x3.yaml"
            config.write_text("pretrain: {}\n", encoding="utf-8")
            feature_path = root / "features"
            ref = root / "real.mp4"
            gen = root / "fake.mp4"
            ref.write_bytes(b"r")
            gen.write_bytes(b"g")

            def fake_runner(command, **_kwargs):
                feature_path.mkdir(parents=True, exist_ok=True)
                payload = {
                    "status": "ok",
                    "score": 3.25,
                    "feature_layer": "finetuned attentive pooler (ssv2-probe) when present",
                }
                (feature_path / "jedi-result.json").write_text(json.dumps(payload), encoding="utf-8")
                return CommandResult(list(command), 0, json.dumps(payload), "", str(feature_path))

            with patch(
                "or_video_reproduction.evaluation.finegrained.official.python_module_probe",
                self._ok_probe,
            ):
                result = run_jedi(
                    [ref],
                    [gen],
                    feature_path=feature_path,
                    python="/opt/jedi/bin/python",
                    model_dir=model_dir,
                    config_path=config,
                    runner=fake_runner,
                )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["score"], 3.25)
        self.assertEqual(result["role"], "relative_ranking_only")
        self.assertIn("not an absolute", result["warning"])
        self.assertIn("jedi_driver", " ".join(result["command"]))

    def test_subprocess_failure_is_failed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "ckpts"
            model_dir.mkdir()
            (model_dir / "vith16.pth.tar").write_bytes(b"enc")
            (model_dir / "ssv2-probe.pth.tar").write_bytes(b"probe")
            config = root / "cfg.yaml"
            config.write_text("x: 1\n", encoding="utf-8")

            def fake_runner(command, **_kwargs):
                return CommandResult(list(command), 1, "", "CUDA OOM", None)

            with patch(
                "or_video_reproduction.evaluation.finegrained.official.python_module_probe",
                self._ok_probe,
            ):
                payload = run_jedi(
                    [root / "a.mp4"],
                    [root / "b.mp4"],
                    feature_path=root / "out",
                    python="/opt/jedi/bin/python",
                    model_dir=model_dir,
                    config_path=config,
                    runner=fake_runner,
                )
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("score", payload)

    def test_malformed_driver_output_is_failed(self) -> None:
        with self.assertRaises(ModelUnavailable) as caught:
            parse_jedi_driver_output("not-json")
        self.assertEqual(caught.exception.status, "failed")


class VbenchAdapterTests(unittest.TestCase):
    def test_missing_root_is_blocked(self) -> None:
        payload = inspect_vbench_root(None)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("--vbench-root", payload["reason"])

    def test_vbench1_evaluate_py_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluate.py").write_text("from vbench import VBench\n", encoding="utf-8")
            payload = inspect_vbench_root(root)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("VBench-1.0", payload["reason"])

    def test_missing_evaluate_py_is_blocked(self) -> None:
        with TemporaryDirectory() as directory:
            payload = inspect_vbench_root(Path(directory))
        self.assertEqual(payload["status"], "blocked")

    def test_successful_official_command_records_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "VBench-2.0"
            (root / "vbench2").mkdir(parents=True)
            (root / "evaluate.py").write_text(
                "from vbench2 import VBench2\n# Human_Anatomy\n", encoding="utf-8"
            )
            video = Path(directory) / "clip.mp4"
            video.write_bytes(b"mp4")
            output_dir = Path(directory) / "out"

            def fake_runner(command, **_kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                payload = {"Human_Anatomy": [{"video_path": str(video), "score": 0.8}]}
                (output_dir / "results_eval_results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                return CommandResult(list(command), 0, "done\n", "", str(root))

            result = run_vbench_anatomy(
                [video],
                vbench_root=root,
                python="/opt/vbench/bin/python",
                output_dir=output_dir,
                runner=fake_runner,
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["dimension"], "Human_Anatomy")
            self.assertTrue(Path(result["raw_output_location"]).is_file())
            self.assertTrue(any(str(item).endswith("evaluate.py") for item in result["command"]))

    def test_subprocess_failure_is_failed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "VBench-2.0"
            (root / "vbench2").mkdir(parents=True)
            (root / "evaluate.py").write_text(
                "from vbench2 import VBench2\nHuman_Anatomy\n", encoding="utf-8"
            )
            video = Path(directory) / "clip.mp4"
            video.write_bytes(b"mp4")

            def fake_runner(command, **_kwargs):
                return CommandResult(list(command), 1, "", "missing ckpt", str(root))

            payload = run_vbench_anatomy(
                [video],
                vbench_root=root,
                python=sys.executable,
                output_dir=Path(directory) / "out",
                runner=fake_runner,
            )
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("score", payload)

    def test_malformed_results_are_failed(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "bad.json").write_text("{", encoding="utf-8")
            with self.assertRaises(ModelUnavailable) as caught:
                parse_vbench_results(output_dir)
        self.assertEqual(caught.exception.status, "failed")


if __name__ == "__main__":
    unittest.main()

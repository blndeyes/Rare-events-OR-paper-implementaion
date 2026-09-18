"""Adapters that invoke official FVMD, JEDi, and VBench-2.0 implementations.

These metrics are never reimplemented here. A missing interpreter, package,
checkout, or checkpoint is ``blocked``. A launched official command that fails
or returns malformed/non-finite output is ``failed``. No FVD, optical-flow,
CLIP, DINO, or I3D substitute is used.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from .external import (
    STATUS_FAILED,
    CommandResult,
    ModelUnavailable,
    blocked_metric,
    failed_metric,
    git_revision,
    package_src_root,
    python_module_probe,
    require_finite_float,
    run_logged_command,
    write_text,
)
from .protocol import (
    FROZEN_MODELS,
    JEDI_DEFAULT_CONFIG_NAME,
    JEDI_ENCODER_CHECKPOINT,
    JEDI_PROBE_CHECKPOINT,
)


FVMD_SCORE_RE = re.compile(r"FVMD:\s*([-+0-9.eE]+)")
VBENCH2_MARKERS = ("vbench2", "VBench2", "VBench-2.0", "Human_Anatomy")


def _tail(text: str, size: int = 4000) -> str:
    return text[-size:] if text else ""


def inspect_fvmd_python(python: str | None) -> dict[str, Any]:
    if not python:
        return blocked_metric(
            "fvmd",
            "--fvmd-python was not provided; official fvmd must run in a compatible "
            "interpreter and must not be imported into the main evaluator",
        )
    probe = python_module_probe(python, "fvmd")
    if not probe.get("available"):
        return blocked_metric(
            "fvmd",
            f"official fvmd is not importable in {python}: {probe.get('error')}",
            probe=probe,
        )
    probe["status"] = "ok"
    probe["metric"] = "fvmd"
    return probe


def parse_fvmd_score(*, log_dir: Path, stdout: str, stderr: str) -> float:
    json_path = log_dir / "fvmd.json"
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ModelUnavailable(
                "fvmd",
                f"official fvmd.json is malformed: {error}",
                status=STATUS_FAILED,
            ) from error
        if isinstance(payload, dict):
            for key in ("fvmd", "FVMD", "score"):
                if key in payload:
                    return require_finite_float(payload[key], metric="fvmd", source=f"fvmd.json[{key}]")
        else:
            return require_finite_float(payload, metric="fvmd", source="fvmd.json")
    for text in (stdout, stderr):
        match = FVMD_SCORE_RE.search(text)
        if match:
            return require_finite_float(match.group(1), metric="fvmd", source="stdout")
    log_txt = log_dir / "log.txt"
    if log_txt.is_file():
        match = FVMD_SCORE_RE.search(log_txt.read_text(encoding="utf-8", errors="replace"))
        if match:
            return require_finite_float(match.group(1), metric="fvmd", source="log.txt")
    raise ModelUnavailable(
        "fvmd",
        "official FVMD produced no parseable finite score (missing fvmd.json and 'FVMD:' line)",
        status=STATUS_FAILED,
    )


def write_fvmd_npy(path: Path, videos: Sequence[np.ndarray]) -> Path:
    stacked = np.stack([np.asarray(video) for video in videos], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stacked)
    return path


def write_fvmd_frame_folders(root: Path, clips: Sequence[tuple[str, np.ndarray]]) -> Path:
    """Official VideoDataset layout: one subfolder of sequential PNG frames per clip."""

    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    for clip_id, frames in clips:
        directory = root / str(clip_id)
        directory.mkdir(parents=True, exist_ok=True)
        array = np.asarray(frames)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ModelUnavailable(
                "fvmd",
                f"clip {clip_id} frames must have shape [T,H,W,3]",
                status=STATUS_FAILED,
            )
        for index, frame in enumerate(array):
            Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(
                directory / f"{index:05d}.png"
            )
    return root


def run_fvmd(
    generated_path: Path,
    reference_path: Path,
    log_dir: Path,
    *,
    python: str | None,
    runner: Callable[..., CommandResult] = run_logged_command,
) -> dict[str, Any]:
    probe = inspect_fvmd_python(python)
    if probe.get("status") == "blocked":
        return probe
    if python is None:
        return blocked_metric("fvmd", "--fvmd-python was not provided")
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "fvmd.stdout"
    stderr_path = log_dir / "fvmd.stderr"
    command = [
        python,
        "-m",
        "fvmd",
        "--log_dir",
        str(log_dir),
        str(generated_path),
        str(reference_path),
    ]
    result = runner(command, cwd=log_dir)
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    provenance = {
        "implementation": FROZEN_MODELS["fvmd"]["implementation"],
        "package": probe.get("version") or FROZEN_MODELS["fvmd"]["package"],
        "package_path": probe.get("path"),
        "python": python,
        "python_version": probe.get("python"),
        "command": command,
        "feature_protocol": FROZEN_MODELS["fvmd"]["official_sampling"],
        "generated_path": str(generated_path),
        "reference_path": str(reference_path),
        "log_dir": str(log_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": result.returncode,
        "role": "primary_motion_distribution",
        "no_fallback": FROZEN_MODELS["fvmd"]["no_fallback"],
    }
    if result.returncode != 0:
        return failed_metric(
            "fvmd",
            f"official fvmd exited {result.returncode}: {_tail(result.stderr, 1000) or _tail(result.stdout, 1000)}",
            **provenance,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )
    try:
        score = parse_fvmd_score(log_dir=log_dir, stdout=result.stdout, stderr=result.stderr)
    except ModelUnavailable as error:
        return failed_metric("fvmd", error.reason, **provenance)
    return {
        "status": "ok",
        "score": score,
        **provenance,
    }


def jedi_checkpoint_paths(model_dir: Path | None) -> dict[str, Path | None]:
    directory = Path(model_dir) if model_dir is not None else None
    return {
        "model_dir": directory,
        "encoder": (directory / JEDI_ENCODER_CHECKPOINT) if directory else None,
        "probe": (directory / JEDI_PROBE_CHECKPOINT) if directory else None,
    }


def inspect_jedi(
    *,
    python: str | None,
    model_dir: Path | None,
    config_path: Path | None,
) -> dict[str, Any]:
    executable = python
    if executable is None:
        import sys

        executable = sys.executable
    probe = python_module_probe(executable, "videojedi")
    checkpoints = jedi_checkpoint_paths(model_dir)
    missing: list[str] = []
    if not probe.get("available"):
        missing.append(
            f"videojedi is not importable in {executable}: {probe.get('error')}"
        )
    if model_dir is None:
        missing.append("--jedi-model-dir is required so V-JEPA weights are not downloaded at runtime")
    else:
        encoder = checkpoints["encoder"]
        probe_ckpt = checkpoints["probe"]
        if encoder is None or not encoder.is_file():
            missing.append(f"missing V-JEPA encoder checkpoint {JEDI_ENCODER_CHECKPOINT} in {model_dir}")
        if probe_ckpt is None or not probe_ckpt.is_file():
            missing.append(f"missing V-JEPA probe checkpoint {JEDI_PROBE_CHECKPOINT} in {model_dir}")
    resolved_config = Path(config_path) if config_path is not None else None
    if resolved_config is None and model_dir is not None:
        candidate = Path(model_dir) / JEDI_DEFAULT_CONFIG_NAME
        if candidate.is_file():
            resolved_config = candidate
    if resolved_config is None or not resolved_config.is_file():
        missing.append(
            f"--jedi-config is required ({JEDI_DEFAULT_CONFIG_NAME}); refusing the official runtime YAML download"
        )
    payload = {
        "probe": probe,
        "python": executable,
        "external_python": python,
        "model_dir": str(model_dir) if model_dir else None,
        "encoder_checkpoint": str(checkpoints["encoder"]) if checkpoints["encoder"] else None,
        "probe_checkpoint": str(checkpoints["probe"]) if checkpoints["probe"] else None,
        "config_path": str(resolved_config) if resolved_config else None,
        "role": "relative_ranking_only",
        "warning": "JEDi is not an absolute calibrated quality score",
        "feature": FROZEN_MODELS["jedi"]["features"],
        "preprocessing": FROZEN_MODELS["jedi"]["preprocessing"],
        "aggregation": FROZEN_MODELS["jedi"]["aggregation"],
    }
    if missing:
        return blocked_metric("jedi", "; ".join(missing), **payload)
    payload["status"] = "ok"
    return payload


def parse_jedi_driver_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ModelUnavailable("jedi", "JEDi driver returned empty stdout", status=STATUS_FAILED)
    try:
        payload = json.loads(text.splitlines()[-1])
    except json.JSONDecodeError as error:
        raise ModelUnavailable(
            "jedi",
            f"JEDi driver returned malformed JSON: {error}",
            status=STATUS_FAILED,
        ) from error
    if not isinstance(payload, dict):
        raise ModelUnavailable("jedi", "JEDi driver JSON is not an object", status=STATUS_FAILED)
    if payload.get("status") != "ok":
        raise ModelUnavailable(
            "jedi",
            str(payload.get("reason") or "JEDi driver reported failure"),
            status=str(payload.get("status") or STATUS_FAILED),
        )
    require_finite_float(payload.get("score"), metric="jedi", source="driver.score")
    return payload


def run_jedi(
    reference_videos: Sequence[Path],
    generated_videos: Sequence[Path],
    *,
    feature_path: Path,
    python: str | None = None,
    model_dir: Path | None = None,
    config_path: Path | None = None,
    runner: Callable[..., CommandResult] = run_logged_command,
) -> dict[str, Any]:
    preflight = inspect_jedi(python=python, model_dir=model_dir, config_path=config_path)
    if preflight.get("status") == "blocked":
        preflight["role"] = "relative_ranking_only"
        return preflight
    if len(reference_videos) != len(generated_videos) or not reference_videos:
        return blocked_metric(
            "jedi",
            "JEDi requires equally many real and generated videos",
            role="relative_ranking_only",
        )
    feature_path.mkdir(parents=True, exist_ok=True)
    request = {
        "reference_videos": [str(path) for path in reference_videos],
        "generated_videos": [str(path) for path in generated_videos],
        "feature_path": str(feature_path),
        "model_dir": str(model_dir) if model_dir is not None else None,
        "config_path": preflight.get("config_path"),
    }
    request_path = feature_path / "jedi-request.json"
    result_path = feature_path / "jedi-result.json"
    stdout_path = feature_path / "jedi.stdout"
    stderr_path = feature_path / "jedi.stderr"
    write_text(request_path, json.dumps(request, indent=2, sort_keys=True) + "\n")
    env = os.environ.copy()
    src_root = str(package_src_root())
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    executable = str(preflight["python"])
    command = [
        executable,
        "-m",
        "or_video_reproduction.evaluation.finegrained.jedi_driver",
        "--request",
        str(request_path),
        "--output",
        str(result_path),
    ]
    result = runner(command, env=env)
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    provenance = {
        "implementation": FROZEN_MODELS["jedi"]["implementation"],
        "package": (preflight.get("probe") or {}).get("version") or FROZEN_MODELS["jedi"]["package"],
        "package_path": (preflight.get("probe") or {}).get("path"),
        "python": executable,
        "command": command,
        "request_path": str(request_path),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": result.returncode,
        "n_videos": len(reference_videos),
        "feature": FROZEN_MODELS["jedi"]["features"],
        "encoder_checkpoint": preflight.get("encoder_checkpoint"),
        "probe_checkpoint": preflight.get("probe_checkpoint"),
        "config_path": preflight.get("config_path"),
        "preprocessing": FROZEN_MODELS["jedi"]["preprocessing"],
        "aggregation": FROZEN_MODELS["jedi"]["aggregation"],
        "role": "relative_ranking_only",
        "warning": "JEDi is not an absolute calibrated quality score",
        "no_silent_fallback": FROZEN_MODELS["jedi"]["no_silent_fallback"],
    }
    if result.returncode != 0:
        return failed_metric(
            "jedi",
            f"official JEDi driver exited {result.returncode}: {_tail(result.stderr, 1000) or _tail(result.stdout, 1000)}",
            **provenance,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )
    try:
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                raise ModelUnavailable(
                    "jedi",
                    str((payload or {}).get("reason") if isinstance(payload, dict) else "malformed result file"),
                    status=STATUS_FAILED,
                )
            score = require_finite_float(payload.get("score"), metric="jedi", source="jedi-result.json")
            provenance.update({key: payload[key] for key in payload if key not in {"status", "score"}})
        else:
            payload = parse_jedi_driver_output(result.stdout)
            score = float(payload["score"])
            provenance.update({key: payload[key] for key in payload if key not in {"status", "score"}})
    except ModelUnavailable as error:
        return failed_metric("jedi", error.reason, **provenance)
    except json.JSONDecodeError as error:
        return failed_metric("jedi", f"JEDi result JSON is malformed: {error}", **provenance)
    return {
        "status": "ok",
        "score": score,
        **provenance,
    }


def compute_jedi_inprocess(
    reference_videos: Sequence[Path],
    generated_videos: Sequence[Path],
    *,
    feature_path: Path,
    model_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Official in-process path used by jedi_driver. Not a CLIP/DINO substitute."""

    try:
        from torch.utils.data import DataLoader, Dataset
        from videojedi import JEDiMetric
    except ImportError as error:
        raise ModelUnavailable("jedi", f"official videojedi package is not installed: {error}") from error
    try:
        import torch
    except ImportError as error:
        raise ModelUnavailable("jedi", f"PyTorch is not installed in the JEDi interpreter: {error}") from error
    from or_video_reproduction.evaluation.finegrained.backends import frames_to_jedi_video_tensor
    from or_video_reproduction.evaluation.video import decode_rgb_video

    if len(reference_videos) != len(generated_videos) or not reference_videos:
        raise ModelUnavailable("jedi", "JEDi requires equally many real and generated videos")

    class _JediVideos(Dataset):
        def __init__(self, paths: Sequence[Path]) -> None:
            self.paths = list(paths)

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int):
            frames = decode_rgb_video(self.paths[index])
            return torch.from_numpy(frames_to_jedi_video_tensor(frames))

    feature_path.mkdir(parents=True, exist_ok=True)
    metric = JEDiMetric(
        feature_path=str(feature_path),
        model_dir=str(model_dir),
        config_path=str(config_path),
    )
    count = len(reference_videos)
    metric.load_features(
        train_loader=DataLoader(_JediVideos(reference_videos), batch_size=1, shuffle=False),
        test_loader=DataLoader(_JediVideos(generated_videos), batch_size=1, shuffle=False),
        num_samples=count,
    )
    score = require_finite_float(metric.compute_metric(), metric="jedi", source="JEDiMetric.compute_metric")
    return {
        "status": "ok",
        "score": score,
        "n_videos": count,
        "feature": "official V-JEPA via videojedi.JEDiMetric.load_features",
        "feature_layer": "finetuned attentive pooler (ssv2-probe) when present",
        "encoder_checkpoint": str(Path(model_dir) / JEDI_ENCODER_CHECKPOINT),
        "probe_checkpoint": str(Path(model_dir) / JEDI_PROBE_CHECKPOINT),
        "config_path": str(config_path),
        "preprocessing": FROZEN_MODELS["jedi"]["preprocessing"],
        "aggregation": FROZEN_MODELS["jedi"]["aggregation"],
        "role": "relative_ranking_only",
        "warning": "JEDi is not an absolute calibrated quality score",
    }


def inspect_vbench_root(vbench_root: Path | None) -> dict[str, Any]:
    if vbench_root is None:
        return blocked_metric(
            "vbench2_anatomy",
            "--vbench-root was not provided; official VBench-2.0 evaluate.py is required",
        )
    root = Path(vbench_root)
    script = root / "evaluate.py"
    if not script.is_file():
        return blocked_metric(
            "vbench2_anatomy",
            f"official evaluator missing: {script}",
            root=str(root),
        )
    text = script.read_text(encoding="utf-8", errors="replace")
    vbench2_dir = (root / "vbench2").is_dir()
    full_info = root / "vbench2" / "VBench2_full_info.json"
    if not vbench2_dir or not any(marker in text for marker in VBENCH2_MARKERS):
        return blocked_metric(
            "vbench2_anatomy",
            f"{script} is not a VBench-2.0 evaluator (vbench2 package/Human_Anatomy missing); "
            "VBench-1.0 evaluate.py is rejected",
            root=str(root),
            evaluate_py=str(script),
        )
    return {
        "status": "ok",
        "metric": "vbench2_anatomy",
        "root": str(root),
        "evaluate_py": str(script),
        "full_json_dir": str(full_info) if full_info.is_file() else None,
        "commit": git_revision(root),
        "dimension": "Human_Anatomy",
        "mode": "custom_input",
    }


def parse_vbench_results(output_dir: Path) -> dict[str, Any]:
    candidates = sorted(output_dir.rglob("*.json"))
    if not candidates:
        raise ModelUnavailable(
            "vbench2_anatomy",
            f"official VBench-2.0 wrote no JSON results under {output_dir}",
            status=STATUS_FAILED,
        )
    parsed: list[dict[str, Any]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ModelUnavailable(
                "vbench2_anatomy",
                f"malformed VBench JSON {path}: {error}",
                status=STATUS_FAILED,
            ) from error
        parsed.append({"path": str(path.resolve()), "payload": payload})
    preferred = [path for path in candidates if "eval_results" in path.name]
    primary = (preferred[-1] if preferred else candidates[-1]).resolve()
    return {"result_files": parsed, "primary": str(primary)}


def stage_video_folder(videos: Sequence[Path], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for video in videos:
        target = destination / video.name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(video.resolve())
        except OSError:
            import shutil

            shutil.copy2(video, target)
        staged.append(str(target))
    return destination


def run_vbench_anatomy(
    videos: Sequence[Path],
    *,
    vbench_root: Path | None,
    python: str | None = None,
    output_dir: Path | None = None,
    runner: Callable[..., CommandResult] = run_logged_command,
) -> dict[str, Any]:
    inspection = inspect_vbench_root(vbench_root)
    if inspection.get("status") == "blocked":
        return inspection
    assert vbench_root is not None
    if not videos:
        return blocked_metric("vbench2_anatomy", "no generated videos were provided for VBench-2.0")
    interpreter = python
    if interpreter is None:
        import sys

        interpreter = sys.executable
    if output_dir is None:
        output_dir = Path(vbench_root) / "evaluation_results" / "Human_Anatomy"
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_video_folder(videos, output_dir / "staged-videos")
    stdout_path = output_dir / "vbench.stdout"
    stderr_path = output_dir / "vbench.stderr"
    command = [
        interpreter,
        str(Path(vbench_root) / "evaluate.py"),
        "--dimension",
        "Human_Anatomy",
        "--videos_path",
        str(staged),
        "--output_path",
        str(output_dir),
        "--mode",
        "custom_input",
    ]
    result = runner(command, cwd=Path(vbench_root))
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    provenance = {
        "implementation": FROZEN_MODELS["vbench2"]["implementation"],
        "dimension": "Human_Anatomy",
        "mode": "custom_input",
        "vbench_root": str(vbench_root),
        "evaluate_py": inspection.get("evaluate_py"),
        "commit": inspection.get("commit"),
        "python": interpreter,
        "command": command,
        "videos_path": str(staged),
        "output_path": str(output_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": result.returncode,
        "full_json_dir": inspection.get("full_json_dir"),
    }
    if result.returncode != 0:
        return failed_metric(
            "vbench2_anatomy",
            f"official VBench-2.0 evaluate.py exited {result.returncode}: "
            f"{_tail(result.stderr, 1000) or _tail(result.stdout, 1000)}",
            **provenance,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )
    try:
        parsed = parse_vbench_results(output_dir)
    except ModelUnavailable as error:
        return failed_metric(
            "vbench2_anatomy",
            error.reason,
            **provenance,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )
    return {
        "status": "ok",
        "raw_output_location": parsed["primary"],
        "result_files": parsed["result_files"],
        **provenance,
    }

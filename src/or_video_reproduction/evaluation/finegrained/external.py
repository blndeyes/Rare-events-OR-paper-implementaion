"""Subprocess helpers for official metric implementations.

Missing tools are recorded as ``blocked``. A launched official command that
exits non-zero or returns non-finite output is ``failed``. Neither case is
converted into a numeric score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import subprocess
from typing import Any


STATUS_OK = "ok"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"


class ModelUnavailable(RuntimeError):
    def __init__(self, metric: str, reason: str, *, status: str = STATUS_BLOCKED) -> None:
        self.metric = metric
        self.reason = reason
        self.status = status
        super().__init__(f"{metric} {status}: {reason}")


def blocked_metric(metric: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": STATUS_BLOCKED, "metric": metric, "reason": reason}
    payload.update(extra)
    return payload


def failed_metric(metric: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": STATUS_FAILED, "metric": metric, "reason": reason}
    payload.update(extra)
    return payload


def unavailable_metric(metric: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": STATUS_UNAVAILABLE, "metric": metric, "reason": reason}
    payload.update(extra)
    return payload


@dataclass(frozen=True)
class OfficialTooling:
    vbench_root: Path | None = None
    vbench_python: str | None = None
    fvmd_python: str | None = None
    jedi_python: str | None = None
    jedi_model_dir: Path | None = None
    jedi_config: Path | None = None
    hand_landmarker_model: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    cwd: str | None

    def as_dict(self, *, tail: int = 4000) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "cwd": self.cwd,
            "stdout_tail": self.stdout[-tail:],
            "stderr_tail": self.stderr[-tail:],
        }


def package_src_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_logged_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> CommandResult:
    result = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return CommandResult(
        command=list(command),
        returncode=int(result.returncode),
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        cwd=str(cwd) if cwd is not None else None,
    )


def git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def python_module_probe(python: str, module: str) -> dict[str, Any]:
    script = (
        "import importlib.metadata, importlib.util, json, sys\n"
        f"name = {module!r}\n"
        "spec = importlib.util.find_spec(name)\n"
        "version = None\n"
        "path = None\n"
        "error = None\n"
        "if spec is None:\n"
        "    error = 'module not found'\n"
        "else:\n"
        "    path = getattr(spec, 'origin', None)\n"
        "    try:\n"
        "        version = importlib.metadata.version(name)\n"
        "    except Exception as exc:\n"
        "        version = None\n"
        "        error = type(exc).__name__ + ': ' + str(exc)\n"
        "print(json.dumps({'module': name, 'available': spec is not None, "
        "'version': version, 'path': path, 'error': error, "
        "'executable': sys.executable, 'python': sys.version}))\n"
    )
    result = run_logged_command([python, "-c", script])
    if result.returncode != 0:
        return {
            "available": False,
            "module": module,
            "executable": python,
            "error": result.stderr.strip() or f"probe exited {result.returncode}",
            "command": result.command,
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return {
            "available": False,
            "module": module,
            "executable": python,
            "error": f"probe returned non-JSON: {error}",
            "stdout_tail": result.stdout[-1000:],
        }
    payload["command"] = result.command
    return payload


def require_finite_float(value: Any, *, metric: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelUnavailable(
            metric,
            f"{source} is not a numeric score: {value!r}",
            status=STATUS_FAILED,
        )
    number = float(value)
    if not math.isfinite(number):
        raise ModelUnavailable(
            metric,
            f"{source} is non-finite: {number}",
            status=STATUS_FAILED,
        )
    return number


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path

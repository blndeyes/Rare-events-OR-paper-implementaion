"""Validate LoRA checkpoints from a training report rather than directory names."""

from __future__ import annotations

import json
from pathlib import Path

from or_video_reproduction.evaluation.video import sha256_file


def load_training_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training report is not an object: {path}")
    return payload


def audited_checkpoint_step(
    *,
    training_report: Path,
    checkpoint: Path,
    expected_step: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Return the audited training step for ``checkpoint``.

    The step is taken from the training-report checkpoint audit, not from the
    parent directory name. A filename ``step_NNNNN`` token must agree with the
    audit when present; disagreement is a hard failure.
    """

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    report = load_training_report(training_report)
    audit = report.get("checkpoint_audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError(f"Training report does not contain a passing checkpoint audit: {training_report}")
    samples = audit.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Checkpoint audit has no samples")

    resolved = checkpoint.resolve()
    matches = []
    for row in samples:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("path")
        if not isinstance(raw_path, str):
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (training_report.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate == resolved or candidate.name == checkpoint.name:
            matches.append(row)
    if not matches:
        raise ValueError(f"Checkpoint {checkpoint} is not listed in {training_report}")
    if len(matches) != 1:
        raise ValueError(f"Checkpoint {checkpoint.name} matches multiple audit rows")
    row = matches[0]
    if row.get("passed") is not True:
        raise ValueError(f"Checkpoint audit did not pass for {checkpoint}")
    step = row.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError(f"Checkpoint audit row has no integer step: {row!r}")
    size = row.get("size_bytes")
    if isinstance(size, int) and size != checkpoint.stat().st_size:
        raise ValueError(
            f"Checkpoint size {checkpoint.stat().st_size} != audited size {size}"
        )
    report_steps = report.get("steps")
    if isinstance(report_steps, int) and report_steps != step:
        raise ValueError(f"Report steps {report_steps} disagree with audited checkpoint step {step}")

    from or_video_reproduction.evaluation.corrected_inference import _checkpoint_step

    try:
        filename_step = _checkpoint_step(checkpoint)
    except ValueError:
        filename_step = None
    if filename_step is not None and filename_step != step:
        raise ValueError(
            f"Filename step {filename_step} disagrees with audited step {step}; "
            "refusing to infer the training step from the path"
        )
    if expected_step is not None and step != expected_step:
        raise ValueError(f"Audited checkpoint step is {step}, expected {expected_step}")
    digest = sha256_file(checkpoint)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checkpoint SHA-256 {digest} does not match expected {expected_sha256}"
        )
    return {
        "step": step,
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": checkpoint.stat().st_size,
        "tensor_count": row.get("tensor_count"),
        "training_report": str(training_report),
    }

"""Strict video decoding for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np


def probe_video(path: Path) -> dict[str, int | float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": numerator / denominator,
    }


def decode_rgb_video(path: Path, *, resize: tuple[int, int] | None = None) -> np.ndarray:
    """Decode all RGB frames with ffmpeg, optionally resizing to (width, height)."""

    metadata = probe_video(path)
    width, height = int(metadata["width"]), int(metadata["height"])
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    if resize is not None:
        width, height = resize
        command.extend(["-vf", f"scale={width}:{height}:flags=bilinear"])
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    result = subprocess.run(command, check=True, capture_output=True)
    pixels_per_frame = width * height * 3
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    if len(raw) % pixels_per_frame:
        raise RuntimeError(f"ffmpeg returned an incomplete RGB frame for {path}")
    frames = raw.reshape(-1, height, width, 3)
    if len(frames) != metadata["frames"]:
        raise RuntimeError(
            f"Decoded frame count differs from probe for {path}: {len(frames)} != {metadata['frames']}"
        )
    return frames


def validate_pair(reference: Path, generated: Path) -> dict[str, int | float]:
    reference_metadata = probe_video(reference)
    generated_metadata = probe_video(generated)
    for field in ("width", "height", "frames"):
        if reference_metadata[field] != generated_metadata[field]:
            raise ValueError(
                f"Paired videos differ in {field}: {reference_metadata[field]} != "
                f"{generated_metadata[field]}"
            )
    if abs(float(reference_metadata["fps"]) - float(generated_metadata["fps"])) > 1e-6:
        raise ValueError("Paired videos differ in frame rate")
    return reference_metadata

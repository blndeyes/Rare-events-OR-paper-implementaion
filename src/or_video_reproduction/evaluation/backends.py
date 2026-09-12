"""Lazy adapters for the official/heavy metric implementations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
from typing import Sequence

import numpy as np

from .video import decode_rgb_video


DOVER_FUSED_PATTERN = re.compile(
    r"Normalized fused overall score \(scale in \[0,1\]\):\s*([-+0-9.eE]+)"
)
PINNED_DOVER_COMMIT = "f1ddc96215bc7fbcf8f315c65d47905f339c3419"
PINNED_GOOGLE_RESEARCH_COMMIT = "08a8d6736475776f42ffac23b2c13111a28e5795"


def _require_git_commit(root: Path, expected: str, name: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise ValueError(f"{name} checkout must be pinned to {expected}; found {actual}")


def parse_dover_fused_score(output: str) -> float:
    match = DOVER_FUSED_PATTERN.search(output)
    if match is None:
        raise ValueError("Official DOVER fused score was not found in command output")
    return float(match.group(1))


def dover_score(video: Path, *, dover_root: Path, python: str = "python") -> float:
    """Run the official DOVER single-video evaluator and return its fused score."""

    _require_git_commit(dover_root, PINNED_DOVER_COMMIT, "DOVER")
    script = dover_root / "evaluate_one_video.py"
    if not script.is_file():
        raise FileNotFoundError(f"DOVER evaluator not found: {script}")
    result = subprocess.run(
        [python, str(script), "-v", str(video), "-f"],
        cwd=dover_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_dover_fused_score(result.stdout + "\n" + result.stderr)


def lpips_video(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    device: str = "cuda",
    batch_size: int = 16,
    network: str = "alex",
) -> float:
    """Frame-mean official LPIPS v0.1 score."""

    import lpips
    import torch

    if reference.shape != generated.shape or reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError("LPIPS videos must have equal [T,H,W,3] shapes")
    model = lpips.LPIPS(net=network, version="0.1").to(device).eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(reference), batch_size):
            left = torch.from_numpy(reference[start : start + batch_size]).permute(0, 3, 1, 2)
            right = torch.from_numpy(generated[start : start + batch_size]).permute(0, 3, 1, 2)
            left = left.to(device=device, dtype=torch.float32) / 127.5 - 1.0
            right = right.to(device=device, dtype=torch.float32) / 127.5 - 1.0
            values.append(model(left, right).flatten().cpu().numpy())
    return float(np.concatenate(values).mean())


def inception_logits(
    videos: Sequence[Path], *, device: str = "cuda", batch_size: int = 32
) -> np.ndarray:
    """Return torchvision Inception-v3 ImageNet logits for every generated frame."""

    import torch
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False).to(device).eval()
    transform = weights.transforms()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for video in videos:
            frames = decode_rgb_video(video)
            for start in range(0, len(frames), batch_size):
                tensor = torch.from_numpy(frames[start : start + batch_size]).permute(0, 3, 1, 2)
                tensor = transform(tensor).to(device)
                outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs)


def _load_official_fvd_module(google_research_root: Path):
    _require_git_commit(
        google_research_root, PINNED_GOOGLE_RESEARCH_COMMIT, "google-research FVD"
    )
    path = google_research_root / "frechet_video_distance" / "frechet_video_distance.py"
    if not path.is_file():
        raise FileNotFoundError(f"Official Google FVD module not found: {path}")
    spec = importlib.util.spec_from_file_location("official_google_fvd", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official FVD module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fvd_i3d_embeddings(
    videos: Sequence[Path], *, google_research_root: Path, batch_size: int = 16
) -> np.ndarray:
    """Extract official Google/DeepMind Kinetics-400 I3D video embeddings.

    The official graph requires a fixed batch of 16.  A final short batch is
    padded by repeating its last video; padded embeddings are discarded.
    """

    if not videos:
        raise ValueError("At least one video is required for FVD embeddings")
    if batch_size != 16:
        raise ValueError("The official Google FVD graph requires batch_size=16")
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    official = _load_official_fvd_module(google_research_root)
    first = decode_rgb_video(videos[0], resize=(224, 224))
    frame_count = len(first)
    graph = tf.Graph()
    with graph.as_default():
        placeholder = tf.placeholder(tf.float32, [16, frame_count, 224, 224, 3])
        preprocessed = official.preprocess(placeholder, (224, 224))
        embeddings = official.create_id3_embedding(preprocessed)
        initializer = tf.group(tf.global_variables_initializer(), tf.tables_initializer())
    output: list[np.ndarray] = []
    with tf.Session(graph=graph) as session:
        session.run(initializer)
        for start in range(0, len(videos), 16):
            paths = videos[start : start + 16]
            decoded = []
            for path in paths:
                frames = first if start == 0 and path == videos[0] else decode_rgb_video(path, resize=(224, 224))
                if len(frames) != frame_count:
                    raise ValueError("All FVD videos must have the same frame count")
                # The official preprocess function expects RGB values in [0, 255]
                # and performs the conversion to the I3D [-1, 1] range itself.
                decoded.append(frames.astype(np.float32))
            valid = len(decoded)
            decoded.extend([decoded[-1]] * (16 - valid))
            output.append(session.run(embeddings, feed_dict={placeholder: np.stack(decoded)})[:valid])
    return np.concatenate(output)

"""Reusable Video Depth Anything adapter for paper-shaped clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Sequence

import numpy as np

from or_video_reproduction.data.clips import (
    OUTPUT_FPS,
    OUTPUT_FRAME_COUNT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)


PINNED_VDA_COMMIT = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _set_native_decord_bridge() -> None:
    """Restore the array type expected by pinned VDA's ``dc_utils`` reader."""

    import decord

    decord.bridge.set_bridge("native")


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_depth_output(depths: np.ndarray, fps: float) -> dict[str, object]:
    errors: list[str] = []
    expected = (OUTPUT_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH)
    if depths.shape != expected:
        errors.append(f"depth shape is {depths.shape}, expected {expected}")
    if not np.issubdtype(depths.dtype, np.floating):
        errors.append(f"depth dtype must be floating, got {depths.dtype}")
    if np.issubdtype(depths.dtype, np.floating) and not np.isfinite(depths).all():
        errors.append("depth output contains NaN or infinite values")
    if abs(float(fps) - OUTPUT_FPS) > 1e-6:
        errors.append(f"depth output fps is {fps}, expected {OUTPUT_FPS}")
    return {
        "passed": not errors,
        "errors": errors,
        "shape": list(depths.shape),
        "dtype": str(depths.dtype),
        "fps": float(fps),
    }


class VideoDepthRunner:
    """Keep one official relative-depth model resident across many clips."""

    def __init__(
        self,
        vda_root: Path,
        checkpoint_path: Path,
        *,
        encoder: str = "vitl",
        device: str = "cuda",
        verify_revision: bool = True,
    ) -> None:
        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported encoder: {encoder}")
        if verify_revision and _git_head(vda_root) != PINNED_VDA_COMMIT:
            raise ValueError(f"VDA checkout must be pinned to {PINNED_VDA_COMMIT}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        # VDA uses a top-level package named ``utils``. Some environments preload an
        # unrelated package with that generic name, so bind it to the pinned checkout.
        for module_name in list(sys.modules):
            if module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]
        vda_utils = types.ModuleType("utils")
        vda_utils.__path__ = [str(vda_root / "utils")]
        sys.modules["utils"] = vda_utils
        sys.path.insert(0, str(vda_root))
        import torch
        from utils.dc_utils import read_video_frames
        from video_depth_anything.video_depth import VideoDepthAnything

        model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self.read_video_frames = read_video_frames
        self.device = device
        self.encoder = encoder
        self.checkpoint_path = checkpoint_path

    def run(self, video_path: Path, output_path: Path) -> dict[str, object]:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        # Decord's bridge is process-global.  Trainer/SAM2 imports may switch it
        # to torch, whereas pinned VDA calls ``get_batch(...).asnumpy()``.
        _set_native_decord_bridge()
        frames, source_fps = self.read_video_frames(str(video_path), -1, -1, 1280)
        depths, output_fps = self.model.infer_video_depth(
            frames,
            source_fps,
            input_size=518,
            device=self.device,
            fp32=False,
        )
        depths = np.asarray(depths, dtype=np.float32)
        validation = validate_depth_output(depths, output_fps)
        if not validation["passed"]:
            raise RuntimeError("; ".join(validation["errors"]))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, depths=depths)
        metadata = {
            "schema_version": 1,
            "kind": "video_depth_anything_relative_depth",
            "vda_revision": PINNED_VDA_COMMIT,
            "encoder": self.encoder,
            "checkpoint": str(self.checkpoint_path),
            "input_video": str(video_path),
            "output": str(output_path),
            "model_reuse": "one loaded model may process multiple clips",
            "validation": validation,
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--vda-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vitl")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = VideoDepthRunner(args.vda_root, args.checkpoint, encoder=args.encoder)
    metadata = runner.run(args.video, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

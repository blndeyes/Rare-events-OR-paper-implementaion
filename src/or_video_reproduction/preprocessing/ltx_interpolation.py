"""Plan or execute official LTX 0.9.7 multi-keyframe interpolation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from or_video_reproduction.data.clips import validate_clip_manifest


PINNED_LTX_COMMIT = "20799e51cd739986d98d9b1aab55cc2067c1eabb"
PIPELINE_CONFIGS = {
    "bf16": "configs/ltxv-13b-0.9.7-dev.yaml",
    "fp8": "configs/ltxv-13b-0.9.7-dev-fp8.yaml",
}
PIPELINE_CONFIG = PIPELINE_CONFIGS["bf16"]


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_interpolation_command(
    manifest: dict[str, object],
    *,
    dataset_root: Path,
    ltx_root: Path,
    python_executable: Path,
    output_dir: Path,
    prompt: str,
    seed: int,
    precision: str = "bf16",
    verify_revision: bool = True,
) -> list[str]:
    validate_clip_manifest(manifest)
    if not prompt.strip():
        raise ValueError("An explicit interpolation prompt is required")
    if precision not in PIPELINE_CONFIGS:
        raise ValueError(
            f"Unsupported precision {precision!r}; choose one of {sorted(PIPELINE_CONFIGS)}"
        )
    if verify_revision:
        actual_commit = _git_head(ltx_root)
        if actual_commit != PINNED_LTX_COMMIT:
            raise ValueError(
                f"LTX revision mismatch: expected {PINNED_LTX_COMMIT}, got {actual_commit}"
            )

    contract = manifest["paper_contract"]
    clip = manifest["clip"]
    keyframes = clip["source_keyframes"]
    media_paths = [str(dataset_root / row["rgb_path"]) for row in keyframes]
    start_frames = [str(row["output_frame_index"]) for row in keyframes]
    strengths = ["1.0"] * len(keyframes)
    width, height = contract["resolution"]

    return [
        str(python_executable),
        str(ltx_root / "inference.py"),
        "--prompt",
        prompt,
        "--conditioning_media_paths",
        *media_paths,
        "--conditioning_start_frames",
        *start_frames,
        "--conditioning_strengths",
        *strengths,
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_frames",
        str(contract["output_frames"]),
        "--frame_rate",
        str(contract["output_fps"]),
        "--seed",
        str(seed),
        "--pipeline_config",
        str(ltx_root / PIPELINE_CONFIGS[precision]),
        "--offload_to_cpu",
        "--output_path",
        str(output_dir),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--ltx-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path, dest="python_executable")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--precision",
        choices=sorted(PIPELINE_CONFIGS),
        default="bf16",
        help="Official 0.9.7-dev checkpoint precision (default: bf16)",
    )
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    command = build_interpolation_command(
        manifest,
        dataset_root=args.dataset_root,
        ltx_root=args.ltx_root,
        python_executable=args.python_executable,
        output_dir=args.output_dir,
        prompt=args.prompt,
        seed=args.seed,
        precision=args.precision,
    )
    plan = {
        "schema_version": 1,
        "kind": "ltx_0.9.7_multi_keyframe_interpolation",
        "paper_fields": {
            "height": 768,
            "width": 1024,
            "num_frames": 97,
            "frame_rate": 24,
        },
        "reproduction_hypotheses": {
            "ltx_revision": PINNED_LTX_COMMIT,
            "prompt": args.prompt,
            "seed": args.seed,
            "conditioning_strength": 1.0,
            "cpu_offload": True,
            "checkpoint_precision": args.precision,
        },
        "argv": command,
    }
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(rendered, encoding="utf-8")
    if args.execute:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        subprocess.run(command, cwd=args.ltx_root, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

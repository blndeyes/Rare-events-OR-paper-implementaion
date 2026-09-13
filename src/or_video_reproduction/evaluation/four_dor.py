"""Track generated 4D-OR instances and build the six-video evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from or_video_reproduction.preprocessing.sam2_propagation import (
    Sam2VideoRunner,
    point_prompt_output_is_current,
)


def _resolve(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing path field {field}")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def build_four_dor_evaluation(
    inference_manifest: Path,
    pair_manifest: Path,
    *,
    output_root: Path,
    sam2_runner: object,
    require_six: bool = True,
) -> dict[str, object]:
    """Run identical prompts on generated videos and pair all metric inputs."""

    inference = json.loads(inference_manifest.read_text(encoding="utf-8"))
    pairs = json.loads(pair_manifest.read_text(encoding="utf-8"))
    if inference.get("schema_version") != 1 or not isinstance(inference.get("samples"), list):
        raise ValueError("Inference manifest must have schema_version 1 and samples")
    if pairs.get("schema_version") != 1 or not isinstance(pairs.get("samples"), list):
        raise ValueError("Pair manifest must have schema_version 1 and samples")
    inference_rows = inference["samples"]
    if require_six and len(inference_rows) != 6:
        raise ValueError(
            "Table 1 4D-OR evaluation requires exactly six videos, "
            f"got {len(inference_rows)}"
        )
    pairs_by_id = {row.get("id"): row for row in pairs["samples"]}
    if len(pairs_by_id) != len(pairs["samples"]):
        raise ValueError("Pair manifest contains duplicate sample ids")

    masks_root = output_root / "generated-masks"
    rows: list[dict[str, object]] = []
    for inference_row in inference_rows:
        clip_id = inference_row.get("id")
        if clip_id not in pairs_by_id:
            raise ValueError(f"No geometry pair exists for inference sample {clip_id!r}")
        pair = pairs_by_id[clip_id]
        prompt_manifest = _resolve(
            pair_manifest.parent, pair.get("prompt_manifest"), "prompt_manifest"
        )
        generated_video = _resolve(
            inference_manifest.parent, inference_row.get("generated_video"), "generated_video"
        )
        generated_masks = masks_root / f"{clip_id}.npz"
        if not point_prompt_output_is_current(generated_masks, prompt_manifest):
            sam2_runner.run_points(generated_video, prompt_manifest, generated_masks)
        rows.append(
            {
                "id": clip_id,
                "reference_video": str(
                    _resolve(
                        inference_manifest.parent,
                        inference_row.get("reference_video"),
                        "reference_video",
                    )
                ),
                "generated_video": str(generated_video),
                "reference_masks": str(
                    _resolve(pair_manifest.parent, pair.get("labels"), "labels")
                ),
                "generated_masks": str(generated_masks),
                "prompt_manifest": str(prompt_manifest),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "4dor_table1_evaluation_inputs",
        "dataset": "4DOR",
        "classification": "reproduction_hypothesis_exact_author_clips_undisclosed",
        "paper_table1_metrics": ["fvd", "ssim", "psnr", "lpips"],
        "secondary_structural_metrics": ["box_iou", "mask_iou"],
        "samples": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "evaluation-manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--allow-non-six", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Sam2VideoRunner(args.sam2_root, args.sam2_checkpoint)
    report = build_four_dor_evaluation(
        args.inference_manifest,
        args.pair_manifest,
        output_root=args.output_root,
        sam2_runner=runner,
        require_six=not args.allow_non_six,
    )
    print(
        json.dumps(
            {
                "samples": len(report["samples"]),
                "output": str(args.output_root / "evaluation-manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Propagate an MMOR first-frame semantic mask through an interpolated clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from or_video_reproduction.data.semantics import (
    ENTITY_CLASSES,
    MMOR_ARTIFACT_LABELS,
    MMOR_SEGMENTATION_LABELS,
)
from or_video_reproduction.data.four_dor import FOUR_DOR_TO_MMOR_MAPPING


PINNED_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def tracked_labels(first_frame_labels: NDArray[np.integer]) -> list[int]:
    """Return foreground labels that correspond to paper entity nodes."""

    result: list[int] = []
    for value in np.unique(first_frame_labels):
        raw_label = int(value)
        if raw_label == 0 or raw_label in MMOR_ARTIFACT_LABELS:
            continue
        class_name = MMOR_SEGMENTATION_LABELS.get(raw_label)
        if class_name in ENTITY_CLASSES:
            result.append(raw_label)
    return result


def compose_label_frame(
    object_ids: Sequence[int], mask_logits: NDArray[np.floating]
) -> NDArray[np.uint8]:
    """Resolve overlapping SAM2 objects by maximum positive mask logit."""

    logits = np.asarray(mask_logits)
    if logits.ndim != 3:
        raise ValueError(f"Expected object x height x width logits, got {logits.shape}")
    if logits.shape[0] != len(object_ids):
        raise ValueError("Object id count does not match mask-logit count")
    if not object_ids:
        return np.zeros(logits.shape[1:], dtype=np.uint8)

    winner = np.argmax(logits, axis=0)
    confidence = np.max(logits, axis=0)
    labels = np.zeros(logits.shape[1:], dtype=np.uint8)
    object_array = np.asarray(object_ids, dtype=np.uint8)
    foreground = confidence > 0.0
    labels[foreground] = object_array[winner[foreground]]
    return labels


def _propagate_with_predictor(
    predictor,
    video_path: Path,
    first_mask_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path,
) -> dict[str, object]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not first_mask_path.is_file():
        raise FileNotFoundError(first_mask_path)

    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            str(video_path),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        target_size = (state["video_width"], state["video_height"])
        first_labels = np.asarray(
            Image.open(first_mask_path)
            .convert("L")
            .resize(target_size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        object_ids = tracked_labels(first_labels)
        if not object_ids:
            raise ValueError("First-frame mask contains no trackable entity labels")
        for object_id in object_ids:
            predictor.add_new_mask(
                state,
                frame_idx=0,
                obj_id=object_id,
                mask=first_labels == object_id,
            )

        frames: dict[int, NDArray[np.uint8]] = {}
        for frame_index, propagated_ids, mask_logits in predictor.propagate_in_video(state):
            logits = mask_logits[:, 0].float().cpu().numpy()
            frames[int(frame_index)] = compose_label_frame(propagated_ids, logits)

    expected_frames = int(state["num_frames"])
    missing = sorted(set(range(expected_frames)) - frames.keys())
    if missing:
        raise RuntimeError(f"SAM2 did not return frames: {missing}")
    labels = np.stack([frames[index] for index in range(expected_frames)])
    labels[0] = first_labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, labels=labels)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "sam2.1_hiera_large_first_frame_mask_propagation",
        "sam2_revision": PINNED_SAM2_COMMIT,
        "sam2_config": SAM2_CONFIG,
        "video_path": str(video_path),
        "first_mask_path": str(first_mask_path),
        "shape": list(labels.shape),
        "dtype": str(labels.dtype),
        "tracked_labels": [
            {
                "raw_label": object_id,
                "class_name": MMOR_SEGMENTATION_LABELS[object_id],
            }
            for object_id in object_ids
        ],
        "overlap_resolution": "maximum positive SAM2 mask logit; otherwise background",
        "frame_zero_policy": "preserve resized ground-truth label map exactly",
        "memory_policy": "offload video frames and inference state to CPU",
        "model_reuse": "one loaded predictor may process multiple clips",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


class Sam2VideoRunner:
    """Keep one official SAM2.1 Hiera Large predictor loaded across clips."""

    def __init__(
        self,
        sam2_root: Path,
        checkpoint_path: Path,
        *,
        verify_revision: bool = True,
    ) -> None:
        if verify_revision:
            actual_commit = _git_head(sam2_root)
            if actual_commit != PINNED_SAM2_COMMIT:
                raise ValueError(
                    f"SAM2 revision mismatch: expected {PINNED_SAM2_COMMIT}, "
                    f"got {actual_commit}"
                )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        from sam2.build_sam import build_sam2_video_predictor

        self.predictor = build_sam2_video_predictor(
            SAM2_CONFIG,
            str(checkpoint_path),
            device="cuda",
            vos_optimized=False,
        )
        self.checkpoint_path = checkpoint_path

    def run(
        self, video_path: Path, first_mask_path: Path, output_path: Path
    ) -> dict[str, object]:
        return _propagate_with_predictor(
            self.predictor,
            video_path,
            first_mask_path,
            output_path,
            checkpoint_path=self.checkpoint_path,
        )

    def run_points(
        self, video_path: Path, prompt_manifest: Path, output_path: Path
    ) -> dict[str, object]:
        return _propagate_point_prompts_with_predictor(
            self.predictor,
            video_path,
            prompt_manifest,
            output_path,
            checkpoint_path=self.checkpoint_path,
        )


def load_point_prompt_manifest(path: Path) -> list[dict[str, object]]:
    """Validate manual normalized 4D-OR point prompts and semantic mappings."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("dataset") != "4DOR":
        raise ValueError("Point prompt manifest must be schema version 1 for 4DOR")
    if payload.get("coordinate_system") != "normalized_xy":
        raise ValueError("4D-OR point prompts must use normalized_xy coordinates")
    instances = payload.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"Manual point prompts are incomplete: {path}")
    validated: list[dict[str, object]] = []
    seen: set[int] = set()
    for instance in instances:
        if not isinstance(instance, dict):
            raise ValueError("Each point-prompt instance must be an object")
        object_id = instance.get("object_id")
        source_class = instance.get("four_dor_class")
        mapped_class = instance.get("mmor_class")
        points = instance.get("points")
        labels = instance.get("point_labels")
        if not isinstance(object_id, int) or not 1 <= object_id <= 255 or object_id in seen:
            raise ValueError("object_id values must be unique integers in [1,255]")
        if not isinstance(source_class, str) or source_class not in FOUR_DOR_TO_MMOR_MAPPING:
            raise ValueError(f"Unsupported or ambiguous 4D-OR class: {source_class!r}")
        expected_class = FOUR_DOR_TO_MMOR_MAPPING[source_class]
        if mapped_class != expected_class or mapped_class not in ENTITY_CLASSES:
            raise ValueError(
                f"Mapping for {source_class!r} must be MMOR class {expected_class!r}"
            )
        if not isinstance(points, list) or not points or not isinstance(labels, list):
            raise ValueError(f"Object {object_id} requires points and point_labels")
        if len(points) != len(labels) or any(label not in (0, 1) for label in labels):
            raise ValueError(f"Object {object_id} point and binary-label counts differ")
        normalized_points = []
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(isinstance(value, (int, float)) for value in point)
                or not all(0.0 <= float(value) <= 1.0 for value in point)
            ):
                raise ValueError(f"Object {object_id} has invalid normalized point {point!r}")
            normalized_points.append([float(point[0]), float(point[1])])
        validated.append(
            {
                "object_id": object_id,
                "four_dor_class": source_class,
                "mmor_class": mapped_class,
                "points": normalized_points,
                "point_labels": [int(value) for value in labels],
            }
        )
        seen.add(object_id)
    return validated


def _propagate_point_prompts_with_predictor(
    predictor,
    video_path: Path,
    prompt_manifest: Path,
    output_path: Path,
    *,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Initialize SAM2 with manual 4D-OR points and propagate unique instances."""

    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    instances = load_point_prompt_manifest(prompt_manifest)
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            str(video_path), offload_video_to_cpu=True, offload_state_to_cpu=True
        )
        width, height = int(state["video_width"]), int(state["video_height"])
        for instance in instances:
            points = np.asarray(instance["points"], dtype=np.float32)
            points[:, 0] *= max(0, width - 1)
            points[:, 1] *= max(0, height - 1)
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=int(instance["object_id"]),
                points=points,
                labels=np.asarray(instance["point_labels"], dtype=np.int32),
            )

        frames: dict[int, NDArray[np.uint8]] = {}
        for frame_index, propagated_ids, mask_logits in predictor.propagate_in_video(state):
            logits = mask_logits[:, 0].float().cpu().numpy()
            frames[int(frame_index)] = compose_label_frame(propagated_ids, logits)

    expected_frames = int(state["num_frames"])
    missing = sorted(set(range(expected_frames)) - frames.keys())
    if missing:
        raise RuntimeError(f"SAM2 did not return frames: {missing}")
    labels = np.stack([frames[index] for index in range(expected_frames)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, labels=labels)
    class_map = {str(row["object_id"]): row["mmor_class"] for row in instances}
    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "sam2.1_hiera_large_manual_point_propagation",
        "dataset": "4DOR",
        "sam2_revision": PINNED_SAM2_COMMIT,
        "sam2_config": SAM2_CONFIG,
        "video_path": str(video_path),
        "prompt_manifest": str(prompt_manifest),
        "shape": list(labels.shape),
        "dtype": str(labels.dtype),
        "instance_classes": class_map,
        "instances": instances,
        "coordinate_conversion": "normalized xy multiplied by (width-1,height-1)",
        "overlap_resolution": "maximum positive SAM2 mask logit; otherwise background",
        "memory_policy": "offload video frames and inference state to CPU",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def propagate_first_frame_mask(
    video_path: Path,
    first_mask_path: Path,
    sam2_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    verify_revision: bool = True,
) -> dict[str, object]:
    runner = Sam2VideoRunner(
        sam2_root,
        checkpoint_path,
        verify_revision=verify_revision,
    )
    return runner.run(video_path, first_mask_path, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--first-mask", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = propagate_first_frame_mask(
        args.video,
        args.first_mask,
        args.sam2_root,
        args.checkpoint,
        args.output,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

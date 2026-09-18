"""Dependency-light scoring for detections, hand crops, and keypoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .protocol import PERSON_MATCH_IOU, box_iou, image_diagonal, linear_sum_assignment

COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12


def detection_rate(detections: Sequence[Mapping[str, object]], frame_count: int) -> dict[str, object]:
    frames = {int(row["frame"]) for row in detections}
    return {
        "frames_with_detection": len(frames),
        "frame_count": frame_count,
        "frame_detection_rate": (len(frames) / frame_count) if frame_count else 0.0,
        "detection_count": len(detections),
        "clips_or_frames_without_detection": frame_count - len(frames),
    }


def nearest_anchor_scores(
    generated: np.ndarray, references: np.ndarray
) -> dict[str, object]:
    if generated.ndim != 2 or references.ndim != 2:
        raise ValueError("Hand embeddings must be [N,D]")
    if not len(generated) or not len(references):
        return {
            "status": "unavailable",
            "reason": "no valid crops on one or both sides",
            "generated_crops": int(len(generated)),
            "reference_crops": int(len(references)),
            "mean_max_cosine": None,
            "mean_cosine": None,
            "score_policy": "no_detection_is_not_zero_or_one",
        }
    generated_norm = generated / np.maximum(np.linalg.norm(generated, axis=1, keepdims=True), 1e-12)
    reference_norm = references / np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-12)
    cosine = generated_norm @ reference_norm.T
    return {
        "status": "ok",
        "generated_crops": int(len(generated)),
        "reference_crops": int(len(references)),
        "mean_max_cosine": float(cosine.max(axis=1).mean()),
        "mean_cosine": float(cosine.mean()),
    }


def summarize_hands_metric(
    *,
    detector_identity: str,
    detector_version: str | None,
    model_asset: str | None,
    embedding_identity: str,
    embedding_version: str | None,
    real_rate: Mapping[str, object],
    generated_rate: Mapping[str, object],
    real_crop_count: int,
    generated_crop_count: int,
    real_failed_frames: int,
    generated_failed_frames: int,
    per_clip: Sequence[Mapping[str, object]],
    embedding_scores: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble a hands result. Missing crops never become score 0 or 1."""

    payload: dict[str, object] = {
        "status": "ok",
        "detector": detector_identity,
        "detector_version": detector_version,
        "model_asset": model_asset,
        "embedding_model": embedding_identity,
        "embedding_version": embedding_version,
        "real": real_rate,
        "generated": generated_rate,
        "valid_real_crops": int(real_crop_count),
        "valid_generated_crops": int(generated_crop_count),
        "real_failed_frames": int(real_failed_frames),
        "generated_failed_frames": int(generated_failed_frames),
        "per_clip": list(per_clip),
        "embedding_scores": dict(embedding_scores),
        "note": "Embedding scores are never reported without detection-failure counts",
        "no_detection_score": "unavailable; never 0 or 1",
    }
    if extra:
        payload.update(dict(extra))
    if embedding_scores.get("status") != "ok":
        payload["score"] = None
        payload["score_status"] = embedding_scores.get("status", "unavailable")
    else:
        payload["score"] = embedding_scores.get("mean_max_cosine")
        payload["score_status"] = "ok"
    return payload


def match_people_by_iou(
    real: Sequence[Mapping[str, object]],
    generated: Sequence[Mapping[str, object]],
    *,
    min_iou: float = PERSON_MATCH_IOU,
) -> dict[str, object]:
    """Same-frame IoU matching. Unmatched people are not given zero MPJPE."""

    matches: list[dict[str, object]] = []
    unmatched_real = 0
    unmatched_generated = 0
    ambiguous = False
    frames = sorted({int(row["frame"]) for row in list(real) + list(generated)})
    for frame in frames:
        real_rows = [row for row in real if int(row["frame"]) == frame]
        gen_rows = [row for row in generated if int(row["frame"]) == frame]
        if not real_rows or not gen_rows:
            unmatched_real += len(real_rows)
            unmatched_generated += len(gen_rows)
            continue
        cost = np.zeros((len(real_rows), len(gen_rows)), dtype=np.float64)
        for i, left in enumerate(real_rows):
            for j, right in enumerate(gen_rows):
                cost[i, j] = 1.0 - box_iou(left["box_xyxy"], right["box_xyxy"])
        rows, cols = linear_sum_assignment(cost)
        used_r, used_c = set(), set()
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            iou = 1.0 - float(cost[row, col])
            if iou < min_iou:
                continue
            matches.append(
                {
                    "frame": frame,
                    "iou": iou,
                    "real": real_rows[row],
                    "generated": gen_rows[col],
                }
            )
            used_r.add(row)
            used_c.add(col)
        unmatched_real += len(real_rows) - len(used_r)
        unmatched_generated += len(gen_rows) - len(used_c)
        if len(real_rows) > 1 and len(gen_rows) > 1 and len(used_r) != min(len(real_rows), len(gen_rows)):
            ambiguous = True
    return {
        "matches": matches,
        "unmatched_real": unmatched_real,
        "unmatched_generated": unmatched_generated,
        "ambiguous": ambiguous,
        "min_iou": min_iou,
    }


def _torso_scale(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    needed = [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]
    if np.any(scores[needed] < threshold):
        return None
    shoulder = 0.5 * (keypoints[COCO_LEFT_SHOULDER] + keypoints[COCO_RIGHT_SHOULDER])
    hip = 0.5 * (keypoints[COCO_LEFT_HIP] + keypoints[COCO_RIGHT_HIP])
    length = float(np.linalg.norm(shoulder - hip))
    return length if length > 1e-6 else None


def mpjpe_from_matches(
    matched: Mapping[str, object],
    *,
    width: int,
    height: int,
    score_threshold: float = 0.3,
) -> dict[str, object]:
    if matched.get("ambiguous"):
        return {
            "status": "unavailable",
            "reason": "identity-consistent person matching is ambiguous; MPJPE not invented",
            "unmatched_real": matched["unmatched_real"],
            "unmatched_generated": matched["unmatched_generated"],
        }
    pixel_errors: list[float] = []
    diagonal_errors: list[float] = []
    torso_errors: list[float] = []
    valid_keypoints = 0
    total_keypoints = 0
    matched_people = 0
    diagonal = image_diagonal(width, height)
    for row in matched["matches"]:
        real_k = np.asarray(row["real"]["keypoints"], dtype=np.float64)
        gen_k = np.asarray(row["generated"]["keypoints"], dtype=np.float64)
        real_s = np.asarray(row["real"]["keypoint_scores"], dtype=np.float64)
        gen_s = np.asarray(row["generated"]["keypoint_scores"], dtype=np.float64)
        if real_k.shape != gen_k.shape:
            return {
                "status": "unavailable",
                "reason": "real/generated keypoint layouts differ",
            }
        valid = (real_s >= score_threshold) & (gen_s >= score_threshold)
        total_keypoints += int(len(valid))
        valid_keypoints += int(valid.sum())
        if not np.any(valid):
            continue
        distances = np.linalg.norm(real_k[valid] - gen_k[valid], axis=1)
        pixel_errors.append(float(distances.mean()))
        diagonal_errors.append(float(distances.mean() / diagonal))
        scale = _torso_scale(real_k, real_s, score_threshold)
        if scale:
            torso_errors.append(float(distances.mean() / scale))
        matched_people += 1
    if not pixel_errors:
        return {
            "status": "unavailable",
            "reason": "no jointly valid keypoints after matching",
            "unmatched_real": matched["unmatched_real"],
            "unmatched_generated": matched["unmatched_generated"],
            "zero_mpjpe_for_misses": False,
        }
    return {
        "status": "ok",
        "matched_people_frames": matched_people,
        "unmatched_real": matched["unmatched_real"],
        "unmatched_generated": matched["unmatched_generated"],
        "valid_keypoint_fraction": valid_keypoints / total_keypoints if total_keypoints else 0.0,
        "mpjpe_pixels": float(np.mean(pixel_errors)),
        "mpjpe_over_image_diagonal": float(np.mean(diagonal_errors)),
        "mpjpe_over_torso": float(np.mean(torso_errors)) if torso_errors else None,
        "zero_mpjpe_for_misses": False,
        "matching": "same-frame Hungarian IoU, not temporal identity tracking",
    }

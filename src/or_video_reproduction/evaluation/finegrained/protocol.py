"""Shared sampling, kernels, and aggregation for every fine-grained metric.

Frames from one video are correlated.  Point estimates may pool selected frame
embeddings when a published metric requires it; uncertainty is obtained by
resampling complete videos, never individual frames.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path

import numpy as np

from or_video_reproduction.data.clips import OUTPUT_FPS, OUTPUT_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH
from or_video_reproduction.evaluation.distribution import frechet_distance

SCHEMA_VERSION = 1
VIDEO_CONTRACT = {
    "width": TARGET_WIDTH,
    "height": TARGET_HEIGHT,
    "frames": OUTPUT_FRAME_COUNT,
    "fps": float(OUTPUT_FPS),
    "pixel_format": "rgb24",
}

# Inclusive endpoints, 16 unique frames from the 97-frame contract.
# linspace(0, 96, 16) rounds to 16 distinct indices and always keeps frame 0.
DEFAULT_SAMPLE_COUNT = 16
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 200
CLIP_RBF_SIGMA = 10.0
MMD_SCALE = 1000.0
POLY_DEGREE = 3
KID_SUBSETS = 100
PRDC_NEAREST_K = 5
CORESET_SIZE = 2048
HAND_MARGIN = 0.20
HAND_MIN_SIDE = 16
HAND_CONFIDENCE = 0.50
HAND_MAX_HANDS = 4
HAND_DETECTOR = "mediapipe.tasks.vision.HandLandmarker"
HAND_MODEL_ASSET = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
JEDI_ENCODER_CHECKPOINT = "vith16.pth.tar"
JEDI_PROBE_CHECKPOINT = "ssv2-probe.pth.tar"
JEDI_DEFAULT_CONFIG_NAME = "vith16_ssv2_16x2x3.yaml"
POSE_CONFIDENCE = 0.50
PERSON_MATCH_IOU = 0.30
IDENTITY_SWITCH_DISTANCE = 80.0

CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float64)
CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float64)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float64)

FROZEN_MODELS = {
    "clip": {
        "name": "CLIP ViT-L/14@336",
        "implementation": "transformers.CLIPModel.get_image_features",
        "huggingface_id": "openai/clip-vit-large-patch14-336",
        "official_reference": "google-research/cmmd Scenic vit_l14_336px",
        "input": "bicubic resize to 336x336, no center crop, RGB in [0,1], CLIP mean/std",
        "feature": "projected image embedding from get_image_features",
        "normalize_embedding": False,
        "kernel": "unbiased Gaussian RBF MMD, sigma=10, scaled by 1000",
    },
    "dinov2": {
        "name": "DINOv2 ViT-L/14",
        "huggingface_id": "facebook/dinov2-large",
        "input": "bicubic resize 256, center crop 224, ImageNet mean/std",
        "feature": "CLS token last_hidden_state[:, 0]",
        "frechet_normalize": False,
        "prdc_normalize": True,
        "nearest_k": PRDC_NEAREST_K,
    },
    "dinov3": {
        "name": "DINOv3 ViT-L/16",
        "huggingface_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "input": "official AutoImageProcessor for the model id",
        "feature": "patch tokens after CLS and register tokens",
        "pooling": "spatial 3x3 average pool, kernel=3, stride=1, padding=1",
        "coreset": "greedy farthest-first k-center, Euclidean, size 2048, seed 42",
        "kernel": "unbiased Gaussian RBF MMD, sigma=10, scaled by 1000",
        "no_silent_fallback": "DINOv2 must not be substituted",
    },
    "depth": {
        "name": "Depth Anything V2 Large",
        "huggingface_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "reason": "independent of Video Depth Anything used to author the ellipse controls",
    },
    "hands": {
        "detector": HAND_DETECTOR,
        "model_asset": HAND_MODEL_ASSET,
        "official_asset_url": HAND_MODEL_URL,
        "no_silent_fallback": "mediapipe.solutions.hands is never substituted",
        "confidence": HAND_CONFIDENCE,
        "max_hands": HAND_MAX_HANDS,
        "crop_margin": HAND_MARGIN,
        "embedding": "CLIP ViT-L/14@336, same preprocessing as CMMD",
        "no_detection_score": "unavailable; never 0 or 1",
    },
    "pose": {
        "person_detector": "facebook/detr-resnet-50 COCO person class",
        "keypoints": "usyd-community/vitpose-plus-base",
        "match": "same-frame Hungarian IoU; MPJPE unavailable if matching is ambiguous",
    },
    "vbench2": {
        "implementation": "Vchitect/VBench VBench-2.0 Human_Anatomy",
        "dimension": "Human_Anatomy",
        "evaluator": "evaluate.py from a VBench-2.0 checkout",
        "mode": "custom_input",
        "no_vendor": "VBench-1.0 evaluate.py is rejected",
    },
    "fvmd": {
        "implementation": "DSL-Lab/FVMD-frechet-video-motion-distance",
        "package": "fvmd==1.0.0",
        "role": "primary motion-distribution metric",
        "invocation": "explicit --fvmd-python -m fvmd --log_dir GEN_PATH GT_PATH",
        "official_sampling": "16-frame segments, stride 1, 256x256, 400 PIPs++ points",
        "input": "per-clip PNG folders or [N,T,H,W,C] npy; not mp4 symlinks",
        "no_fallback": "FVD, optical flow, and in-process sklearn substitutes are forbidden",
    },
    "jedi": {
        "implementation": "oooolga/JEDi videojedi",
        "package": "videojedi==1.1.0",
        "role": "relative ranking only; not an absolute calibrated score",
        "features": "official V-JEPA via videojedi.JEDiMetric.load_features",
        "encoder_checkpoint": JEDI_ENCODER_CHECKPOINT,
        "probe_checkpoint": JEDI_PROBE_CHECKPOINT,
        "config": JEDI_DEFAULT_CONFIG_NAME,
        "preprocessing": "TCHW RGB in [0,1], ImageNet mean/std inside VJEPA",
        "aggregation": "finetuned attentive pooler; polynomial MMD degree=2 coef0=0 x100",
        "no_silent_fallback": "CLIP, DINOv2, I3D, and other extractors are never substituted",
    },
}

WEAK_SAMPLE_WARNING = (
    "Six videos yield a weak distributional estimate: selected frames from one "
    "clip are correlated, covariance estimates for 1024-d features are rank-deficient, "
    "and video-level bootstrap intervals are wide. Do not treat these scores as a "
    "completed cross-method benchmark until every compared method shares this split."
)
WAN_DIAGNOSTIC_WARNING = (
    "The current WAN pair is a single take-9 clip and is not a member of the frozen "
    "six-clip 4D-OR split (takes 2 and 6). Its scores are diagnostic only and must "
    "not be compared with six-video distributional scores."
)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_frame_indices(
    frame_count: int,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    *,
    include_frame_zero: bool = True,
) -> list[int]:
    """Return a deterministic uniform sample that always includes the last frame."""

    if frame_count <= 0 or sample_count <= 0:
        raise ValueError("frame_count and sample_count must be positive")
    if sample_count > frame_count:
        raise ValueError("Cannot sample more frames than the video contains")
    if include_frame_zero:
        raw = np.linspace(0, frame_count - 1, sample_count)
    else:
        if frame_count == 1:
            raise ValueError("Cannot exclude frame zero from a one-frame video")
        raw = np.linspace(1, frame_count - 1, sample_count)
    indices = np.round(raw).astype(np.int64)
    unique = np.unique(indices)
    if len(unique) != sample_count:
        raise ValueError(
            f"Uniform sampling collided: requested {sample_count} frames from "
            f"{frame_count}, got {unique.tolist()}"
        )
    if include_frame_zero and unique[0] != 0:
        raise ValueError("Frame-zero policy was violated")
    if not include_frame_zero and unique[0] == 0:
        raise ValueError("Frame zero was sampled despite being excluded")
    if unique[-1] != frame_count - 1:
        raise ValueError("Sampling must include the final frame")
    return [int(value) for value in unique]


def frame_sampling_manifest(
    *,
    frame_count: int = OUTPUT_FRAME_COUNT,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    include_frame_zero: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    indices = select_frame_indices(
        frame_count, sample_count, include_frame_zero=include_frame_zero
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": "uniform_inclusive_endpoints",
        "frame_count": frame_count,
        "sample_count": sample_count,
        "include_frame_zero": include_frame_zero,
        "include_last_frame": True,
        "seed": seed,
        "seed_role": (
            "Sampling itself is a deterministic linspace and does not consume the seed. "
            "The seed is used for KID subsets, coreset first-point selection, and "
            "video-level bootstrap."
        ),
        "indices": indices,
        "applies_to": "every frame-based embedding metric; real and generated members of a pair share these indices",
        "official_protocol_exceptions": {
            "fvmd": FROZEN_MODELS["fvmd"]["official_sampling"],
            "jedi": "official V-JEPA video-level aggregation",
        },
    }


def rbf_kernel(left: np.ndarray, right: np.ndarray, *, sigma: float = CLIP_RBF_SIGMA) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("RBF bandwidth must be positive")
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Kernel inputs must be [N,D] arrays with the same D")
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True)
    squared = np.maximum(left_norm + right_norm.T - 2.0 * left @ right.T, 0.0)
    return np.exp(-squared / (2.0 * sigma**2))


def polynomial_kernel(left: np.ndarray, right: np.ndarray, *, degree: int = POLY_DEGREE) -> np.ndarray:
    """Standard KID polynomial kernel k(x,y) = (x^T y / d + 1)^degree."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Kernel inputs must be [N,D] arrays with the same D")
    if degree < 1:
        raise ValueError("Polynomial degree must be >= 1")
    dimension = left.shape[1]
    return (left @ right.T / dimension + 1.0) ** degree


def unbiased_mmd2(
    reference: np.ndarray,
    generated: np.ndarray,
    kernel: Callable[..., np.ndarray],
) -> float:
    """U-statistic MMD^2; xx/yy diagonals are excluded. The estimator may be negative."""

    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if len(reference) != len(generated):
        raise ValueError("MMD comparisons require equally many real and generated embeddings")
    count = len(reference)
    if count < 2:
        raise ValueError("Unbiased MMD requires at least two embeddings in each set")
    k_xx = kernel(reference, reference)
    k_yy = kernel(generated, generated)
    k_xy = kernel(reference, generated)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    return float(
        k_xx.sum() / (count * (count - 1))
        + k_yy.sum() / (count * (count - 1))
        - 2.0 * k_xy.mean()
    )


def scaled_unbiased_rbf_mmd(reference: np.ndarray, generated: np.ndarray) -> float:
    return MMD_SCALE * unbiased_mmd2(
        reference, generated, lambda left, right: rbf_kernel(left, right, sigma=CLIP_RBF_SIGMA)
    )


def kid_polynomial(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    subsets: int = KID_SUBSETS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Subset MMD with the degree-3 polynomial kernel.

    ``n`` is the number of embeddings in each set.  Here that is
    ``n_clips * n_sampled_frames`` after the shared frame sampler.  Every
    subset draws ``n // 2`` real embeddings and the same number of generated
    embeddings.  The same n is used on both sides of every comparison.
    """

    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if len(reference) != len(generated):
        raise ValueError("KID requires equally many real and generated embeddings")
    count = len(reference)
    subset_size = max(2, count // 2)
    if subset_size > count:
        raise ValueError("KID subset size exceeds n")
    if subsets < 1:
        raise ValueError("KID subset count must be positive")
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(subsets):
        left = rng.choice(count, size=subset_size, replace=False)
        right = rng.choice(count, size=subset_size, replace=False)
        scores.append(
            unbiased_mmd2(
                reference[left],
                generated[right],
                lambda first, second: polynomial_kernel(first, second, degree=POLY_DEGREE),
            )
        )
    values = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1) if len(values) > 1 else 0.0),
        "subsets": subsets,
        "n": count,
        "n_definition": "number of embeddings in each of the real and generated sets",
        "subset_size": subset_size,
        "seed": seed,
        "kernel": f"(x^T y / d + 1)^{POLY_DEGREE}",
        "estimator": "unbiased MMD^2",
    }


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, np.finfo(np.float64).tiny)
    return values / norms


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True)
    return np.sqrt(np.maximum(left_norm + right_norm.T - 2.0 * left @ right.T, 0.0))


def density_coverage(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    nearest_k: int = PRDC_NEAREST_K,
) -> dict[str, float | int]:
    """Naeem et al. density and coverage (PRDC) with Euclidean k-NN radii."""

    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if len(reference) <= nearest_k or len(generated) <= nearest_k:
        raise ValueError(f"Density/coverage require more than k={nearest_k} embeddings in each set")
    real_real = _pairwise_distances(reference, reference)
    np.fill_diagonal(real_real, np.inf)
    radii = np.partition(real_real, nearest_k - 1, axis=1)[:, nearest_k - 1]
    real_fake = _pairwise_distances(reference, generated)
    density = float((1.0 / nearest_k) * (real_fake < radii[:, None]).sum(axis=0).mean())
    coverage = float((real_fake.min(axis=1) < radii).mean())
    return {
        "density": density,
        "coverage": coverage,
        "nearest_k": nearest_k,
    }


def greedy_kcenter_coreset(
    features: np.ndarray,
    size: int,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Deterministic farthest-first traversal (Gonzalez k-center)."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not len(features):
        raise ValueError("Coreset features must be a non-empty [N,D] array")
    count = min(int(size), len(features))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(features)))
    selected = [start]
    min_distance = _pairwise_distances(features, features[start : start + 1]).reshape(-1)
    min_distance[start] = 0.0
    for _ in range(1, count):
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        updated = _pairwise_distances(features, features[next_index : next_index + 1]).reshape(-1)
        min_distance = np.minimum(min_distance, updated)
        min_distance[next_index] = 0.0
    indices = np.asarray(selected, dtype=np.int64)
    return {
        "indices": indices,
        "features": features[indices],
        "algorithm": "greedy_farthest_first_kcenter",
        "distance": "euclidean",
        "size": count,
        "requested_size": int(size),
        "seed": seed,
        "first_index": start,
    }


def spatial_average_pool_3x3(patches: np.ndarray) -> np.ndarray:
    """Average-pool a [H,W,C] patch grid with a 3x3 window, stride 1, zero-pad 1."""

    grid = np.asarray(patches, dtype=np.float64)
    if grid.ndim != 3:
        raise ValueError("Patch grid must have shape [H,W,C]")
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))
    return windows.mean(axis=(-1, -2))


def greedy_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique min-cost matching by globally sorting edges. Used when n is large."""

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("Assignment cost must be a non-empty 2D array")
    pairs = [
        (float(matrix[row, col]), row, col)
        for row in range(matrix.shape[0])
        for col in range(matrix.shape[1])
    ]
    pairs.sort()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    chosen: list[tuple[int, int]] = []
    limit = min(matrix.shape)
    for _, row, col in pairs:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        chosen.append((row, col))
        if len(chosen) == limit:
            break
    chosen.sort()
    return (
        np.asarray([row for row, _ in chosen], dtype=np.int64),
        np.asarray([col for _, col in chosen], dtype=np.int64),
    )


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact search for small matrices; greedy unique matching otherwise."""

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("Assignment cost must be a non-empty 2D array")
    rows, cols = matrix.shape
    if max(rows, cols) > 8:
        return greedy_assignment(matrix)
    used_cols = np.zeros(cols, dtype=bool)
    best_cost = [np.inf]
    best_choice: list[int] = [-1] * rows

    def search(row: int, acc: float, choice: list[int]) -> None:
        if acc >= best_cost[0]:
            return
        if row == rows:
            best_cost[0] = acc
            best_choice[:] = choice
            return
        order = np.argsort(matrix[row])
        for column in order:
            if used_cols[column]:
                continue
            used_cols[column] = True
            choice[row] = int(column)
            search(row + 1, acc + float(matrix[row, column]), choice)
            used_cols[column] = False
            choice[row] = -1

    if rows <= cols:
        search(0, 0.0, [-1] * rows)
        return np.arange(rows), np.asarray(best_choice, dtype=np.int64)
    assigned_rows, assigned_cols = linear_sum_assignment(matrix.T)
    return assigned_cols, assigned_rows


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    inter_w = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    inter_h = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    inter = inter_w * inter_h
    union = (lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0) - inter
    return float(inter / union) if union else 0.0


def video_bootstrap(
    reference_by_video: Sequence[np.ndarray],
    generated_by_video: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Resample complete videos with replacement and recompute a pooled statistic."""

    if len(reference_by_video) != len(generated_by_video):
        raise ValueError("Bootstrap requires paired real/generated video embedding lists")
    count = len(reference_by_video)
    if count < 2:
        return {
            "status": "unavailable",
            "reason": "video-level bootstrap requires at least two clips",
            "n_videos": count,
        }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, count, size=count)
        reference = np.concatenate([reference_by_video[index] for index in indices], axis=0)
        generated = np.concatenate([generated_by_video[index] for index in indices], axis=0)
        values.append(float(statistic(reference, generated)))
    array = np.asarray(values, dtype=np.float64)
    return {
        "status": "ok",
        "n_videos": count,
        "samples": samples,
        "seed": seed,
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        "unit": "video",
        "warning": WEAK_SAMPLE_WARNING if count <= 6 else None,
    }


def concatenate_video_embeddings(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("No embeddings were provided")
    return np.concatenate([np.asarray(row, dtype=np.float64) for row in rows], axis=0)


def statistical_flags(*, n_videos: int, n_embeddings: int, feature_dim: int | None) -> list[str]:
    flags = []
    if n_videos < 2:
        flags.append("single_video_diagnostic")
    if n_videos <= 6:
        flags.append("weak_video_level_distribution")
    if feature_dim is not None and n_embeddings <= feature_dim:
        flags.append("frechet_covariance_rank_deficient")
    return flags


def clip_preprocess_numpy(frames: np.ndarray, *, size: int = 336) -> np.ndarray:
    """Bicubic warp to square CLIP resolution with official CMMD-style normalization."""

    from PIL import Image

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("CLIP frames must have shape [T,H,W,3]")
    resized = np.empty((len(frames), size, size, 3), dtype=np.float32)
    for index, frame in enumerate(frames):
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
        resized[index] = np.asarray(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return (resized - CLIP_MEAN) / CLIP_STD


def dinov2_preprocess_numpy(frames: np.ndarray) -> np.ndarray:
    from PIL import Image

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("DINOv2 frames must have shape [T,H,W,3]")
    output = np.empty((len(frames), 3, 224, 224), dtype=np.float32)
    for index, frame in enumerate(frames):
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
        resized = image.resize((256, 256), Image.Resampling.BICUBIC)
        width, height = resized.size
        left = (width - 224) // 2
        top = (height - 224) // 2
        crop = np.asarray(resized.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255.0
        output[index] = np.transpose((crop - IMAGENET_MEAN) / IMAGENET_STD, (2, 0, 1))
    return output


def expand_margin(box: Sequence[int], *, width: int, height: int, margin: float) -> list[int]:
    x0, y0, x1, y1 = (int(value) for value in box)
    extra = margin * max(x1 - x0, y1 - y0, 1)
    left = max(0, int(np.floor(x0 - extra)))
    top = max(0, int(np.floor(y0 - extra)))
    right = min(width, int(np.ceil(x1 + extra)))
    bottom = min(height, int(np.ceil(y1 + extra)))
    return [left, top, right, bottom]


def image_diagonal(width: int, height: int) -> float:
    return float(np.hypot(width, height))


__all__ = [
    "CLIP_RBF_SIGMA",
    "CORESET_SIZE",
    "DEFAULT_SAMPLE_COUNT",
    "DEFAULT_SEED",
    "FROZEN_MODELS",
    "HAND_CONFIDENCE",
    "HAND_DETECTOR",
    "HAND_MAX_HANDS",
    "HAND_MODEL_ASSET",
    "JEDI_DEFAULT_CONFIG_NAME",
    "JEDI_ENCODER_CHECKPOINT",
    "JEDI_PROBE_CHECKPOINT",
    "MMD_SCALE",
    "PRDC_NEAREST_K",
    "SCHEMA_VERSION",
    "VIDEO_CONTRACT",
    "WAN_DIAGNOSTIC_WARNING",
    "WEAK_SAMPLE_WARNING",
    "box_iou",
    "clip_preprocess_numpy",
    "concatenate_video_embeddings",
    "density_coverage",
    "dinov2_preprocess_numpy",
    "expand_margin",
    "frame_sampling_manifest",
    "frechet_distance",
    "greedy_kcenter_coreset",
    "image_diagonal",
    "kid_polynomial",
    "l2_normalize",
    "linear_sum_assignment",
    "scaled_unbiased_rbf_mmd",
    "select_frame_indices",
    "spatial_average_pool_3x3",
    "statistical_flags",
    "unbiased_mmd2",
    "video_bootstrap",
    "write_json",
]

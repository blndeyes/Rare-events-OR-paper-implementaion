"""Dependency-light paired, structural, and downstream metrics.

The target paper names these metrics but does not state aggregation details.  The
functions below make the reproduction convention explicit: image metrics are
averaged over frames; IoU is a macro average over every present frame/class pair.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _paired_rgb(reference: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference)
    generated = np.asarray(generated)
    if reference.shape != generated.shape:
        raise ValueError(f"Paired RGB arrays differ in shape: {reference.shape} != {generated.shape}")
    if reference.ndim not in (3, 4) or reference.shape[-1] != 3:
        raise ValueError("RGB input must have shape [H,W,3] or [T,H,W,3]")
    if reference.ndim == 3:
        reference = reference[None]
        generated = generated[None]
    return reference.astype(np.float64), generated.astype(np.float64)


def peak_signal_to_noise_ratio(
    reference: np.ndarray, generated: np.ndarray, *, data_range: float = 255.0
) -> float:
    """Return the arithmetic mean of per-frame RGB PSNR values."""

    reference, generated = _paired_rgb(reference, generated)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = np.mean((reference - generated) ** 2, axis=(1, 2, 3))
    values = np.full(mse.shape, np.inf, dtype=np.float64)
    nonzero = mse != 0
    values[nonzero] = 10.0 * np.log10(data_range**2 / mse[nonzero])
    return float(np.mean(values))


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 != 1 or size < 3 or sigma <= 0:
        raise ValueError("Gaussian window must have odd size >= 3 and positive sigma")
    coordinates = np.arange(size, dtype=np.float64) - size // 2
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _filter_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=axis)
    return np.tensordot(windows, kernel, axes=([-1], [0]))


def _gaussian_filter(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _filter_axis(_filter_axis(values, kernel, 0), kernel, 1)


def structural_similarity(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    data_range: float = 255.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Return frame-mean RGB SSIM using the standard 11x11 Gaussian window."""

    reference, generated = _paired_rgb(reference, generated)
    if min(reference.shape[1:3]) < window_size:
        raise ValueError(f"SSIM images must be at least {window_size} pixels on each side")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    kernel = _gaussian_kernel(window_size, sigma)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    radius = window_size // 2
    frame_scores: list[float] = []
    for x, y in zip(reference, generated, strict=True):
        ux = _gaussian_filter(x, kernel)
        uy = _gaussian_filter(y, kernel)
        uxx = _gaussian_filter(x * x, kernel)
        uyy = _gaussian_filter(y * y, kernel)
        uxy = _gaussian_filter(x * y, kernel)
        vx = np.maximum(0.0, uxx - ux * ux)
        vy = np.maximum(0.0, uyy - uy * uy)
        vxy = uxy - ux * uy
        score = ((2.0 * ux * uy + c1) * (2.0 * vxy + c2)) / (
            (ux * ux + uy * uy + c1) * (vx + vy + c2)
        )
        valid = score[radius:-radius, radius:-radius]
        frame_scores.append(float(np.mean(valid)))
    return float(np.mean(frame_scores))


def _paired_labels(reference: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference)
    generated = np.asarray(generated)
    if reference.shape != generated.shape:
        raise ValueError(f"Paired label arrays differ in shape: {reference.shape} != {generated.shape}")
    if reference.ndim == 2:
        reference = reference[None]
        generated = generated[None]
    if reference.ndim != 3:
        raise ValueError("Label input must have shape [H,W] or [T,H,W]")
    return reference, generated


def _iou_summary(rows: list[tuple[int, int, float]]) -> dict[str, object]:
    if not rows:
        raise ValueError("No foreground frame/class unions were available for IoU")
    per_class: dict[str, list[float]] = {}
    for _, label, value in rows:
        per_class.setdefault(str(label), []).append(value)
    return {
        "mean": float(np.mean([row[2] for row in rows])),
        "observations": len(rows),
        "per_class": {key: float(np.mean(values)) for key, values in sorted(per_class.items())},
    }


def segmentation_iou(
    reference: np.ndarray, generated: np.ndarray, *, background_label: int = 0
) -> dict[str, object]:
    """Macro-average mask IoU over foreground labels present in either input."""

    reference, generated = _paired_labels(reference, generated)
    rows: list[tuple[int, int, float]] = []
    for frame_index, (truth, prediction) in enumerate(zip(reference, generated, strict=True)):
        labels = np.union1d(np.unique(truth), np.unique(prediction))
        for label_value in labels:
            label = int(label_value)
            if label == background_label:
                continue
            truth_mask = truth == label_value
            prediction_mask = prediction == label_value
            union = int(np.count_nonzero(truth_mask | prediction_mask))
            if union:
                intersection = int(np.count_nonzero(truth_mask & prediction_mask))
                rows.append((frame_index, label, intersection / union))
    return _iou_summary(rows)


def _tight_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _box_iou(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def bounding_box_iou(
    reference: np.ndarray, generated: np.ndarray, *, background_label: int = 0
) -> dict[str, object]:
    """Macro-average tight-mask bounding-box IoU over present frame/classes."""

    reference, generated = _paired_labels(reference, generated)
    rows: list[tuple[int, int, float]] = []
    for frame_index, (truth, prediction) in enumerate(zip(reference, generated, strict=True)):
        labels = np.union1d(np.unique(truth), np.unique(prediction))
        for label_value in labels:
            label = int(label_value)
            if label == background_label:
                continue
            a = _tight_box(truth == label_value)
            b = _tight_box(prediction == label_value)
            rows.append((frame_index, label, _box_iou(a, b)))
    return _iou_summary(rows)


def binary_classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], *, positive_label: int = 1
) -> dict[str, float | int | None]:
    """Return accuracy and positive-class recall for the downstream classifier."""

    truth = np.asarray(targets)
    predicted = np.asarray(predictions)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted) or not len(truth):
        raise ValueError("targets and predictions must be non-empty one-dimensional arrays of equal length")
    positives = truth == positive_label
    true_positives = int(np.count_nonzero(positives & (predicted == positive_label)))
    positive_count = int(np.count_nonzero(positives))
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "recall": true_positives / positive_count if positive_count else None,
        "samples": int(len(truth)),
        "positives": positive_count,
    }

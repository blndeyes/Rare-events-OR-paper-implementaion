"""Per-instance depth aggregation and normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray


def mean_valid_depth(
    depth: NDArray[np.number], mask: NDArray[np.bool_], invalid_value: float = 0.0
) -> float:
    depth_array = np.asarray(depth, dtype=np.float64)
    binary = np.asarray(mask, dtype=bool)
    if depth_array.shape != binary.shape:
        raise ValueError(f"Depth {depth_array.shape} and mask {binary.shape} must match")
    values = depth_array[binary]
    valid = values[np.isfinite(values) & (values != invalid_value)]
    if not valid.size:
        raise ValueError("Instance mask contains no valid depth samples")
    return float(valid.mean())


def normalize_depths(
    values: Mapping[str, float], *, smaller_is_nearer: bool
) -> dict[str, float]:
    """Min-max normalize depths so 1.0 always means nearer."""

    if not values:
        return {}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Depth values must be finite")
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {key: 0.5 for key in values}
    normalized = {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}
    if smaller_is_nearer:
        normalized = {key: 1.0 - value for key, value in normalized.items()}
    return normalized


def depth_to_blue(normalized_depth: float) -> int:
    if not 0.0 <= normalized_depth <= 1.0:
        raise ValueError(f"Normalized depth outside [0, 1]: {normalized_depth}")
    return int(round(normalized_depth * 255.0))

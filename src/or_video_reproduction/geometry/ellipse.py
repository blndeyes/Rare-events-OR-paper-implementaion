"""Deterministic moment-matched ellipse fitting and rasterization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Ellipse:
    """Ellipse in image coordinates.

    ``angle_degrees`` is the major-axis angle measured clockwise from the
    positive x axis because image y coordinates increase downwards.
    """

    center_x: float
    center_y: float
    major_diameter: float
    minor_diameter: float
    angle_degrees: float
    source_pixels: int


def fit_ellipse(mask: NDArray[np.bool_], minimum_pixels: int = 5) -> Ellipse:
    """Fit the filled ellipse whose first two moments match a binary mask.

    For a uniformly filled ellipse, covariance eigenvalues are ``a²/4`` and
    ``b²/4``. Therefore full axis diameters are four times the square roots of
    the eigenvalues. This gives an exact continuous solution and avoids an
    OpenCV-specific contour convention.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {binary.shape}")
    y, x = np.nonzero(binary)
    count = int(x.size)
    if count < minimum_pixels:
        raise ValueError(f"Need at least {minimum_pixels} mask pixels, got {count}")

    center_x = float(x.mean())
    center_y = float(y.mean())
    centered = np.stack((x - center_x, y - center_y), axis=0)
    covariance = centered @ centered.T / count
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_value, minor_value = np.maximum(eigenvalues[order], 0.0)
    major_vector = eigenvectors[:, order[0]]

    major_diameter = 4.0 * math.sqrt(float(major_value))
    minor_diameter = 4.0 * math.sqrt(float(minor_value))
    angle = math.degrees(math.atan2(float(major_vector[1]), float(major_vector[0]))) % 180.0
    return Ellipse(
        center_x=center_x,
        center_y=center_y,
        major_diameter=major_diameter,
        minor_diameter=minor_diameter,
        angle_degrees=angle,
        source_pixels=count,
    )


def rasterize_ellipse(
    ellipse: Ellipse, shape: tuple[int, int]
) -> NDArray[np.bool_]:
    """Rasterize an ellipse into a boolean image of ``(height, width)``."""

    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid output shape: {shape}")
    if ellipse.major_diameter <= 0 or ellipse.minor_diameter <= 0:
        raise ValueError("Ellipse diameters must be positive")

    y, x = np.ogrid[:height, :width]
    dx = x - ellipse.center_x
    dy = y - ellipse.center_y
    theta = math.radians(ellipse.angle_degrees)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    along_major = dx * cosine + dy * sine
    along_minor = -dx * sine + dy * cosine
    major_radius = ellipse.major_diameter / 2.0
    minor_radius = ellipse.minor_diameter / 2.0
    return (along_major / major_radius) ** 2 + (along_minor / minor_radius) ** 2 <= 1.0

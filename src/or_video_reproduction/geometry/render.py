"""Render class/depth-encoded ellipse conditioning frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from or_video_reproduction.data.semantics import ENTITY_CLASSES

from .depth import depth_to_blue
from .ellipse import Ellipse, rasterize_ellipse
from .palette import PAPER_36_PALETTE


@dataclass(frozen=True)
class RenderInstance:
    key: str
    class_name: str
    ellipse: Ellipse
    normalized_depth: float
    raw_depth: float | None = None


def render_conditioning(
    instances: list[RenderInstance],
    shape: tuple[int, int],
    *,
    smaller_is_nearer: bool,
) -> NDArray[np.uint8]:
    """Render far-to-near so nearer ellipses deterministically occlude farther ones."""

    height, width = shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def drawing_key(instance: RenderInstance) -> tuple[float, str, str]:
        if instance.raw_depth is None:
            depth_key = float("-inf")
        else:
            depth_key = -instance.raw_depth if smaller_is_nearer else instance.raw_depth
        return depth_key, instance.class_name, instance.key

    for instance in sorted(instances, key=drawing_key):
        if instance.class_name not in ENTITY_CLASSES:
            raise ValueError(
                f"Only segmented entity nodes may be rendered as ellipses; got "
                f"{instance.class_name!r}"
            )
        try:
            red, green = PAPER_36_PALETTE[instance.class_name]
        except KeyError as error:
            raise KeyError(f"Class is outside paper-36 profile: {instance.class_name}") from error
        pixels = rasterize_ellipse(instance.ellipse, shape)
        canvas[pixels] = (red, green, depth_to_blue(instance.normalized_depth))
    return canvas

"""Deterministic red/green semantic encoding for the paper-36 profile."""

from __future__ import annotations

from typing import Final

from or_video_reproduction.data.semantics import vocabulary


# A 6x6 lattice gives 36 well-separated, non-black red/green pairs. The paper
# does not disclose its exact pairs, so this mapping is an explicit hypothesis.
PALETTE_LEVELS: Final = (32, 70, 108, 146, 184, 222)


def build_palette() -> dict[str, tuple[int, int]]:
    labels = vocabulary()
    return {
        label: (PALETTE_LEVELS[index // 6], PALETTE_LEVELS[index % 6])
        for index, label in enumerate(labels)
    }


PAPER_36_PALETTE: Final = build_palette()


def decode_red_green(red: int, green: int) -> str:
    pair = (int(red), int(green))
    matches = [label for label, color in PAPER_36_PALETTE.items() if color == pair]
    if not matches:
        raise KeyError(f"Unknown paper-36 red/green pair: {pair}")
    return matches[0]

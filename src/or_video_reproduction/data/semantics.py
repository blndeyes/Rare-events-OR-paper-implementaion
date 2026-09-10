"""Versioned MMOR semantic vocabulary used by the reproduction.

The target paper reports 36 classes.  The ``paper_36`` interpretation combines
the supplementary material's 21 entity classes with the 15 non-proximity
predicates.  This module keeps the two kinds of label separate: entities may
label geometric instances, while predicates label edges between entities.
"""

from __future__ import annotations

import argparse
import json
from typing import Final


PROFILE_NAME: Final = "paper_36"

ENTITY_CLASSES: Final = (
    "anaesthetist",
    "anesthesia_equipment",
    "assistant_surgeon",
    "c_arm",
    "circulator",
    "drape",
    "drill",
    "hammer",
    "head_surgeon",
    "instrument",
    "instrument_table",
    "mako_robot",
    "monitor",
    "mps",
    "mps_station",
    "nurse",
    "operating_table",
    "patient",
    "saw",
    "student",
    "tracker",
)

PREDICATE_CLASSES: Final = (
    "assisting",
    "calibrating",
    "cementing",
    "cleaning",
    "cutting",
    "drilling",
    "hammering",
    "holding",
    "lying_on",
    "manipulating",
    "preparing",
    "sawing",
    "scanning",
    "suturing",
    "touching",
)

# ``closeTo`` is generated from proximity and is ignored by multiple official
# scene-graph paths.  Excluding it is the only source-backed 21 + 15 reading.
EXCLUDED_SPATIAL_PREDICATES: Final = ("close_to",)

# These occur in official code/data but are not part of the selected paper-36
# vocabulary.  Keeping them explicit prevents silent remapping during parsing.
OFFICIAL_EXTRA_ENTITIES: Final = (
    "secondary_table",
    "unrelated_person",
    "cementer",
)

RAW_LABEL_ALIASES: Final = {
    "ae": "anesthesia_equipment",
    "anest": "anaesthetist",
    "closeto": "close_to",
    "lyingon": "lying_on",
    "ot": "operating_table",
}

# Raw grayscale values used by MMOR's ``segmentation_export_*`` PNGs.  These
# values come from the official panoptic dataset loader, not from the target
# paper. Values 14, 19, 20, 22, and 23 are documented upstream as artifacts.
MMOR_SEGMENTATION_LABELS: Final = {
    1: "instrument_table",
    2: "anesthesia_equipment",
    3: "operating_table",
    4: "mps_station",
    5: "patient",
    6: "drape",
    7: "anaesthetist",
    8: "circulator",
    9: "assistant_surgeon",
    10: "head_surgeon",
    11: "mps",
    12: "nurse",
    13: "drill",
    15: "hammer",
    16: "saw",
    17: "tracker",
    18: "mako_robot",
    24: "monitor",
    25: "c_arm",
    26: "unrelated_person",
    27: "student",
    28: "secondary_table",
    29: "cementer",
}

MMOR_ARTIFACT_LABELS: Final = (14, 19, 20, 22, 23)


def vocabulary() -> tuple[str, ...]:
    """Return the ordered 36-label paper vocabulary."""

    return ENTITY_CLASSES + PREDICATE_CLASSES


def validate_profile() -> None:
    """Raise if an edit breaks the selected semantic contract."""

    assert len(ENTITY_CLASSES) == 21
    assert len(PREDICATE_CLASSES) == 15
    assert len(vocabulary()) == 36
    assert len(set(vocabulary())) == 36
    assert not set(EXCLUDED_SPATIAL_PREDICATES) & set(vocabulary())


def summary() -> dict[str, object]:
    validate_profile()
    return {
        "profile": PROFILE_NAME,
        "entity_count": len(ENTITY_CLASSES),
        "predicate_count": len(PREDICATE_CLASSES),
        "semantic_label_count": len(vocabulary()),
        "entity_classes": list(ENTITY_CLASSES),
        "predicate_classes": list(PREDICATE_CLASSES),
        "excluded_spatial_predicates": list(EXCLUDED_SPATIAL_PREDICATES),
        "official_extra_entities": list(OFFICIAL_EXTRA_ENTITIES),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the MMOR semantic profile")
    parser.add_argument("--compact", action="store_true", help="print JSON on one line")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(summary(), indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

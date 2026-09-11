"""Named precision profiles shared by training gates and future full runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


QUANTIZATION_MODES = {
    "no_change",
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
}
MIXED_PRECISION_MODES = {"bf16", "fp16"}
TRANSFORMER_DTYPES = {"bf16", "fp16"}


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    description: str
    quantization: str
    mixed_precision: str
    transformer_load_dtype: str
    paper_faithful: bool


def load_training_profile(path: Path, name: str) -> TrainingProfile:
    """Load and strictly validate one named hardware/precision profile."""

    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Training profile schema_version must be 1")
    profiles = document.get("profiles", {})
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown training profile {name!r}; available: {available}")

    values = profiles[name]
    profile = TrainingProfile(name=name, **values)
    if profile.quantization not in QUANTIZATION_MODES:
        raise ValueError(f"Unsupported quantization mode: {profile.quantization}")
    if profile.mixed_precision not in MIXED_PRECISION_MODES:
        raise ValueError(f"Unsupported mixed precision: {profile.mixed_precision}")
    if profile.transformer_load_dtype not in TRANSFORMER_DTYPES:
        raise ValueError(
            f"Unsupported transformer load dtype: {profile.transformer_load_dtype}"
        )
    if profile.paper_faithful and (
        profile.quantization != "no_change"
        or profile.mixed_precision != "bf16"
        or profile.transformer_load_dtype != "bf16"
    ):
        raise ValueError("A paper-faithful profile must use unquantized BF16")
    return profile


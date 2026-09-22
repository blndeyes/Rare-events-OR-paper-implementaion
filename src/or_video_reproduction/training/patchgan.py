"""Configurable conditional PatchGAN components for the LTX reproduction.

The paper names PatchGAN and cites pix2pix, but does not disclose its exact
architecture or optimization settings.  This module therefore implements an
explicit reproduction hypothesis rather than claiming author-equivalent code.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ObjectiveName = Literal["bce", "hinge"]


@dataclass(frozen=True)
class PatchGANConfig:
    """Fully specified settings for the optional adversarial objective."""

    enabled: bool
    architecture: str
    sample_channels: int
    condition_channels: int
    base_channels: int
    layers: int
    max_channels: int
    normalization: str
    input_normalization: str
    objective: ObjectiveName
    adversarial_weight: float
    start_step: int
    frame_stride: int
    frame_batch_size: int
    gradient_checkpointing: bool
    learning_rate: float
    beta1: float
    beta2: float
    discriminator_updates: int

    def __post_init__(self) -> None:
        if self.architecture != "pix2pix_70x70_2d":
            raise ValueError(f"Unsupported PatchGAN architecture: {self.architecture}")
        if self.normalization != "instance":
            raise ValueError(f"Unsupported PatchGAN normalization: {self.normalization}")
        if self.input_normalization != "zero_one_to_minus_one_one":
            raise ValueError(
                f"Unsupported PatchGAN input normalization: {self.input_normalization}"
            )
        if self.objective not in ("bce", "hinge"):
            raise ValueError(f"Unsupported adversarial objective: {self.objective}")
        positive_ints = {
            "sample_channels": self.sample_channels,
            "condition_channels": self.condition_channels,
            "base_channels": self.base_channels,
            "layers": self.layers,
            "max_channels": self.max_channels,
            "frame_stride": self.frame_stride,
            "frame_batch_size": self.frame_batch_size,
            "discriminator_updates": self.discriminator_updates,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"PatchGAN integer settings must be positive: {invalid}")
        if self.layers != 3:
            raise ValueError("pix2pix_70x70_2d requires exactly three stride-2 layers")
        if not self.enabled:
            raise ValueError("PatchGANConfig represents an enabled adversarial path")
        if self.adversarial_weight <= 0:
            raise ValueError("adversarial_weight must be positive")
        if self.start_step < 0:
            raise ValueError("start_step cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")

    @property
    def input_channels(self) -> int:
        return self.sample_channels + self.condition_channels

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PatchGANConfig:
        """Parse a strict mapping and reject missing, null, or unknown settings."""

        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - fields)
        missing = sorted(fields - set(value))
        null_fields = sorted(name for name in fields & set(value) if value[name] is None)
        if unknown or missing or null_fields:
            raise ValueError(
                "Invalid PatchGAN configuration: "
                f"unknown={unknown}, missing={missing}, null={null_fields}"
            )
        return cls(**dict(value))


def load_patchgan_config(path: Path) -> PatchGANConfig:
    """Load a standalone or top-level ``patchgan`` YAML mapping."""

    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("PatchGAN YAML must contain a mapping")
    value = payload.get("patchgan", payload)
    if not isinstance(value, Mapping):
        raise TypeError("patchgan must be a mapping")
    return PatchGANConfig.from_mapping(value)


class PatchDiscriminator2D(nn.Module):
    """Conditional pix2pix 70x70 PatchGAN discriminator.

    Inputs are channel-concatenated condition and RGB sample frames.  Three
    stride-2 blocks followed by two stride-1 convolutions produce a spatial
    authenticity map whose logits each see a 70x70 input receptive field.
    """

    def __init__(self, config: PatchGANConfig) -> None:
        super().__init__()
        channels = config.base_channels
        blocks: list[nn.Module] = [
            nn.Conv2d(config.input_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        previous = channels
        for layer in range(1, config.layers):
            channels = min(config.base_channels * (2**layer), config.max_channels)
            blocks.extend(
                [
                    nn.Conv2d(previous, channels, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(channels, affine=False, track_running_stats=False),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            previous = channels
        channels = min(config.base_channels * (2**config.layers), config.max_channels)
        blocks.extend(
            [
                nn.Conv2d(previous, channels, kernel_size=4, stride=1, padding=1),
                nn.InstanceNorm2d(channels, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1),
            ]
        )
        self.network = nn.Sequential(*blocks)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError(f"PatchGAN expects [N,C,H,W], got {tuple(value.shape)}")
        return self.network(value)


def selected_frame_indices(frame_count: int, stride: int) -> tuple[int, ...]:
    """Return deterministic frame indices, always including the final frame."""

    if frame_count <= 0 or stride <= 0:
        raise ValueError("frame_count and stride must be positive")
    indices = list(range(0, frame_count, stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return tuple(indices)


def conditional_video_frames(condition: Tensor, sample: Tensor, *, stride: int) -> Tensor:
    """Build conditional frame pairs from ``[B,C,T,H,W]`` videos."""

    if condition.ndim != 5 or sample.ndim != 5:
        raise ValueError("condition and sample must have shape [B,C,T,H,W]")
    if condition.shape[0] != sample.shape[0] or condition.shape[2:] != sample.shape[2:]:
        raise ValueError(
            "condition and sample must have equal batch, frame, height, and width dimensions"
        )
    indices = selected_frame_indices(sample.shape[2], stride)
    paired = torch.cat([condition[:, :, indices], sample[:, :, indices]], dim=1)
    return paired.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()


class PatchAdversarialLoss(nn.Module):
    """BCE or hinge losses over discriminator patch logits."""

    def __init__(self, objective: ObjectiveName) -> None:
        super().__init__()
        if objective not in ("bce", "hinge"):
            raise ValueError(f"Unsupported adversarial objective: {objective}")
        self.objective = objective

    def discriminator(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        if self.objective == "bce":
            real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            return 0.5 * (real + fake)
        return 0.5 * (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())

    def generator(self, fake_logits: Tensor) -> Tensor:
        if self.objective == "bce":
            return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        return -fake_logits.mean()


class ConditionalPatchGAN(nn.Module):
    """Architecture plus safe discriminator/generator objective entry points."""

    def __init__(self, config: PatchGANConfig) -> None:
        super().__init__()
        self.config = config
        self.discriminator = PatchDiscriminator2D(config)
        self.loss = PatchAdversarialLoss(config.objective)

    def _frames(self, condition: Tensor, sample: Tensor) -> Tensor:
        frames = conditional_video_frames(
            condition,
            sample,
            stride=self.config.frame_stride,
        )
        return frames.mul(2).sub(1)

    def _logits(self, frames: Tensor) -> Tensor:
        discriminator_parameter = next(self.discriminator.parameters())
        frames = frames.to(
            device=discriminator_parameter.device,
            dtype=discriminator_parameter.dtype,
        )
        outputs = []
        for chunk in frames.split(self.config.frame_batch_size):
            if self.config.gradient_checkpointing and torch.is_grad_enabled():
                outputs.append(checkpoint(self.discriminator, chunk, use_reentrant=False))
            else:
                outputs.append(self.discriminator(chunk))
        return torch.cat(outputs)

    def discriminator_loss(self, condition: Tensor, real: Tensor, fake: Tensor) -> Tensor:
        real_input = self._frames(condition, real)
        fake_input = self._frames(condition, fake.detach())
        return self.loss.discriminator(
            self._logits(real_input),
            self._logits(fake_input),
        )

    def generator_loss(self, condition: Tensor, fake: Tensor) -> Tensor:
        fake_input = self._frames(condition, fake)
        return self.loss.generator(self._logits(fake_input))


@contextmanager
def frozen(module: nn.Module):
    """Temporarily freeze parameters while preserving input gradients."""

    states = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield module
    finally:
        for parameter, state in zip(module.parameters(), states, strict=True):
            parameter.requires_grad_(state)


def receptive_field(config: PatchGANConfig) -> int:
    """Calculate the discriminator logit's spatial receptive field."""

    field = 1
    jump = 1
    for stride in [2] * config.layers + [1, 1]:
        field += 3 * jump
        jump *= stride
    return field


def architecture_record(config: PatchGANConfig, model: nn.Module) -> dict[str, object]:
    """Return serializable provenance for reports and checkpoint audits."""

    return {
        "implementation": "conditional_spatial_patchgan_reproduction_hypothesis",
        "paper_equivalent": False,
        "paper_disclosure": "PatchGAN name and pix2pix citation only",
        "config": asdict(config),
        "input_channels": config.input_channels,
        "receptive_field": receptive_field(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def audit_discriminator_checkpoints(
    checkpoints_dir: Path,
    *,
    expected_steps: Sequence[int],
    minimum_bytes: int = 1_000_000,
    opener: Callable[[str], object] | None = None,
) -> dict[str, object]:
    """Audit separate discriminator-only checkpoints without accepting them as generators."""

    if opener is None:
        from safetensors import safe_open

        opener = lambda path: safe_open(path, framework="pt", device="cpu")

    samples: list[dict[str, object]] = []
    for step in expected_steps:
        path = checkpoints_dir / f"discriminator_weights_step_{step:05d}.safetensors"
        sample: dict[str, object] = {
            "step": step,
            "path": str(path),
            "kind": "patchgan_discriminator_only",
            "passed": False,
        }
        errors: list[str] = []
        if not path.is_file():
            sample["errors"] = ["missing checkpoint"]
            samples.append(sample)
            continue
        size = path.stat().st_size
        sample["size_bytes"] = size
        if size < minimum_bytes:
            errors.append(f"size is below {minimum_bytes} bytes")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sample["sha256"] = digest.hexdigest()
        try:
            with opener(str(path)) as checkpoint:
                keys = list(checkpoint.keys())
            sample["tensor_count"] = len(keys)
            if not keys:
                errors.append("checkpoint contains no tensors")
        except Exception as error:  # noqa: BLE001 - persist every audit failure.
            errors.append(f"unreadable safetensors: {type(error).__name__}: {error}")
        sample["errors"] = errors
        sample["passed"] = not errors
        samples.append(sample)
    return {
        "passed": all(bool(sample["passed"]) for sample in samples),
        "expected_steps": list(expected_steps),
        "samples": samples,
    }

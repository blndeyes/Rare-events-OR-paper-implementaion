"""Opt-in PatchGAN integration for the pinned LTX-Video-Trainer.

The integration is implemented as a local mixin so the pinned upstream checkout
stays unchanged. Ordinary LTX training continues to instantiate the upstream
trainer directly.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import Adam

from .patchgan import ConditionalPatchGAN, PatchGANConfig, architecture_record, frozen


def assert_patchgan_compatible_upstream(trainer_class: type) -> None:
    """Fail before model loading if pinned upstream hooks no longer match."""

    markers = {
        "_training_step": (
            "prepare_batch(batch, self._timestep_sampler)",
            "prepare_model_inputs(training_batch)",
            "compute_loss(model_pred, training_batch)",
        ),
        "_save_checkpoint": (
            'filename = f"{prefix}_weights_step_',
            "get_peft_model_state_dict",
        ),
    }
    missing: list[str] = []
    for method_name, required in markers.items():
        source = inspect.getsource(getattr(trainer_class, method_name))
        missing.extend(
            f"{method_name}:{marker}" for marker in required if marker not in source
        )
    if missing:
        raise RuntimeError(
            "Pinned LTX trainer is incompatible with PatchGAN integration; "
            f"missing source markers: {missing}"
        )


def predict_clean_latents(noisy_latents: Tensor, flow_prediction: Tensor, sigmas: Tensor) -> Tensor:
    """Recover x0 from rectified-flow state ``xt = x0 + sigma * velocity``."""

    if noisy_latents.shape != flow_prediction.shape:
        raise ValueError("noisy_latents and flow_prediction must have identical shapes")
    if sigmas.ndim == 1:
        sigmas = sigmas[:, None, None]
    if sigmas.ndim != 3 or sigmas.shape[0] != noisy_latents.shape[0]:
        raise ValueError("sigmas must broadcast as [batch,1,1]")
    return noisy_latents - sigmas.to(noisy_latents) * flow_prediction


def configure_patchgan_vae(vae: Any) -> dict[str, bool]:
    """Enable official Diffusers tiled decode so 97-frame VAE reconstruction can fit a 24 GiB GPU.

    The one-step INT2 gate OOM'd while decoding the full latent volume in one
    ``vae.decode`` call. Tiling, slicing, framewise decode, and VAE gradient
    checkpointing are 4090 plumbing, not a paper-disclosed training setting.
    """

    flags = {
        "tiling": False,
        "slicing": False,
        "gradient_checkpointing": False,
        "framewise_decoding": False,
    }
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
        flags["tiling"] = True
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
        flags["slicing"] = True
    if hasattr(vae, "enable_gradient_checkpointing"):
        vae.enable_gradient_checkpointing()
        flags["gradient_checkpointing"] = True
    if hasattr(vae, "use_framewise_decoding"):
        vae.use_framewise_decoding = True
        flags["framewise_decoding"] = True
    return flags


def target_token_sigmas(sigmas: Tensor, conditioning_mask: Tensor) -> Tensor:
    """Expand sample sigmas and force clean conditioning tokens to timestep zero."""

    if sigmas.ndim != 3 or sigmas.shape[1:] != (1, 1):
        raise ValueError("sigmas must have shape [batch,1,1]")
    if conditioning_mask.ndim != 2 or conditioning_mask.shape[0] != sigmas.shape[0]:
        raise ValueError("conditioning_mask must have shape [batch,tokens]")
    expanded = sigmas.expand(-1, conditioning_mask.shape[1], -1).clone()
    expanded.masked_fill_(conditioning_mask.unsqueeze(-1), 0)
    return expanded


class PatchGANLtxTrainerMixin:
    """Mixin that adds an isolated adversarial objective to the pinned trainer."""

    def __init__(self, trainer_config: Any, patchgan_config: PatchGANConfig) -> None:
        if trainer_config.conditioning.mode != "reference_video":
            raise ValueError("Conditional PatchGAN requires reference_video conditioning")
        if trainer_config.optimization.gradient_accumulation_steps != 1:
            raise ValueError("PatchGAN integration currently requires gradient accumulation of 1")
        super().__init__(trainer_config)
        if self._accelerator.num_processes != 1:
            raise ValueError("PatchGAN integration currently supports one training process")
        self._patchgan_config = patchgan_config
        self._patchgan = ConditionalPatchGAN(patchgan_config).to(self._accelerator.device)
        optimizer = Adam(
            self._patchgan.discriminator.parameters(),
            lr=patchgan_config.learning_rate,
            betas=(patchgan_config.beta1, patchgan_config.beta2),
        )
        self._patchgan, self._patchgan_optimizer = self._accelerator.prepare(
            self._patchgan, optimizer
        )
        self._vae.eval().to(self._accelerator.device)
        self._patchgan_vae_flags = configure_patchgan_vae(self._vae)
        self._last_patchgan_metrics: dict[str, float] = {}

    def _decode_latent_batch(
        self,
        packed_latents: Tensor,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tensor:
        from ltxv_trainer.ltxv_utils import decode_video

        decode_dtype = getattr(self._vae, "dtype", torch.float32)
        decoded = [
            decode_video(
                self._vae,
                packed_latents[index],
                num_frames=num_frames,
                height=height,
                width=width,
                device=self._accelerator.device,
                dtype=decode_dtype,
            )
            for index in range(packed_latents.shape[0])
        ]
        return torch.cat(decoded)

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> Tensor:
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)
        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)
        model_prediction = self._transformer(**model_inputs)[0]
        flow_loss = self._training_strategy.compute_loss(model_prediction, training_batch)

        if self._global_step < self._patchgan_config.start_step:
            self._last_patchgan_metrics = {
                "train/flow_loss": float(flow_loss.detach()),
                "train/adversarial_generator_loss": 0.0,
                "train/adversarial_discriminator_loss": 0.0,
            }
            return flow_loss

        target_length = training_batch.targets.shape[1]
        target_prediction = model_prediction[:, -target_length:]
        noisy_target = training_batch.latents[:, -target_length:]
        target_conditioning_mask = training_batch.conditioning_mask[:, -target_length:]
        effective_sigmas = target_token_sigmas(
            training_batch.sigmas,
            target_conditioning_mask,
        )
        predicted_clean = predict_clean_latents(
            noisy_target,
            target_prediction,
            effective_sigmas,
        )
        decode_args = {
            "num_frames": training_batch.num_frames,
            "height": training_batch.height,
            "width": training_batch.width,
        }
        del model_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fake_video = self._decode_latent_batch(predicted_clean, **decode_args)
        with torch.no_grad():
            real_video = self._decode_latent_batch(batch["latents"]["latents"], **decode_args)
            condition_video = self._decode_latent_batch(
                batch["ref_latents"]["latents"], **decode_args
            )

        discriminator_loss = torch.zeros((), device=fake_video.device)
        for _ in range(self._patchgan_config.discriminator_updates):
            self._patchgan_optimizer.zero_grad(set_to_none=True)
            discriminator_loss = self._patchgan.discriminator_loss(
                condition_video, real_video, fake_video
            )
            self._accelerator.backward(discriminator_loss)
            self._patchgan_optimizer.step()

        with frozen(self._patchgan.discriminator):
            generator_loss = self._patchgan.generator_loss(condition_video, fake_video)
        total_loss = flow_loss + self._patchgan_config.adversarial_weight * generator_loss
        self._last_patchgan_metrics = {
            "train/flow_loss": float(flow_loss.detach()),
            "train/adversarial_generator_loss": float(generator_loss.detach()),
            "train/adversarial_discriminator_loss": float(discriminator_loss.detach()),
        }
        return total_loss

    def _log_metrics(self, metrics: dict[str, object]) -> None:
        super()._log_metrics({**metrics, **self._last_patchgan_metrics})

    def _save_checkpoint(self) -> Path:
        generator_path = super()._save_checkpoint()
        if not self._accelerator.is_main_process:
            return generator_path

        from safetensors.torch import save_file

        discriminator = self._accelerator.unwrap_model(self._patchgan).discriminator
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in discriminator.state_dict().items()
        }
        step = int(self._global_step)
        path = generator_path.parent / f"discriminator_weights_step_{step:05d}.safetensors"
        record = architecture_record(self._patchgan_config, discriminator)
        save_file(
            state,
            path,
            metadata={
                "global_step": str(step),
                "kind": "patchgan_discriminator_only",
                "architecture": json.dumps(record, sort_keys=True),
            },
        )
        return generator_path


def create_patchgan_trainer(trainer_config: Any, patchgan_config: PatchGANConfig) -> Any:
    """Construct an opt-in trainer while importing upstream only at runtime."""

    from ltxv_trainer.trainer import LtxvTrainer

    assert_patchgan_compatible_upstream(LtxvTrainer)

    class PatchGANLtxTrainer(PatchGANLtxTrainerMixin, LtxvTrainer):
        pass

    return PatchGANLtxTrainer(trainer_config, patchgan_config)

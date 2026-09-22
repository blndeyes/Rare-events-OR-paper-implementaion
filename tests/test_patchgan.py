import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from or_video_reproduction.training.patchgan import (
    ConditionalPatchGAN,
    PatchDiscriminator2D,
    PatchGANConfig,
    architecture_record,
    audit_discriminator_checkpoints,
    conditional_video_frames,
    frozen,
    load_patchgan_config,
    receptive_field,
    selected_frame_indices,
)
from or_video_reproduction.training.patchgan_ltx import (
    assert_patchgan_compatible_upstream,
    call_with_cuda_oom_offload,
    configure_patchgan_vae,
    module_device,
    predict_clean_latents,
    target_token_sigmas,
)


class _ReadableCheckpoint:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def keys(self):
        return ["network.0.weight"]


def get_peft_model_state_dict(value):
    return value


class _CompatibleTrainer:
    def _training_step(self, batch):
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)
        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)
        model_pred = self._transformer(**model_inputs)[0]
        return self._training_strategy.compute_loss(model_pred, training_batch)

    def _save_checkpoint(self):
        prefix = "lora"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        return get_peft_model_state_dict(self._transformer), filename


class _IncompatibleTrainer:
    def _training_step(self, batch):
        return batch

    def _save_checkpoint(self):
        return None


def _config(**overrides) -> PatchGANConfig:
    values = {
        "enabled": True,
        "architecture": "pix2pix_70x70_2d",
        "sample_channels": 3,
        "condition_channels": 3,
        "base_channels": 8,
        "layers": 3,
        "max_channels": 64,
        "normalization": "instance",
        "input_normalization": "zero_one_to_minus_one_one",
        "objective": "bce",
        "adversarial_weight": 0.1,
        "start_step": 0,
        "frame_stride": 8,
        "frame_batch_size": 1,
        "gradient_checkpointing": True,
        "learning_rate": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
        "discriminator_updates": 1,
    }
    values.update(overrides)
    return PatchGANConfig(**values)


class PatchGANTests(unittest.TestCase):
    def test_pix2pix_architecture_has_70px_field_and_patch_output(self) -> None:
        config = _config()
        model = PatchDiscriminator2D(config)

        result = model(torch.randn(2, 6, 256, 256))
        record = architecture_record(config, model)

        self.assertEqual(result.shape, (2, 1, 30, 30))
        self.assertEqual(receptive_field(config), 70)
        self.assertEqual(record["receptive_field"], 70)
        self.assertGreater(record["parameter_count"], 0)
        self.assertFalse(record["paper_equivalent"])

    def test_video_pairing_is_ordered_and_includes_final_frame(self) -> None:
        condition = torch.zeros(1, 3, 10, 64, 64)
        sample = torch.zeros(1, 3, 10, 64, 64)
        for frame in range(10):
            condition[:, :, frame] = frame
            sample[:, :, frame] = 100 + frame

        paired = conditional_video_frames(condition, sample, stride=4)

        self.assertEqual(selected_frame_indices(10, 4), (0, 4, 8, 9))
        self.assertEqual(paired.shape, (4, 6, 64, 64))
        self.assertEqual(paired[:, 0, 0, 0].tolist(), [0, 4, 8, 9])
        self.assertEqual(paired[:, 3, 0, 0].tolist(), [100, 104, 108, 109])

    def test_discriminator_detaches_fake_but_generator_reaches_fake(self) -> None:
        patchgan = ConditionalPatchGAN(_config(frame_stride=2))
        condition = torch.randn(1, 3, 3, 64, 64)
        real = torch.randn(1, 3, 3, 64, 64)
        fake = torch.randn(1, 3, 3, 64, 64, requires_grad=True)

        discriminator_loss = patchgan.discriminator_loss(condition, real, fake)
        discriminator_loss.backward()
        self.assertIsNone(fake.grad)

        patchgan.discriminator.zero_grad(set_to_none=True)
        with frozen(patchgan.discriminator):
            generator_loss = patchgan.generator_loss(condition, fake)
            generator_loss.backward()

        self.assertIsNotNone(fake.grad)
        self.assertGreater(float(fake.grad.abs().sum()), 0)
        self.assertTrue(
            all(parameter.grad is None for parameter in patchgan.discriminator.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in patchgan.discriminator.parameters())
        )

    def test_discriminator_casts_bfloat16_frames_to_parameter_dtype(self) -> None:
        patchgan = ConditionalPatchGAN(_config(frame_stride=2))
        condition = torch.rand(1, 3, 3, 96, 96, dtype=torch.bfloat16)
        sample = torch.rand(1, 3, 3, 96, 96, dtype=torch.bfloat16, requires_grad=True)

        loss = patchgan.generator_loss(condition, sample)
        loss.backward()

        self.assertEqual(loss.dtype, torch.float32)
        self.assertIsNotNone(sample.grad)

    def test_strict_config_load_rejects_undisclosed_null_weight(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.yaml"
            valid.write_text(
                """patchgan:
  enabled: true
  architecture: pix2pix_70x70_2d
  sample_channels: 3
  condition_channels: 3
  base_channels: 8
  layers: 3
  max_channels: 64
  normalization: instance
  input_normalization: zero_one_to_minus_one_one
  objective: hinge
  adversarial_weight: 0.1
  start_step: 10
  frame_stride: 8
  frame_batch_size: 1
  gradient_checkpointing: true
  learning_rate: 0.0002
  beta1: 0.5
  beta2: 0.999
  discriminator_updates: 1
""",
                encoding="utf-8",
            )
            invalid = root / "invalid.yaml"
            invalid.write_text(
                valid.read_text(encoding="utf-8").replace(
                    "  adversarial_weight: 0.1", "  adversarial_weight: null"
                ),
                encoding="utf-8",
            )

            loaded = load_patchgan_config(valid)
            with self.assertRaisesRegex(ValueError, "null=.*adversarial_weight"):
                load_patchgan_config(invalid)

        self.assertEqual(loaded.objective, "hinge")
        self.assertEqual(loaded.start_step, 10)

    def test_rejects_non_70x70_layer_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly three"):
            _config(layers=4)

    def test_checked_in_hypothesis_is_complete_and_explicit(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "configs"
            / "patchgan_ltx_step600_hypothesis.yaml"
        )

        config = load_patchgan_config(path)

        self.assertEqual(config.architecture, "pix2pix_70x70_2d")
        self.assertEqual(config.input_normalization, "zero_one_to_minus_one_one")
        self.assertEqual(config.frame_batch_size, 1)
        self.assertTrue(config.gradient_checkpointing)

    def test_rectified_flow_reconstruction_recovers_clean_latents(self) -> None:
        clean = torch.randn(2, 5, 4)
        noise = torch.randn_like(clean)
        sigmas = torch.tensor([0.2, 0.8])[:, None, None]
        velocity = noise - clean
        noisy = clean + sigmas * velocity

        recovered = predict_clean_latents(noisy, velocity, sigmas)

        torch.testing.assert_close(recovered, clean)

    def test_conditioned_target_tokens_keep_zero_sigma(self) -> None:
        sigmas = torch.tensor([0.25, 0.75])[:, None, None]
        mask = torch.tensor([[True, False, False], [False, True, False]])

        result = target_token_sigmas(sigmas, mask)

        torch.testing.assert_close(
            result.squeeze(-1),
            torch.tensor([[0.0, 0.25, 0.25], [0.75, 0.0, 0.75]]),
        )

    def test_upstream_contract_rejects_missing_training_hooks(self) -> None:
        assert_patchgan_compatible_upstream(_CompatibleTrainer)
        with self.assertRaisesRegex(RuntimeError, "missing source markers"):
            assert_patchgan_compatible_upstream(_IncompatibleTrainer)

    def test_discriminator_audit_labels_weights_as_discriminator_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "discriminator_weights_step_00600.safetensors"
            path.write_bytes(b"audited-checkpoint")
            audit = audit_discriminator_checkpoints(
                root,
                expected_steps=(600,),
                minimum_bytes=1,
                opener=lambda _: _ReadableCheckpoint(),
            )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["samples"][0]["kind"], "patchgan_discriminator_only")
        self.assertEqual(len(audit["samples"][0]["sha256"]), 64)

    def test_configure_patchgan_vae_enables_tiled_framewise_decode(self) -> None:
        class _Vae:
            def __init__(self) -> None:
                self.use_framewise_decoding = False
                self.calls: list[str] = []

            def enable_tiling(self) -> None:
                self.calls.append("tiling")

            def enable_slicing(self) -> None:
                self.calls.append("slicing")

            def enable_gradient_checkpointing(self) -> None:
                self.calls.append("gradient_checkpointing")

        vae = _Vae()
        flags = configure_patchgan_vae(vae)

        self.assertEqual(
            flags,
            {
                "tiling": True,
                "slicing": True,
                "gradient_checkpointing": True,
                "framewise_decoding": True,
                "tile_sample_min_height": 256,
                "tile_sample_min_width": 256,
                "tile_sample_stride_height": 224,
                "tile_sample_stride_width": 224,
            },
        )
        self.assertTrue(vae.use_framewise_decoding)
        self.assertEqual(vae.tile_sample_min_height, 256)
        self.assertEqual(vae.tile_sample_stride_height, 224)
        self.assertEqual(vae.calls, ["tiling", "slicing", "gradient_checkpointing"])

    def test_cuda_oom_offloads_modules_to_cpu_then_retries(self) -> None:
        class _FakeParam:
            def __init__(self) -> None:
                self.device = torch.device("cuda")

        class _FakeModule:
            def __init__(self, name: str) -> None:
                self.name = name
                self._param = _FakeParam()

            def parameters(self):
                yield self._param

            def to(self, device):
                self._param.device = torch.device(device)
                return self

        transformer = _FakeModule("transformer")
        vae = _FakeModule("vae")
        calls: list[str] = []

        def decode():
            calls.append(f"{module_device(transformer).type}:{module_device(vae).type}")
            if module_device(transformer).type != "cpu":
                raise torch.cuda.OutOfMemoryError("need transformer offload")
            if module_device(vae).type != "cpu":
                raise torch.cuda.OutOfMemoryError("need vae offload")
            return "ok"

        result, offloaded = call_with_cuda_oom_offload(decode, (transformer, vae))

        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["cuda:cuda", "cpu:cuda", "cpu:cpu"])
        self.assertEqual(len(offloaded), 2)
        self.assertEqual(module_device(transformer).type, "cpu")
        self.assertEqual(module_device(vae).type, "cpu")

    def test_cuda_oom_restores_modules_when_every_retry_fails(self) -> None:
        class _FakeParam:
            def __init__(self) -> None:
                self.device = torch.device("cuda")

        class _FakeModule:
            def __init__(self) -> None:
                self._param = _FakeParam()

            def parameters(self):
                yield self._param

            def to(self, device):
                self._param.device = torch.device(device)
                return self

        module = _FakeModule()

        def always_fail():
            raise torch.cuda.OutOfMemoryError("still oom")

        with self.assertRaises(torch.cuda.OutOfMemoryError):
            call_with_cuda_oom_offload(always_fail, (module,))

        self.assertEqual(module_device(module).type, "cuda")


if __name__ == "__main__":
    unittest.main()

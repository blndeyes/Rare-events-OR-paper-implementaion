"""Focused checks for cross-host PatchGAN transport and resumable state."""

from __future__ import annotations

import math
import socket
import threading
import time

import numpy as np
import torch

from or_video_reproduction.training.dual_host_patchgan import PatchGANWorker, RemotePatchGANClient, _receive, _send, serve
from or_video_reproduction.training.patchgan import PatchGANConfig


def _config() -> PatchGANConfig:
    return PatchGANConfig(
        enabled=True,
        architecture="latent_patchgan_16x16_2d",
        sample_channels=128,
        condition_channels=128,
        base_channels=4,
        layers=1,
        max_channels=8,
        normalization="instance",
        input_normalization="identity",
        objective="bce",
        adversarial_weight=0.1,
        start_step=0,
        frame_stride=4,
        frame_batch_size=2,
        gradient_checkpointing=False,
        learning_rate=2e-4,
        beta1=0.5,
        beta2=0.999,
        discriminator_updates=1,
        domain="latent",
    )


def test_framed_tensor_roundtrip() -> None:
    left, right = socket.socketpair()
    values = [np.arange(12, dtype=np.float32).reshape(1, 3, 4)]
    thread = threading.Thread(target=_send, args=(left, {"command": "train", "step": 1}, values))
    try:
        thread.start()
        message, decoded = _receive(right)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert message == {"command": "train", "step": 1}
        np.testing.assert_array_equal(decoded[0], values[0])
    finally:
        left.close()
        right.close()


def test_worker_gradient_and_full_state_resume(tmp_path) -> None:
    torch.manual_seed(42)
    worker = PatchGANWorker(_config(), tmp_path / "worker", device="cpu")
    shape = (1, 128, 5, 24, 32)
    rng = np.random.default_rng(42)
    values = [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]
    response, gradients = worker.train(1, values)
    assert response["step"] == 1
    assert math.isfinite(response["generator_loss"])
    assert math.isfinite(response["discriminator_loss"])
    assert gradients[0].shape == shape
    assert np.isfinite(gradients[0]).all()
    assert np.any(gradients[0] != 0)
    checkpoint = worker.checkpoint(1)
    assert checkpoint["bytes"] > 0
    worker.train(2, values)
    assert worker.resume(1, checkpoint["sha256"]) == {"status": "ok", "step": 1}
    assert worker.step == 1
    resumed, _ = worker.train(2, values)
    assert resumed["step"] == 2


def test_client_worker_protocol_on_loopback(tmp_path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    worker = PatchGANWorker(_config(), tmp_path / "loopback", device="cpu")
    server = threading.Thread(target=serve, args=("127.0.0.1", port, "t" * 40, worker), daemon=True)
    server.start()
    rng = np.random.default_rng(8)
    values = [torch.from_numpy(rng.standard_normal((1, 128, 5, 24, 32)).astype(np.float32)) for _ in range(3)]
    client = None
    for _ in range(100):
        try:
            client = RemotePatchGANClient("127.0.0.1", port, "t" * 40)
            break
        except ConnectionRefusedError:
            time.sleep(0.01)
    assert client is not None
    try:
        gradient, losses = client.train(1, *values)
        assert gradient.shape == values[2].shape
        assert math.isfinite(losses["generator_loss"])
        checkpoint = client.checkpoint(1)
        client.resume(1, checkpoint["sha256"])
        client.shutdown()
    finally:
        client.close()
    server.join(timeout=5)
    assert not server.is_alive()

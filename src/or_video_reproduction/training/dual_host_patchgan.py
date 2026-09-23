"""Run one PatchGAN discriminator on a second host's GPU.

The LTX generator remains BF16/rank-128 on the first host.  This transport
returns the discriminator's gradient with respect to clean generated latents,
so the generator can backpropagate through its own graph without placing a
second copy of the 13B transformer on the discriminator host.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .patchgan import ConditionalPatchGAN, PatchGANConfig, frozen

_MAX_HEADER = 1_000_000
_MAX_ARRAY = 128 * 1024 * 1024


def _read_exact(sock: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        part = sock.recv(min(count - len(chunks), 1024 * 1024))
        if not part:
            raise ConnectionError("PatchGAN peer disconnected")
        chunks.extend(part)
    return bytes(chunks)


def _send(sock: socket.socket, message: dict[str, Any], arrays: list[np.ndarray] | None = None) -> None:
    normalized = [np.ascontiguousarray(value, dtype=np.float32) for value in arrays or []]
    header = {**message, "arrays": [{"shape": list(value.shape), "bytes": value.nbytes} for value in normalized]}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_HEADER:
        raise ValueError("PatchGAN message header is too large")
    sock.sendall(struct.pack("!Q", len(encoded)))
    sock.sendall(encoded)
    for value in normalized:
        if value.nbytes > _MAX_ARRAY:
            raise ValueError("PatchGAN tensor is too large")
        sock.sendall(memoryview(value).cast("B"))


def _receive(sock: socket.socket) -> tuple[dict[str, Any], list[np.ndarray]]:
    length = struct.unpack("!Q", _read_exact(sock, 8))[0]
    if length > _MAX_HEADER:
        raise ValueError("PatchGAN message header is too large")
    header = json.loads(_read_exact(sock, length))
    arrays = []
    for spec in header.pop("arrays", []):
        shape = tuple(int(value) for value in spec["shape"])
        count = int(spec["bytes"])
        if count < 0 or count > _MAX_ARRAY or int(np.prod(shape)) * 4 != count:
            raise ValueError("Invalid PatchGAN tensor shape or byte count")
        arrays.append(np.frombuffer(_read_exact(sock, count), dtype=np.float32).reshape(shape).copy())
    return header, arrays


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RemotePatchGANClient:
    def __init__(self, host: str, port: int, token: str, *, timeout: float = 180.0) -> None:
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.socket.settimeout(timeout)
        self.token = token

    def _call(self, command: str, *, arrays: list[np.ndarray] | None = None, **fields: Any) -> tuple[dict[str, Any], list[np.ndarray]]:
        _send(self.socket, {"command": command, "token": self.token, **fields}, arrays)
        response, values = _receive(self.socket)
        if response.get("status") != "ok":
            raise RuntimeError(f"Remote PatchGAN {command} failed: {response.get('error')}")
        return response, values

    def train(self, step: int, condition: torch.Tensor, real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        tensors = [value.detach().float().cpu().numpy() for value in (condition, real, fake)]
        response, arrays = self._call("train", step=step, arrays=tensors)
        if len(arrays) != 1 or arrays[0].shape != tuple(fake.shape):
            raise ValueError("Remote PatchGAN returned a mismatched gradient")
        gradient = torch.from_numpy(arrays[0]).to(device=fake.device, dtype=fake.dtype)
        if not torch.isfinite(gradient).all():
            raise ValueError("Remote PatchGAN returned a nonfinite gradient")
        return gradient, {"generator_loss": float(response["generator_loss"]), "discriminator_loss": float(response["discriminator_loss"])}

    def checkpoint(self, step: int) -> dict[str, Any]:
        response, _ = self._call("checkpoint", step=step)
        return {key: response[key] for key in ("step", "path", "sha256", "bytes")}

    def resume(self, step: int, sha256: str) -> None:
        self._call("resume", step=step, sha256=sha256)

    def shutdown(self) -> None:
        self._call("shutdown")

    def close(self) -> None:
        self.socket.close()


class PatchGANWorker:
    def __init__(self, config: PatchGANConfig, output_dir: Path, *, resume: bool = False, device: str = "cuda") -> None:
        if config.domain != "latent":
            raise ValueError("Remote PatchGAN worker currently requires latent-domain input")
        output_dir.mkdir(parents=True, exist_ok=resume)
        if not resume and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing nonempty PatchGAN output: {output_dir}")
        self.output_dir = output_dir
        self.config = config
        self.device = torch.device(device)
        self.model = ConditionalPatchGAN(config).to(self.device).train()
        self.optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.step = 0

    def train(self, step: int, values: list[np.ndarray]) -> tuple[dict[str, Any], list[np.ndarray]]:
        if step != self.step + 1 or len(values) != 3:
            raise ValueError(f"Expected PatchGAN step {self.step + 1} and three tensors")
        condition, real, fake = [torch.from_numpy(value).to(self.device) for value in values]
        if condition.shape != real.shape or real.shape != fake.shape or not all(torch.isfinite(x).all() for x in (condition, real, fake)):
            raise ValueError("PatchGAN inputs have mismatched shapes or nonfinite values")
        for _ in range(self.config.discriminator_updates):
            self.optimizer.zero_grad(set_to_none=True)
            d_loss = self.model.discriminator_loss(condition, real, fake)
            d_loss.backward()
            self.optimizer.step()
        leaf = fake.detach().requires_grad_(True)
        with frozen(self.model.discriminator):
            g_loss = self.model.generator_loss(condition, leaf)
        (gradient,) = torch.autograd.grad(g_loss, leaf)
        if not all(torch.isfinite(x).all() for x in (d_loss, g_loss, gradient)):
            raise ValueError("PatchGAN loss or gradient is nonfinite")
        self.step = step
        return ({"status": "ok", "step": step, "generator_loss": float(g_loss.detach()), "discriminator_loss": float(d_loss.detach())}, [gradient.detach().cpu().numpy()])

    def checkpoint(self, step: int) -> dict[str, Any]:
        if step != self.step:
            raise ValueError(f"PatchGAN checkpoint step {step} differs from worker step {self.step}")
        destination = self.output_dir / f"discriminator_full_state_step_{step:05d}.pt"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        temporary = destination.with_suffix(".tmp")
        torch.save({"step": step, "config": self.config.__dict__, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if self.device.type == "cuda" else None}, temporary)
        os.replace(temporary, destination)
        return {"status": "ok", "step": step, "path": str(destination), "sha256": _sha256(destination), "bytes": destination.stat().st_size}

    def resume(self, step: int, expected_sha256: str) -> dict[str, Any]:
        path = self.output_dir / f"discriminator_full_state_step_{step:05d}.pt"
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise ValueError("Remote discriminator checkpoint is missing or has wrong SHA-256")
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload["step"] != step or payload["config"] != self.config.__dict__:
            raise ValueError("Remote discriminator checkpoint configuration differs")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["cpu_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"].cpu())
        self.step = step
        return {"status": "ok", "step": step}


def serve(host: str, port: int, token: str, worker: PatchGANWorker) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        print(f"PatchGAN worker listening on {host}:{port}", flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(3600)
                while True:
                    try:
                        request, values = _receive(connection)
                    except ConnectionError:
                        break
                    try:
                        if not hmac.compare_digest(str(request.get("token", "")), token):
                            raise PermissionError("PatchGAN worker token differs")
                        command = request.get("command")
                        if command == "train":
                            response, outputs = worker.train(int(request["step"]), values)
                        elif command == "checkpoint":
                            response, outputs = worker.checkpoint(int(request["step"])), []
                        elif command == "resume":
                            response, outputs = worker.resume(int(request["step"]), request["sha256"]), []
                        elif command == "shutdown":
                            _send(connection, {"status": "ok"})
                            return
                        else:
                            raise ValueError(f"Unknown PatchGAN command: {command}")
                        _send(connection, response, outputs)
                    except Exception as error:
                        _send(connection, {"status": "error", "error": f"{type(error).__name__}: {error}"})
                        break

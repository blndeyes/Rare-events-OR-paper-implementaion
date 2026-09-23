"""One BF16/rank-128 coverage run with LTX on Stone and PatchGAN on PC.

The reviewed coverage runtime remains the source of the training loop, sample
cursor, validation, and generator full-state checkpoints.  This launcher adds a
remote latent PatchGAN gradient at every optimizer update and binds its own
discriminator checkpoint to each generator checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .dual_host_patchgan import PatchGANWorker, RemotePatchGANClient, serve
from .patchgan import load_patchgan_config
from .patchgan_ltx import predict_clean_latents, target_token_sigmas, unpack_packed_latents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _import_stone_runtime(run_path: Path, run: dict[str, Any]) -> tuple[Any, Any]:
    entrypoint = run_path.parent / "train_runner.py"
    spec = importlib.util.spec_from_file_location("coverage_train_runner", entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import coverage admission from {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path[:0] = [run["runtime"], str(Path(run["upstream"]) / "src")]
    from coverage_trainer import CoverageTrainer

    return module, CoverageTrainer


def _load_batch(row: dict[str, Any], trainer: Any, check_file: Any) -> dict[str, Any]:
    from torch.utils.data import default_collate

    payload = {}
    for field in ("latents", "ref_latents", "conditions"):
        artifact = row["encoded"][field]
        check_file(artifact)
        payload[field] = torch.load(artifact["path"], map_location="cpu", weights_only=True)
    if payload["latents"]["latents"].shape != payload["ref_latents"]["latents"].shape:
        raise ValueError("Target/control latent shapes differ")
    if tuple(payload["latents"]["latents"].shape) != (13 * 24 * 32, 128):
        raise ValueError("Expected 97-frame 1024x768 packed latents")

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise ValueError("Nonfinite encoded sample")
            return value.to(trainer._accelerator.device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return move(default_collate([payload]))


def _run_generator(args: argparse.Namespace) -> int:
    from accelerate.utils import set_seed

    run = json.loads(args.run.read_text(encoding="utf-8"))
    if run.get("arm") != "A" or run.get("expected_train_count") != 645:
        raise ValueError("Coverage PatchGAN requires all 645 train pairs in arm A")
    if args.until < 1 or args.until > 3000:
        raise ValueError("Endpoint must be between 1 and 3000")
    if subprocess.check_output(["git", "-C", run["upstream"], "rev-parse", "HEAD"], text=True).strip() != run["upstream_commit"]:
        raise ValueError("Pinned upstream trainer commit differs")
    admission, CoverageTrainer = _import_stone_runtime(args.run, run)
    from ltxv_trainer.config import LtxvTrainerConfig
    for item in run["bound_files"]:
        admission.check_file(item)
    train, development = admission.admission(run)
    if len(train) != 645 or len(development) != 18:
        raise ValueError("Coverage cohort counts differ")
    prepared = json.loads(Path(run["encoded_manifest"]).read_text(encoding="utf-8"))
    if {row["id"] for row in prepared["samples"] if row["role"] == "train"} != {row["id"] for row in train}:
        raise ValueError("Trainer cohort differs from encoded training split")
    patchgan = load_patchgan_config(args.patchgan_config)
    if patchgan.domain != "latent" or patchgan.start_step != 0:
        raise ValueError("Dual-host run requires PatchGAN on every update from step zero")
    if args.resume is None:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.is_dir():
        raise FileNotFoundError("Resume output directory is missing")
    config_data = json.loads(json.dumps(run["trainer_config"]))
    config_data["output_dir"] = str(args.output)
    config = LtxvTrainerConfig.model_validate(config_data)
    if config.lora.rank != 128 or config.lora.alpha != 128 or config.acceleration.mixed_precision_mode != "bf16" or config.acceleration.quantization is not None:
        raise ValueError("Generator must remain BF16, rank/alpha 128/128, without quantization")
    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise ValueError("PatchGAN connection token is too short")
    client = RemotePatchGANClient(args.pc_host, args.port, token)
    if args.resume is not None:
        step = int(args.resume.name.removeprefix("step_"))
        record_path = args.output / "checkpoint_records" / f"step_{step:05d}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        remote = record["patchgan_discriminator"]
        if remote["step"] != step:
            raise ValueError("Discriminator and generator checkpoint steps differ")
        client.resume(step, remote["sha256"])
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    torch.set_num_threads(4)
    set_seed(config.seed)
    trainer = CoverageTrainer(config, component_paths=run["components"])
    current: dict[str, Any] = {"batch": None, "sample_id": None}
    lookup = {row["id"]: row for row in train + development}
    log_dir = args.output / "patchgan_segments"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}.jsonl")
    log = log_path.open("x", encoding="utf-8")
    strategy = trainer._training_strategy
    original_compute = strategy.compute_loss

    def compute_with_patchgan(prediction: torch.Tensor, prepared_batch: Any) -> torch.Tensor:
        flow_loss = original_compute(prediction, prepared_batch)
        batch = current["batch"]
        if batch is None:
            return flow_loss
        sample_id = current["sample_id"]
        current["batch"] = None
        current["sample_id"] = None
        length = prepared_batch.targets.shape[1]
        sigmas = target_token_sigmas(prepared_batch.sigmas, prepared_batch.conditioning_mask[:, -length:])
        clean = predict_clean_latents(prepared_batch.latents[:, -length:], prediction[:, -length:], sigmas)
        dimensions = {"num_frames": prepared_batch.num_frames, "height": prepared_batch.height, "width": prepared_batch.width}
        fake = unpack_packed_latents(clean, **dimensions)
        real = unpack_packed_latents(batch["latents"]["latents"], **dimensions)
        condition = unpack_packed_latents(batch["ref_latents"]["latents"], **dimensions)
        step = trainer._global_step + 1
        gradient, losses = client.train(step, condition, real, fake)
        surrogate = (fake.float() * gradient.float()).sum()
        g_value = torch.tensor(losses["generator_loss"], device=fake.device, dtype=torch.float32)
        total = flow_loss + patchgan.adversarial_weight * (surrogate - surrogate.detach() + g_value)
        if not torch.isfinite(total).item() or not all(math.isfinite(x) for x in losses.values()):
            raise ValueError("Nonfinite combined PatchGAN objective")
        log.write(json.dumps({"step_attempted": step, "sample_id": sample_id, "flow_loss": float(flow_loss.detach()), **losses}) + "\n")
        log.flush()
        os.fsync(log.fileno())
        return total

    strategy.compute_loss = compute_with_patchgan
    original_save = trainer._save_complete

    def save_both(cursor: Any, contract: Any) -> dict[str, Any]:
        step = trainer._global_step
        remote = client.checkpoint(step)
        result = original_save(cursor, contract)
        result["patchgan_discriminator"] = remote
        _write_json(args.output / "checkpoint_records" / f"step_{step:05d}.json", result)
        return result

    trainer._save_complete = save_both

    def train_batch(sample_id: str) -> dict[str, Any]:
        value = _load_batch(lookup[sample_id], trainer, admission.check_file)
        current["batch"], current["sample_id"] = value, sample_id
        return value

    def dev_batch(sample_id: str) -> dict[str, Any]:
        current["batch"], current["sample_id"] = None, None
        return _load_batch(lookup[sample_id], trainer, admission.check_file)

    started = time.monotonic()

    def resources() -> None:
        if shutil.disk_usage(args.output).free < args.minimum_free_disk_gib * 1024**3:
            raise RuntimeError("Stone scratch disk floor reached")
        if time.monotonic() - started > args.maximum_hours * 3600:
            raise TimeoutError("Dual-host PatchGAN wall-time cap reached")

    def decide(_step: int, _state: Any) -> dict[str, str]:
        return {"action": "continue", "reason": "Continue until the requested 3000-step endpoint unless a technical guard fails"}

    contract = {"kind": "dual_host_bf16_rank128_patchgan_reproduction_hypothesis", "training_ids": sorted(row["id"] for row in train),
                "development_ids": sorted(row["id"] for row in development), "encoded_manifest_sha256": _sha256(Path(run["encoded_manifest"])),
                "patchgan_config_sha256": _sha256(args.patchgan_config), "upstream_commit": run["upstream_commit"],
                "local_repo_commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True).strip(),
                "optimizer_update_endpoint": 3000, "scheduler_total_updates": config.optimization.steps,
                "paper_deviations": ["645 training pairs instead of the paper's 338",
                                     "mask-depth conditioning selected by the user",
                                     "latent 16x16 PatchGAN, BCE and weight 0.1 are reproduction hypotheses because the paper does not specify them",
                                     "training stops after 3000 updates instead of the paper's 8000"],
                "patchgan_config": patchgan.__dict__,
                "gpu_topology": {"generator": "irtazastone RTX 6000 Ada", "discriminator": "irtazapc RTX 4090"}}
    _write_json(args.output / "dual_host_contract.json", contract)
    try:
        result = trainer.run_coverage(train_ids=[row["id"] for row in train], development_rows=development,
                                      load_train_batch=train_batch, load_development_batch=dev_batch,
                                      contract=contract, until_step=args.until, resource_guard=resources,
                                      checkpoint_decider=decide, resume=args.resume)
        if result.get("completed_step") == 3000 and result.get("exit_code") == 0:
            client.shutdown()
    finally:
        log.close()
        client.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--host", required=True)
    worker.add_argument("--port", type=int, required=True)
    worker.add_argument("--token-file", type=Path, required=True)
    worker.add_argument("--patchgan-config", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--resume", action="store_true")
    generator = sub.add_parser("generator")
    generator.add_argument("--run", type=Path, required=True)
    generator.add_argument("--patchgan-config", type=Path, required=True)
    generator.add_argument("--output", type=Path, required=True)
    generator.add_argument("--pc-host", required=True)
    generator.add_argument("--port", type=int, required=True)
    generator.add_argument("--token-file", type=Path, required=True)
    generator.add_argument("--until", type=int, required=True)
    generator.add_argument("--resume", type=Path)
    generator.add_argument("--minimum-free-disk-gib", type=float, default=40.0)
    generator.add_argument("--maximum-hours", type=float, default=96.0)
    args = parser.parse_args()
    if args.mode == "worker":
        token = args.token_file.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise ValueError("PatchGAN connection token is too short")
        torch.manual_seed(42)
        serve(args.host, args.port, token, PatchGANWorker(load_patchgan_config(args.patchgan_config), args.output, resume=args.resume))
        return 0
    return _run_generator(args)


if __name__ == "__main__":
    raise SystemExit(main())

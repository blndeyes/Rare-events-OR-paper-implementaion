"""Preflight and execute the frozen fine-grained evaluation split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from or_video_reproduction.evaluation.video import decode_sampled_frames

from .backends import (
    ModelUnavailable,
    crop_detection,
    detect_hands_mediapipe,
    detect_people_detr,
    embed_clip_frames,
    embed_dinov2_frames,
    embed_dinov3_patches,
    estimate_vitpose_keypoints,
    load_clip,
    load_depth,
    load_dinov2,
    load_dinov3,
    predict_depth,
    run_fvmd,
    run_jedi,
    run_vbench_anatomy,
)
from .cache import ArtifactCache
from .control import compare_tracks, parse_ellipse_video, track_instances, detections_to_frame_instances
from .inventory import build_evaluation_manifest, save_evaluation_manifest
from .protocol import (
    CORESET_SIZE,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    FROZEN_MODELS,
    SCHEMA_VERSION,
    VIDEO_CONTRACT,
    WAN_DIAGNOSTIC_WARNING,
    WEAK_SAMPLE_WARNING,
    concatenate_video_embeddings,
    density_coverage,
    frame_sampling_manifest,
    greedy_kcenter_coreset,
    kid_polynomial,
    l2_normalize,
    scaled_unbiased_rbf_mmd,
    statistical_flags,
    video_bootstrap,
    write_json,
)
from or_video_reproduction.evaluation.distribution import frechet_distance
from .scoring import detection_rate, match_people_by_iou, mpjpe_from_matches, nearest_anchor_scores

ALL_METRICS = (
    "clip_cmmd",
    "clip_kid",
    "dinov2",
    "dinov3",
    "hands",
    "anatomy",
    "control",
    "fvmd",
    "jedi",
)


def _maybe_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def collect_environment(output_root: Path, *, hf_cache: Path | None, torch_cache: Path | None) -> dict[str, object]:
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "hf_cache": str(hf_cache) if hf_cache else os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "torch_cache": str(torch_cache) if torch_cache else os.environ.get("TORCH_HOME"),
        "nvidia_smi": gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip(),
        "pip_freeze": pip.stdout.splitlines() if pip.returncode == 0 else [pip.stderr],
        "cwd": str(Path.cwd()),
        "output_root": str(output_root),
    }


def _group_status(group: dict[str, object]) -> dict[str, object]:
    n_videos = int(group["sample_count"])
    n_embeddings = n_videos * DEFAULT_SAMPLE_COUNT
    flags = statistical_flags(n_videos=n_videos, n_embeddings=n_embeddings, feature_dim=1024)
    return {
        "n_videos": n_videos,
        "n_embeddings": n_embeddings,
        "n_definition": "n_videos * sampled_frames; equal real and generated counts",
        "flags": flags,
        "comparison_class": group["comparison_class"],
        "warning": WAN_DIAGNOSTIC_WARNING if group["comparison_class"] == "diagnostic_single_video" else WEAK_SAMPLE_WARNING,
    }


def probe_model_availability(*, vbench_root: Path | None) -> dict[str, object]:
    status = {}
    status["torch"] = {"available": _maybe_import("torch")}
    status["transformers"] = {"available": _maybe_import("transformers")}
    status["mediapipe"] = {"available": _maybe_import("mediapipe")}
    status["fvmd"] = {"available": _maybe_import("fvmd")}
    status["videojedi"] = {"available": _maybe_import("videojedi")}
    status["vbench2"] = {
        "available": bool(vbench_root and (vbench_root / "evaluate.py").is_file()),
        "root": str(vbench_root) if vbench_root else None,
    }
    for metric, spec in FROZEN_MODELS.items():
        status.setdefault(metric, {})
        status[metric]["spec"] = spec
    return status


def build_preflight(
    manifest: dict[str, object],
    sampling: dict[str, object],
    *,
    output_root: Path,
    vbench_root: Path | None,
    hf_cache: Path | None,
    torch_cache: Path | None,
) -> dict[str, object]:
    bytes_per_frame = VIDEO_CONTRACT["width"] * VIDEO_CONTRACT["height"] * 3
    sampled_videos = sum(int(group["sample_count"]) * 2 for group in manifest["groups"])
    sampled_bytes = sampled_videos * len(sampling["indices"]) * bytes_per_frame
    availability = probe_model_availability(vbench_root=vbench_root)
    invalid = []
    for group in manifest["groups"]:
        info = _group_status(group)
        if "frechet_covariance_rank_deficient" in info["flags"]:
            invalid.append(f"{group['id']}: DINOv2 Frechet is rank-deficient (n={info['n_embeddings']}, d=1024)")
        if group["comparison_class"] == "diagnostic_single_video":
            invalid.append(f"{group['id']}: single-video WAN scores are diagnostic only")
    missing = []
    if not availability["torch"]["available"]:
        missing.append("torch")
    if not availability["transformers"]["available"]:
        missing.append("transformers (CLIP, DINOv2, DINOv3, Depth Anything V2, DETR)")
    if not availability["mediapipe"]["available"]:
        missing.append("mediapipe (hand detector)")
    if not availability["fvmd"]["available"]:
        missing.append("fvmd (official FVMD)")
    if not availability["videojedi"]["available"]:
        missing.append("videojedi (official JEDi); ranking metric only")
    if not availability["vbench2"]["available"]:
        missing.append("VBench-2.0 evaluate.py (pass --vbench-root)")
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(manifest.get("ok")),
        "input_root": manifest["input_root"],
        "output_root": str(output_root),
        "video_contract": VIDEO_CONTRACT,
        "pairing_rule": manifest["pairing_rule"],
        "groups": [
            {
                "id": group["id"],
                "clip_ids": group["clip_ids"],
                "comparison_class": group["comparison_class"],
                **_group_status(group),
            }
            for group in manifest["groups"]
        ],
        "wan_in_frozen_4dor_split": manifest["wan_in_frozen_4dor_split"],
        "wan_note": manifest["wan_note"],
        "planned_sampled_frames": sampling,
        "models": FROZEN_MODELS,
        "availability": availability,
        "missing_dependencies": missing,
        "statistically_invalid_or_weak": invalid,
        "expected_storage": {
            "sampled_rgb_uncompressed_gb": sampled_bytes / (1024**3),
            "note": "embeddings and detections are much smaller; model weights use the configured caches",
        },
        "caches": {"huggingface": str(hf_cache) if hf_cache else None, "torch": str(torch_cache) if torch_cache else None},
        "errors": manifest.get("errors", []),
        "duplicate_files": manifest.get("duplicate_files", []),
    }


def _load_or_decode(cache: ArtifactCache, name: str, path: Path, indices: Sequence[int]) -> np.ndarray:
    if cache.has_array(name):
        return cache.load_array(name)["frames"]
    frames = decode_sampled_frames(path, indices)
    cache.save_array(name, frames=frames)
    return frames


def _unavailable(metric: str, reason: str) -> dict[str, object]:
    return {"status": "unavailable", "metric": metric, "reason": reason}


def _metric_from_embeddings(
    reference: list[np.ndarray],
    generated: list[np.ndarray],
    *,
    comparison_class: str,
) -> dict[str, object]:
    real = concatenate_video_embeddings(reference)
    fake = concatenate_video_embeddings(generated)
    n_videos = len(reference)
    flags = statistical_flags(n_videos=n_videos, n_embeddings=len(real), feature_dim=real.shape[1])
    cmmd = scaled_unbiased_rbf_mmd(real, fake)
    kid = kid_polynomial(real, fake, seed=DEFAULT_SEED)
    bootstrap = video_bootstrap(reference, generated, scaled_unbiased_rbf_mmd, seed=DEFAULT_SEED)
    per_clip = []
    for index, (real_clip, fake_clip) in enumerate(zip(reference, generated, strict=True)):
        per_clip.append(
            {
                "index": index,
                "clip_rbf_mmd_x1000": scaled_unbiased_rbf_mmd(real_clip, fake_clip) if len(real_clip) > 1 else None,
            }
        )
    return {
        "status": "ok",
        "comparison_class": comparison_class,
        "n_videos": n_videos,
        "n": len(real),
        "n_definition": "embeddings = clips x sampled frames; equal real/generated counts",
        "feature_dim": int(real.shape[1]),
        "clip_cmmd_unbiased_rbf_x1000": cmmd,
        "clip_kid_polynomial": kid,
        "per_clip": per_clip,
        "video_bootstrap_cmmd": bootstrap,
        "flags": flags,
        "diagnostic_only": comparison_class == "diagnostic_single_video",
    }


def _dinov2_from_embeddings(reference: list[np.ndarray], generated: list[np.ndarray], comparison_class: str) -> dict[str, object]:
    real = concatenate_video_embeddings(reference)
    fake = concatenate_video_embeddings(generated)
    flags = statistical_flags(n_videos=len(reference), n_embeddings=len(real), feature_dim=real.shape[1])
    prdc = density_coverage(l2_normalize(real), l2_normalize(fake))
    result = {
        "status": "ok",
        "frechet": frechet_distance(real, fake),
        "density": prdc["density"],
        "coverage": prdc["coverage"],
        "nearest_k": prdc["nearest_k"],
        "n": len(real),
        "feature_dim": int(real.shape[1]),
        "cls_token": "last_hidden_state[:,0]",
        "frechet_embedding_normalized": False,
        "prdc_embedding_normalized": True,
        "flags": flags,
        "diagnostic_only": comparison_class == "diagnostic_single_video",
        "video_bootstrap_frechet": video_bootstrap(reference, generated, frechet_distance),
    }
    if "frechet_covariance_rank_deficient" in flags:
        result["frechet_warning"] = "n <= feature_dim; Frechet covariance is rank-deficient"
    return result


def _dinov3_from_patches(reference: list[np.ndarray], generated: list[np.ndarray], comparison_class: str) -> dict[str, object]:
    real = np.concatenate([row.reshape(-1, row.shape[-1]) for row in reference], axis=0)
    fake = np.concatenate([row.reshape(-1, row.shape[-1]) for row in generated], axis=0)
    if len(real) != len(fake):
        raise ValueError("DINOv3 real/generated patch counts differ")
    real_core = greedy_kcenter_coreset(real, CORESET_SIZE, seed=DEFAULT_SEED)
    fake_core = greedy_kcenter_coreset(fake, CORESET_SIZE, seed=DEFAULT_SEED)
    size = min(len(real_core["features"]), len(fake_core["features"]))
    score = scaled_unbiased_rbf_mmd(real_core["features"][:size], fake_core["features"][:size])
    return {
        "status": "ok",
        "mmd_unbiased_rbf_x1000": score,
        "coreset_size": size,
        "coreset_algorithm": real_core["algorithm"],
        "distance": "euclidean",
        "pooling": "3x3 avg pool stride 1 pad 1",
        "seed": DEFAULT_SEED,
        "n_patch_tokens": len(real),
        "diagnostic_only": comparison_class == "diagnostic_single_video",
    }


def _symlink_eval_set(output_root: Path, group_id: str, pairs: list[dict[str, object]]) -> tuple[Path, Path]:
    root = output_root / "scratch" / "motion" / group_id
    generated_dir = root / "generated"
    reference_dir = root / "reference"
    generated_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        clip_id = pair["id"]
        for directory, source in (
            (generated_dir, Path(pair["generated_video"])),
            (reference_dir, Path(pair["reference_video"])),
        ):
            target = directory / f"{clip_id}.mp4"
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source)
    return generated_dir, reference_dir


def evaluate_group(
    group: dict[str, object],
    *,
    sampling: dict[str, object],
    output_root: Path,
    device: str,
    metrics: set[str],
    hf_cache: str | None,
    vbench_root: Path | None,
    logger,
) -> dict[str, object]:
    indices = list(sampling["indices"])
    cache = ArtifactCache(
        output_root / "embeddings",
        namespace=str(group["id"]),
        config={"sampling": sampling, "video_contract": VIDEO_CONTRACT},
    )
    detection_cache = ArtifactCache(
        output_root / "detections",
        namespace=str(group["id"]),
        config={"sampling": sampling},
    )
    pairs = group["pairs"]
    result: dict[str, object] = {
        "group_id": group["id"],
        "domain": group["domain"],
        "comparison_class": group["comparison_class"],
        "clip_ids": group["clip_ids"],
        "metrics": {},
        "per_clip": [],
    }
    clip_frames: list[dict[str, np.ndarray]] = []
    clip_ref: list[np.ndarray] | None = None
    loaded_clip = None
    for pair in pairs:
        clip_id = pair["id"]
        generated = _load_or_decode(cache, f"{clip_id}-generated-frames", Path(pair["generated_video"]), indices)
        reference = _load_or_decode(cache, f"{clip_id}-reference-frames", Path(pair["reference_video"]), indices)
        control = None
        if pair.get("control_video"):
            control = _load_or_decode(cache, f"{clip_id}-control-frames", Path(pair["control_video"]), indices)
        clip_frames.append({"id": clip_id, "generated": generated, "reference": reference, "control": control, "pair": pair})
        result["per_clip"].append({"id": clip_id, "sampled_frames": indices})

    if metrics & {"clip_cmmd", "clip_kid"}:
        try:
            clip_ref, clip_gen = [], []
            clip_info = None
            for row in clip_frames:
                ref_name = f"{row['id']}-clip-reference"
                gen_name = f"{row['id']}-clip-generated"
                if cache.has_array(ref_name) and cache.has_array(gen_name):
                    clip_ref.append(cache.load_array(ref_name)["x"])
                    clip_gen.append(cache.load_array(gen_name)["x"])
                    continue
                if loaded_clip is None:
                    loaded_clip, clip_info = load_clip(device, hf_cache)
                real = embed_clip_frames(row["reference"], loaded_clip)
                fake = embed_clip_frames(row["generated"], loaded_clip)
                cache.save_array(ref_name, x=real)
                cache.save_array(gen_name, x=fake)
                clip_ref.append(real)
                clip_gen.append(fake)
            clip_scores = _metric_from_embeddings(
                clip_ref, clip_gen, comparison_class=str(group["comparison_class"])
            )
            if clip_info:
                clip_scores["model"] = {
                    "identifier": clip_info.identifier,
                    "revision": clip_info.revision,
                    **clip_info.extra,
                }
            result["metrics"]["clip_cmmd"] = {
                key: clip_scores[key]
                for key in clip_scores
                if key != "clip_kid_polynomial"
            }
            result["metrics"]["clip_kid"] = {
                "status": "ok",
                **clip_scores["clip_kid_polynomial"],
                "diagnostic_only": clip_scores["diagnostic_only"],
            }
        except ModelUnavailable as error:
            result["metrics"]["clip_cmmd"] = _unavailable("clip_cmmd", error.reason)
            result["metrics"]["clip_kid"] = _unavailable("clip_kid", error.reason)
            clip_ref = None

    if "dinov2" in metrics:
        try:
            loaded = None
            info = None
            ref_rows, gen_rows = [], []
            for row in clip_frames:
                ref_name = f"{row['id']}-dinov2-reference"
                gen_name = f"{row['id']}-dinov2-generated"
                if cache.has_array(ref_name) and cache.has_array(gen_name):
                    ref_rows.append(cache.load_array(ref_name)["x"])
                    gen_rows.append(cache.load_array(gen_name)["x"])
                    continue
                if loaded is None:
                    loaded, info = load_dinov2(device, hf_cache)
                real = embed_dinov2_frames(row["reference"], loaded)
                fake = embed_dinov2_frames(row["generated"], loaded)
                cache.save_array(ref_name, x=real)
                cache.save_array(gen_name, x=fake)
                ref_rows.append(real)
                gen_rows.append(fake)
            scored = _dinov2_from_embeddings(ref_rows, gen_rows, str(group["comparison_class"]))
            if info:
                scored["model"] = {"identifier": info.identifier, "revision": info.revision, **info.extra}
            result["metrics"]["dinov2"] = scored
        except ModelUnavailable as error:
            result["metrics"]["dinov2"] = _unavailable("dinov2", error.reason)

    if "dinov3" in metrics:
        try:
            loaded = None
            info = None
            ref_rows, gen_rows = [], []
            for row in clip_frames:
                ref_name = f"{row['id']}-dinov3-reference"
                gen_name = f"{row['id']}-dinov3-generated"
                if cache.has_array(ref_name) and cache.has_array(gen_name):
                    ref_rows.append(cache.load_array(ref_name)["x"])
                    gen_rows.append(cache.load_array(gen_name)["x"])
                    continue
                if loaded is None:
                    loaded, info = load_dinov3(device, hf_cache)
                real = embed_dinov3_patches(row["reference"], loaded)
                fake = embed_dinov3_patches(row["generated"], loaded)
                cache.save_array(ref_name, x=real)
                cache.save_array(gen_name, x=fake)
                ref_rows.append(real)
                gen_rows.append(fake)
            scored = _dinov3_from_patches(ref_rows, gen_rows, str(group["comparison_class"]))
            if info:
                scored["model"] = {"identifier": info.identifier, "revision": info.revision, **info.extra}
            result["metrics"]["dinov3"] = scored
        except ModelUnavailable as error:
            result["metrics"]["dinov3"] = _unavailable("dinov3", error.reason)

    if "hands" in metrics:
        try:
            real_crops: list[np.ndarray] = []
            fake_crops: list[np.ndarray] = []
            per_clip_hands = []
            empty_real = 0
            empty_gen = 0
            clip_model = None
            for row in clip_frames:
                name = f"{row['id']}-hands"
                if detection_cache.has_json(name):
                    payload = detection_cache.load_json(name)
                else:
                    real_det = detect_hands_mediapipe(row["reference"])
                    gen_det = detect_hands_mediapipe(row["generated"])
                    payload = {"real": real_det, "generated": gen_det}
                    detection_cache.save_json(name, payload)
                real_rate = detection_rate(payload["real"]["detections"], len(indices))
                gen_rate = detection_rate(payload["generated"]["detections"], len(indices))
                if real_rate["detection_count"] == 0:
                    empty_real += 1
                if gen_rate["detection_count"] == 0:
                    empty_gen += 1
                def _crops(frames, detections):
                    crops = []
                    for det in detections:
                        crop = crop_detection(frames[int(det["frame"])], det["box_xyxy"])
                        if crop is not None:
                            crops.append(crop)
                    return crops
                real_c = _crops(row["reference"], payload["real"]["detections"])
                gen_c = _crops(row["generated"], payload["generated"]["detections"])
                per_clip_hands.append(
                    {
                        "id": row["id"],
                        "real": real_rate,
                        "generated": gen_rate,
                        "real_valid_crops": len(real_c),
                        "generated_valid_crops": len(gen_c),
                    }
                )
                if real_c or gen_c:
                    if clip_model is None:
                        clip_model, _ = load_clip(device, hf_cache)
                    for crop in real_c:
                        real_crops.append(embed_clip_frames(crop[None], clip_model)[0])
                    for crop in gen_c:
                        fake_crops.append(embed_clip_frames(crop[None], clip_model)[0])
            real_arr = np.stack(real_crops) if real_crops else np.zeros((0, 1))
            fake_arr = np.stack(fake_crops) if fake_crops else np.zeros((0, 1))
            embed_scores = nearest_anchor_scores(fake_arr, real_arr) if real_crops and fake_crops else {
                "status": "unavailable",
                "reason": "missing real or generated hand crops",
                "generated_crops": len(fake_crops),
                "reference_crops": len(real_crops),
            }
            if embed_scores["status"] == "ok" and len(real_arr) > 1 and len(fake_arr) > 1:
                n = min(len(real_arr), len(fake_arr))
                embed_scores["clip_rbf_mmd_x1000"] = scaled_unbiased_rbf_mmd(real_arr[:n], fake_arr[:n])
            result["metrics"]["hands"] = {
                "status": "ok",
                "detector": "mediapipe.solutions.hands",
                "confidence_threshold": payload["real"]["confidence_threshold"],
                "crop_margin": payload["real"]["crop_margin"],
                "embedding_model": FROZEN_MODELS["clip"]["huggingface_id"],
                "real_clips_with_no_hands": empty_real,
                "generated_clips_with_no_hands": empty_gen,
                "valid_real_crops": len(real_crops),
                "valid_generated_crops": len(fake_crops),
                "per_clip": per_clip_hands,
                "embedding_scores": embed_scores,
                "note": "Embedding scores are never reported without detection-failure counts",
            }
        except ModelUnavailable as error:
            result["metrics"]["hands"] = _unavailable("hands", error.reason)

    if "anatomy" in metrics:
        anatomy: dict[str, object] = {"status": "ok", "parts": {}}
        try:
            if vbench_root is not None:
                videos = [Path(pair["generated_video"]) for pair in pairs]
                anatomy["parts"]["vbench2_human_anatomy"] = run_vbench_anatomy(videos, vbench_root=vbench_root)
            else:
                anatomy["parts"]["vbench2_human_anatomy"] = _unavailable(
                    "vbench2_anatomy", "--vbench-root was not provided"
                )
        except ModelUnavailable as error:
            anatomy["parts"]["vbench2_human_anatomy"] = _unavailable("vbench2_anatomy", error.reason)
        try:
            per_clip_people = []
            real_det_all = []
            gen_det_all = []
            clip_mpjpe: list[dict[str, object]] = []
            for row in clip_frames:
                name = f"{row['id']}-people"
                if detection_cache.has_json(name):
                    payload = detection_cache.load_json(name)
                else:
                    payload = {
                        "real": detect_people_detr(row["reference"], device, hf_cache),
                        "generated": detect_people_detr(row["generated"], device, hf_cache),
                    }
                    detection_cache.save_json(name, payload)
                real_rate = detection_rate(payload["real"]["detections"], len(indices))
                gen_rate = detection_rate(payload["generated"]["detections"], len(indices))
                per_clip_people.append({"id": row["id"], "real": real_rate, "generated": gen_rate})
                real_det_all.extend(payload["real"]["detections"])
                gen_det_all.extend(payload["generated"]["detections"])
                pose_name = f"{row['id']}-vitpose"
                try:
                    if detection_cache.has_json(pose_name):
                        posed = detection_cache.load_json(pose_name)
                    elif payload["real"]["detections"] or payload["generated"]["detections"]:
                        posed = {
                            "real": estimate_vitpose_keypoints(
                                row["reference"], payload["real"]["detections"], device, hf_cache
                            ),
                            "generated": estimate_vitpose_keypoints(
                                row["generated"], payload["generated"]["detections"], device, hf_cache
                            ),
                        }
                        detection_cache.save_json(pose_name, posed)
                    else:
                        posed = {
                            "real": {"identifier": FROZEN_MODELS["pose"]["keypoints"], "detections": []},
                            "generated": {
                                "identifier": FROZEN_MODELS["pose"]["keypoints"],
                                "detections": [],
                            },
                        }
                    matched = match_people_by_iou(
                        posed["real"]["detections"], posed["generated"]["detections"]
                    )
                    height, width = row["generated"].shape[:2]
                    mpjpe = mpjpe_from_matches(matched, width=width, height=height)
                except ModelUnavailable as pose_error:
                    mpjpe = _unavailable("mpjpe", pose_error.reason)
                per_clip_people[-1]["mpjpe"] = mpjpe
                clip_mpjpe.append(mpjpe)
            anatomy["parts"]["person_detection"] = {
                "status": "ok",
                "real": detection_rate(real_det_all, len(indices) * len(clip_frames)),
                "generated": detection_rate(gen_det_all, len(indices) * len(clip_frames)),
                "per_clip": per_clip_people,
                "detector": "facebook/detr-resnet-50",
                "confidence_threshold": 0.5,
            }
            ok_mpjpe = [row for row in clip_mpjpe if row.get("status") == "ok"]
            if ok_mpjpe:
                anatomy["parts"]["mpjpe"] = {
                    "status": "ok",
                    "pose_model": FROZEN_MODELS["pose"]["keypoints"],
                    "mean_mpjpe_pixels": float(np.mean([row["mpjpe_pixels"] for row in ok_mpjpe])),
                    "mean_mpjpe_over_image_diagonal": float(
                        np.mean([row["mpjpe_over_image_diagonal"] for row in ok_mpjpe])
                    ),
                    "clips_with_mpjpe": len(ok_mpjpe),
                    "clips_without_mpjpe": len(clip_mpjpe) - len(ok_mpjpe),
                    "zero_mpjpe_for_misses": False,
                    "per_clip": per_clip_people,
                }
            else:
                anatomy["parts"]["mpjpe"] = {
                    "status": "unavailable",
                    "reason": "no clip produced defensible matched ViTPose keypoints",
                    "zero_mpjpe_for_misses": False,
                    "per_clip": per_clip_people,
                }
        except ModelUnavailable as error:
            anatomy["parts"]["person_detection"] = _unavailable("pose", error.reason)
            anatomy["parts"]["mpjpe"] = _unavailable("mpjpe", error.reason)
        result["metrics"]["anatomy"] = anatomy

    if "control" in metrics:
        if any(row["control"] is None for row in clip_frames):
            result["metrics"]["control"] = _unavailable(
                "control", "one or more clips have no ellipse-control video"
            )
        else:
            try:
                depth_loaded = None
                depth_info = None
                per_clip_control = []
                for row in clip_frames:
                    control_tracks = track_instances(parse_ellipse_video(row["control"]))
                    name = f"{row['id']}-independent-entities"
                    if detection_cache.has_json(name):
                        payload = detection_cache.load_json(name)
                    else:
                        people = detect_people_detr(row["generated"], device, hf_cache)
                        if depth_loaded is None:
                            depth_loaded, depth_info = load_depth(device, hf_cache)
                        detections = []
                        for det in people["detections"]:
                            box = [int(v) for v in det["box_xyxy"]]
                            frame = row["generated"][int(det["frame"])]
                            depth_map = predict_depth(frame, depth_loaded)
                            x0, y0, x1, y1 = box
                            crop = depth_map[y0:y1, x0:x1]
                            rel = float(np.nanmean(crop)) if crop.size else None
                            detections.append(
                                {
                                    "frame": det["frame"],
                                    "class_name": "person",
                                    "box_xyxy": box,
                                    "relative_depth": rel,
                                }
                            )
                        payload = {
                            "generated_detections": detections,
                            "depth_model": {
                                "identifier": depth_info.identifier if depth_info else None,
                                "revision": depth_info.revision if depth_info else None,
                            },
                        }
                        detection_cache.save_json(name, payload)
                    generated_tracks = detections_to_frame_instances(payload["generated_detections"])
                    scored = compare_tracks(control_tracks, generated_tracks)
                    scored["clip_id"] = row["id"]
                    scored["independent_depth_model"] = payload.get("depth_model")
                    scored["generated_class_note"] = (
                        "Independent RGB recovery currently uses COCO person boxes plus "
                        "Depth Anything V2. Non-person ellipse classes remain unmatched "
                        "rather than being forced into a correspondence."
                    )
                    per_clip_control.append(scored)
                result["metrics"]["control"] = {
                    "status": "ok",
                    "independent_depth_model": FROZEN_MODELS["depth"],
                    "psnr_against_ellipse_rgb": "forbidden",
                    "per_clip": per_clip_control,
                    "aggregate_entity_detection_rate": float(
                        np.mean([row.get("entity_detection_rate") or 0.0 for row in per_clip_control])
                    ),
                }
            except ModelUnavailable as error:
                result["metrics"]["control"] = _unavailable("control", error.reason)

    if "fvmd" in metrics:
        try:
            gen_dir, ref_dir = _symlink_eval_set(output_root, str(group["id"]), pairs)
            result["metrics"]["fvmd"] = run_fvmd(gen_dir, ref_dir, output_root / "scratch" / "fvmd-logs" / str(group["id"]))
            result["metrics"]["fvmd"]["diagnostic_only"] = group["comparison_class"] == "diagnostic_single_video"
        except ModelUnavailable as error:
            result["metrics"]["fvmd"] = _unavailable("fvmd", error.reason)
        except Exception as error:  # noqa: BLE001
            result["metrics"]["fvmd"] = _unavailable("fvmd", f"official FVMD failed: {error}")

    if "jedi" in metrics:
        try:
            result["metrics"]["jedi"] = run_jedi(
                [Path(pair["reference_video"]) for pair in pairs],
                [Path(pair["generated_video"]) for pair in pairs],
                feature_path=output_root / "scratch" / "jedi" / str(group["id"]),
                model_dir=Path(hf_cache) if hf_cache else None,
            )
            result["metrics"]["jedi"]["diagnostic_only"] = group["comparison_class"] == "diagnostic_single_video"
        except ModelUnavailable as error:
            result["metrics"]["jedi"] = _unavailable("jedi", error.reason)
            result["metrics"]["jedi"]["role"] = "relative_ranking_only"
            result["metrics"]["jedi"]["package_present"] = _maybe_import("videojedi")

    logger(f"finished group {group['id']}")
    return result


def render_summary(payload: dict[str, object]) -> str:
    lines = [
        "# Fine-grained OR video evaluation",
        "",
        f"Input: `{payload['input_root']}`",
        f"Output: `{payload['output_root']}`",
        "",
        "## 1. Exact data",
        "",
        payload["pairing_rule"],
        "",
        f"WAN in frozen 4D-OR split: `{payload['wan_in_frozen_4dor_split']}`",
        payload["wan_note"],
        "",
        WAN_DIAGNOSTIC_WARNING,
        "",
        "## 2. Models and preprocessing",
        "",
        json.dumps(FROZEN_MODELS, indent=2),
        "",
        "## 3. Sample counts",
        "",
    ]
    for group in payload["groups"]:
        lines.append(
            f"- `{group['id']}` ({group['comparison_class']}): {group['sample_count']} clips; "
            f"IDs: {', '.join(group['clip_ids'])}"
        )
    lines.extend(["", "## 4. Detection rates", ""])
    for group_id, group in payload["results"].items():
        hands = group.get("metrics", {}).get("hands", {})
        anatomy = group.get("metrics", {}).get("anatomy", {})
        lines.append(f"### {group_id}")
        lines.append(f"- hands: `{json.dumps({k: hands.get(k) for k in ('real_clips_with_no_hands','generated_clips_with_no_hands','valid_real_crops','valid_generated_crops','status')})}`")
        person = anatomy.get("parts", {}).get("person_detection", {}) if isinstance(anatomy, dict) else {}
        lines.append(f"- anatomy person detection: `{json.dumps(person)[:800]}`")
        lines.append("")
    lines.extend(["## 5. Per-clip results", "", "See `per-clip-results.json`.", ""])
    lines.extend(["## 6. Aggregate scores and uncertainty", ""])
    lines.append("| Group | Class | CLIP-CMMD x1000 | CLIP-KID mean | FD-DINOv2 | Density | Coverage | DINOv3 MMD x1000 | FVMD | JEDi |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group_id, group in payload["results"].items():
        metrics = group.get("metrics", {})
        clip = metrics.get("clip_cmmd", {})
        kid = metrics.get("clip_kid", {})
        dino = metrics.get("dinov2", {})
        dino3 = metrics.get("dinov3", {})
        fvmd = metrics.get("fvmd", {})
        jedi = metrics.get("jedi", {})
        def cell(record, *keys):
            if not isinstance(record, dict) or record.get("status") == "unavailable":
                return record.get("status", "n/a") if isinstance(record, dict) else "n/a"
            for key in keys:
                if key in record:
                    value = record[key]
                    return f"{value:.4f}" if isinstance(value, float) else str(value)
            return "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    group_id,
                    str(group.get("comparison_class")),
                    cell(clip, "clip_cmmd_unbiased_rbf_x1000"),
                    cell(kid, "mean"),
                    cell(dino, "frechet"),
                    cell(dino, "density"),
                    cell(dino, "coverage"),
                    cell(dino3, "mmd_unbiased_rbf_x1000"),
                    cell(fvmd, "score"),
                    cell(jedi, "score"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "WAN and any other single-clip row is diagnostic. Do not compare it with six-video scores.",
            "",
            "## 7. Failures and missing detections",
            "",
            "See `metric-status.json` and per-clip hand/anatomy records. Misses are not scored as zero MPJPE.",
            "",
            "## 8. Metrics that could not be completed",
            "",
        ]
    )
    for metric, status in payload["metric_status"].items():
        if status.get("status") != "ok":
            lines.append(f"- `{metric}`: {status.get('status')} — {status.get('reason')}")
    lines.extend(
        [
            "",
            "## 9. Statistical limitations",
            "",
            WEAK_SAMPLE_WARNING,
            "",
            "## 10. Compact comparison table",
            "",
            "Groups are not merged. Cross-method ranking is incomplete until WAN and every other method share this frozen six-clip split.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def flatten_metric_status(results: dict[str, dict[str, object]]) -> dict[str, object]:
    status: dict[str, object] = {}
    for group_id, group in results.items():
        for name, record in group.get("metrics", {}).items():
            key = f"{group_id}:{name}"
            if isinstance(record, dict):
                status[key] = {
                    "status": record.get("status", "ok"),
                    "reason": record.get("reason"),
                    "diagnostic_only": record.get("diagnostic_only"),
                }
            else:
                status[key] = {"status": "ok"}
    return status


def merge_group_results(
    previous: dict[str, object] | None, current: dict[str, object]
) -> dict[str, object]:
    """Keep metrics from earlier stages when this run requested a subset."""

    if not previous:
        return current
    merged = dict(previous)
    merged.update({key: value for key, value in current.items() if key != "metrics"})
    previous_metrics = previous.get("metrics") if isinstance(previous.get("metrics"), dict) else {}
    current_metrics = current.get("metrics") if isinstance(current.get("metrics"), dict) else {}
    merged["metrics"] = {**previous_metrics, **current_metrics}
    return merged


def run_pipeline(
    input_root: Path,
    output_root: Path,
    *,
    stage: str,
    device: str,
    metrics: set[str],
    vbench_root: Path | None,
    hf_cache: Path | None,
    torch_cache: Path | None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "visualizations").mkdir(exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "finegrained-evaluation.log"

    def logger(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line)

    if hf_cache:
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache / "hub")
    if torch_cache:
        os.environ["TORCH_HOME"] = str(torch_cache)

    logger("building frozen evaluation manifest")
    manifest = build_evaluation_manifest(input_root)
    save_evaluation_manifest(manifest, output_root)
    sampling = frame_sampling_manifest()
    write_json(output_root / "frame-sampling.json", sampling)
    environment = collect_environment(output_root, hf_cache=hf_cache, torch_cache=torch_cache)
    write_json(output_root / "environment-lock.json", environment)
    preflight = build_preflight(
        manifest,
        sampling,
        output_root=output_root,
        vbench_root=vbench_root,
        hf_cache=hf_cache,
        torch_cache=torch_cache,
    )
    write_json(output_root / "preflight-report.json", preflight)
    logger(f"preflight ok={preflight['ok']} missing={preflight['missing_dependencies']}")
    if stage == "preflight":
        return {"preflight": preflight, "manifest": manifest}

    results: dict[str, dict[str, object]] = {}
    previous_path = output_root / "per-clip-results.json"
    previous_results: dict[str, object] = {}
    if previous_path.is_file():
        try:
            loaded = json.loads(previous_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous_results = loaded
        except json.JSONDecodeError:
            previous_results = {}
    for group in manifest["groups"]:
        logger(f"evaluating {group['id']}")
        group_id = str(group["id"])
        try:
            current = evaluate_group(
                group,
                sampling=sampling,
                output_root=output_root,
                device=device,
                metrics=metrics,
                hf_cache=str(hf_cache) if hf_cache else None,
                vbench_root=vbench_root,
                logger=logger,
            )
        except Exception as error:  # noqa: BLE001
            logger(traceback.format_exc())
            current = {"status": "failed", "reason": str(error)}
        previous = previous_results.get(group_id)
        results[group_id] = merge_group_results(
            previous if isinstance(previous, dict) else None, current
        )
        write_json(output_root / "per-clip-results.json", results)

    metric_status = flatten_metric_status(results)
    write_json(output_root / "metric-status.json", metric_status)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "groups": {
            group_id: {
                "comparison_class": record.get("comparison_class"),
                "metrics": record.get("metrics"),
            }
            for group_id, record in results.items()
        },
        "do_not_merge_mmor_and_4dor": True,
        "cross_method_benchmark_complete": False,
        "reason_incomplete": (
            "WAN is a single take-9 diagnostic clip and is not on the frozen six-clip 4D-OR split"
        ),
    }
    write_json(output_root / "aggregate-results.json", aggregate)
    summary_payload = {
        "input_root": manifest["input_root"],
        "output_root": str(output_root),
        "pairing_rule": manifest["pairing_rule"],
        "wan_in_frozen_4dor_split": manifest["wan_in_frozen_4dor_split"],
        "wan_note": manifest["wan_note"],
        "groups": manifest["groups"],
        "results": results,
        "metric_status": metric_status,
    }
    (output_root / "results-summary.md").write_text(render_summary(summary_payload), encoding="utf-8")
    logger("wrote results-summary.md")
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "run"), default="run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics", default=",".join(ALL_METRICS))
    parser.add_argument("--vbench-root", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--torch-cache", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = {item.strip() for item in args.metrics.split(",") if item.strip()}
    unknown = metrics - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    run_pipeline(
        args.input_root,
        args.output_root,
        stage=args.stage,
        device=args.device,
        metrics=metrics,
        vbench_root=args.vbench_root,
        hf_cache=args.hf_cache,
        torch_cache=args.torch_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

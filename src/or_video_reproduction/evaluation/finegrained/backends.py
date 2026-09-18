"""Official-model adapters.  Missing weights fail the metric; they never substitute."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import (
    FROZEN_MODELS,
    HAND_CONFIDENCE,
    HAND_MARGIN,
    HAND_MIN_SIDE,
    POSE_CONFIDENCE,
    clip_preprocess_numpy,
    dinov2_preprocess_numpy,
    expand_margin,
    spatial_average_pool_3x3,
)


class ModelUnavailable(RuntimeError):
    def __init__(self, metric: str, reason: str) -> None:
        self.metric = metric
        self.reason = reason
        super().__init__(f"{metric} unavailable: {reason}")


@dataclass(frozen=True)
class LoadedModel:
    metric: str
    identifier: str
    revision: str | None
    extra: dict[str, Any]


def _torch_module():
    try:
        import torch
    except ImportError as error:
        raise ModelUnavailable("torch", "PyTorch is not installed") from error
    return torch


def _record_revision(model: Any, fallback: str | None = None) -> str | None:
    config = getattr(model, "config", None)
    return getattr(config, "_commit_hash", None) or fallback


def load_clip(device: str, cache_dir: str | None = None) -> tuple[Any, LoadedModel]:
    spec = FROZEN_MODELS["clip"]
    try:
        from transformers import CLIPModel
    except ImportError as error:
        raise ModelUnavailable("clip_cmmd", "transformers is not installed") from error
    torch = _torch_module()
    try:
        model = CLIPModel.from_pretrained(spec["huggingface_id"], cache_dir=cache_dir)
    except Exception as error:  # noqa: BLE001
        raise ModelUnavailable("clip_cmmd", f"could not load {spec['huggingface_id']}: {error}") from error
    model = model.to(device).eval()
    info = LoadedModel(
        metric="clip_cmmd",
        identifier=spec["huggingface_id"],
        revision=_record_revision(model),
        extra={"preprocessing": spec["input"], "feature": spec["feature"]},
    )
    return (model, torch), info


def unwrap_clip_features(value: Any) -> Any:
    """Extract the CLIP image embedding from a tensor or transformers 5 ModelOutput.

    Transformers 4 ``get_image_features`` returns a tensor. Transformers 5.14 on
    irtazapc returned ``BaseModelOutputWithPooling``.  The CMMD embedding is the
    projected image feature, not a raw vision ``last_hidden_state``.
    """

    if isinstance(value, (tuple, list)):
        if not value:
            raise ModelUnavailable("clip_cmmd", "CLIP get_image_features returned an empty tuple")
        value = value[0]
    image_embeds = getattr(value, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    if hasattr(value, "cpu") and hasattr(value, "float") and not hasattr(value, "last_hidden_state"):
        return value
    return None


def _clip_last_dim(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if not shape:
        return None
    return int(shape[-1])


def coerce_clip_image_features(model: Any, value: Any) -> Any:
    """Map a CLIP ``get_image_features`` return value to the projected image embedding.

    Transformers 5 may return ``BaseModelOutputWithPooling`` whose ``pooler_output``
    is already the 768-d projected CLIP image feature. Applying ``visual_projection``
    again is invalid (1024x768 against 768). Raw vision ``last_hidden_state`` is
    never used as the CMMD embedding.
    """

    unwrapped = unwrap_clip_features(value)
    if unwrapped is not None:
        return unwrapped
    projection = getattr(model, "visual_projection", None)
    pooler = getattr(value, "pooler_output", None)
    hidden = getattr(value, "last_hidden_state", None)
    in_features = getattr(projection, "in_features", None) if projection is not None else None
    out_features = getattr(projection, "out_features", None) if projection is not None else None
    pooler_dim = _clip_last_dim(pooler)
    if pooler is not None:
        if out_features is not None and pooler_dim == int(out_features):
            return pooler
        if in_features is not None and pooler_dim == int(in_features):
            return projection(pooler)
        if projection is None:
            return pooler
    if hidden is not None and projection is not None:
        cls = hidden[:, 0]
        cls_dim = _clip_last_dim(cls)
        if in_features is not None and cls_dim == int(in_features):
            return projection(cls)
        if out_features is not None and cls_dim == int(out_features):
            return cls
    raise ModelUnavailable(
        "clip_cmmd",
        f"get_image_features returned {type(value).__name__} with pooler_dim={pooler_dim} "
        f"projection=({in_features}->{out_features}); refusing a silent backbone substitute",
    )


def clip_image_feature_tensor(model: Any, pixel_values: Any) -> Any:
    features = model.get_image_features(pixel_values=pixel_values)
    return coerce_clip_image_features(model, features)


def embed_clip_frames(
    frames: np.ndarray,
    loaded: tuple[Any, Any],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    model, torch = loaded
    pixels = clip_preprocess_numpy(frames)
    tensor = torch.from_numpy(np.transpose(pixels, (0, 3, 1, 2))).to(
        device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
    )
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            features = clip_image_feature_tensor(model, tensor[start : start + batch_size])
            outputs.append(features.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def load_dinov2(device: str, cache_dir: str | None = None) -> tuple[Any, LoadedModel]:
    spec = FROZEN_MODELS["dinov2"]
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise ModelUnavailable("dinov2", "transformers is not installed") from error
    torch = _torch_module()
    try:
        model = AutoModel.from_pretrained(spec["huggingface_id"], cache_dir=cache_dir)
    except Exception as error:  # noqa: BLE001
        raise ModelUnavailable("dinov2", f"could not load {spec['huggingface_id']}: {error}") from error
    model = model.to(device).eval()
    return (model, torch), LoadedModel(
        metric="dinov2",
        identifier=spec["huggingface_id"],
        revision=_record_revision(model),
        extra={"preprocessing": spec["input"], "feature": spec["feature"]},
    )


def embed_dinov2_frames(
    frames: np.ndarray, loaded: tuple[Any, Any], *, batch_size: int = 8
) -> np.ndarray:
    model, torch = loaded
    pixels = dinov2_preprocess_numpy(frames)
    tensor = torch.from_numpy(pixels).to(
        device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
    )
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            hidden = model(pixel_values=tensor[start : start + batch_size]).last_hidden_state
            outputs.append(hidden[:, 0].float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def load_dinov3(device: str, cache_dir: str | None = None) -> tuple[Any, LoadedModel]:
    spec = FROZEN_MODELS["dinov3"]
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise ModelUnavailable("dinov3", "transformers is not installed") from error
    torch = _torch_module()
    try:
        processor = AutoImageProcessor.from_pretrained(
            spec["huggingface_id"], cache_dir=cache_dir, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            spec["huggingface_id"], cache_dir=cache_dir, trust_remote_code=True
        )
    except Exception as error:  # noqa: BLE001
        raise ModelUnavailable(
            "dinov3",
            f"official DINOv3 weights {spec['huggingface_id']} are unavailable ({error}). "
            "DINOv2 was not substituted.",
        ) from error
    if "dinov3" not in type(model).__name__.lower() and "dinov3" not in model.config.__class__.__name__.lower():
        raise ModelUnavailable("dinov3", "loaded model is not a DINOv3 class; refusing to continue")
    model = model.to(device).eval()
    return (model, processor, torch), LoadedModel(
        metric="dinov3",
        identifier=spec["huggingface_id"],
        revision=_record_revision(model),
        extra={
            "preprocessing": spec["input"],
            "feature": spec["feature"],
            "pooling": spec["pooling"],
            "num_register_tokens": getattr(model.config, "num_register_tokens", None),
            "patch_size": getattr(model.config, "patch_size", None),
        },
    )


def embed_dinov3_patches(
    frames: np.ndarray, loaded: tuple[Any, Any, Any], *, batch_size: int = 4
) -> np.ndarray:
    """Return pooled patch tokens with shape [T, H, W, C]."""

    model, processor, torch = loaded
    device = next(model.parameters()).device
    grids: list[np.ndarray] = []
    registers = int(getattr(model.config, "num_register_tokens", 0) or 0)
    patch_size = int(getattr(model.config, "patch_size", 16))
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = [np.asarray(frame) for frame in frames[start : start + batch_size]]
            inputs = processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            hidden = model(pixel_values=pixel_values).last_hidden_state
            patches = hidden[:, 1 + registers :, :]
            _, _, height, width = pixel_values.shape
            grid_h, grid_w = height // patch_size, width // patch_size
            if patches.shape[1] != grid_h * grid_w:
                raise ModelUnavailable(
                    "dinov3",
                    f"patch count {patches.shape[1]} != {grid_h}x{grid_w}; refusing to reshape",
                )
            grid = patches.reshape(patches.shape[0], grid_h, grid_w, patches.shape[-1])
            pooled = []
            for item in grid.float().cpu().numpy():
                pooled.append(spatial_average_pool_3x3(item))
            grids.append(np.stack(pooled, axis=0))
    return np.concatenate(grids, axis=0)


def load_depth(device: str, cache_dir: str | None = None) -> tuple[Any, LoadedModel]:
    spec = FROZEN_MODELS["depth"]
    try:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    except ImportError as error:
        raise ModelUnavailable("depth_anything_v2", "transformers is not installed") from error
    try:
        processor = AutoImageProcessor.from_pretrained(spec["huggingface_id"], cache_dir=cache_dir)
        model = AutoModelForDepthEstimation.from_pretrained(spec["huggingface_id"], cache_dir=cache_dir)
    except Exception as error:  # noqa: BLE001
        raise ModelUnavailable(
            "depth_anything_v2",
            f"could not load {spec['huggingface_id']}: {error}",
        ) from error
    if "video" in spec["huggingface_id"].lower():
        raise ModelUnavailable("depth_anything_v2", "refusing Video Depth Anything")
    torch = _torch_module()
    model = model.to(device).eval()
    return (model, processor, torch), LoadedModel(
        metric="depth_anything_v2",
        identifier=spec["huggingface_id"],
        revision=_record_revision(model),
        extra={"reason": spec["reason"]},
    )


def predict_depth(frame: np.ndarray, loaded: tuple[Any, Any, Any]) -> np.ndarray:
    model, processor, torch = loaded
    inputs = processor(images=np.asarray(frame), return_tensors="pt")
    with torch.inference_mode():
        outputs = model(pixel_values=inputs["pixel_values"].to(next(model.parameters()).device))
        depth = outputs.predicted_depth
        upsampled = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
    return upsampled.squeeze().float().cpu().numpy()


def detect_hands_mediapipe(frames: np.ndarray) -> dict[str, Any]:
    try:
        import mediapipe as mp
    except ImportError as error:
        raise ModelUnavailable("hands", "mediapipe is not installed") from error
    detections: list[dict[str, Any]] = []
    height, width = frames.shape[1:3]
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=HAND_CONFIDENCE,
    ) as hands:
        for frame_index, frame in enumerate(frames):
            result = hands.process(np.asarray(frame))
            if not result.multi_hand_landmarks:
                continue
            for hand in result.multi_hand_landmarks:
                xs = [landmark.x * width for landmark in hand.landmark]
                ys = [landmark.y * height for landmark in hand.landmark]
                box = [
                    int(max(0, np.floor(min(xs)))),
                    int(max(0, np.floor(min(ys)))),
                    int(min(width, np.ceil(max(xs)))),
                    int(min(height, np.ceil(max(ys)))),
                ]
                detections.append(
                    {
                        "frame": int(frame_index),
                        "box_xyxy": box,
                        "confidence": HAND_CONFIDENCE,
                    }
                )
    return {
        "detector": "mediapipe.solutions.hands",
        "confidence_threshold": HAND_CONFIDENCE,
        "crop_margin": HAND_MARGIN,
        "detections": detections,
        "frames_with_hands": len({row["frame"] for row in detections}),
        "frame_count": int(len(frames)),
    }


def crop_detection(frame: np.ndarray, box: Sequence[int], *, margin: float = HAND_MARGIN) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = expand_margin(box, width=width, height=height, margin=margin)
    if (x1 - x0) < HAND_MIN_SIDE or (y1 - y0) < HAND_MIN_SIDE:
        return None
    return np.asarray(frame[y0:y1, x0:x1])


def detect_people_detr(frames: np.ndarray, device: str, cache_dir: str | None = None) -> dict[str, Any]:
    try:
        from transformers import DetrImageProcessor, DetrForObjectDetection
    except ImportError as error:
        raise ModelUnavailable("pose", "transformers is not installed") from error
    torch = _torch_module()
    identifier = "facebook/detr-resnet-50"
    try:
        processor = DetrImageProcessor.from_pretrained(identifier, cache_dir=cache_dir)
        model = DetrForObjectDetection.from_pretrained(identifier, cache_dir=cache_dir).to(device).eval()
    except Exception as error:  # noqa: BLE001
        raise ModelUnavailable("pose", f"could not load {identifier}: {error}") from error
    detections: list[dict[str, Any]] = []
    with torch.inference_mode():
        for frame_index, frame in enumerate(frames):
            inputs = processor(images=np.asarray(frame), return_tensors="pt")
            outputs = model(**{key: value.to(device) for key, value in inputs.items()})
            target_sizes = torch.tensor([frame.shape[:2]], device=device)
            results = processor.post_process_object_detection(
                outputs, threshold=POSE_CONFIDENCE, target_sizes=target_sizes
            )[0]
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"], strict=True
            ):
                if int(label) != 1:
                    continue
                detections.append(
                    {
                        "frame": int(frame_index),
                        "class_name": "person",
                        "confidence": float(score),
                        "box_xyxy": [float(value) for value in box.tolist()],
                    }
                )
    return {
        "detector": identifier,
        "revision": _record_revision(model),
        "confidence_threshold": POSE_CONFIDENCE,
        "detections": detections,
        "frames_with_people": len({row["frame"] for row in detections}),
        "frame_count": int(len(frames)),
    }


def run_vbench_anatomy(videos: Sequence[Path], *, vbench_root: Path, python: str = "python") -> dict[str, Any]:
    if vbench_root is None:
        raise ModelUnavailable("vbench2_anatomy", "--vbench-root was not provided")
    script = vbench_root / "evaluate.py"
    if not script.is_file():
        raise ModelUnavailable("vbench2_anatomy", f"official evaluator missing: {script}")
    import subprocess

    scores = []
    logs = []
    for video in videos:
        result = subprocess.run(
            [
                python,
                str(script),
                "--dimension",
                "Human_Anatomy",
                "--videos_path",
                str(video),
                "--mode",
                "custom_input",
            ],
            cwd=vbench_root,
            capture_output=True,
            text=True,
            check=False,
        )
        logs.append({"video": str(video), "returncode": result.returncode, "stdout": result.stdout[-2000:]})
        if result.returncode != 0:
            raise ModelUnavailable(
                "vbench2_anatomy",
                f"official VBench-2.0 evaluator failed for {video}: {result.stderr[-1000:]}",
            )
        scores.append({"video": str(video), "raw_output": result.stdout})
    return {
        "implementation": "Vchitect/VBench VBench-2.0 Human_Anatomy",
        "per_video": scores,
        "logs": logs,
    }


def run_fvmd(generated_dir: Path, reference_dir: Path, log_dir: Path) -> dict[str, Any]:
    try:
        from fvmd import fvmd
    except ImportError as error:
        raise ModelUnavailable("fvmd", "official fvmd package is not installed") from error
    log_dir.mkdir(parents=True, exist_ok=True)
    value = fvmd(log_dir=str(log_dir), gen_path=str(generated_dir), gt_path=str(reference_dir))
    return {
        "implementation": "DSL-Lab/FVMD-frechet-video-motion-distance",
        "package": "fvmd",
        "score": float(value),
        "role": "primary_motion_distribution",
        "official_sampling": FROZEN_MODELS["fvmd"]["official_sampling"],
    }


def run_jedi(reference: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    try:
        from videojedi import JEDiMetric
    except ImportError as error:
        raise ModelUnavailable("jedi", "official videojedi package is not installed") from error
    metric = JEDiMetric()
    metric.train_features = np.asarray(reference)
    metric.test_features = np.asarray(generated)
    score = float(metric.compute_metric())
    return {
        "implementation": "oooolga/JEDi",
        "package": "videojedi",
        "score": score,
        "role": "relative_ranking_only",
        "warning": "JEDi is not an absolute calibrated quality score",
    }

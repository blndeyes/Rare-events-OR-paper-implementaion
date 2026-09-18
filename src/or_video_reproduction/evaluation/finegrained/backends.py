"""Official-model adapters.  Missing weights fail the metric; they never substitute."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .external import ModelUnavailable
from .protocol import (
    FROZEN_MODELS,
    HAND_CONFIDENCE,
    HAND_DETECTOR,
    HAND_MARGIN,
    HAND_MAX_HANDS,
    HAND_MIN_SIDE,
    HAND_MODEL_ASSET,
    POSE_CONFIDENCE,
    clip_preprocess_numpy,
    dinov2_preprocess_numpy,
    expand_margin,
    spatial_average_pool_3x3,
)


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


def select_hand_landmarker_api() -> dict[str, Any]:
    """Return the explicit MediaPipe Tasks HandLandmarker API, or fail.

    ``mediapipe.solutions.hands`` is never used, even when it is importable.
    """

    try:
        import mediapipe as mp
    except ImportError as error:
        raise ModelUnavailable("hands", "mediapipe is not installed") from error
    solutions_hands = getattr(getattr(mp, "solutions", None), "hands", None)
    try:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision
    except ImportError as error:
        if solutions_hands is not None:
            raise ModelUnavailable(
                "hands",
                "mediapipe.solutions.hands exists but is not the selected backend; "
                "this evaluator requires mediapipe.tasks.vision.HandLandmarker and a "
                f"{HAND_MODEL_ASSET} asset",
            ) from error
        raise ModelUnavailable(
            "hands",
            f"mediapipe.tasks.vision.HandLandmarker is unavailable ({error})",
        ) from error
    if getattr(mp_tasks_vision, "HandLandmarker", None) is None:
        if solutions_hands is not None:
            raise ModelUnavailable(
                "hands",
                "mediapipe.solutions.hands exists but is not the selected backend; "
                "this evaluator requires mediapipe.tasks.vision.HandLandmarker",
            )
        raise ModelUnavailable("hands", "mediapipe.tasks.python.vision.HandLandmarker is missing")

    def create(model_path: str) -> Any:
        options_kwargs: dict[str, Any] = {
            "base_options": mp_tasks_python.BaseOptions(model_asset_path=model_path),
            "num_hands": HAND_MAX_HANDS,
            "min_hand_detection_confidence": HAND_CONFIDENCE,
        }
        running_mode = getattr(mp_tasks_vision, "RunningMode", None)
        if running_mode is not None:
            options_kwargs["running_mode"] = running_mode.IMAGE
        options = mp_tasks_vision.HandLandmarkerOptions(**options_kwargs)
        return mp_tasks_vision.HandLandmarker.create_from_options(options)

    return {
        "create": create,
        "Image": getattr(mp, "Image", None),
        "ImageFormat": getattr(mp, "ImageFormat", None),
        "version": getattr(mp, "__version__", None),
        "detector": HAND_DETECTOR,
        "package": "mediapipe",
    }


def _hand_image(frame: np.ndarray, api: dict[str, Any]) -> Any:
    image_cls = api.get("Image")
    image_format = api.get("ImageFormat")
    pixels = np.ascontiguousarray(frame, dtype=np.uint8)
    if image_cls is None or image_format is None:
        raise ModelUnavailable(
            "hands",
            "mediapipe.Image / ImageFormat is unavailable; cannot run HandLandmarker",
        )
    srgb = getattr(image_format, "SRGB", None)
    if srgb is None:
        raise ModelUnavailable("hands", "mediapipe.ImageFormat.SRGB is unavailable")
    return image_cls(image_format=srgb, data=pixels)


def _hand_confidence(result: Any, hand_index: int) -> float:
    handedness = getattr(result, "handedness", None)
    if not handedness or hand_index >= len(handedness):
        return HAND_CONFIDENCE
    categories = handedness[hand_index]
    if not categories:
        return HAND_CONFIDENCE
    first = categories[0]
    score = getattr(first, "score", None)
    return float(score) if score is not None else HAND_CONFIDENCE


def detect_hands_mediapipe(
    frames: np.ndarray,
    *,
    model_asset_path: Path | str | None,
    landmarker_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Detect hands with MediaPipe Tasks HandLandmarker only."""

    if model_asset_path is None:
        raise ModelUnavailable(
            "hands",
            "--hand-landmarker-model is required; refusing to download "
            f"{HAND_MODEL_ASSET} at runtime",
        )
    asset = Path(model_asset_path)
    if not asset.is_file():
        raise ModelUnavailable("hands", f"HandLandmarker model asset missing: {asset}")
    api = None if landmarker_factory is not None else select_hand_landmarker_api()
    landmarker = (
        landmarker_factory(str(asset))
        if landmarker_factory is not None
        else api["create"](str(asset))
    )
    detections: list[dict[str, Any]] = []
    height, width = frames.shape[1:3]
    close = getattr(landmarker, "close", None)
    try:
        for frame_index, frame in enumerate(frames):
            if landmarker_factory is not None:
                result = landmarker.detect(np.asarray(frame))
            else:
                assert api is not None
                result = landmarker.detect(_hand_image(frame, api))
            hands = getattr(result, "hand_landmarks", None) or []
            for hand_index, hand in enumerate(hands):
                xs = [float(landmark.x) * width for landmark in hand]
                ys = [float(landmark.y) * height for landmark in hand]
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
                        "confidence": _hand_confidence(result, hand_index),
                    }
                )
    finally:
        if callable(close):
            close()
    version = None if api is None else api.get("version")
    return {
        "detector": HAND_DETECTOR,
        "detector_version": version,
        "model_asset": str(asset),
        "confidence_threshold": HAND_CONFIDENCE,
        "crop_margin": HAND_MARGIN,
        "max_hands": HAND_MAX_HANDS,
        "detections": detections,
        "frames_with_hands": len({row["frame"] for row in detections}),
        "frame_count": int(len(frames)),
        "failed_frame_count": int(len(frames) - len({row["frame"] for row in detections})),
    }


def crop_detection(frame: np.ndarray, box: Sequence[int], *, margin: float = HAND_MARGIN) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = expand_margin(box, width=width, height=height, margin=margin)
    if (x1 - x0) < HAND_MIN_SIDE or (y1 - y0) < HAND_MIN_SIDE:
        return None
    return np.asarray(frame[y0:y1, x0:x1])


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x0, y0, x1, y1 = (float(value) for value in box)
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def frames_to_jedi_video_tensor(frames: np.ndarray) -> np.ndarray:
    """Convert THWC uint8 RGB to JEDi's TCHW float32 in [0, 1]."""

    video = np.asarray(frames)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("JEDi video must have shape [T,H,W,3]")
    return np.transpose(video.astype(np.float32) / 255.0, (0, 3, 1, 2))


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


def estimate_vitpose_keypoints(
    frames: np.ndarray,
    detections: Sequence[dict[str, Any]],
    device: str,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Attach COCO keypoints to DETR person boxes with official ViTPose++."""

    identifier = str(FROZEN_MODELS["pose"]["keypoints"])
    try:
        from transformers import AutoProcessor, VitPoseForPoseEstimation
    except ImportError as error:
        raise ModelUnavailable(
            "mpjpe",
            f"transformers.VitPoseForPoseEstimation is unavailable ({error})",
        ) from error
    torch = _torch_module()
    try:
        processor = AutoProcessor.from_pretrained(identifier, cache_dir=cache_dir)
        model = VitPoseForPoseEstimation.from_pretrained(identifier, cache_dir=cache_dir).to(device).eval()
    except Exception as error:  # noqa: BLE001
        raise ModelUnavailable("mpjpe", f"could not load {identifier}: {error}") from error

    grouped: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        grouped.setdefault(int(detection["frame"]), []).append(dict(detection))
    enriched: list[dict[str, Any]] = []
    with torch.inference_mode():
        for frame_index, rows in grouped.items():
            image = np.asarray(frames[frame_index])
            boxes = np.asarray([xyxy_to_xywh(row["box_xyxy"]) for row in rows], dtype=np.float32)
            inputs = processor(images=image, boxes=[boxes], return_tensors="pt")
            tensors = {
                key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
            }
            dataset_index = torch.zeros(1, dtype=torch.int64, device=device)
            try:
                outputs = model(**tensors, dataset_index=dataset_index)
            except TypeError:
                outputs = model(**tensors)
            poses = processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]
            if len(poses) != len(rows):
                raise ModelUnavailable(
                    "mpjpe",
                    f"ViTPose returned {len(poses)} poses for {len(rows)} boxes; refusing invented matches",
                )
            for row, pose in zip(rows, poses, strict=True):
                keypoints = pose["keypoints"]
                scores = pose["scores"]
                if hasattr(keypoints, "detach"):
                    keypoints = keypoints.detach().cpu().numpy()
                if hasattr(scores, "detach"):
                    scores = scores.detach().cpu().numpy()
                row["keypoints"] = np.asarray(keypoints, dtype=np.float64).tolist()
                row["keypoint_scores"] = np.asarray(scores, dtype=np.float64).tolist()
                row["pose_model"] = identifier
                enriched.append(row)
    return {
        "identifier": identifier,
        "revision": _record_revision(model),
        "dataset_index": 0,
        "detections": enriched,
    }

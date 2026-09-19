"""Resumable per-configuration caches for frames, embeddings, and detections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import write_json


def config_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class ArtifactCache:
    """Store arrays and JSON under a metric/model/preprocessing namespace."""

    def __init__(self, root: Path, *, namespace: str, config: dict[str, Any]) -> None:
        self.namespace = namespace
        self.config = dict(config)
        self.key = config_key(self.config)
        self.root = root / namespace / self.key
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "config.json", {"namespace": namespace, **self.config})

    def array_path(self, name: str) -> Path:
        return self.root / f"{name}.npz"

    def json_path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def has_array(self, name: str) -> bool:
        return self.array_path(name).is_file()

    def has_json(self, name: str) -> bool:
        return self.json_path(name).is_file()

    def save_array(self, name: str, **arrays: np.ndarray) -> Path:
        path = self.array_path(name)
        temporary = path.with_name(path.name + ".tmp")
        from io import BytesIO

        buffer = BytesIO()
        np.savez_compressed(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(path)
        return path

    def load_array(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.array_path(name), allow_pickle=False) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.json_path(name)
        write_json(path, payload)
        return path

    def load_json(self, name: str) -> dict[str, Any]:
        return json.loads(self.json_path(name).read_text(encoding="utf-8"))

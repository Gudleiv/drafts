from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    models_dir: str = "models"
    onnx_dir: str = "onnx"
    engines_dir: str = "engines"
    device: str = "cuda"
    backend: str = "torch"
    precision: str = "fp16"
    tile: int = 1024
    overlap: int = 32
    batch: int = 4
    channels_last: bool = True
    compile: bool = False
    jpeg_quality: int = 92
    max_loaded_models: int = 3
    max_upload_mb: int = 64
    preload: list[str] = field(default_factory=list)
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    ort_providers: list[str] = field(
        default_factory=lambda: [
            "NvTensorRTRTXExecutionProvider",
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
        ]
    )
    ort_provider_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    ort_provider_library: str = ""
    log_level: str = "info"

    @classmethod
    def load(cls, path: str | None = None, overrides: dict[str, Any] | None = None) -> "Settings":
        data: dict[str, Any] = {}
        if path and Path(path).is_file():
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        names = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in names}
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None and k in names})
        return cls(**data)

    def resolve(self, base: Path) -> "Settings":
        for name in ("models_dir", "onnx_dir", "engines_dir"):
            value = Path(getattr(self, name))
            if not value.is_absolute():
                setattr(self, name, str(base / value))
        return self

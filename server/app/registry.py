from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from spandrel import ImageModelDescriptor, ModelLoader

from .backends import OrtBackend, TorchBackend, resolve_dtype
from .config import Settings
from .export import export_onnx, onnx_file_name

log = logging.getLogger("registry")

MODEL_SUFFIXES = {".pth", ".pt", ".safetensors", ".ckpt"}


@dataclass
class ModelInfo:
    name: str
    path: str
    arch: str = ""
    scale: int = 0
    input_channels: int = 0
    output_channels: int = 0
    supports_half: bool = False
    multiple_of: int = 1
    minimum: int = 0
    params: int = 0
    loaded: bool = False
    backend: str = ""
    dtype: str = ""
    tile: int = 0
    error: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedModel:
    info: ModelInfo
    backend: Any
    device: torch.device
    dtype: torch.dtype
    scale: int
    tile: int
    lock: threading.Lock = field(default_factory=threading.Lock)

    def infer(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.backend(tiles)


def _effective_tile(tile: int, multiple_of: int, minimum: int) -> int:
    t = max(tile, minimum)
    if multiple_of > 1:
        t = int(math.ceil(t / multiple_of) * multiple_of)
    return t


class Registry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device(settings.device if torch.cuda.is_available() or settings.device == "cpu" else "cpu")
        self._infos: dict[str, ModelInfo] = {}
        self._loaded: "OrderedDict[str, LoadedModel]" = OrderedDict()
        self._lock = threading.RLock()
        self.scan()

    def scan(self) -> None:
        root = Path(self.settings.models_dir)
        root.mkdir(parents=True, exist_ok=True)
        found: dict[str, ModelInfo] = {}
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in MODEL_SUFFIXES and p.is_file():
                name = p.relative_to(root).with_suffix("").as_posix()
                found[name] = self._infos.get(name) or ModelInfo(name=name, path=str(p))
        with self._lock:
            self._infos = found
        for info in found.values():
            if not info.arch and not info.error:
                self._probe(info)

    def _probe(self, info: ModelInfo) -> None:
        try:
            desc = ModelLoader(torch.device("cpu")).load_from_file(info.path)
            self._fill(info, desc)
            del desc
        except Exception as exc:
            info.error = f"{type(exc).__name__}: {exc}"
            log.warning("probe failed for %s: %s", info.name, info.error)

    def _fill(self, info: ModelInfo, desc: ImageModelDescriptor) -> None:
        info.arch = desc.architecture.name
        info.scale = int(desc.scale)
        info.input_channels = int(desc.input_channels)
        info.output_channels = int(desc.output_channels)
        info.supports_half = bool(desc.supports_half)
        req = desc.size_requirements
        info.multiple_of = int(getattr(req, "multiple_of", 1) or 1)
        info.minimum = int(getattr(req, "minimum", 0) or 0)
        info.params = sum(p.numel() for p in desc.model.parameters())
        info.error = ""

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [i.public() for i in self._infos.values()]

    def names(self) -> list[str]:
        with self._lock:
            return list(self._infos.keys())

    def info(self, name: str) -> ModelInfo:
        with self._lock:
            if name not in self._infos:
                raise KeyError(name)
            return self._infos[name]

    def get(self, name: str) -> LoadedModel:
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return self._loaded[name]
            info = self.info(name)
            loaded = self._load(info)
            self._loaded[name] = loaded
            while len(self._loaded) > max(1, self.settings.max_loaded_models):
                old_name, old = self._loaded.popitem(last=False)
                self._release(old)
                self._infos[old_name].loaded = False
                log.info("evicted %s", old_name)
            return loaded

    def unload(self, name: str) -> bool:
        with self._lock:
            loaded = self._loaded.pop(name, None)
            if loaded is None:
                return False
            self._release(loaded)
            self._infos[name].loaded = False
            return True

    def _release(self, loaded: LoadedModel) -> None:
        try:
            with loaded.lock:
                loaded.backend.close()
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _load(self, info: ModelInfo) -> LoadedModel:
        s = self.settings
        t0 = time.perf_counter()
        desc = ModelLoader(self.device).load_from_file(info.path)
        if not isinstance(desc, ImageModelDescriptor):
            raise ValueError(f"{info.name}: unsupported model type {type(desc).__name__}")
        self._fill(info, desc)
        if info.input_channels != 3 or info.output_channels != 3:
            raise ValueError(f"{info.name}: only 3-channel models are supported")
        dtype = resolve_dtype(s.precision, desc.supports_half, desc.supports_bfloat16, self.device)
        tile = _effective_tile(s.tile, info.multiple_of, info.minimum)

        if s.backend == "ort":
            onnx_path = Path(s.onnx_dir) / onnx_file_name(info.name.replace("/", "__"), tile, s.batch, dtype)
            if not onnx_path.is_file():
                log.info("exporting %s", onnx_path)
                export_onnx(desc, onnx_path, tile, s.batch, dtype, self.device)
            options = {k: dict(v) for k, v in s.ort_provider_options.items()}
            options.setdefault("TensorrtExecutionProvider", {})
            options["TensorrtExecutionProvider"].setdefault("trt_fp16_enable", dtype == torch.float16)
            options["TensorrtExecutionProvider"].setdefault("trt_engine_cache_enable", True)
            options["TensorrtExecutionProvider"].setdefault("trt_engine_cache_path", s.engines_dir)
            Path(s.engines_dir).mkdir(parents=True, exist_ok=True)
            backend: Any = OrtBackend(
                onnx_path,
                s.ort_providers,
                options,
                s.ort_provider_library,
                tile,
                s.batch,
                info.input_channels,
                info.output_channels,
                info.scale,
                dtype,
                self.device,
            )
            del desc
            info.backend = f"ort:{backend.provider}"
            dtype = backend.dtype
            device = backend.device
        else:
            backend = TorchBackend(desc, self.device, dtype, s.channels_last, s.compile)
            info.backend = "torch"
            device = self.device

        info.loaded = True
        info.dtype = str(dtype).replace("torch.", "")
        info.tile = tile
        log.info("loaded %s (%s, x%d, %s, tile %d) in %.2fs", info.name, info.arch, info.scale, info.dtype, tile, time.perf_counter() - t0)
        return LoadedModel(info=info, backend=backend, device=device, dtype=dtype, scale=info.scale, tile=tile)

    def warmup(self, loaded: LoadedModel) -> float:
        t0 = time.perf_counter()
        x = torch.rand((self.settings.batch, 3, loaded.tile, loaded.tile), dtype=loaded.dtype, device=loaded.device)
        with loaded.lock:
            loaded.infer(x)
        if loaded.device.type == "cuda":
            torch.cuda.synchronize(loaded.device)
        return (time.perf_counter() - t0) * 1000.0

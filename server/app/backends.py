from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F


class Backend(Protocol):
    dtype: torch.dtype

    def __call__(self, tiles: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...


def resolve_dtype(precision: str, supports_half: bool, supports_bf16: bool, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if precision == "fp16" and supports_half:
        return torch.float16
    if precision in ("fp16", "bf16") and supports_bf16:
        return torch.bfloat16
    return torch.float32


class TorchBackend:
    def __init__(
        self,
        descriptor: Any,
        device: torch.device,
        dtype: torch.dtype,
        channels_last: bool = True,
        compile_model: bool = False,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.descriptor = descriptor.to(device).eval()
        if dtype == torch.float16:
            self.descriptor = self.descriptor.half()
        elif dtype == torch.bfloat16:
            self.descriptor = self.descriptor.bfloat16()
        self.memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.descriptor.model.to(memory_format=self.memory_format)
        self._forward = self.descriptor.model
        if compile_model:
            self._forward = torch.compile(self._forward, mode="max-autotune-no-cudagraphs")
        self._call_fn = getattr(self.descriptor, "_call_fn", None)

    @torch.inference_mode()
    def __call__(self, tiles: torch.Tensor) -> torch.Tensor:
        x = tiles.to(self.dtype).contiguous(memory_format=self.memory_format)
        if self._call_fn is not None:
            return self._call_fn(self._forward, x)
        return self._forward(x)

    def close(self) -> None:
        self.descriptor = None
        self._forward = None


class OrtBackend:
    def __init__(
        self,
        onnx_path: Path,
        providers: list[str],
        provider_options: dict[str, dict[str, Any]],
        provider_library: str,
        tile: int,
        batch: int,
        in_channels: int,
        out_channels: int,
        scale: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        import onnxruntime as ort

        if provider_library:
            try:
                ort.register_execution_provider_library(providers[0], provider_library)
            except Exception:
                pass
        available = set(ort.get_available_providers())
        chosen: list[tuple[str, dict[str, Any]]] = [
            (p, provider_options.get(p, {})) for p in providers if p in available
        ]
        if not chosen:
            chosen = [("CPUExecutionProvider", {})]
            device = torch.device("cpu")
        self.provider = chosen[0][0]
        self.device = device
        self.batch = batch
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), so, providers=chosen)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        self.dtype = torch.float16 if "float16" in model_input.type else torch.float32
        dtype = self.dtype
        self.np_dtype = {torch.float16: np.float16, torch.float32: np.float32}[dtype]
        self.out = torch.empty(
            (batch, out_channels, tile * scale, tile * scale), dtype=dtype, device=device
        )
        self._in_channels = in_channels

    def __call__(self, tiles: torch.Tensor) -> torch.Tensor:
        n = tiles.shape[0]
        x = tiles.to(device=self.device, dtype=self.dtype).contiguous()
        if n < self.batch:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, self.batch - n))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dev_type = self.device.type
        dev_id = self.device.index or 0
        io = self.session.io_binding()
        io.bind_input(
            self.input_name,
            device_type=dev_type,
            device_id=dev_id,
            element_type=self.np_dtype,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        io.bind_output(
            self.output_name,
            device_type=dev_type,
            device_id=dev_id,
            element_type=self.np_dtype,
            shape=tuple(self.out.shape),
            buffer_ptr=self.out.data_ptr(),
        )
        self.session.run_with_iobinding(io)
        return self.out[:n]

    def close(self) -> None:
        self.session = None
        self.out = None

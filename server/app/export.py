from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def onnx_file_name(model_name: str, tile: int, batch: int, dtype: torch.dtype) -> str:
    prec = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, "fp32")
    return f"{model_name}_b{batch}_t{tile}_{prec}.onnx"


def export_onnx(
    descriptor: Any,
    path: Path,
    tile: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    opset: int = 17,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = descriptor.to(device).eval()
    if dtype == torch.float16:
        desc = desc.half()
    elif dtype == torch.bfloat16:
        raise ValueError("bf16 export is not supported; use fp16 or fp32")
    model = desc.model
    dummy = torch.zeros((batch, desc.input_channels, tile, tile), dtype=dtype, device=device)
    kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )
    with torch.inference_mode():
        try:
            torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(model, (dummy,), str(path), **kwargs)
    return path

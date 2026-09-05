from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .codec import (
    apply_orientation,
    decode_image,
    encode_jpeg_bytes,
    encode_png_bytes,
    read_orientation,
    to_rgb_bytes,
    to_rgba_bytes,
)
from .registry import LoadedModel
from .tiling import tiled_forward

OUTPUT_FORMATS = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "rgb": "application/octet-stream",
    "rgba": "application/octet-stream",
}


class StageTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.marks: dict[str, float] = {}
        self._t = time.perf_counter()
        self._start = self._t

    def mark(self, name: str) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self.marks[name] = (now - self._t) * 1000.0
        self._t = now

    def total(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


@dataclass
class Result:
    data: bytes
    media_type: str
    width: int
    height: int
    scale: int
    timings: dict[str, float] = field(default_factory=dict)


def process(
    loaded: LoadedModel,
    data: bytes,
    output: str,
    quality: int,
    overlap: int,
    batch: int,
) -> Result:
    if output not in OUTPUT_FORMATS:
        raise ValueError(f"unknown output format: {output}")
    device = loaded.device
    timer = StageTimer(device)

    orientation = read_orientation(data)
    img = decode_image(data, device)
    if orientation != 1:
        img = apply_orientation(img, orientation).contiguous()
    timer.mark("decode")

    x = img.unsqueeze(0).to(loaded.dtype).div_(255.0)
    with loaded.lock:
        y = tiled_forward(x, loaded.infer, loaded.tile, overlap, loaded.scale, batch)
    y0 = y[0].float() if y.dtype == torch.bfloat16 else y[0]
    out = y0.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
    del x, y
    timer.mark("infer")

    if output == "jpeg":
        payload = encode_jpeg_bytes(out, quality)
    elif output == "png":
        payload = encode_png_bytes(out)
    elif output == "rgb":
        payload = to_rgb_bytes(out)
    else:
        payload = to_rgba_bytes(out)
    timer.mark("encode")

    _, h, w = out.shape
    timings = dict(timer.marks)
    timings["total"] = timer.total()
    return Result(payload, OUTPUT_FORMATS[output], w, h, loaded.scale, timings)

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

Infer = Callable[[torch.Tensor], torch.Tensor]


def tiled_forward(
    x: torch.Tensor,
    fn: Infer,
    tile: int,
    overlap: int,
    scale: int,
    batch: int,
) -> torch.Tensor:
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("expected a (1, C, H, W) tensor")
    core = tile - 2 * overlap
    if core <= 0:
        raise ValueError("tile must be larger than 2 * overlap")

    _, _, h, w = x.shape
    ny = math.ceil(h / core)
    nx = math.ceil(w / core)
    pad_h = ny * core - h
    pad_w = nx * core - w
    xp = F.pad(x, (overlap, overlap + pad_w, overlap, overlap + pad_h), mode="replicate")

    coords = [(iy, ix) for iy in range(ny) for ix in range(nx)]
    out: torch.Tensor | None = None
    o = overlap * scale
    for i in range(0, len(coords), max(1, batch)):
        chunk = coords[i : i + max(1, batch)]
        tiles = torch.stack(
            [xp[0, :, iy * core : iy * core + tile, ix * core : ix * core + tile] for iy, ix in chunk]
        )
        res = fn(tiles)
        if out is None:
            out = res.new_empty((1, res.shape[1], h * scale, w * scale))
        for k, (iy, ix) in enumerate(chunk):
            y0 = iy * core * scale
            x0 = ix * core * scale
            ch = min(core, h - iy * core) * scale
            cw = min(core, w - ix * core) * scale
            out[0, :, y0 : y0 + ch, x0 : x0 + cw] = res[k, :, o : o + ch, o : o + cw]
    assert out is not None
    return out


def tile_count(h: int, w: int, tile: int, overlap: int) -> int:
    core = tile - 2 * overlap
    return math.ceil(h / core) * math.ceil(w / core)

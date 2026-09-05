from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

try:
    from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg

    _HAS_TV = True
except Exception:  # pragma: no cover
    _HAS_TV = False

_EXIF_ORIENTATION = 0x0112


def read_orientation(data: bytes) -> int:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return int(im.getexif().get(_EXIF_ORIENTATION, 1))
    except Exception:
        return 1


def apply_orientation(img: torch.Tensor, orientation: int) -> torch.Tensor:
    dims = (-2, -1)
    if orientation == 2:
        return img.flip(-1)
    if orientation == 3:
        return img.rot90(2, dims)
    if orientation == 4:
        return img.flip(-2)
    if orientation == 5:
        return img.transpose(-2, -1)
    if orientation == 6:
        return img.rot90(-1, dims)
    if orientation == 7:
        return img.transpose(-2, -1).rot90(2, dims)
    if orientation == 8:
        return img.rot90(1, dims)
    return img


def decode_image(data: bytes, device: torch.device) -> torch.Tensor:
    if _HAS_TV and data[:2] == b"\xff\xd8":
        try:
            buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            if device.type == "cuda":
                return decode_jpeg(buf, mode=ImageReadMode.RGB, device=device)
            return decode_jpeg(buf, mode=ImageReadMode.RGB).to(device)
        except Exception:
            pass
    with Image.open(io.BytesIO(data)) as im:
        arr = np.asarray(im.convert("RGB"))
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().to(device)


def _pil_encode(img: torch.Tensor, fmt: str, **kwargs) -> bytes:
    arr = img.permute(1, 2, 0).contiguous().cpu().numpy()
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format=fmt, **kwargs)
    return out.getvalue()


def encode_jpeg_bytes(img: torch.Tensor, quality: int) -> bytes:
    if _HAS_TV:
        try:
            return encode_jpeg(img.contiguous(), quality=quality).cpu().numpy().tobytes()
        except Exception:
            pass
    return _pil_encode(img, "JPEG", quality=quality, subsampling=0)


def encode_png_bytes(img: torch.Tensor) -> bytes:
    return _pil_encode(img, "PNG", compress_level=1)


def to_rgb_bytes(img: torch.Tensor) -> bytes:
    return img.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()


def to_rgba_bytes(img: torch.Tensor) -> bytes:
    _, h, w = img.shape
    alpha = torch.full((h, w, 1), 255, dtype=torch.uint8, device=img.device)
    rgba = torch.cat([img.permute(1, 2, 0), alpha], dim=-1).contiguous()
    return rgba.cpu().numpy().tobytes()

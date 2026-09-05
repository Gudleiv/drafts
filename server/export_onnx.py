from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from spandrel import ModelLoader

from app.backends import resolve_dtype
from app.config import Settings
from app.export import export_onnx, onnx_file_name
from app.registry import _effective_tile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tile", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--precision", choices=["fp16", "fp32"])
    parser.add_argument("--device")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out")
    args = parser.parse_args()

    overrides = {k: getattr(args, k) for k in ("tile", "batch", "precision", "device")}
    settings = Settings.load(args.config, overrides)
    base = Path(args.config).resolve().parent if Path(args.config).is_file() else Path.cwd()
    settings.resolve(base)

    device = torch.device(settings.device if torch.cuda.is_available() or settings.device == "cpu" else "cpu")
    root = Path(settings.models_dir)
    candidates = [p for p in root.rglob("*") if p.is_file() and p.relative_to(root).with_suffix("").as_posix() == args.model]
    if not candidates:
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1
    desc = ModelLoader(device).load_from_file(str(candidates[0]))
    dtype = resolve_dtype(settings.precision, desc.supports_half, False, device)
    req = desc.size_requirements
    tile = _effective_tile(settings.tile, int(getattr(req, "multiple_of", 1) or 1), int(getattr(req, "minimum", 0) or 0))
    out = Path(args.out) if args.out else Path(settings.onnx_dir) / onnx_file_name(args.model.replace("/", "__"), tile, settings.batch, dtype)
    export_onnx(desc, out, tile, settings.batch, dtype, device, args.opset)
    print(f"{out}  input=({settings.batch},{desc.input_channels},{tile},{tile}) dtype={str(dtype).replace('torch.', '')} scale={desc.scale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

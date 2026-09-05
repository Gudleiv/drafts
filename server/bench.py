from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from app.config import Settings
from app.pipeline import process
from app.registry import Registry


def fmt(v: float) -> str:
    return f"{v:8.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="jpeg")
    parser.add_argument("--save", default="")
    parser.add_argument("--device")
    parser.add_argument("--backend", choices=["torch", "ort"])
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--tile", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--compile", action="store_true", default=None)
    args = parser.parse_args()

    overrides = {k: getattr(args, k) for k in ("device", "backend", "precision", "tile", "overlap", "batch", "compile")}
    settings = Settings.load(args.config, overrides)
    base = Path(args.config).resolve().parent if Path(args.config).is_file() else Path.cwd()
    settings.resolve(base)
    settings.max_loaded_models = 1

    registry = Registry(settings)
    names = registry.names() if args.all or not args.models else args.models
    if not names:
        print("no models found", file=sys.stderr)
        return 1

    data = Path(args.image).read_bytes()
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={registry.device} backend={settings.backend} precision={settings.precision} "
          f"tile={settings.tile} overlap={settings.overlap} batch={settings.batch} input={len(data) / 1e6:.1f} MB")
    header = f"{'model':40s} {'arch':22s} {'x':>2s} {'tile':>5s} {'decode':>8s} {'infer':>8s} {'encode':>8s} {'total':>8s} {'MP/s':>7s} {'vram MB':>8s}"
    print(header)
    print("-" * len(header))

    for name in names:
        try:
            loaded = registry.get(name)
            registry.warmup(loaded)
        except Exception as exc:
            print(f"{name:40s} load failed: {type(exc).__name__}: {exc}")
            continue
        if registry.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        best = None
        result = None
        for _ in range(max(1, args.runs)):
            t0 = time.perf_counter()
            result = process(loaded, data, args.output, settings.jpeg_quality, settings.overlap, settings.batch)
            wall = (time.perf_counter() - t0) * 1000.0
            if best is None or wall < best[0]:
                best = (wall, dict(result.timings))
        assert best is not None and result is not None
        wall, t = best
        in_mp = (result.width / loaded.scale) * (result.height / loaded.scale) / 1e6
        mps = in_mp / (t["infer"] / 1000.0) if t.get("infer") else 0.0
        vram = torch.cuda.max_memory_allocated() / 1e6 if registry.device.type == "cuda" else 0.0
        print(f"{name:40s} {loaded.info.arch[:22]:22s} {loaded.scale:2d} {loaded.tile:5d} "
              f"{fmt(t['decode'])} {fmt(t['infer'])} {fmt(t['encode'])} {fmt(wall)} {mps:7.1f} {vram:8.0f}")
        if save_dir:
            ext = {"jpeg": "jpg", "png": "png"}.get(args.output, "bin")
            (save_dir / f"{name.replace('/', '__')}.{ext}").write_bytes(result.data)
        registry.unload(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .pipeline import OUTPUT_FORMATS, process
from .registry import Registry

log = logging.getLogger("server")

EXPOSED_HEADERS = ["X-Width", "X-Height", "X-Scale", "X-Model", "X-Format", "Server-Timing"]


def create_app(settings: Settings) -> FastAPI:
    registry = Registry(settings)
    gpu_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for name in settings.preload:
            try:
                loaded = await run_in_threadpool(registry.get, name)
                ms = await run_in_threadpool(registry.warmup, loaded)
                log.info("preloaded %s, warmup %.0f ms", name, ms)
            except Exception as exc:
                log.error("preload failed for %s: %s", name, exc)
        yield

    app = FastAPI(title="Image Processing Service", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=EXPOSED_HEADERS,
    )
    app.state.settings = settings
    app.state.registry = registry

    @app.get("/health")
    async def health():
        cuda = torch.cuda.is_available()
        return {
            "status": "ok",
            "device": str(registry.device),
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "backend": settings.backend,
            "precision": settings.precision,
            "tile": settings.tile,
            "overlap": settings.overlap,
            "batch": settings.batch,
        }

    @app.get("/models")
    async def models(rescan: bool = False):
        if rescan:
            await run_in_threadpool(registry.scan)
        return registry.list()

    @app.post("/models/{name:path}/load")
    async def load_model(name: str):
        try:
            loaded = await run_in_threadpool(registry.get, name)
        except KeyError:
            raise HTTPException(404, f"unknown model: {name}")
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
        ms = await run_in_threadpool(registry.warmup, loaded)
        return {"model": loaded.info.public(), "warmup_ms": round(ms, 1)}

    @app.post("/models/{name:path}/unload")
    async def unload_model(name: str):
        ok = await run_in_threadpool(registry.unload, name)
        return {"unloaded": ok}

    @app.post("/process")
    async def process_endpoint(
        request: Request,
        model: str = Query(...),
        output: str = Query("jpeg"),
        quality: int | None = Query(None, ge=1, le=100),
        overlap: int | None = Query(None, ge=0),
        batch: int | None = Query(None, ge=1),
    ):
        if output not in OUTPUT_FORMATS:
            raise HTTPException(400, f"output must be one of {sorted(OUTPUT_FORMATS)}")
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file") or form.get("image")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(400, "multipart form must contain a 'file' field")
            data = await upload.read()
        else:
            data = await request.body()
        if not data:
            raise HTTPException(400, "empty body")
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"body exceeds {settings.max_upload_mb} MB")

        try:
            loaded = await run_in_threadpool(registry.get, model)
        except KeyError:
            raise HTTPException(404, f"unknown model: {model}")
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")

        q = quality if quality is not None else settings.jpeg_quality
        ov = overlap if overlap is not None else settings.overlap
        bs = batch if batch is not None else settings.batch

        def work():
            with gpu_lock:
                return process(loaded, data, output, q, ov, bs)

        try:
            result = await run_in_threadpool(work)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(507, "out of GPU memory; reduce tile or batch")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            log.exception("processing failed")
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")

        server_timing = ", ".join(f"{k};dur={v:.1f}" for k, v in result.timings.items())
        headers = {
            "X-Width": str(result.width),
            "X-Height": str(result.height),
            "X-Scale": str(result.scale),
            "X-Model": model,
            "X-Format": output,
            "Server-Timing": server_timing,
            "Cache-Control": "no-store",
        }
        return Response(content=result.data, media_type=result.media_type, headers=headers)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app


def build_settings(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--models-dir", dest="models_dir")
    parser.add_argument("--device")
    parser.add_argument("--backend", choices=["torch", "ort"])
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--tile", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--preload", nargs="*")
    parser.add_argument("--log-level", dest="log_level")
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    settings = Settings.load(args.config, overrides)
    base = Path(args.config).resolve().parent if Path(args.config).is_file() else Path.cwd()
    return settings.resolve(base)


def main(argv: list[str] | None = None) -> None:
    settings = build_settings(argv)
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()

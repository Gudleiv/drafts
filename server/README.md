# Image Processing Service

Local HTTP service that runs image restoration and super-resolution models on an NVIDIA GPU.
Send a JPEG or PNG, get back the processed image as JPEG, PNG, or raw RGB/RGBA pixels.

- Model files are loaded through [spandrel](https://github.com/chaiNNer-org/spandrel): drop any supported `.pth` / `.safetensors` into `models/`.
- JPEG decode and encode run on the GPU through nvJPEG (torchvision), with a CPU fallback.
- Fixed-size tiling with overlap keeps VRAM bounded and keeps tensor shapes static.
- Two inference backends: PyTorch (default, works with every model) and ONNX Runtime with the TensorRT-RTX / TensorRT execution providers.
- `bench.py` measures decode / inference / encode time per model so models can be compared on the same input.

## Requirements

- Windows 11 (native) or Linux. WSL2 works as well, but a native Windows process avoids the extra layer.
- NVIDIA GPU, Ampere or newer, with a current driver.
- Python 3.11 or 3.12.

## Install

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

Pick the wheel index that matches the CUDA runtime you want (`cu128`, `cu130`, `cu132`, ...); the newest index the driver supports is usually the best choice, and builds older than `cu128` do not cover the newest GPU generations. Verify the install:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If the ONNX Runtime backend is used as well, install an `onnxruntime` package built against the same major CUDA version as the torch wheel, so that one process does not load two incompatible runtimes.

Optional, for the ONNX Runtime backend (one of):

```powershell
pip install onnxruntime-trt-rtx        # TensorRT-RTX execution provider
pip install onnxruntime-gpu            # TensorRT + CUDA execution providers
```

See the [TensorRT-RTX execution provider docs](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html) for the current package name and driver requirements. On Windows the provider DLL may need explicit registration; set `ort_provider_library` in `config.yaml` to the path of `onnxruntime_providers_nv_tensorrt_rtx.dll` if `ort.get_available_providers()` does not list it.

Some architectures live in a separate package with a different license:

```powershell
pip install spandrel_extra_arches
```

## Models

Put model files into `models/` (subfolders are allowed; the subfolder becomes part of the model name).
Accepted extensions: `.pth`, `.pt`, `.safetensors`, `.ckpt`. Only 3-channel RGB models are served.

A `.safetensors` file stores tensors without architecture metadata, so spandrel infers the architecture from weight names and shapes. If a file cannot be identified, it still appears in `GET /models` with a non-empty `error` field; use the `.pth` release of the same model in that case.

Suggested starting points from [OpenModelDB](https://openmodeldb.info/) (filter by architecture `Compact`, `SPAN`, `RealPLKSR` and by scale `1x` / `2x`):

- `1x` restoration models (denoise, anti-aliasing, JPEG artifact removal) keep the output size and are the cheapest option.
- `2x` Compact / SPAN models are fast enough for large inputs; heavier architectures (ESRGAN/RRDB, DAT, HAT, NAFNet, SCUNet) are several times slower.

Check that every model was recognized:

```powershell
python -c "from app.config import Settings; from app.registry import Registry; import json; print(json.dumps(Registry(Settings.load('config.yaml')).list(), indent=1))"
```

## Configure

```powershell
copy config.example.yaml config.yaml
```

Key options:

| Option | Meaning |
|---|---|
| `backend` | `torch` or `ort` |
| `precision` | `fp16`, `bf16`, `fp32`; falls back per model capability |
| `tile` | tile edge in pixels; rounded up to the model's size requirement |
| `overlap` | overlap in pixels around each tile; must cover the model's receptive field (32 is enough for Compact / SPAN, use 64 or more for deep networks) |
| `batch` | tiles per forward pass |
| `channels_last` | NHWC memory format for the PyTorch backend |
| `compile` | wrap the model with `torch.compile` (longer first run) |
| `max_loaded_models` | LRU limit of resident models |
| `preload` | model names loaded and warmed up at startup |
| `cors_origins` | allowed origins; `chrome-extension://<id>` entries can be listed explicitly |

Every option can also be passed on the command line (`--tile 2048 --batch 2`).

VRAM per tile grows with `tile^2 * batch * channels`. Starting points for a 16 GB card: `tile: 1024, batch: 4` or `tile: 2048, batch: 1`. If a request fails with HTTP 507, lower `tile` or `batch`.

## Run

```powershell
python -m app.main --config config.yaml
```

Open `client/test.html` in a browser to send a file and see per-stage timings.

## API

`GET /health`
Returns device, GPU name and active defaults.

`GET /models?rescan=false`
Lists models with architecture, scale, dtype and load state. `rescan=true` re-reads the models folder.

`POST /models/{name}/load`
Loads the model and runs one warm-up pass. `POST /models/{name}/unload` releases it.

`POST /process?model=NAME&output=jpeg|png|rgb|rgba&quality=92&overlap=32&batch=4`
Body: raw image bytes (`Content-Type: image/jpeg`) or `multipart/form-data` with a `file` field.

Response body is the processed image. Headers:

| Header | Content |
|---|---|
| `X-Width`, `X-Height` | output size in pixels |
| `X-Scale` | model scale factor |
| `Server-Timing` | `decode`, `infer`, `encode`, `total` in milliseconds |

Output formats:

- `jpeg` encoded on the GPU; `quality` sets the JPEG quality.
- `png` encoded on the CPU; slow for large images, intended for inspection.
- `rgb` / `rgba` raw 8-bit interleaved pixels, row-major, no header. Use with `ImageData` + `putImageData` on a canvas to skip the browser-side decode.

EXIF orientation is applied before processing; the output carries no EXIF.

Example:

```powershell
curl.exe -s -X POST "http://127.0.0.1:8765/process?model=MODEL_NAME&output=jpeg" `
  -H "Content-Type: image/jpeg" --data-binary "@input.jpg" -o output.jpg -D -
```

## Benchmark

```powershell
python bench.py --image input.jpg --all --runs 3 --save out
python bench.py --image input.jpg --models MODEL_A MODEL_B --tile 2048 --batch 1
python bench.py --image input.jpg --all --backend ort
```

Prints decode / infer / encode / total milliseconds, throughput in megapixels per second, and peak VRAM for each model. `--save` writes every output next to each other for visual comparison.

## ONNX Runtime backend

With `backend: ort` the service exports each model to ONNX on first use with static shapes `(batch, 3, tile, tile)` into `onnx/` and builds a session with the first available provider from `ort_providers`. TensorRT engine caches are written to `engines/`. The first load of a model takes longer while the engine is built; subsequent starts reuse the cache.

To export ahead of time:

```powershell
python export_onnx.py --model MODEL_NAME --tile 1024 --batch 4 --precision fp16
```

Provider options are passed through `ort_provider_options`, keyed by provider name.

## Notes

- The service processes one request at a time on the GPU; concurrent requests are queued.
- `rgb` / `rgba` output for a 6000x6000 input is 108 / 144 MB per response; keep the service on loopback.
- If `torchvision.io.decode_jpeg` cannot handle a file (rare JPEG variants), decoding silently falls back to Pillow on the CPU, which shows up as a longer `decode` timing.

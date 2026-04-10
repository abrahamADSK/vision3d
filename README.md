# Vision3D

> GPU inference server for AI-powered 3D asset generation using Hunyuan3D-2

> [!WARNING]
> **Experimental project — use at your own risk.**
> This is an independent, unofficial experiment created with [Claude Code](https://claude.com/claude-code). It is **not** affiliated with, endorsed by, or officially supported by Tencent (Hunyuan3D-2) or Stability AI (SDXL Turbo) in any way. All model names and trademarks belong to their respective owners.
>
> Running AI inference pipelines on GPU hardware carries real risks: **high VRAM usage, potential system instability, large model downloads, and non-deterministic outputs.** Always monitor GPU resources and validate outputs before using in production. The author(s) accept no responsibility for hardware issues, incorrect results, or any other damage resulting from its use.

AI-powered 3D model generation server. Turns images (or text prompts) into textured 3D meshes using GPU inference.

Built on [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) (Tencent) for shape and texture generation, and [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) (Stability AI) for the intermediate image generation step in the text-to-3D flow. Exposed as a REST API with a web UI, real-time progress streaming, and automatic polygon decimation.

## Features

- **Image-to-3D**: Upload a reference image, get a textured 3D model (`.glb`, `.obj` + baked texture)
- **Text-to-3D**: Describe an object in English, get a 3D mesh (uses SDXL Turbo as intermediate step)
- **Full pipeline**: Shape generation + decimation + texturing in one call
- **Polygon decimation**: Reduce dense meshes to a target face count (default 50k)
- **Real-time progress**: Server-Sent Events (SSE) stream for live feedback
- **Web UI**: Browser-based interface — no CLI or MCP client needed
- **REST API**: Integrate from any language, tool, or pipeline

## Requirements

- **GPU** — one of:
  - NVIDIA GPU with 16+ GB VRAM (CUDA — tested on RTX 3090, 24 GB)
  - Apple Silicon Mac with 16+ GB unified RAM (MPS — tested on M4 Pro / M5 Pro)
- Python 3.10+
- **Hunyuan3D-2**:
  - CUDA: [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) (original fork) + turbo model
  - MPS: [abrahamADSK/Hunyuan3D-2-Mac](https://github.com/abrahamADSK/Hunyuan3D-2-Mac) (Mac fork with MPS texturing fixes) + base model v2-0 (turbo requires a scheduler not available in the Mac fork)
- [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) (~6 GB, auto-downloaded on first text-to-3D use)
- `diffusers`, `transformers`, `accelerate` (for SDXL Turbo text-to-image)
- `rembg` + `onnxruntime` (background removal for text-to-3D)
- See [requirements.txt](requirements.txt) for the full dependency list

## Quick Start

### 1. Clone Vision3D

```bash
git clone https://github.com/abrahamADSK/vision3d.git
cd vision3d
```

### 2. Install

```bash
bash install.sh
```

`install.sh` detects your platform (CUDA / MPS / CPU) automatically and handles:
- Creating a `.venv/` virtual environment
- Installing all dependencies from `requirements.txt`
- Finding and installing the correct Hunyuan3D-2 fork in editable mode
- Installing mesh-processing extras (pymeshlab, xatlas, etc.)
- Verifying all critical imports and torch backend

Safe to run multiple times (idempotent).

### 3. Run

```bash
.venv/bin/python server.py --host 0.0.0.0 --port 8000
```

Open `http://YOUR_GPU_HOST:8000` in a browser.

### 4. Backend selection (local MPS vs remote CUDA) — v1.5.0+

At startup, `server.py` asks where to run inference:

```
Run locally? [y/N]: n
Remote host: <gpu-host>
→ Connected to http://<gpu-host>:8000
```

- Answer `y` → run inference **locally** (Apple Silicon MPS or local CUDA).
- Answer `n` (or just press Enter) → enter a **remote host**. The local server validates `GET http://{host}:8000/api/health` with a 5 s timeout and loops until a remote answers.
- The prompt offers a default only if the env var `VISION3D_DEFAULT_REMOTE_HOST` is set in your shell (e.g. in `~/.zshrc`). If unset, the prompt has no default and the hostname is required. **No machine name is hardcoded in the repo.**

In remote mode the local process becomes a thin **HTTP façade**: every `/api/*` call (generation, polling, file downloads, SSE progress stream) is proxied to the remote vision3d instance. There is **zero local job state** — job IDs are owned by the remote. The local web UI is unaware of the difference. This is useful when you want to develop and test from a Mac while delegating the heavy CUDA work to a Linux GPU box.

#### Skipping the prompt (CI / scripts)

```bash
.venv/bin/python server.py --local                    # force local
.venv/bin/python server.py --remote <gpu-host>        # force remote
.venv/bin/python server.py --reload                   # forces local (prompt incompatible with reload)
```

#### Environment variables

| Variable | Default | Description |
|---|---|---|
| `VISION3D_DEFAULT_REMOTE_HOST` | unset | Hostname offered as the default in the interactive prompt. Personal config — set in your shell, never committed. |
| `VISION3D_REMOTE_HOST` | unset | If set, the server starts in remote mode without prompting. Set automatically by `--remote` and by the interactive prompt (so uvicorn workers and `--reload` inherit it). |
| `VISION3D_REMOTE_PORT` | `8000` | Port on the remote host. |
| `VISION3D_REMOTE_KEY` | unset | API key forwarded to the remote on every proxied request. If unset, the inbound `x-api-key` from the local client is forwarded as-is. |

The local `GET /api/health` endpoint reports the active mode in its JSON response (`mode: "local"` or `"remote"`, plus `remote_host` / `remote_port` when applicable).

### Apple Silicon (MPS) notes

- The Mac fork is [abrahamADSK/Hunyuan3D-2-Mac](https://github.com/abrahamADSK/Hunyuan3D-2-Mac) (a fork of Maxim-Lanskoy/Hunyuan3D-2-Mac with the MPS texturing fixes). Clone it **outside** the vision3d repo — `install.sh` searches for it in `../hunyuan3d-mac`, `~/Projects/hunyuan3d-mac`, and `~/Claude_projects/hunyuan3d-mac`.
- Uses the base model `hunyuan3d-dit-v2-0` (not turbo) with `variant='fp16'`, `use_safetensors=True`.
- Weights are downloaded automatically from HuggingFace on first run (~18.8 GB for `hunyuan3d-dit-v2-0`).
- `custom_rasterizer` compiles on Mac via the [hunyuan3d-mac](https://github.com/abrahamADSK/Hunyuan3D-2-Mac) fork. Texturing works on MPS since v1.3.0.
- Generation verified: 355k vertices, 710k faces.

#### Verified stack on Apple Silicon (M4 Pro, 48 GB)

After an extensive debugging session, the following stack is confirmed working end-to-end (shape + texture) on macOS via the [hunyuan3d-mac](https://github.com/abrahamADSK/hunyuan3d-mac) fork:

| Component | Version |
|---|---|
| macOS | 26.3.1 (Tahoe) |
| Python | 3.13.9 |
| PyTorch | **2.6.0** (pinned) |
| torchvision | 0.21.0 |
| torchaudio | 2.6.0 |
| diffusers | 0.37.1 |
| transformers | 5.5.0 |
| accelerate | 1.13.0 |

> [!WARNING]
> **Do not upgrade PyTorch beyond 2.6.0 on macOS/MPS.** PyTorch 2.7+ introduced an MPS regression that causes diffusion pipelines to produce noise/NaN output even in fp32 (confirmed on torch 2.11.0). The previous known-good 2.4.1 has no cp313 wheels, so 2.6.0 is the oldest version usable on Python 3.13. CUDA is unaffected.

**fp16 is unsafe on MPS for some SD-based UNets.** Specifically, the Hunyuan3D multiview UNet and the SD x4 upscaler emit `invalid value encountered in cast` (NaN) warnings in fp16 on MPS with torch 2.6.0. The fix — load those two pipelines in fp32 when `device == 'mps'` — is applied in the [hunyuan3d-mac](https://github.com/abrahamADSK/hunyuan3d-mac) fork. The delight pipeline remains stable in fp16. This is a per-model issue, not a blanket MPS limitation.

**Rebuild C++ extensions after any torch version change.** PyTorch native extensions (e.g. `custom_rasterizer_kernel`, `mesh_processor` in the hunyuan3d-mac fork) break ABI between minor torch releases. Symptom when not rebuilt: `Symbol not found: __ZNK3c10*` at import time. Reinstall the extensions in editable mode after upgrading or downgrading torch.

**Texturing benchmark (M4 Pro, 48 GB unified memory)**: full image-to-textured-mesh pipeline 626 s end-to-end, peak memory 23.8 GB (49.5 % of unified RAM), zero NaN warnings, visual quality verified.

### 4. (Optional) Install as systemd service

```bash
sudo bash setup.sh
```

This creates a `vision3d.service` that starts on boot and generates an API key.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Health check + GPU info |
| `GET` | `/api/presets` | List available quality presets |
| `POST` | `/api/generate-full` | Full pipeline: image → shape → decimate → texture |
| `POST` | `/api/generate-shape` | Image → mesh (shape only) |
| `POST` | `/api/generate-text` | Text prompt → mesh (uses SDXL Turbo + Hunyuan3D-2) |
| `POST` | `/api/texture-mesh` | Mesh + image → textured mesh |
| `GET` | `/api/jobs/{id}` | Poll job status |
| `GET` | `/api/jobs/{id}/stream` | SSE real-time progress |
| `GET` | `/api/jobs/{id}/files/{name}` | Download result file |

### Quality presets

Each preset controls both the generation quality (mesh resolution, inference steps) and the post-processing (polygon decimation):

| Preset | Faces | Octree Resolution | Inference Steps | Use case |
|--------|-------|-------------------|-----------------|----------|
| `low` | 10k | 256 | 20 | Mobile/web, fast previews |
| `medium` | 50k | 384 | 30 | General use (default) |
| `high` | 150k | 384 | 50 | Detailed models |
| `ultra` | no limit | 512 | 50 | Maximum geometric detail |

### Example: Full pipeline with curl

```bash
# Using a preset:
curl -X POST http://localhost:8000/api/generate-full \
  -F "image=@photo.png" \
  -F "preset=high"

# Or with explicit face count:
curl -X POST http://localhost:8000/api/generate-full \
  -F "image=@photo.png" \
  -F "target_faces=50000"

# Poll until done:
curl http://localhost:8000/api/jobs/abc123
# Download textured model:
curl -O http://localhost:8000/api/jobs/abc123/files/textured.glb
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_API_KEY` | _(empty = open)_ | API key for authentication. Leave empty for LAN use. Clients send the key via `X-API-Key` header, or `?key=<value>` query param for SSE connections (EventSource does not support custom headers). |
| `GPU_MODELS_DIR` | `./hf_models` | Path to Hunyuan3D-2 model weights |
| `GPU_WORK_DIR` | `./output` | Working directory for job outputs |
| `GPU_VISION_DIR` | `.` (script dir) | Vision3D installation directory |

## Architecture

```
Browser / MCP client / curl
         │
         ▼
   ┌─────────────┐
   │  Vision3D    │  FastAPI + uvicorn
   │  server.py   │  REST API + Web UI + SSE
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │ Hunyuan3D-2 │  Shape generation (DiT)
   │ + SDXL Turbo│  Text → image (text-to-3D)
   │  pipelines  │  Texture painting
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │  NVIDIA GPU  │  CUDA inference (shape + texture)
   │  — or —      │
   │  Apple GPU   │  MPS inference (shape + texture)
   └─────────────┘
```

## Models used

| Model | Source | Size | Function |
|-------|--------|------|----------|
| `hunyuan3d-dit-v2-0-turbo` | Tencent/Hunyuan3D-2 | ~400 MB | 3D shape generation (CUDA — turbo) |
| `hunyuan3d-dit-v2-0-fast` | Tencent/Hunyuan3D-2 | ~4.93 GB | 3D shape generation (CUDA + MPS) |
| `hunyuan3d-dit-v2-0` | Tencent/Hunyuan3D-2 | ~18.8 GB | 3D shape generation (CUDA + MPS — full/base model) |
| `hunyuan3d-paint-v2-0-turbo` | Tencent/Hunyuan3D-2 | ~14 GB | Texture painting (CUDA only) |
| `hunyuan3d-paint-v2-0` | Tencent/Hunyuan3D-2 | ~14 GB | Texture painting (CUDA + MPS) |
| `hunyuan3d-delight-v2-0` | Tencent/Hunyuan3D-2 | ~4 GB | Relighting (paint dependency) |
| `sdxl-turbo` | Stability AI | ~6 GB | Text → image (intermediate step) |

### Model compatibility by platform

| Model | Type | CUDA | MPS | Notes |
|-------|------|------|-----|-------|
| `hunyuan3d-dit-v2-0-turbo` | Shape | Yes | No | Requires `ConsistencyFlowMatchEulerDiscreteScheduler` (missing in Mac fork) |
| `hunyuan3d-dit-v2-0-fast` | Shape | Yes | Yes | Requires `PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically by `server.py`) |
| `hunyuan3d-dit-v2-0` (full) | Shape | Yes | Yes | Default on MPS |
| `hunyuan3d-paint-v2-0-turbo` | Paint | Yes | No | Needs fork changes for MPS |
| `hunyuan3d-paint-v2-0` (normal) | Paint | Yes | Yes | Since v1.3.0 via hunyuan3d-mac fork |

> **Note:** On Apple Silicon, `server.py` automatically selects `full` as the default shape model and the normal paint model. The turbo variants require scheduler/fork changes not yet available on MPS.

## Project Structure

```
vision3d/
├── server.py          # FastAPI server — all endpoints, pipelines, and web UI
├── install.sh         # Automated installer (CUDA/MPS/CPU detection, venv, deps)
├── setup.sh           # GPU deployment script (systemd service, API key)
├── requirements.txt   # Python dependencies
└── .env.example       # Environment variables template
```

## Troubleshooting

**CUDA out of memory**
- Reduce quality preset (use `draft` or `medium` instead of `high`)
- Check GPU memory with `nvidia-smi` — the full pipeline requires ~16 GB VRAM
- Only one generation job should run at a time (no concurrent GPU access)

**Model weights not found**
- Run the download script from Hunyuan3D-2: `python download_weights.py`
- Verify weights are in the path specified by `CHECKPOINT_DIR` in your environment

**Server starts but generation fails**
- Check that `custom_rasterizer` compiled successfully (required for texturing on CUDA)
- Verify Hunyuan3D-2 installation: `python -c "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline"`
- Check server logs for stack traces

**MPS: turbo model not loading**
- The turbo model requires `ConsistencyFlowMatchEulerDiscreteScheduler`, which does not exist in the Mac fork
- Use the base model `hunyuan3d-dit-v2-0` instead — `server.py` selects it automatically on MPS

**MPS: texturing issues**
- Texturing works on MPS since v1.3.0 via the [hunyuan3d-mac](https://github.com/abrahamADSK/Hunyuan3D-2-Mac) fork. If you see rasterization errors, ensure you are using the fork (not upstream) and that `custom_rasterizer` compiled successfully.
- Metal optimization of `custom_rasterizer` for better MPS performance is a future improvement (127 CUDA lines to port).

## Ecosystem

`vision3d` is part of a four-component VFX pipeline. Each component has a defined role:

| Repo | Role |
|------|------|
| [flame-mcp](https://github.com/abrahamADSK/flame-mcp) | Controls Autodesk Flame for compositing, conform, and finishing |
| [maya-mcp](https://github.com/abrahamADSK/maya-mcp) | Controls Autodesk Maya for 3D modeling, animation, and rendering |
| [fpt-mcp](https://github.com/abrahamADSK/fpt-mcp) | Connects to Autodesk Flow Production Tracking (ShotGrid) for production tracking, asset management, and publishes |
| [vision3d](https://github.com/abrahamADSK/vision3d) | GPU inference server for AI-powered 3D generation — the remote backend for maya-mcp's image-to-3D and text-to-3D tools |

`vision3d` is the GPU computation layer of the ecosystem. It exposes no MCP interface directly — its primary consumer is `maya-mcp`, which submits generation jobs via HTTP and imports the resulting `.glb` files into Maya. `flame-mcp` and `fpt-mcp` do not connect to `vision3d`.

## License

MIT

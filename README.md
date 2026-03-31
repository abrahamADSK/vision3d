# Vision3D

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

- NVIDIA GPU with 16+ GB VRAM (tested on RTX 3090, 24 GB)
- Python 3.10+
- [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) installed in a venv with model weights downloaded
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

### 2. Install Hunyuan3D-2 (prerequisite)

Vision3D uses Hunyuan3D-2 for inference. Clone it inside the vision3d directory and create a shared venv:

```bash
python3 -m venv .venv
git clone https://github.com/Tencent/Hunyuan3D-2.git
.venv/bin/pip install -e Hunyuan3D-2/
```

### 3. Compile custom_rasterizer (required for texturing)

The paint pipeline depends on a C++ extension that must be compiled from source:

```bash
.venv/bin/pip install ./Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
```

### 4. Download model weights

**Hunyuan3D-2** (~10 GB — shape and texture generation):

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('tencent/Hunyuan3D-2', allow_patterns='hunyuan3d-dit-v2-0-turbo/*')
snapshot_download('tencent/Hunyuan3D-2', allow_patterns='hunyuan3d-paint-v2-0-turbo/*')
"
```

Move or symlink the downloaded weights into `hf_models/` inside the vision3d directory, or set `GPU_MODELS_DIR` to point at wherever they were downloaded.

**SDXL Turbo** (~6 GB — required for text-to-3D):

Downloaded automatically from HuggingFace on first text-to-3D use. No manual action needed. The model is cached in `~/.cache/huggingface/`.

### 5. Install Vision3D dependencies

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### 6. Run

```bash
.venv/bin/python server.py --host 0.0.0.0 --port 8000
```

Open `http://YOUR_GPU_HOST:8000` in a browser.

### 7. (Optional) Install as systemd service

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
| `GPU_API_KEY` | _(empty = open)_ | API key for authentication. Leave empty for LAN use. |
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
   │  NVIDIA GPU  │  CUDA inference
   └─────────────┘
```

## Models used

| Model | Source | Size | Function |
|-------|--------|------|----------|
| `hunyuan3d-dit-v2-0-turbo` | Tencent/Hunyuan3D-2 | ~400 MB | 3D shape generation |
| `hunyuan3d-paint-v2-0-turbo` | Tencent/Hunyuan3D-2 | ~14 GB | Texture painting |
| `hunyuan3d-delight-v2-0` | Tencent/Hunyuan3D-2 | ~4 GB | Relighting (paint dependency) |
| `sdxl-turbo` | Stability AI | ~6 GB | Text → image (intermediate step) |

## Integration with maya-mcp

Vision3D works standalone or as the GPU backend for [maya-mcp](https://github.com/abrahamADSK/maya-mcp). In that setup, maya-mcp calls Vision3D's API to generate 3D assets and imports them into Autodesk Maya.

## License

MIT

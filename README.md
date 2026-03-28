# Vision3D

AI-powered 3D model generation server. Turns images (or text prompts) into textured 3D meshes using GPU inference.

Built on [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) by Tencent, exposed as a REST API with a web UI, real-time progress streaming, and automatic polygon decimation.

## Features

- **Image-to-3D**: Upload a reference image, get a textured 3D model (`.glb`, `.obj` + baked texture)
- **Text-to-3D**: Describe an object in English, get a 3D mesh
- **Full pipeline**: Shape generation + decimation + texturing in one call
- **Polygon decimation**: Reduce dense meshes to a target face count (default 50k)
- **Real-time progress**: Server-Sent Events (SSE) stream for live feedback
- **Web UI**: Browser-based interface — no CLI or MCP client needed
- **REST API**: Integrate from any language, tool, or pipeline

## Requirements

- NVIDIA GPU with 16+ GB VRAM (tested on RTX 3090, 24 GB)
- Python 3.10+
- [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) installed in a venv with model weights downloaded

## Quick Start

### 1. Install Hunyuan3D-2 (prerequisite)

Vision3D uses Hunyuan3D-2 for inference. Install it first on your GPU machine:

```bash
git clone https://github.com/Tencent/Hunyuan3D-2.git
cd Hunyuan3D-2
python3 -m venv .venv
.venv/bin/pip install -e .
```

Download the model weights (~10 GB):

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('tencent/Hunyuan3D-2', allow_patterns='hunyuan3d-dit-v2-0-turbo/*')
snapshot_download('tencent/Hunyuan3D-2', allow_patterns='hunyuan3d-paint-v2-0-turbo/*')
"
```

Move or symlink the weights to your models directory (default: `~/ai-studio/vision/hf_models`), or set `GPU_MODELS_DIR` to where they were downloaded.

### 2. Clone Vision3D

```bash
git clone https://github.com/abrahamADSK/vision3d.git
cd vision3d
```

### 3. Install Vision3D dependencies

Into the Hunyuan3D-2 venv:

```bash
/path/to/Hunyuan3D-2/.venv/bin/pip install -r requirements.txt
```

For better polygon decimation (recommended):

```bash
/path/to/Hunyuan3D-2/.venv/bin/pip install pyfqmr
```

### 4. Run

```bash
/path/to/Hunyuan3D-2/.venv/bin/python server.py --host 0.0.0.0 --port 8000
```

Open `http://YOUR_GPU_HOST:8000` in a browser.

### 5. (Optional) Install as systemd service

```bash
sudo bash setup.sh
```

This creates a `vision3d.service` that starts on boot, generates an API key, and optionally configures Caddy for HTTPS.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Health check + GPU info |
| `POST` | `/api/generate-full` | Full pipeline: image → shape → decimate → texture |
| `POST` | `/api/generate-shape` | Image → mesh (shape only) |
| `POST` | `/api/generate-text` | Text prompt → mesh |
| `POST` | `/api/texture-mesh` | Mesh + image → textured mesh |
| `GET` | `/api/jobs/{id}` | Poll job status |
| `GET` | `/api/jobs/{id}/stream` | SSE real-time progress |
| `GET` | `/api/jobs/{id}/files/{name}` | Download result file |

### Example: Full pipeline with curl

```bash
curl -X POST http://localhost:8000/api/generate-full \
  -F "image=@photo.png" \
  -F "target_faces=50000" \
  -F "output_subdir=my_asset"
# Returns: {"job_id": "abc123", "status": "running", "poll": "/api/jobs/abc123", "stream": "/api/jobs/abc123/stream"}

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
   │  pipelines  │  Texture painting
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │  NVIDIA GPU  │  CUDA inference
   └─────────────┘
```

## Integration with maya-mcp

Vision3D is designed to work standalone or as the GPU backend for [maya-mcp](https://github.com/abrahamADSK/maya-mcp-project). In that setup, maya-mcp calls Vision3D's API to generate 3D assets and imports them into Autodesk Maya.

## License

MIT

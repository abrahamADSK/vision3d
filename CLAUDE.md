# Vision3D — Critical Context for Claude

> **Last updated**: 2026-04-01

---

## 1. Architecture

**Vision3D** is a FastAPI server running on GPU machine **glorfindel** (Rocky Linux), exposing a REST API for 3D model generation using **Hunyuan3D-2** (Tencent) and **SDXL Turbo** (Stability AI).

Four main endpoints:
- **image-to-3D** (`/api/generate-shape`): Image → 3D mesh
- **text-to-3D** (`/api/generate-text`): Text prompt → intermediate image (SDXL Turbo) → 3D mesh
- **texture painting** (`/api/texture-mesh`): Mesh + image → textured mesh
- **full-pipeline** (`/api/generate-full`): Image → shape + decimation + texture in one call

---

## 2. Execution Environment

### GPU Machine
- **Server**: `glorfindel` (GPU with CUDA, Rocky Linux)
- **User**: root (via systemd)
- **Working directory**: `/home/flame/ai-studio/vision3d/`

### Directory structure on glorfindel
```
~/ai-studio/vision3d/
├── .venv/                          # Virtual environment
├── server.py                        # FastAPI server
├── hy3dgen/                         # Local Hunyuan3D-2 code
├── Hunyuan3D-2/                     # Tencent/Hunyuan3D-2 git clone
└── hf_models/                       # Downloaded models
    ├── hunyuan3d-dit-v2-0-turbo/   # ~400 MB, ~1 min generation
    ├── hunyuan3d-dit-v2-0-fast/    # ~1 GB, ~2-3 min
    ├── hunyuan3d-dit-v2-0/         # ~3 GB, ~5 min (max quality)
    └── hunyuan3d-paint-v2-0-turbo/ # ~14 GB, texture model
```

### Systemd service
- **File**: `/etc/systemd/system/vision3d.service`
- **Restart**: `sudo systemctl daemon-reload && sudo systemctl restart vision3d`
- **Logs**: `sudo journalctl -u vision3d -f`

---

## 3. Pipelines and Models

### Shape Models

Geometry generation via `Hunyuan3DDiTFlowMatchingPipeline` from Tencent/Hunyuan3D-2.

| Model | Time | Steps | Use case |
|-------|------|-------|----------|
| `turbo` | ~1 min | 5-10 | Quick prototyping |
| `fast` | ~2-3 min | 15-20 | Iterations |
| `full` | ~5 min | 30-50 | Final renders |

**CRITICAL**: The shape pipeline ONLY accepts `image=`, NEVER `text=`:
```python
pipeline(image=pil_image, ...)  # Correct
pipeline(text="...", ...)        # Error: unsupported parameter
```

### Text-to-3D (3-phase pipeline)

`_run_shape_from_text()` implements:

**Phase 1/3 — Text → Image**:
1. Automatic prompt enhancement (white background, studio lighting, centered)
2. **SDXL Turbo** generates 512x512 image with 4 steps (~2s), upscaled to 1024x1024
3. SDXL Turbo unloaded from VRAM immediately (`_unload_t2i_pipeline()`)
4. `BackgroundRemover` (rembg) removes background

**Phase 2/3 — Shape Generation**:
5. Clean image → shape pipeline (image-to-3D)
6. Decimation to `target_faces`
7. Shape pipeline unloaded from VRAM (`_unload_shape_pipeline()`)

**Phase 3/3 — Texturing**:
8. Paint pipeline generates texture using reference image
9. Output: `textured.glb`, `mesh_uv.obj`, `texture_baked.png`, `mesh.glb`

### Text-to-image model
- **Pipeline**: `AutoPipelineForText2Image` (from `diffusers`) — **SDXL Turbo**
- **Model**: `stabilityai/sdxl-turbo` (~6 GB in fp16)
- **Auto-downloaded** on first use (via HuggingFace)
- **Dependencies**: `diffusers`, `transformers`, `accelerate`
- **VRAM management**: loaded on demand, fully unloaded after generation
- **Parameters**: 4 inference steps, guidance_scale=0.0, 512x512 + upscale to 1024x1024

### Paint model (Texture)
- **Model**: `hunyuan3d-paint-v2-0-turbo` (~14 GB)
- **Dependency**: requires `hunyuan3d-delight-v2-0` (~4 GB, relighting model — DO NOT delete)
- Used in both image-to-3D (full-pipeline) and text-to-3D

### VRAM Management (CRITICAL — RTX 3090, 24 GB)
GPU is shared. Models do NOT fit simultaneously:
- Shape turbo: ~10 GB | Paint turbo: ~14 GB | SDXL Turbo: ~6 GB
- **Mandatory sequence**: load → use → unload before loading next
- If shape and paint loaded together: silent OOM
- CUDA extensions (`custom_rasterizer`, `differentiable_renderer`) must be recompiled if PyTorch is updated

---

## 4. Quality Presets

`_resolve_preset()` in `server.py` maps `quality` → config:

| Preset | Model | Octree | Steps | Max faces | Use case |
|--------|-------|--------|-------|-----------|----------|
| `low` | turbo | 256 | 10 | 10k | Web preview |
| `medium` | turbo | 384 | 20 | 50k | Standard use |
| `high` | full | 384 | 30 | 150k | High quality |
| `ultra` | full | 512 | 50 | no limit | Production |

---

## 5. REST API

### Shape Generation
```
POST /api/generate-shape
Content-Type: multipart/form-data

Parameters:
  - image: PNG/JPG file
  - model: "turbo" | "fast" | "full" (default: "turbo")
  - quality: "low" | "medium" | "high" | "ultra"
  - steps: inference steps (overrides quality)
  - seed: seed for reproducibility (default: random)
```

### Text Generation
```
POST /api/generate-text
Content-Type: application/json

{
  "prompt": "describe the 3D object in detail",
  "model": "turbo" | "fast" | "full",
  "quality": "low" | "medium" | "high" | "ultra",
  "seed": int (optional)
}
```

### Texturing
```
POST /api/texture-mesh
Content-Type: multipart/form-data

Parameters:
  - mesh: .glb file (3D mesh)
  - image: PNG/JPG file (texture reference)
```

### Full Pipeline
```
POST /api/generate-full
Content-Type: multipart/form-data

Parameters:
  - image: PNG/JPG file
  - model: "turbo" | "fast" | "full"
  - quality: preset
  - texture_image: texture image (optional)
```

### Status and Download
```
GET /api/jobs/{job_id}
GET /api/jobs/{job_id}/files/{filename}
GET /api/jobs/{job_id}/stream  (SSE)
```

### Info
```
GET /api/health
GET /api/models
GET /api/presets
```

---

## 6. Operational Notes

- Server runs as **root** via systemd
- Source code edited from local Mac
- After `git pull` on glorfindel: `sudo systemctl daemon-reload && sudo systemctl restart vision3d`
- Web UI embedded in `server.py` (inline HTML at root endpoint `/`)
- Web UI has 2 tabs: "Image → 3D" and "Text → 3D"
- GLB results displayed in interactive 3D viewer (`<model-viewer>`, orbit controls)
- Use `.venv/bin/python -m pip` on glorfindel (pip's shebang may be broken)

---

## 7. Integration with Other Projects

```
vision3d (FastAPI on glorfindel)
    ↑
    │ HTTP REST (port 8000)
    ↓
maya-mcp (MCP server, Vision3D API client)
    ↑
    │ MCP protocol (stdio)
    ↓
Claude Code / Claude Desktop (local Mac)
```

**Locations**:
- `vision3d`: `/home/flame/ai-studio/vision3d/` (glorfindel)
- `maya-mcp`: `~/Claude_projects/maya-mcp-project/` (Mac)
- `fpt-mcp`: `~/Claude_projects/fpt-mcp/` (Mac)

---

## 8. Development and Deployment

### After git pull on glorfindel
```bash
ssh glorfindel
cd ~/ai-studio/vision3d && git pull && sudo systemctl restart vision3d && sudo journalctl -u vision3d -f -n 20
```

### Local development (on glorfindel)
```bash
cd ~/ai-studio/vision3d/
.venv/bin/python server.py --port 8000
```

### Code Structure (server.py)

**Main functions**:
- `_get_shape_pipeline(model)` — Load shape model
- `_get_paint_pipeline()` — Load paint model
- `_load_t2i_pipeline()` — Load SDXL Turbo
- `_unload_t2i_pipeline()` — Unload SDXL Turbo from VRAM
- `_unload_shape_pipeline()` — Unload shape model from VRAM
- `_run_shape_from_image()` — Process image-to-3D
- `_run_shape_from_text()` — Process text-to-3D (3 phases)
- `_run_texture()` — Apply texture to mesh
- `_run_full_pipeline()` — Shape + texture in one step
- `_decimate_mesh()` — Reduce polygon count

---

## 9. Key Points to Remember

1. **text-to-3D needs an intermediate image**: Never pass `text=` directly to the shape pipeline.
2. **Shape models are stateful**: Loading is expensive. Cached in `_shape_pipeline`.
3. **GPU has memory limits**: `ultra` preset may require 24 GB+ VRAM.
4. **Systemd manages the lifecycle**: Don't use `kill -9`, use `systemctl restart`.
5. **API key is optional**: Leave `GPU_API_KEY=""` for open access during development.
6. **Output in `GPU_WORK_DIR`**: Defaults to `./output`, organized by `job_id`.
7. **SDXL Turbo auto-downloads**: First text-to-3D run downloads ~6 GB.
8. **Text-to-3D dependencies**: `diffusers`, `transformers`, `accelerate` must be in the venv.

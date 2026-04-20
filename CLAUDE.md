# Vision3D — Critical Context for Claude

> **Last updated**: 2026-04-17 — v1.5.1+; Chat 44 concept-registry rollout
> **Audited against**: `server.py` (~2510 lines)
>
> **Concept registry active.** Before editing `server.py`, README, or
> install.sh, read `.concepts.yml` at the repo root. The pre-commit hook
> runs `scripts/verify_concepts.py` on every commit and blocks drift
> between code and the declared invariants (8 active, all passing against
> HEAD). Soft-launch mode (`strict: false`) for two weeks, then flip to
> strict.

---

## 1. Architecture

**Vision3D** is a FastAPI server for 3D model generation using **Hunyuan3D-2** (Tencent) and **SDXL Turbo** (Stability AI). Since v1.5.0 it can run in **two modes**:

- **LOCAL** — inference runs on this machine (Apple Silicon MPS, NVIDIA CUDA, or CPU). Original behaviour. Used by the dedicated GPU box (Rocky Linux + RTX 3090) and by Mac developers with M-series chips.
- **REMOTE** — the local server acts as a thin **HTTP façade**: every `/api/*` call is proxied via `httpx` async to another vision3d instance. Zero local job state, zero GPU load. SSE streams pass through byte-for-byte. This lets a Mac front-end delegate heavy CUDA work to a Linux GPU box while presenting the same web UI locally.

The mode is chosen at startup by an interactive prompt (`Run locally? [y/N]:` → `Remote host [<default>]:`) and persisted via env var `VISION3D_REMOTE_HOST` so uvicorn workers and `--reload` pick it up. CLI flags `--local` and `--remote HOST` skip the prompt. The prompt's default value is read from `VISION3D_DEFAULT_REMOTE_HOST` (a personal env var, never committed) — no hostname is hardcoded in the repo.

Eleven REST endpoints + embedded Web UI:
- **image-to-3D** (`POST /api/generate-shape`): Image → 3D mesh (`.glb`)
- **text-to-3D** (`POST /api/generate-text`): Text prompt → SDXL Turbo image → shape → decimate → textured mesh
- **texture painting** (`POST /api/texture-mesh`): Mesh + image → textured mesh
- **full-pipeline** (`POST /api/generate-full`): Image → shape + decimation + texture in one call
- **job polling** (`GET /api/jobs/{job_id}`): Job status, log, and download links
- **file download** (`GET /api/jobs/{job_id}/files/{filename}`): Download result files
- **SSE stream** (`GET /api/jobs/{job_id}/stream`): Real-time Server-Sent Events progress
- **health** (`GET /api/health`): GPU info, available models, text-to-3D availability
- **models** (`GET /api/models`): Available shape models with weights on disk
- **presets** (`GET /api/presets`): Quality preset configurations
- **web UI** (`GET /`): Embedded HTML UI with image-to-3D and text-to-3D tabs

---

## 2. Execution Environment

### GPU Machine
- **Server**: dedicated GPU host (CUDA, Rocky Linux). Hostname is operator-configured, never hardcoded in this repo.
- **User**: root (via systemd)
- **Working directory**: `/home/flame/ai-studio/vision3d/`

### Directory structure on the GPU host
```
~/ai-studio/vision3d/
├── .venv/                          # Virtual environment
├── server.py                        # FastAPI server (single file, ~1646 lines)
├── hy3dgen/                         # Local Hunyuan3D-2 code
├── Hunyuan3D-2/                     # Tencent/Hunyuan3D-2 git clone
└── hf_models/                       # Downloaded models (GPU_MODELS_DIR)
    ├── hunyuan3d-dit-v2-0-turbo/   # ~400 MB, ~1 min generation
    ├── hunyuan3d-dit-v2-0-fast/    # ~1 GB, ~2-3 min
    ├── hunyuan3d-dit-v2-0/         # ~3 GB, ~5 min (max quality)
    ├── hunyuan3d-paint-v2-0-turbo/ # ~14 GB, texture model
    └── hunyuan3d-delight-v2-0/     # ~4 GB, relighting model (DO NOT delete)
```

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `GPU_API_KEY` | `""` (empty = open access) | API key for authentication |
| `GPU_MODELS_DIR` | `./hf_models` (relative to script) | Model weights directory |
| `GPU_WORK_DIR` | `./output` (relative to script) | Job output directory |
| `GPU_VISION_DIR` | `.` (script directory) | Vision3D installation root |
| `VISION3D_REMOTE_HOST` | unset | If set, server runs in **remote proxy mode** (set automatically by `--remote` and by the interactive prompt) |
| `VISION3D_REMOTE_PORT` | `8000` | Port on the remote vision3d host |
| `VISION3D_REMOTE_KEY` | unset | API key forwarded to the remote (default: pass through inbound `x-api-key`) |

### Systemd service
- **File**: `/etc/systemd/system/vision3d.service`
- **Restart**: `sudo systemctl daemon-reload && sudo systemctl restart vision3d`
- **Logs**: `sudo journalctl -u vision3d -f`

### CLI Arguments
```bash
.venv/bin/python server.py --host 0.0.0.0 --port 8000   # defaults — interactive prompt
.venv/bin/python server.py --port 9000                    # custom port
.venv/bin/python server.py --reload                       # auto-reload (forces local; prompt incompatible)
.venv/bin/python server.py --local                        # skip prompt, force local backend
.venv/bin/python server.py --remote <gpu-host>            # skip prompt, proxy to a remote
```
- `--host`: bind address (default: `0.0.0.0`)
- `--port`: port (default: `8000`)
- `--local` / `--remote HOST`: bypass the interactive backend prompt (v1.5.0+)
- `--reload`: uvicorn auto-reload for development

---

## 3. Pipelines and Models

### Shape Models

Geometry generation via `Hunyuan3DDiTFlowMatchingPipeline` from Tencent/Hunyuan3D-2.

| User-facing name | Subfolder | Time | Steps | Use case |
|---|---|---|---|---|
| `turbo` | `hunyuan3d-dit-v2-0-turbo` | ~1 min | 5-10 | Quick prototyping |
| `fast` | `hunyuan3d-dit-v2-0-fast` | ~2-3 min | 15-20 | Iterations |
| `full` | `hunyuan3d-dit-v2-0` | ~5 min | 30-50 | Final renders |

Model availability is detected at runtime by checking for `model.fp16.safetensors` or `model.fp16.ckpt` in the subfolder.

**CRITICAL**: The shape pipeline ONLY accepts `image=`, NEVER `text=`:
```python
pipeline(image=pil_image, octree_resolution=384, num_inference_steps=30)  # Correct
pipeline(text="...", ...)   # Error: unsupported parameter
```

### Text-to-3D (3-phase pipeline)

`_run_shape_from_text()` implements a full pipeline (NOT just shape — also textures):

**Phase 1/3 — Text → Image** (steps 1-2/8):
1. Load **SDXL Turbo** (`_load_t2i_pipeline()`)
2. Automatic prompt enhancement: appends `"isolated object, centered, no floor, no ground, no shadow, white background, studio lighting, photorealistic, clean lines, product photography"`
3. Generate 512×512 image with 4 steps, guidance_scale=0.0
4. Upscale to 1024×1024 (PIL LANCZOS)
5. Unload SDXL Turbo from VRAM (`_unload_t2i_pipeline()`)
6. Save raw image as `text2img_reference.png`
7. `BackgroundRemover` (rembg) removes background if image has <5% transparent pixels
8. Save clean image as `text2img_clean.png` (not included in download list)

**Phase 2/3 — Shape Generation** (steps 3-5/8):
1. Load shape pipeline for requested model
2. Generate 3D mesh from clean image
3. Decimate if `target_faces > 0` and `orig_faces > target_faces`
4. Save `mesh.glb`
5. Unload shape pipeline from VRAM (`_unload_shape_pipeline()`)

**Phase 3/3 — Texturing** (steps 6-8/8):
1. Load paint pipeline
2. Paint texture using reference image
3. Save: `textured.glb`, `mesh_uv.obj`, `texture_baked.png` (if extractable), `mesh.glb`

### Text-to-image model
- **Pipeline**: `AutoPipelineForText2Image` (from `diffusers`) — **SDXL Turbo**
- **Model**: `stabilityai/sdxl-turbo` (~6 GB in fp16)
- **Auto-downloaded** on first use (via HuggingFace)
- **Dependencies**: `diffusers`, `transformers`, `accelerate`
- **VRAM management**: loaded on demand, fully unloaded after generation (moved to CPU then deleted, `gc.collect()` ×2, `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`)
- **Parameters**: 4 inference steps, guidance_scale=0.0, 512×512 + upscale to 1024×1024

### Paint model (Texture)
- **Model**: `hunyuan3d-paint-v2-0-turbo` (~14 GB)
- **Dependency**: requires `hunyuan3d-delight-v2-0` (~4 GB, relighting model — DO NOT delete)
- Used in: `generate-full`, `generate-text`, `texture-mesh`
- **Idle-timer unload** (v1.6.2+): cached across requests for latency; a background
  task (`_paint_idle_loop`) unloads it after `PAINT_IDLE_SECONDS` (default 900 s
  = 15 min) of inactivity, protected by `_gpu_semaphore` so it never races an
  in-flight job. The unload calls `.to("cpu")` + `del` + `gc.collect()` x2 +
  `_clear_device_cache()` (which now also runs `torch.cuda.ipc_collect()`
  so the VRAM actually returns to other processes, not just PyTorch's cache).
  Overridable via env: `VISION3D_PAINT_IDLE_SECONDS`,
  `VISION3D_PAINT_IDLE_CHECK_INTERVAL`. Clean shutdown (lifespan exit) also
  unloads before returning.

### VRAM Management (CRITICAL — RTX 3090, 24 GB)
GPU is shared. Models do NOT fit simultaneously:
- Shape turbo: ~10 GB | Paint turbo: ~14 GB | SDXL Turbo: ~6 GB
- **Mandatory sequence**: load → use → unload before loading next
- If shape and paint loaded together: silent OOM
- CUDA extensions (`custom_rasterizer`, `differentiable_renderer`) must be recompiled if PyTorch is updated

VRAM cleanup sequence in code: `del pipeline` → `gc.collect()` ×2 → `torch.cuda.empty_cache()` → `torch.cuda.synchronize()`.

---

## 4. Quality Presets

`_resolve_preset()` maps `preset` → config. Explicit parameter overrides (model, octree_resolution, num_inference_steps) take precedence over preset values.

| Preset | Model | Octree | Steps | Target faces | Label |
|--------|-------|--------|-------|-----------|----------|
| `low` | turbo | 256 | 10 | 10,000 | "Low — turbo, 10k faces, fast" |
| `medium` | turbo | 384 | 20 | 50,000 | "Medium — turbo, 50k faces" |
| `high` | full | 384 | 30 | 150,000 | "High — full model, 150k faces" |
| `ultra` | full | 512 | 50 | 0 (no limit) | "Ultra — full model, max detail" |

Default when no preset given: `target_faces=0, octree_resolution=384, num_inference_steps=30, model=turbo`.

---

## 5. REST API

### Authentication
All endpoints accept `x_api_key` via HTTP header (`X-API-Key`). If `GPU_API_KEY` env is empty, all requests are allowed (open access). Verification uses `secrets.compare_digest` (timing-safe).

Returns `HTTP 401` with `{"detail": "Invalid or missing API key"}` on failure.

### POST /api/generate-shape
Image → 3D mesh (shape only, no texture). Async job.

**Content-Type**: `multipart/form-data`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | File (required) | — | PNG/JPG input image |
| `output_subdir` | Form string | `"0"` | Subdirectory under WORK_DIR |
| `target_faces` | Form int | `0` | Decimation target (0 = no decimation) |
| `preset` | Form string | `""` | Quality preset: low/medium/high/ultra |
| `model` | Form string | `""` | Shape model: turbo/fast/full |
| `octree_resolution` | Form int | `0` | Octree resolution (0 = use preset/default 384) |
| `num_inference_steps` | Form int | `0` | Steps (0 = use preset/default 30) |

**Response** (200):
```json
{"job_id": "abc12345", "status": "running", "poll": "/api/jobs/abc12345"}
```

**Output files**: `mesh.glb`

### POST /api/generate-text
Text → image → shape → decimate → texture. Full pipeline. Async job.

**Content-Type**: `multipart/form-data` (NOT JSON)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_prompt` | Form string (required) | — | Text description of the 3D object |
| `output_subdir` | Form string | `"0"` | Subdirectory under WORK_DIR |
| `target_faces` | Form int | `0` | Decimation target (0 = no decimation) |
| `preset` | Form string | `""` | Quality preset |
| `model` | Form string | `""` | Shape model |
| `octree_resolution` | Form int | `0` | Octree resolution |
| `num_inference_steps` | Form int | `0` | Inference steps |

**Response** (200):
```json
{"job_id": "abc12345", "status": "running", "poll": "/api/jobs/abc12345"}
```

**Output files**: `text2img_reference.png`, `textured.glb`, `mesh_uv.obj`, `texture_baked.png` (conditional), `mesh.glb`

### POST /api/texture-mesh
Apply texture to an existing mesh. Async job.

**Content-Type**: `multipart/form-data`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mesh` | File (required) | — | `.glb` 3D mesh file |
| `image` | File (required) | — | PNG/JPG texture reference image |
| `output_subdir` | Form string | `"0"` | Subdirectory under WORK_DIR |

**Response** (200):
```json
{"job_id": "abc12345", "status": "running", "poll": "/api/jobs/abc12345"}
```

**Output files**: `textured.glb`, `mesh_uv.obj`, `texture_baked.png` (conditional)

### POST /api/generate-full
Image → shape → decimate → texture. Full pipeline. Async job.

**Content-Type**: `multipart/form-data`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | File (required) | — | PNG/JPG input image |
| `output_subdir` | Form string | `"0"` | Subdirectory under WORK_DIR |
| `target_faces` | Form int | `50000` | Decimation target (NOTE: default is 50k, not 0) |
| `preset` | Form string | `""` | Quality preset |
| `model` | Form string | `""` | Shape model |
| `octree_resolution` | Form int | `0` | Octree resolution |
| `num_inference_steps` | Form int | `0` | Inference steps |

**Response** (200):
```json
{
  "job_id": "abc12345",
  "status": "running",
  "quality": "medium",
  "target_faces": 50000,
  "poll": "/api/jobs/abc12345",
  "stream": "/api/jobs/abc12345/stream"
}
```

**Output files**: `textured.glb`, `mesh_uv.obj`, `texture_baked.png` (conditional), `mesh.glb`

### GET /api/jobs/{job_id}
Poll job status.

**Response** (200, running):
```json
{"id": "abc12345", "type": "full-pipeline", "status": "running", "elapsed_s": 42.3, "log": ["[1/6] Loading..."]}
```

**Response** (200, completed):
```json
{
  "id": "abc12345", "type": "full-pipeline", "status": "completed", "elapsed_s": 180.5,
  "log": ["..."],
  "files": [{"name": "textured.glb", "download": "/api/jobs/abc12345/files/textured.glb"}]
}
```

**Response** (200, failed):
```json
{"id": "abc12345", "type": "full-pipeline", "status": "failed", "elapsed_s": 10.2, "log": ["..."], "error": "traceback..."}
```

**HTTP errors**: `404` (job not found)

### GET /api/jobs/{job_id}/files/{filename}
Download a result file.

**HTTP errors**: `404` (job not found), `409` (job not yet completed), `404` (file not in job files list), `404` (file missing on disk)

### GET /api/jobs/{job_id}/stream
Server-Sent Events (SSE) for real-time progress.

**Event types**:
| Event | Data | When |
|---|---|---|
| `log` | Progress text line | Each new log entry |
| `status` | `"running"` / `"completed"` / `"failed"` | Every poll cycle (2s) |
| `done` | JSON: `{"status":"completed","elapsed_s":N,"files":[...]}` | Job finished |
| `done` | JSON: `{"status":"failed","elapsed_s":N,"error":"..."}` | Job failed |
| `error` | `"Job disappeared"` | Job removed from memory |

**Poll interval**: 2 seconds.

**Connection**: `text/event-stream` with headers `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`.

### GET /api/health
**Response** (200):
```json
{
  "status": "ok",
  "api_key_required": false,
  "gpu": "NVIDIA GeForce RTX 3090",
  "vram_gb": 24.0,
  "models": ["turbo", "fast", "full"],
  "text_to_3d": "available (SDXL Turbo)"
}
```

### GET /api/models
**Response** (200):
```json
{
  "models": ["turbo", "fast"],
  "default": "turbo",
  "all": {"turbo": "hunyuan3d-dit-v2-0-turbo", "fast": "hunyuan3d-dit-v2-0-fast", "full": "hunyuan3d-dit-v2-0"}
}
```
`models` lists only those with weights on disk. `all` shows the full map.

### GET /api/presets
**Response** (200): Returns the full `QUALITY_PRESETS` dictionary.

---

## 6. Job System

### Job lifecycle
- Jobs are stored **in-memory** (`_jobs` dict). All jobs are **lost on server restart**.
- Job ID: first 8 characters of a UUID4 (e.g., `"a1b2c3d4"`).
- States: `running` → `completed` | `failed`
- Each job tracks: `id`, `type`, `status`, `detail`, `created` (timestamp), `output_dir`, `files`, `error`, `log` (list of progress strings).

### Job types
| Type | Created by | Pipeline |
|---|---|---|
| `shape-image` | `/api/generate-shape` | Image → mesh |
| `shape-text` | `/api/generate-text` | Text → image → mesh → texture |
| `texture` | `/api/texture-mesh` | Mesh + image → textured mesh |
| `full-pipeline` | `/api/generate-full` | Image → mesh → decimate → texture |

### Background execution
Jobs run via `asyncio.create_task()` wrapping `loop.run_in_executor(None, func, *args)` — the blocking inference runs in the default thread pool executor.

### Output directory structure
Each job writes to `WORK_DIR / output_subdir`:
```
output/
└── web_0/                  # output_subdir value
    ├── input.png           # uploaded image (generate-shape, generate-full)
    ├── mesh.glb            # raw shape mesh
    ├── textured.glb        # textured mesh (full/text pipelines)
    ├── mesh_uv.obj         # UV-mapped mesh (full/text/texture pipelines)
    ├── texture_baked.png   # extracted texture (conditional — may fail)
    ├── text2img_reference.png  # SDXL-generated image (text pipeline only)
    └── text2img_clean.png  # background-removed image (text pipeline only, not in download list)
```

**NOTE**: `texture_baked.png` extraction uses two fallback methods: first tries `textured.visual.material.image`, then `textured.visual.to_texture().image`. If both fail, the file is silently omitted from the files list.

---

## 7. Decimation System

### Adaptive curvature-aware decimation
`_decimate_mesh()` → `_adaptive_decimate()` implements smart polygon reduction:

1. **Curvature analysis** (`_compute_vertex_curvature`): computes per-vertex curvature via face normal variance. Classifies vertices as high-detail (curvature > 0.3).
2. **Face classification**: faces where ANY vertex has curvature > 0.25 are "protected".
3. **If protected faces ≥ target**: falls back to gentle uniform decimation (aggressiveness=3).
4. **Otherwise**: uses `pyfqmr` with aggressiveness=4, preserve_border=True, max_iterations=100.
5. **Post-decimation check**: re-computes curvature to verify detail preservation.

### Fallback chain
1. `pyfqmr` adaptive (curvature-aware) → 2. `pyfqmr` uniform → 3. `trimesh.simplify_quadric_decimation` → 4. Return original mesh.

---

## 8. Operational Notes

- Server runs as **root** via systemd on the GPU host.
- Source code edited from local Mac, pushed via git.
- After `git pull` on the GPU host: `sudo systemctl daemon-reload && sudo systemctl restart vision3d`
- Web UI embedded in `server.py` (inline HTML, ~380 lines at `_WEB_UI_HTML`).
- Web UI has 2 tabs: "Image → 3D" and "Text → 3D".
- Web UI supports the API key via URL query param `?key=<value>` (the key value comes from the `GPU_API_KEY` env var on the server).
- GLB results displayed in interactive 3D viewer (`<model-viewer>` v3.5.0, orbit controls).
- Use `.venv/bin/python -m pip` on the GPU host (pip's shebang may be broken).

### Background removal
Automatic background removal is triggered when the input image has **less than 5% transparent pixels** (alpha channel < 10). Uses `hy3dgen.rembg.BackgroundRemover`. If rembg is not installed, the step is silently skipped.

---

## 9. Integration with Other Projects

```
vision3d (FastAPI on the GPU host)
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
- `vision3d`: `/home/<user>/ai-studio/vision3d/` (GPU host)
- `maya-mcp`: `~/Claude_projects/maya-mcp/` (Mac)
- `fpt-mcp`: `~/Claude_projects/fpt-mcp/` (Mac)

---

## 10. Development and Deployment

### After git pull on the GPU host
```bash
ssh <gpu-host>
cd ~/ai-studio/vision3d && git pull && sudo systemctl restart vision3d && sudo journalctl -u vision3d -f -n 20
```

### Local development (on the GPU host)
```bash
cd ~/ai-studio/vision3d/
.venv/bin/python server.py --port 8000
.venv/bin/python server.py --port 8000 --reload   # auto-reload mode
```

### Code Structure (server.py)

**Configuration** (lines 1-60): env vars, paths, constants.

**Job tracking** (lines 62-99): `_new_job`, `_job_log`, `_job_done`, `_job_fail` — in-memory dict.

**Authentication** (lines 101-109): `_verify_api_key` with `secrets.compare_digest`.

**Pipeline loaders** (lines 112-260):
- `_get_shape_pipeline(model)` — Load/swap shape model (cached, model-aware)
- `_get_paint_pipeline()` — Load paint model (cached, never unloaded)
- `_load_t2i_pipeline()` — Load SDXL Turbo to GPU
- `_unload_t2i_pipeline()` — Unload SDXL Turbo (cpu → del → gc → empty_cache)
- `_unload_shape_pipeline()` — Unload shape model (del → gc → empty_cache)

**Inference functions** (lines 263-905):
- `_run_shape_from_image()` — Image-to-3D (5 steps: load → image → shape → decimate → save)
- `_run_shape_from_text()` — Text-to-3D (8 steps: SDXL → rembg → shape → decimate → texture → save)
- `_run_texture()` — Texture painting (4 steps: load → mesh → paint → save)
- `_run_full_pipeline()` — Full pipeline (6 steps: shape → rembg → generate → decimate → paint → save)

**Decimation** (lines 550-729):
- `_compute_vertex_curvature()` — Per-vertex curvature via face normal variance
- `_adaptive_decimate()` — Curvature-aware decimation with pyfqmr
- `_uniform_decimate()` — Simple decimation fallback
- `_decimate_mesh()` — Entry point (delegates to adaptive)

**Quality presets** (lines 493-547): `QUALITY_PRESETS` dict and `_resolve_preset()`.

**FastAPI endpoints** (lines 924-1233): All REST endpoints.

**Web UI** (lines 1236-1626): Embedded HTML/CSS/JS.

**Entry point** (lines 1629-1646): argparse + uvicorn.run.

---

## 11. Key Points to Remember

1. **text-to-3D needs an intermediate image**: Never pass `text=` directly to the shape pipeline.
2. **text-to-3D produces textured output**: Unlike generate-shape, generate-text runs the full 3-phase pipeline including texturing.
3. **Shape models are stateful**: Loading is expensive. Cached in `_shape_pipeline`. Swapping models requires unloading the current one.
4. **GPU has memory limits**: `ultra` preset may require 24 GB+ VRAM.
5. **Systemd manages the lifecycle**: Don't use `kill -9`, use `systemctl restart`.
6. **API key is optional**: Leave `GPU_API_KEY=""` for open access during development.
7. **Output in `GPU_WORK_DIR`**: Defaults to `./output`, organized by `output_subdir`.
8. **SDXL Turbo auto-downloads**: First text-to-3D run downloads ~6 GB from HuggingFace.
9. **Text-to-3D dependencies**: `diffusers`, `transformers`, `accelerate` must be in the venv.
10. **Jobs are in-memory**: All job state is lost on server restart.
11. **GPU concurrency protected**: `asyncio.Semaphore(1)` prevents concurrent GPU jobs (HTTP 429 if busy).
12. **generate-text uses Form data**: NOT JSON — all params are multipart/form-data.
13. **generate-full default is 50k faces**: Unlike generate-shape (default 0), generate-full defaults to 50,000 target_faces.

---

## 12. Known Issues and Potential Bugs

1. ~~**`_resolve_preset` target_faces override bug**~~: **FIXED (Phase 8.1)** — Changed condition to `if target_faces > 0:` so explicit target_faces overrides preset value. Preset default is kept only when target_faces is 0 or not provided.

2. ~~**No job cleanup / memory leak**~~: **FIXED (Phase 8.2)** — Added `_cleanup_old_jobs()` background task that runs every 5 minutes (`JOB_CLEANUP_INTERVAL=300`), removing completed/failed jobs older than 1 hour (`JOB_TTL_SECONDS=3600`). Added `_check_max_jobs()` that rejects new jobs with HTTP 503 if `len(_jobs) > 100` (`MAX_JOBS=100`). Both constants are configurable.

3. ~~**No concurrent job protection**~~: **FIXED (Phase 8.1)** — Added `asyncio.Semaphore(1)` (`_gpu_semaphore`). All POST endpoints call `_check_gpu_available()` upfront; returns HTTP 429 with `Retry-After: 30` if GPU is busy. The semaphore is held for the full duration of `_run_in_background()`.

4. ~~**SSE auth bypass from Web UI**~~: **FIXED (Phase 8.2)** — `_verify_api_key()` now accepts an optional `query_api_key` fallback parameter. The `stream_job()` endpoint reads `x_api_key` from both Header and Query param (`Query(None, alias="x_api_key")`). Header takes priority; query param is used as fallback for EventSource connections. The Web UI already sends `?x_api_key=` in the URL.

5. ~~**No input validation**~~: **Mostly FIXED** — ~~No sanitization of `output_subdir`~~ **output_subdir sanitized (Phase 8.1)** — `_validate_output_subdir()` rejects `..`, `/`, `\` and verifies `Path.resolve()` stays inside WORK_DIR. **MIME type + size validation added (Phase 8.2)** — `_validate_upload()` checks Content-Type against allowed lists (`ALLOWED_IMAGE_TYPES`: png/jpeg/webp; `ALLOWED_MESH_TYPES`: gltf-binary/octet-stream) and enforces 50 MB max file size. Applied to `generate-shape`, `texture-mesh`, `generate-full`. Returns HTTP 400 on violation.

6. ~~**output_subdir collision**~~: **FIXED (Phase 8.3)** — Added `_resolve_output_subdir()` helper that replaces the default `"0"` with a unique 8-char UUID4 prefix (same format as `job_id`). If the user passes an explicit `output_subdir` (anything other than `"0"`), it is preserved. Applied in all 4 POST endpoints, after `_validate_output_subdir()`.

7. ~~**texture_baked.png is conditional**~~: **FIXED (Phase 8.3)** — Added `_job_log()` warning when texture extraction fails: `"⚠ texture_baked.png extraction failed — file not included"`. Applied in all 3 functions that attempt texture extraction (`_run_shape_from_text`, `_run_texture`, `_run_full_pipeline`). The job still completes successfully (no behavior change), but the client now sees the warning in the job log and SSE stream.

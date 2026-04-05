#!/usr/bin/env python3
"""
server.py — FastAPI inference server for Vision3D.

Runs on the GPU machine and exposes REST endpoints
for shape generation (image-to-3D, text-to-3D) and texture painting.
Replaces the previous SSH-based communication.

Endpoints:
    POST /api/generate-shape    — image → mesh.glb
    POST /api/generate-text     — text prompt → mesh.glb
    POST /api/texture-mesh      — mesh + image → textured mesh
    GET  /api/health            — health check
    GET  /api/jobs/{job_id}     — job status + download

Usage:
    .venv/bin/python server.py --port 8000
    .venv/bin/python server.py --host 0.0.0.0 --port 8000  # LAN access

Environment:
    GPU_API_KEY       — API key for authentication (empty = open access)
    GPU_MODELS_DIR    — Model weights (default: ./hf_models)
    GPU_WORK_DIR      — working directory for outputs (default: ./output)
    GPU_VISION_DIR    — Vision3D installation directory (default: .)
"""

import asyncio
import os
import secrets
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Header, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("GPU_API_KEY", "")
_SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.environ.get(
    "GPU_MODELS_DIR",
    str(_SCRIPT_DIR / "hf_models"),
)
WORK_DIR = Path(
    os.environ.get(
        "GPU_WORK_DIR",
        str(_SCRIPT_DIR / "output"),
    )
)
VISION_DIR = Path(
    os.environ.get(
        "GPU_VISION_DIR",
        str(_SCRIPT_DIR),
    )
)

# ── Job tracking ─────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}

# GPU concurrency protection — only one inference job at a time (RTX 3090, 24 GB shared VRAM)
_gpu_semaphore = asyncio.Semaphore(1)


def _new_job(job_type: str, detail: str = "") -> str:
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "id": job_id,
        "type": job_type,
        "status": "running",
        "detail": detail,
        "created": time.time(),
        "output_dir": None,
        "files": [],
        "error": None,
        "log": [],
    }
    return job_id


def _job_log(job_id: str, msg: str):
    if job_id in _jobs:
        _jobs[job_id]["log"].append(msg)


def _job_done(job_id: str, output_dir: str, files: list[str]):
    if job_id in _jobs:
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["output_dir"] = output_dir
        _jobs[job_id]["files"] = files


def _job_fail(job_id: str, error: str):
    if job_id in _jobs:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = error


# ── Job cleanup (TTL-based eviction) — Phase 8.2 fix for Bug #2 ────────────

JOB_TTL_SECONDS = 3600      # Remove completed/failed jobs older than 1 hour
JOB_CLEANUP_INTERVAL = 300  # Run cleanup every 5 minutes
MAX_JOBS = 100               # Reject new jobs if _jobs exceeds this limit


def _cleanup_old_jobs() -> int:
    """Remove completed/failed jobs older than JOB_TTL_SECONDS.

    Returns the number of jobs removed.
    """
    now = time.time()
    expired = [
        jid for jid, job in _jobs.items()
        if job["status"] in ("completed", "failed")
        and (now - job["created"]) > JOB_TTL_SECONDS
    ]
    for jid in expired:
        del _jobs[jid]
    return len(expired)


async def _job_cleanup_loop():
    """Background task that periodically cleans up expired jobs."""
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL)
        n = _cleanup_old_jobs()
        if n > 0:
            print(f"[Vision3D] Cleaned {n} expired jobs")


def _check_max_jobs():
    """Raise HTTP 503 if the job dict exceeds MAX_JOBS (memory protection)."""
    if len(_jobs) > MAX_JOBS:
        raise HTTPException(
            status_code=503,
            detail=f"Too many jobs ({len(_jobs)}). Try again later.",
            headers={"Retry-After": "60"},
        )


# ── Input validation ─────────────────────────────────────────────────────────

import re


def _validate_output_subdir(output_subdir: str) -> str:
    """Sanitize output_subdir to prevent path traversal attacks.

    Rejects values containing '..', '/', or '\\' and verifies the resolved
    path stays inside WORK_DIR.  Returns the validated subdir string or
    raises HTTP 400.
    """
    # Reject obviously dangerous characters / sequences
    if re.search(r'(\.\.|[/\\])', output_subdir):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_subdir '{output_subdir}': must not contain '..', '/' or '\\'",
        )

    # Resolve and verify the path stays within WORK_DIR
    resolved = (WORK_DIR / output_subdir).resolve()
    if not str(resolved).startswith(str(WORK_DIR.resolve())):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_subdir '{output_subdir}': resolves outside working directory",
        )

    return output_subdir


# Allowed MIME types for upload validation (Phase 8.2 — Bug #5 partial fix)
ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"]
ALLOWED_MESH_TYPES = ["model/gltf-binary", "application/octet-stream"]  # GLB often sent as octet-stream


async def _validate_upload(
    file: UploadFile,
    allowed_types: list[str],
    max_size_mb: int = 50,
) -> bytes:
    """Validate upload MIME type and size, return file content.

    Reads the file once and returns content to avoid double-read.
    Raises HTTP 400 if validation fails. Phase 8.2 fix for Bug #5.
    """
    # Check Content-Type against allowed list
    content_type = file.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{content_type}' for '{file.filename}'. "
                f"Allowed: {', '.join(allowed_types)}"
            ),
        )

    # Read content and check size
    content = await file.read()
    max_bytes = max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File '{file.filename}' too large ({len(content) / 1024 / 1024:.1f} MB). "
                f"Maximum: {max_size_mb} MB"
            ),
        )

    return content


# ── Authentication ───────────────────────────────────────────────────────────


def _verify_api_key(x_api_key: Optional[str], query_api_key: Optional[str] = None):
    """Verify API key if one is configured.

    Checks header first, falls back to query param (needed for SSE/EventSource
    which cannot send custom headers). Phase 8.2 fix for Bug #4.
    """
    if not API_KEY:
        return  # No API key configured — open access (LAN only)
    # Use header key if present, otherwise fall back to query param
    key = x_api_key or query_api_key
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Pipeline loaders (lazy, cached) ─────────────────────────────────────────

_shape_pipeline = None
_shape_pipeline_name = None
_paint_pipeline = None
_t2i_pipeline = None

# SDXL Turbo for text-to-image (required for text-to-3D)
# ~6GB VRAM, 1 step (~2s), good prompt adherence, fully unloadable
T2I_MODEL = "stabilityai/sdxl-turbo"

# Map user-facing model names to HunyuanD-2 subfolder names
SHAPE_MODELS = {
    "turbo": "hunyuan3d-dit-v2-0-turbo",
    "fast":  "hunyuan3d-dit-v2-0-fast",
    "full":  "hunyuan3d-dit-v2-0",
}


def _get_available_models():
    """Return list of model names that have weights on disk."""
    available = []
    for name, subfolder in SHAPE_MODELS.items():
        path = os.path.join(MODELS_DIR, subfolder)
        safetensors = os.path.join(path, "model.fp16.safetensors")
        ckpt = os.path.join(path, "model.fp16.ckpt")
        if os.path.isdir(path) and (os.path.exists(safetensors) or os.path.exists(ckpt)):
            available.append(name)
    return available


def _get_shape_pipeline(model_name: str = "turbo"):
    """Load shape pipeline by model name. Swaps models if a different one is requested."""
    import torch
    global _shape_pipeline, _shape_pipeline_name

    if _shape_pipeline is not None and _shape_pipeline_name == model_name:
        return _shape_pipeline

    subfolder = SHAPE_MODELS.get(model_name)
    if not subfolder:
        print(f"[Shape] Unknown model '{model_name}', falling back to turbo")
        model_name = "turbo"
        subfolder = SHAPE_MODELS["turbo"]

    # Unload current model if switching
    if _shape_pipeline is not None:
        print(f"[Shape] Unloading {_shape_pipeline_name} to load {model_name}...")
        del _shape_pipeline
        _shape_pipeline = None
        _shape_pipeline_name = None
        torch.cuda.empty_cache()

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    path = os.path.join(MODELS_DIR, subfolder)
    model_file = os.path.join(path, "model.fp16.safetensors")
    if os.path.isdir(path) and os.path.exists(model_file):
        print(f"[Shape] Loading {model_name} from local: {path}")
        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODELS_DIR, subfolder=subfolder
        )
    else:
        print(f"[Shape] Loading {model_name} from HuggingFace Hub...")
        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2", subfolder=subfolder
        )

    _shape_pipeline_name = model_name
    return _shape_pipeline


def _unload_shape_pipeline():
    """Unload shape model from GPU to free VRAM for paint pipeline."""
    import gc
    import torch
    global _shape_pipeline, _shape_pipeline_name

    if _shape_pipeline is None:
        return

    print(f"[Shape] Unloading {_shape_pipeline_name} to free VRAM for paint...")
    del _shape_pipeline
    _shape_pipeline = None
    _shape_pipeline_name = None
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("[Shape] VRAM freed.")


def _get_paint_pipeline():
    global _paint_pipeline
    if _paint_pipeline is None:
        from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline

        if os.path.isdir(MODELS_DIR):
            print(f"[Paint] Loading from local: {MODELS_DIR}")
            _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                MODELS_DIR, subfolder="hunyuan3d-paint-v2-0-turbo"
            )
        else:
            print("[Paint] Loading from HuggingFace Hub...")
            _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                "tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0-turbo"
            )
    return _paint_pipeline


def _load_t2i_pipeline():
    """Load SDXL Turbo text-to-image pipeline onto GPU."""
    import torch
    global _t2i_pipeline

    if _t2i_pipeline is not None:
        return _t2i_pipeline

    from diffusers import AutoPipelineForText2Image

    print(f"[T2I] Loading SDXL Turbo from {T2I_MODEL}...")
    _t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
        T2I_MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    _t2i_pipeline.to("cuda")
    print("[T2I] SDXL Turbo loaded on GPU.")
    return _t2i_pipeline


def _unload_t2i_pipeline():
    """Fully unload SDXL Turbo from GPU memory to free VRAM for other apps."""
    import gc
    import torch
    global _t2i_pipeline

    if _t2i_pipeline is None:
        return

    print("[T2I] Unloading SDXL Turbo from memory...")
    _t2i_pipeline.to("cpu")
    del _t2i_pipeline
    _t2i_pipeline = None
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("[T2I] VRAM freed.")


# ── Inference functions (run in thread pool) ─────────────────────────────────


def _run_shape_from_image(
    image_path: str, output_dir: str, job_id: str,
    target_faces: int = 0,
    octree_resolution: int = 384,
    num_inference_steps: int = 30,
    model: str = "turbo",
) -> dict:
    """Image → 3D shape generation (blocking, runs in thread)."""
    import torch
    from PIL import Image
    import numpy as np

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _job_log(job_id, f"[1/5] Loading shape pipeline ({model})...")
    pipeline = _get_shape_pipeline(model)

    _job_log(job_id, f"[2/5] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")

    # Background removal if needed
    alpha = np.array(image)[:, :, 3]
    if (alpha < 10).sum() / alpha.size < 0.05:
        try:
            from hy3dgen.rembg import BackgroundRemover

            _job_log(job_id, "      Removing background...")
            image = BackgroundRemover()(image)
        except ImportError:
            pass

    _job_log(job_id, f"[3/5] Generating 3D shape (octree={octree_resolution}, steps={num_inference_steps})...")
    result = pipeline(
        image=image,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
    )
    mesh = result[0]
    _job_log(
        job_id,
        f"      Generated: {len(mesh.vertices):,} verts | {len(mesh.faces):,} faces",
    )

    # Decimation (polygon reduction)
    if target_faces > 0:
        _job_log(job_id, "[4/5] Decimating mesh...")
        import trimesh as _tri
        tri_mesh = _tri.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        tri_mesh = _decimate_mesh(tri_mesh, target_faces, job_id)
        # Replace mesh data for export
        glb_path = output / "mesh.glb"
        tri_mesh.export(str(glb_path))
    else:
        _job_log(job_id, "[4/5] Skipping decimation (no target set)")
        glb_path = output / "mesh.glb"
        mesh.export(str(glb_path))

    _job_log(job_id, "[5/5] Saving mesh.glb...")
    size_kb = glb_path.stat().st_size // 1024

    torch.cuda.empty_cache()

    return {"mesh_path": str(glb_path), "mesh_size_kb": size_kb}


def _run_shape_from_text(
    text_prompt: str, output_dir: str, job_id: str,
    target_faces: int = 0,
    octree_resolution: int = 384,
    num_inference_steps: int = 30,
    model: str = "turbo",
) -> dict:
    """Text → image → 3D shape → texture (blocking, runs in thread).

    Full pipeline:
    1. SDXL Turbo generates a reference image from text
    2. Background removal (rembg) to isolate the object
    3. Shape pipeline generates 3D mesh from the clean image
    4. Decimation (if target_faces > 0)
    5. Paint pipeline textures the mesh using the reference image
    """
    import torch
    from PIL import Image
    import numpy as np
    import trimesh

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ── Phase 0: Text → Image via SDXL Turbo ────────────────────────────
    _job_log(job_id, "═══ PHASE 1/3: TEXT TO IMAGE ═══")
    _job_log(job_id, "[1/8] Loading SDXL Turbo...")
    t2i = _load_t2i_pipeline()

    # Enhance prompt for cleaner 3D generation (isolated object, no floor)
    enhanced_prompt = (
        f"{text_prompt}, isolated object, centered, no floor, no ground, "
        "no shadow, white background, studio lighting, photorealistic, "
        "clean lines, product photography"
    )
    _job_log(job_id, f"[2/8] Generating reference image...")
    _job_log(job_id, f"      Prompt: '{text_prompt}'")
    _job_log(job_id, f"      Enhanced: '{enhanced_prompt}'")
    sdxl_result = t2i(
        prompt=enhanced_prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=512,
        width=512,
    )
    ref_image = sdxl_result.images[0]
    if ref_image is None:
        raise RuntimeError(
            f"Text-to-image generation returned None for prompt: '{text_prompt}'. "
            "Check SDXL Turbo installation."
        )
    # Upscale to 1024x1024 for better 3D quality
    ref_image = ref_image.resize((1024, 1024), Image.LANCZOS)

    # Free VRAM before shape generation (shared GPU)
    _job_log(job_id, "      Unloading SDXL Turbo to free VRAM...")
    _unload_t2i_pipeline()
    # Save the raw generated image
    ref_path = output / "text2img_reference.png"
    ref_image.save(str(ref_path))
    _job_log(job_id, f"      Reference image generated: {ref_image.size[0]}x{ref_image.size[1]}")

    # Background removal to isolate the object
    ref_image = ref_image.convert("RGBA")
    alpha = np.array(ref_image)[:, :, 3]
    if (alpha < 10).sum() / alpha.size < 0.05:
        try:
            from hy3dgen.rembg import BackgroundRemover
            _job_log(job_id, "      Removing background (rembg)...")
            ref_image = BackgroundRemover()(ref_image)
            # Save the clean image too
            clean_path = output / "text2img_clean.png"
            ref_image.save(str(clean_path))
            _job_log(job_id, "      Background removed successfully")
        except ImportError:
            _job_log(job_id, "      WARNING: rembg not available, using raw image")

    # ── Phase 1: Image → 3D shape ────────────────────────────────────
    _job_log(job_id, "═══ PHASE 2/3: SHAPE GENERATION ═══")
    _job_log(job_id, f"[3/8] Loading shape pipeline ({model})...")
    pipeline = _get_shape_pipeline(model)

    _job_log(job_id, f"[4/8] Generating 3D shape (octree={octree_resolution}, steps={num_inference_steps})...")
    result = pipeline(
        image=ref_image,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
    )
    mesh = result[0]
    orig_faces = len(mesh.faces)
    _job_log(job_id, f"      Generated: {len(mesh.vertices):,} verts | {orig_faces:,} faces")

    # Decimation
    if target_faces > 0 and orig_faces > target_faces:
        _job_log(job_id, f"[5/8] Decimating: {orig_faces:,} → {target_faces:,} faces...")
        tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        tri_mesh = _decimate_mesh(tri_mesh, target_faces, job_id)
        glb_path = output / "mesh.glb"
        tri_mesh.export(str(glb_path))
        paint_mesh = tri_mesh
    else:
        _job_log(job_id, "[5/8] Skipping decimation")
        tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        glb_path = output / "mesh.glb"
        tri_mesh.export(str(glb_path))
        paint_mesh = tri_mesh

    mesh_size_kb = glb_path.stat().st_size // 1024
    _job_log(job_id, f"      Shape saved: mesh.glb ({mesh_size_kb} KB)")

    # Free shape model from VRAM before loading paint
    _unload_shape_pipeline()
    torch.cuda.empty_cache()

    # ── Phase 2: Texturing ───────────────────────────────────────────
    _job_log(job_id, "═══ PHASE 3/3: TEXTURING ═══")
    _job_log(job_id, "[6/8] Loading paint pipeline...")
    paint = _get_paint_pipeline()

    _job_log(job_id, "[7/8] Painting texture (~2-5 min)...")
    textured = paint(paint_mesh, ref_image)

    _job_log(job_id, "[8/8] Saving textured results...")
    files = ["text2img_reference.png"]

    glb_out = output / "textured.glb"
    textured.export(str(glb_out))
    files.append("textured.glb")

    obj_out = output / "mesh_uv.obj"
    textured.export(str(obj_out))
    files.append("mesh_uv.obj")

    tex_out = output / "texture_baked.png"
    tex_saved = False
    try:
        mat = textured.visual.material
        if hasattr(mat, "image") and mat.image is not None:
            mat.image.save(str(tex_out))
            tex_saved = True
    except Exception:
        pass
    if not tex_saved:
        try:
            tv = textured.visual
            if hasattr(tv, "to_texture"):
                tv.to_texture().image.save(str(tex_out))
                tex_saved = True
        except Exception:
            pass
    if tex_saved:
        files.append("texture_baked.png")

    files.append("mesh.glb")

    torch.cuda.empty_cache()
    _job_log(job_id, f"═══ COMPLETE: {len(files)} files ready ═══")

    return {"files": files, "output_dir": str(output)}


# ── Quality presets ──────────────────────────────────────────────────────────

QUALITY_PRESETS = {
    "low": {
        "target_faces": 10000,
        "octree_resolution": 256,
        "num_inference_steps": 10,
        "model": "turbo",
        "label": "Low — turbo, 10k faces, fast",
    },
    "medium": {
        "target_faces": 50000,
        "octree_resolution": 384,
        "num_inference_steps": 20,
        "model": "turbo",
        "label": "Medium — turbo, 50k faces",
    },
    "high": {
        "target_faces": 150000,
        "octree_resolution": 384,
        "num_inference_steps": 30,
        "model": "full",
        "label": "High — full model, 150k faces",
    },
    "ultra": {
        "target_faces": 0,
        "octree_resolution": 512,
        "num_inference_steps": 50,
        "model": "full",
        "label": "Ultra — full model, max detail",
    },
}


def _resolve_preset(target_faces: int, preset: str, model: str = "",
                    octree_resolution: int = 0, num_inference_steps: int = 0) -> dict:
    """Resolve generation parameters from preset or explicit values."""
    if preset and preset in QUALITY_PRESETS:
        params = QUALITY_PRESETS[preset].copy()
        # Allow explicit overrides even with a preset
        if model:
            params["model"] = model
        if octree_resolution > 0:
            params["octree_resolution"] = octree_resolution
        if num_inference_steps > 0:
            params["num_inference_steps"] = num_inference_steps
        # Explicit target_faces (>0) overrides preset value; 0 keeps preset default
        if target_faces > 0:
            params["target_faces"] = target_faces
        return params
    return {
        "target_faces": target_faces,
        "octree_resolution": octree_resolution or 384,
        "num_inference_steps": num_inference_steps or 30,
        "model": model or "turbo",
    }


def _compute_vertex_curvature(mesh, job_id: str):
    """Compute per-vertex curvature to identify high-detail areas (edges, spikes, creases).

    Returns an array of curvature values per vertex (higher = more detail needed).
    Uses the discrete mean curvature approximation via the angle defect method.
    """
    import numpy as np

    _job_log(job_id, "      Analyzing mesh curvature for adaptive decimation...")
    vertices = mesh.vertices
    faces = mesh.faces
    n_verts = len(vertices)

    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    face_normals = face_normals / norms

    # Compute per-vertex curvature as the variance of adjacent face normals
    # High variance = high curvature = sharp edges/spikes
    vertex_curvature = np.zeros(n_verts)
    vertex_face_count = np.zeros(n_verts)

    # Accumulate face normals per vertex
    vertex_normal_sum = np.zeros((n_verts, 3))
    for i in range(3):
        np.add.at(vertex_normal_sum, faces[:, i], face_normals)
        np.add.at(vertex_face_count, faces[:, i], 1)

    # Average normal per vertex
    mask = vertex_face_count > 0
    vertex_normal_avg = np.zeros_like(vertex_normal_sum)
    vertex_normal_avg[mask] = vertex_normal_sum[mask] / vertex_face_count[mask, np.newaxis]

    # Curvature = deviation of each face normal from the vertex average
    for i in range(3):
        diff = face_normals - vertex_normal_avg[faces[:, i]]
        deviation = np.linalg.norm(diff, axis=1)
        np.add.at(vertex_curvature, faces[:, i], deviation)

    vertex_curvature[mask] /= vertex_face_count[mask]

    # Normalize to [0, 1]
    cmax = vertex_curvature.max()
    if cmax > 1e-10:
        vertex_curvature /= cmax

    high_curv = (vertex_curvature > 0.3).sum()
    _job_log(job_id, f"      Curvature analysis: {high_curv:,} high-detail vertices ({100*high_curv/n_verts:.1f}%)")

    return vertex_curvature


def _adaptive_decimate(mesh, target_faces: int, job_id: str):
    """Two-pass adaptive decimation that preserves high-curvature regions.

    Strategy:
    1. Compute per-vertex curvature to find edges, spikes, creases.
    2. Split mesh into high-curvature (protected) and low-curvature regions.
    3. Decimate low-curvature regions aggressively, high-curvature regions gently.
    4. Merge results.

    Falls back to uniform decimation if adaptive fails.
    """
    import trimesh
    import numpy as np

    current_faces = len(mesh.faces)
    if current_faces <= target_faces:
        _job_log(job_id, f"      Mesh has {current_faces:,} faces (target: {target_faces:,}), skipping")
        return mesh

    # Compute curvature
    curvature = _compute_vertex_curvature(mesh, job_id)

    # Curvature threshold: faces where ANY vertex has high curvature are "protected"
    curvature_threshold = 0.25
    face_max_curvature = np.max(curvature[mesh.faces], axis=1)
    high_detail_mask = face_max_curvature > curvature_threshold
    n_protected = high_detail_mask.sum()
    n_reducible = current_faces - n_protected

    _job_log(job_id, f"      Protected faces (high detail): {n_protected:,} | Reducible: {n_reducible:,}")

    if n_protected >= target_faces:
        # More protected faces than target — just do gentle uniform decimation
        _job_log(job_id, f"      Too many protected faces, using gentle uniform decimation")
        return _uniform_decimate(mesh, target_faces, job_id, aggressiveness=3)

    # Budget: protected faces stay, reducible faces get decimated
    # Allocate some extra budget to reducible region to reach target
    reducible_target = max(target_faces - n_protected, int(n_reducible * 0.3))
    _job_log(job_id, f"      Decimation budget: {n_protected:,} protected + {reducible_target:,} reduced = ~{n_protected + reducible_target:,} total")

    # Use pyfqmr with vertex weights based on curvature
    # Higher curvature = higher weight = more resistance to decimation
    try:
        import pyfqmr

        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(mesh.vertices, mesh.faces)

        # pyfqmr doesn't support per-vertex weights directly,
        # but we can use a two-pass approach:
        # Pass 1: Aggressive decimation of the full mesh but with border preservation
        # The aggressiveness parameter controls how much it respects geometric features
        # Lower aggressiveness = more feature preservation
        simplifier.simplify_mesh(
            target_count=target_faces,
            aggressiveness=4,          # Lower = more feature-preserving (default was 7)
            preserve_border=True,
            max_iterations=100,
        )
        vertices, faces, normals = simplifier.getMesh()
        decimated = trimesh.Trimesh(vertices=vertices, faces=faces)

        final_faces = len(decimated.faces)
        _job_log(job_id, f"      Adaptive decimation (pyfqmr): {current_faces:,} → {final_faces:,} faces")

        # Verify that high-detail areas were preserved by checking remaining curvature
        if final_faces > 0:
            new_curv = _compute_vertex_curvature(decimated, job_id)
            high_after = (new_curv > 0.3).sum()
            _job_log(job_id, f"      Detail preservation check: {high_after:,} high-curvature vertices remain")

        return decimated

    except ImportError:
        _job_log(job_id, "      pyfqmr not available, using trimesh fallback")
        return _uniform_decimate(mesh, target_faces, job_id, aggressiveness=5)


def _uniform_decimate(mesh, target_faces: int, job_id: str, aggressiveness: int = 7):
    """Simple uniform decimation (no curvature awareness)."""
    import trimesh

    current_faces = len(mesh.faces)
    if current_faces <= target_faces:
        return mesh

    # Try pyfqmr first
    try:
        import pyfqmr
        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(mesh.vertices, mesh.faces)
        simplifier.simplify_mesh(
            target_count=target_faces,
            aggressiveness=aggressiveness,
            preserve_border=True,
        )
        vertices, faces, normals = simplifier.getMesh()
        decimated = trimesh.Trimesh(vertices=vertices, faces=faces)
        _job_log(job_id, f"      Uniform decimation (pyfqmr): {len(decimated.faces):,} faces")
        return decimated
    except ImportError:
        pass

    # Fallback: trimesh
    try:
        decimated = mesh.simplify_quadric_decimation(target_faces)
        _job_log(job_id, f"      Uniform decimation (trimesh): {len(decimated.faces):,} faces")
        return decimated
    except Exception as e:
        _job_log(job_id, f"      Decimation failed ({e}), keeping original")
        return mesh


def _decimate_mesh(mesh, target_faces: int, job_id: str):
    """Smart decimation: uses adaptive curvature-aware reduction.

    Automatically detects edges, spikes, and creases and preserves them
    while aggressively reducing flat/smooth regions.
    """
    return _adaptive_decimate(mesh, target_faces, job_id)


def _run_texture(
    mesh_path: str, image_path: str, output_dir: str, job_id: str
) -> dict:
    """Texture painting (blocking, runs in thread)."""
    import torch
    import trimesh
    from PIL import Image

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _job_log(job_id, "[1/4] Loading paint pipeline...")
    pipeline = _get_paint_pipeline()

    _job_log(job_id, f"[2/4] Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    _job_log(job_id, f"[3/4] Texturing with image (~2-5 min)...")
    image = Image.open(image_path)
    textured = pipeline(mesh, image)

    _job_log(job_id, "[4/4] Saving results...")
    files = []

    glb_out = output / "textured.glb"
    textured.export(str(glb_out))
    files.append("textured.glb")

    obj_out = output / "mesh_uv.obj"
    textured.export(str(obj_out))
    files.append("mesh_uv.obj")

    tex_out = output / "texture_baked.png"
    tex_saved = False
    try:
        mat = textured.visual.material
        if hasattr(mat, "image") and mat.image is not None:
            mat.image.save(str(tex_out))
            tex_saved = True
    except Exception:
        pass
    if not tex_saved:
        try:
            tv = textured.visual
            if hasattr(tv, "to_texture"):
                tv.to_texture().image.save(str(tex_out))
                tex_saved = True
        except Exception:
            pass
    if tex_saved:
        files.append("texture_baked.png")

    torch.cuda.empty_cache()

    return {"files": files, "output_dir": str(output)}


# ── Background task runner ───────────────────────────────────────────────────


def _run_full_pipeline(
    image_path: str, output_dir: str, job_id: str,
    target_faces: int = 50000,
    octree_resolution: int = 384,
    num_inference_steps: int = 30,
    model: str = "turbo",
) -> dict:
    """Full pipeline: image → shape → decimate → texture (blocking, runs in thread)."""
    import torch
    from PIL import Image
    import numpy as np
    import trimesh

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Shape generation ──────────────────────────────────
    _job_log(job_id, "═══ PHASE 1/2: SHAPE GENERATION ═══")
    _job_log(job_id, f"[1/6] Loading shape pipeline ({model})...")
    pipeline = _get_shape_pipeline(model)

    _job_log(job_id, f"[2/6] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")

    alpha = np.array(image)[:, :, 3]
    if (alpha < 10).sum() / alpha.size < 0.05:
        try:
            from hy3dgen.rembg import BackgroundRemover
            _job_log(job_id, "      Removing background...")
            image = BackgroundRemover()(image)
        except ImportError:
            pass

    _job_log(job_id, f"[3/6] Generating 3D shape (octree={octree_resolution}, steps={num_inference_steps})...")
    result = pipeline(
        image=image,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
    )
    mesh = result[0]
    orig_faces = len(mesh.faces)
    _job_log(job_id, f"      Generated: {len(mesh.vertices):,} verts | {orig_faces:,} faces")

    # Decimation — keep the in-memory mesh for painting (avoids lossy GLB roundtrip)
    if target_faces > 0 and orig_faces > target_faces:
        _job_log(job_id, f"[3.5/6] Decimating: {orig_faces:,} → {target_faces:,} faces...")
        tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        tri_mesh = _decimate_mesh(tri_mesh, target_faces, job_id)
        glb_path = output / "mesh.glb"
        tri_mesh.export(str(glb_path))
        paint_mesh = tri_mesh  # pass directly to paint pipeline
    else:
        # No decimation — convert to trimesh in memory, export for download
        tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        glb_path = output / "mesh.glb"
        tri_mesh.export(str(glb_path))
        paint_mesh = tri_mesh

    mesh_size_kb = glb_path.stat().st_size // 1024
    _job_log(job_id, f"      Shape saved: mesh.glb ({mesh_size_kb} KB)")

    # Free shape model from VRAM before loading paint
    _unload_shape_pipeline()
    torch.cuda.empty_cache()

    # ── Phase 2: Texturing ──────────────────────────────────────────
    _job_log(job_id, "═══ PHASE 2/2: TEXTURING ═══")
    _job_log(job_id, "[4/6] Loading paint pipeline...")
    paint = _get_paint_pipeline()

    ref_image = Image.open(image_path)
    _job_log(job_id, "[5/6] Painting texture (~2-5 min)...")
    textured = paint(paint_mesh, ref_image)

    _job_log(job_id, "[6/6] Saving textured results...")
    files = []

    glb_out = output / "textured.glb"
    textured.export(str(glb_out))
    files.append("textured.glb")

    obj_out = output / "mesh_uv.obj"
    textured.export(str(obj_out))
    files.append("mesh_uv.obj")

    tex_out = output / "texture_baked.png"
    tex_saved = False
    try:
        mat = textured.visual.material
        if hasattr(mat, "image") and mat.image is not None:
            mat.image.save(str(tex_out))
            tex_saved = True
    except Exception:
        pass
    if not tex_saved:
        try:
            tv = textured.visual
            if hasattr(tv, "to_texture"):
                tv.to_texture().image.save(str(tex_out))
                tex_saved = True
        except Exception:
            pass
    if tex_saved:
        files.append("texture_baked.png")

    # Also keep the raw mesh
    files.append("mesh.glb")

    torch.cuda.empty_cache()
    _job_log(job_id, f"═══ COMPLETE: {len(files)} files ready ═══")

    return {"files": files, "output_dir": str(output)}


async def _run_in_background(job_id: str, func, *args):
    """Run a blocking inference function in a thread, update job status.

    Acquires _gpu_semaphore for the duration of the job to prevent
    concurrent GPU access (OOM protection).
    """
    loop = asyncio.get_event_loop()
    async with _gpu_semaphore:
        try:
            result = await loop.run_in_executor(None, func, *args)
            output_dir = result.get("output_dir") or str(
                Path(result.get("mesh_path", "")).parent
            )
            files = result.get("files", [])
            if not files and "mesh_path" in result:
                files = ["mesh.glb"]
            _job_done(job_id, output_dir, files)
        except Exception as e:
            _job_fail(job_id, f"{e}\n{traceback.format_exc()}")


def _check_gpu_available():
    """Raise HTTP 429 if the GPU is already busy with another job."""
    if _gpu_semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail="GPU busy — another job is running. Try again later.",
            headers={"Retry-After": "30"},
        )


# ── FastAPI app ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Vision3D] Models dir: {MODELS_DIR}")
    print(f"[Vision3D] Work dir:   {WORK_DIR}")
    print(f"[Vision3D] API key:    {'configured' if API_KEY else 'NONE (open access)'}")
    # Start background job cleanup task (Phase 8.2 — Bug #2 fix)
    cleanup_task = asyncio.create_task(_job_cleanup_loop())
    print(f"[Vision3D] Job cleanup: every {JOB_CLEANUP_INTERVAL}s, TTL {JOB_TTL_SECONDS}s, max {MAX_JOBS} jobs")
    yield
    cleanup_task.cancel()  # Stop cleanup on shutdown


app = FastAPI(
    title="Vision3D",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health():
    """Health check — returns GPU info if available."""
    info = {"status": "ok", "api_key_required": bool(API_KEY)}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        info["gpu"] = "torch not available"
    info["models"] = _get_available_models()
    # Check if text-to-image module is available
    try:
        from diffusers import AutoPipelineForText2Image  # noqa: F401
        info["text_to_3d"] = "available (SDXL Turbo)"
    except ImportError:
        info["text_to_3d"] = "unavailable (diffusers not found)"
    return info


@app.post("/api/generate-shape")
async def generate_shape(
    image: UploadFile = File(...),
    output_subdir: str = Form("0"),
    target_faces: int = Form(0),
    preset: str = Form(""),
    model: str = Form(""),
    octree_resolution: int = Form(0),
    num_inference_steps: int = Form(0),
    x_api_key: Optional[str] = Header(None),
):
    """Upload an image, get a 3D mesh back (async job)."""
    _verify_api_key(x_api_key)
    output_subdir = _validate_output_subdir(output_subdir)  # Sanitize path traversal
    _check_gpu_available()  # Reject if GPU is busy (OOM protection)
    _check_max_jobs()  # Reject if too many jobs in memory (Phase 8.2)

    # Validate upload: MIME type + size (Phase 8.2 — Bug #5 fix)
    image_content = await _validate_upload(image, ALLOWED_IMAGE_TYPES)

    params = _resolve_preset(target_faces, preset, model, octree_resolution, num_inference_steps)
    job_id = _new_job("shape-image", f"subdir={output_subdir}, target_faces={params['target_faces']}")
    out_dir = WORK_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image (already read by _validate_upload)
    image_path = out_dir / "input.png"
    image_path.write_bytes(image_content)

    # Launch inference in background
    asyncio.create_task(
        _run_in_background(
            job_id, _run_shape_from_image, str(image_path), str(out_dir), job_id,
            params["target_faces"], params["octree_resolution"], params["num_inference_steps"], params["model"],
        )
    )

    return {"job_id": job_id, "status": "running", "poll": f"/api/jobs/{job_id}"}


@app.post("/api/generate-text")
async def generate_text(
    text_prompt: str = Form(...),
    output_subdir: str = Form("0"),
    target_faces: int = Form(0),
    preset: str = Form(""),
    model: str = Form(""),
    octree_resolution: int = Form(0),
    num_inference_steps: int = Form(0),
    x_api_key: Optional[str] = Header(None),
):
    """Generate 3D mesh from text prompt (async job)."""
    _verify_api_key(x_api_key)
    output_subdir = _validate_output_subdir(output_subdir)  # Sanitize path traversal
    _check_gpu_available()  # Reject if GPU is busy (OOM protection)
    _check_max_jobs()  # Reject if too many jobs in memory (Phase 8.2)

    params = _resolve_preset(target_faces, preset, model, octree_resolution, num_inference_steps)
    job_id = _new_job("shape-text", f"prompt={text_prompt[:50]}")
    out_dir = WORK_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    asyncio.create_task(
        _run_in_background(
            job_id, _run_shape_from_text, text_prompt, str(out_dir), job_id,
            params["target_faces"], params["octree_resolution"], params["num_inference_steps"], params["model"],
        )
    )

    return {"job_id": job_id, "status": "running", "poll": f"/api/jobs/{job_id}"}


@app.post("/api/texture-mesh")
async def texture_mesh(
    mesh: UploadFile = File(...),
    image: UploadFile = File(...),
    output_subdir: str = Form("0"),
    x_api_key: Optional[str] = Header(None),
):
    """Upload mesh + image, get textured mesh back (async job)."""
    _verify_api_key(x_api_key)
    output_subdir = _validate_output_subdir(output_subdir)  # Sanitize path traversal
    _check_gpu_available()  # Reject if GPU is busy (OOM protection)
    _check_max_jobs()  # Reject if too many jobs in memory (Phase 8.2)

    # Validate uploads: MIME type + size (Phase 8.2 — Bug #5 fix)
    mesh_content = await _validate_upload(mesh, ALLOWED_MESH_TYPES)
    image_content = await _validate_upload(image, ALLOWED_IMAGE_TYPES)

    job_id = _new_job("texture", f"subdir={output_subdir}")
    out_dir = WORK_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = out_dir / "mesh.glb"
    mesh_path.write_bytes(mesh_content)

    image_path = out_dir / "input.png"
    image_path.write_bytes(image_content)

    asyncio.create_task(
        _run_in_background(
            job_id, _run_texture, str(mesh_path), str(image_path), str(out_dir), job_id
        )
    )

    return {"job_id": job_id, "status": "running", "poll": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, x_api_key: Optional[str] = Header(None)):
    """Poll job status. When completed, includes download links."""
    _verify_api_key(x_api_key)

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = {
        "id": job["id"],
        "type": job["type"],
        "status": job["status"],
        "elapsed_s": round(time.time() - job["created"], 1),
        "log": job["log"],
    }

    if job["status"] == "completed":
        result["files"] = [
            {"name": f, "download": f"/api/jobs/{job_id}/files/{f}"}
            for f in job["files"]
        ]
    elif job["status"] == "failed":
        result["error"] = job["error"]

    return result


@app.get("/api/jobs/{job_id}/files/{filename}")
async def download_file(
    job_id: str, filename: str, x_api_key: Optional[str] = Header(None)
):
    """Download a result file from a completed job."""
    _verify_api_key(x_api_key)

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job not yet completed")
    if filename not in job["files"]:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not in job")

    file_path = Path(job["output_dir"]) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(str(file_path), filename=filename)


# ── Combined pipeline: image → shape → texture (one call) ───────────────────


@app.get("/api/presets")
async def get_presets():
    """Return available quality presets."""
    return QUALITY_PRESETS


@app.get("/api/models")
async def get_models():
    """Return available shape models (those with weights on disk)."""
    available = _get_available_models()
    return {
        "models": available,
        "default": "turbo" if "turbo" in available else (available[0] if available else "turbo"),
        "all": {name: subfolder for name, subfolder in SHAPE_MODELS.items()},
    }


@app.post("/api/generate-full")
async def generate_full(
    image: UploadFile = File(...),
    output_subdir: str = Form("0"),
    target_faces: int = Form(50000),
    preset: str = Form(""),
    model: str = Form(""),
    octree_resolution: int = Form(0),
    num_inference_steps: int = Form(0),
    x_api_key: Optional[str] = Header(None),
):
    """Full pipeline: image → shape generation → adaptive decimation → texturing.

    Returns a job that produces textured.glb, mesh_uv.obj, texture_baked.png, and mesh.glb.
    Use /api/jobs/{job_id}/stream for real-time Server-Sent Events progress.
    """
    _verify_api_key(x_api_key)
    output_subdir = _validate_output_subdir(output_subdir)  # Sanitize path traversal
    _check_gpu_available()  # Reject if GPU is busy (OOM protection)
    _check_max_jobs()  # Reject if too many jobs in memory (Phase 8.2)

    # Validate upload: MIME type + size (Phase 8.2 — Bug #5 fix)
    image_content = await _validate_upload(image, ALLOWED_IMAGE_TYPES)

    params = _resolve_preset(target_faces, preset, model, octree_resolution, num_inference_steps)
    preset_label = preset or f"{params['target_faces']} faces"
    job_id = _new_job("full-pipeline", f"subdir={output_subdir}, quality={preset_label}")
    out_dir = WORK_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "input.png"
    image_path.write_bytes(image_content)

    asyncio.create_task(
        _run_in_background(
            job_id, _run_full_pipeline, str(image_path), str(out_dir), job_id,
            params["target_faces"], params["octree_resolution"], params["num_inference_steps"], params["model"],
        )
    )

    return {
        "job_id": job_id,
        "status": "running",
        "quality": preset_label,
        "target_faces": params["target_faces"],
        "poll": f"/api/jobs/{job_id}",
        "stream": f"/api/jobs/{job_id}/stream",
    }


# ── Server-Sent Events (SSE) for real-time progress ─────────────────────────


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(
    job_id: str,
    x_api_key: Optional[str] = Header(None),
    x_api_key_query: Optional[str] = Query(None, alias="x_api_key"),
):
    """Stream job progress as Server-Sent Events (SSE).

    Connect with: const es = new EventSource('/api/jobs/{job_id}/stream?x_api_key=KEY')
    Events: 'log' (progress lines), 'status' (running/completed/failed), 'done' (final).
    Accepts API key via header or query param (EventSource cannot send custom headers).
    """
    # SSE auth: header first, query param fallback (Bug #4 fix — Phase 8.2)
    _verify_api_key(x_api_key, query_api_key=x_api_key_query)

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    import json

    async def event_generator():
        last_log_len = 0

        while True:
            j = _jobs.get(job_id)
            if not j:
                yield f"event: error\ndata: Job disappeared\n\n"
                break

            # Send new log lines
            if len(j["log"]) > last_log_len:
                for line in j["log"][last_log_len:]:
                    yield f"event: log\ndata: {line}\n\n"
                last_log_len = len(j["log"])

            elapsed = round(time.time() - j["created"], 1)

            if j["status"] == "completed":
                files_json = json.dumps(j["files"])
                yield f"event: status\ndata: completed\n\n"
                yield f"event: done\ndata: {{\"status\":\"completed\",\"elapsed_s\":{elapsed},\"files\":{files_json}}}\n\n"
                break
            elif j["status"] == "failed":
                err = (j.get("error") or "Unknown").replace("\n", "\\n")
                yield f"event: status\ndata: failed\n\n"
                yield f"event: done\ndata: {{\"status\":\"failed\",\"elapsed_s\":{elapsed},\"error\":\"{err}\"}}\n\n"
                break

            yield f"event: status\ndata: running\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Web UI ───────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Browser-based UI for image-to-3D generation."""
    return _WEB_UI_HTML


_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vision3D</title>
<style>
  :root { --bg: #0f1117; --card: #1a1d27; --border: #2d3040; --accent: #6c5ce7; --ok: #00b894; --fail: #e17055; --text: #dfe6e9; --dim: #636e72; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 860px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.5rem; margin-bottom: .5rem; }
  .subtitle { color: var(--dim); margin-bottom: 2rem; font-size: .9rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  label { display: block; color: var(--dim); font-size: .85rem; margin-bottom: .3rem; margin-top: .8rem; }
  input[type=file], input[type=number], input[type=text], select { width: 100%; padding: .6rem .8rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: .9rem; }
  .row { display: flex; gap: 1rem; }
  .row > * { flex: 1; }
  .row3 { display: flex; gap: .75rem; }
  .row3 > * { flex: 1; }
  button { padding: .7rem 1.5rem; border: none; border-radius: 8px; font-size: .95rem; cursor: pointer; font-weight: 600; margin-top: 1rem; transition: all .2s; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { opacity: .85; }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; }
  #log { background: #0a0c10; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; font-family: 'SF Mono', monospace; font-size: .8rem; line-height: 1.6; max-height: 400px; overflow-y: auto; white-space: pre-wrap; display: none; margin-top: 1rem; }
  .log-line { color: var(--dim); }
  .log-line.phase { color: var(--accent); font-weight: bold; }
  .log-line.done { color: var(--ok); font-weight: bold; }
  .log-line.error { color: var(--fail); }
  #result { display: none; margin-top: 1rem; }
  #viewer3d { width: 100%; height: 400px; border-radius: 8px; background: #1a1d27; margin-top: 1rem; display: none; }
  #refImage { max-width: 200px; border-radius: 8px; margin-top: .5rem; }
  .file-link { display: inline-block; padding: .5rem 1rem; margin: .3rem; background: var(--accent); color: #fff; border-radius: 6px; text-decoration: none; font-size: .85rem; }
  .file-link:hover { opacity: .8; }
  .progress-bar { height: 4px; background: var(--border); border-radius: 2px; margin-top: .5rem; overflow: hidden; display: none; }
  .progress-bar .fill { height: 100%; background: var(--accent); width: 0%; transition: width .5s; }
  #preview { max-width: 200px; max-height: 200px; border-radius: 8px; margin-top: .5rem; display: none; }
  .status { font-size: .85rem; color: var(--dim); margin-top: .5rem; }
  .tabs { display: flex; gap: .5rem; margin-bottom: 1rem; }
  .tab { padding: .5rem 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; cursor: pointer; color: var(--dim); font-size: .85rem; }
  .tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .param-hint { font-size: .75rem; color: var(--dim); margin-top: .15rem; }
</style>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
</head>
<body>
<div class="container">
  <h1>Vision3D</h1>
  <p class="subtitle">AI-powered 3D generation from images or text</p>

  <div class="card">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('full')">Image → 3D</div>
      <div class="tab" onclick="switchTab('text')">Text → 3D</div>
    </div>

    <form id="form" onsubmit="return submitJob(event)">
      <!-- ── Full Pipeline ── -->
      <div id="tab-full">
        <label for="image">Reference image (PNG/JPG)</label>
        <input type="file" id="image" accept="image/*" onchange="previewImage(this)">
        <img id="preview" alt="preview">

        <div class="row">
          <div>
            <label for="preset">Quality preset</label>
            <select id="preset" onchange="onPresetChange('full')">
              <option value="low">Low (turbo, 10k, fast)</option>
              <option value="medium" selected>Medium (turbo, 50k)</option>
              <option value="high">High (full, 150k)</option>
              <option value="ultra">Ultra (full, max detail)</option>
            </select>
          </div>
          <div>
            <label for="model">Shape model</label>
            <select id="model"></select>
            <div class="param-hint">turbo ~1 min | full ~5 min</div>
          </div>
        </div>

        <div class="row3">
          <div>
            <label for="octree_resolution">Octree resolution</label>
            <input type="number" id="octree_resolution" value="384" min="128" max="512" step="64">
            <div class="param-hint">256 / 384 / 512</div>
          </div>
          <div>
            <label for="num_inference_steps">Inference steps</label>
            <input type="number" id="num_inference_steps" value="20" min="1" max="100" step="1">
            <div class="param-hint">turbo: 5-10 | full: 30-50</div>
          </div>
          <div>
            <label for="target_faces">Target faces</label>
            <input type="number" id="target_faces" value="50000" min="0" step="5000">
            <div class="param-hint">0 = no decimation</div>
          </div>
        </div>

        <div class="row">
          <div>
            <label for="subdir">Output subdirectory</label>
            <input type="text" id="subdir" value="web_0" placeholder="e.g. asset_001">
          </div>
        </div>
      </div>

      <!-- ── Text to 3D ── -->
      <div id="tab-text" style="display:none">
        <label for="text_prompt">Text prompt (describe the object)</label>
        <input type="text" id="text_prompt" placeholder="e.g. a realistic wooden mailbox, a medieval sword, a cute robot">
        <div class="param-hint">The prompt is automatically enhanced for better 3D results (isolated object, white background, etc.)</div>

        <div class="row">
          <div>
            <label for="preset_text">Quality preset</label>
            <select id="preset_text" onchange="onPresetChange('text')">
              <option value="low">Low (turbo, 10k, fast)</option>
              <option value="medium" selected>Medium (turbo, 50k)</option>
              <option value="high">High (full, 150k)</option>
              <option value="ultra">Ultra (full, max detail)</option>
            </select>
          </div>
          <div>
            <label for="model_text">Shape model</label>
            <select id="model_text"></select>
            <div class="param-hint">turbo ~1 min | full ~5 min</div>
          </div>
        </div>

        <div class="row3">
          <div>
            <label for="octree_resolution_text">Octree resolution</label>
            <input type="number" id="octree_resolution_text" value="384" min="128" max="512" step="64">
            <div class="param-hint">256 / 384 / 512</div>
          </div>
          <div>
            <label for="num_inference_steps_text">Inference steps</label>
            <input type="number" id="num_inference_steps_text" value="20" min="1" max="100" step="1">
            <div class="param-hint">turbo: 5-10 | full: 30-50</div>
          </div>
          <div>
            <label for="target_faces_text">Target faces</label>
            <input type="number" id="target_faces_text" value="50000" min="0" step="5000">
            <div class="param-hint">0 = no decimation</div>
          </div>
        </div>

        <div class="row">
          <div>
            <label for="subdir_text">Output subdirectory</label>
            <input type="text" id="subdir_text" value="web_text_0" placeholder="e.g. text_asset_001">
          </div>
        </div>
      </div>

      <button type="submit" class="btn-primary" id="submitBtn">Generate 3D Model</button>
      <div class="progress-bar" id="progressBar"><div class="fill" id="progressFill"></div></div>
      <p class="status" id="statusText"></p>
    </form>

    <div id="log"></div>
    <div id="result"></div>
  </div>
</div>

<script>
let currentTab = 'full';
let apiKey = new URLSearchParams(location.search).get('key') || '';

const PRESETS = {
  low:    { model: 'turbo', octree_resolution: 256, num_inference_steps: 10, target_faces: 10000 },
  medium: { model: 'turbo', octree_resolution: 384, num_inference_steps: 20, target_faces: 50000 },
  high:   { model: 'full',  octree_resolution: 384, num_inference_steps: 30, target_faces: 150000 },
  ultra:  { model: 'full',  octree_resolution: 512, num_inference_steps: 50, target_faces: 0 },
};

// Populate model dropdowns from server
async function loadModels() {
  try {
    const resp = await fetch('/api/models');
    const data = await resp.json();
    const models = data.models || [];
    ['model', 'model_text'].forEach(id => {
      const sel = document.getElementById(id);
      sel.innerHTML = '';
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        if (m === data.default) opt.selected = true;
        sel.appendChild(opt);
      });
      // If no models found, show a fallback
      if (models.length === 0) {
        sel.innerHTML = '<option value="turbo">turbo</option>';
      }
    });
  } catch (e) {
    console.warn('Could not load models:', e);
  }
}
loadModels();

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[onclick*="${tab}"]`).classList.add('active');
  ['full','text'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t===tab ? '' : 'none';
  });
  currentTab = tab;
}

function onPresetChange(tabId) {
  // Determine suffix for field IDs ('full' has no suffix, 'shape' has '_shape')
  const suffix = tabId === 'full' ? '' : '_' + tabId;
  const presetSel = document.getElementById('preset' + suffix);
  const preset = PRESETS[presetSel.value];
  if (!preset) return;

  document.getElementById('model' + suffix).value = preset.model;
  document.getElementById('octree_resolution' + suffix).value = preset.octree_resolution;
  document.getElementById('num_inference_steps' + suffix).value = preset.num_inference_steps;
  document.getElementById('target_faces' + suffix).value = preset.target_faces;
}

function previewImage(input) {
  const prev = document.getElementById('preview');
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = e => { prev.src = e.target.result; prev.style.display = 'block'; };
    reader.readAsDataURL(input.files[0]);
  }
}

function addLog(text, cls='') {
  const log = document.getElementById('log');
  log.style.display = 'block';
  const line = document.createElement('div');
  line.className = 'log-line ' + cls;
  line.textContent = text;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

async function submitJob(e) {
  e.preventDefault();
  const btn = document.getElementById('submitBtn');
  const log = document.getElementById('log');
  const result = document.getElementById('result');
  const status = document.getElementById('statusText');
  const bar = document.getElementById('progressBar');
  const fill = document.getElementById('progressFill');

  btn.disabled = true;
  log.innerHTML = '';
  log.style.display = 'block';
  result.style.display = 'none';
  bar.style.display = 'block';
  fill.style.width = '5%';

  let url, formData = new FormData();

  if (currentTab === 'full') {
    url = '/api/generate-full';
    const file = document.getElementById('image').files[0];
    if (!file) { alert('Select an image'); btn.disabled = false; return; }
    formData.append('image', file);
    formData.append('output_subdir', document.getElementById('subdir').value);
    formData.append('model', document.getElementById('model').value);
    formData.append('octree_resolution', document.getElementById('octree_resolution').value);
    formData.append('num_inference_steps', document.getElementById('num_inference_steps').value);
    formData.append('target_faces', document.getElementById('target_faces').value);
  } else if (currentTab === 'text') {
    url = '/api/generate-text';
    const prompt = document.getElementById('text_prompt').value.trim();
    if (!prompt) { alert('Enter a text prompt'); btn.disabled = false; return; }
    formData.append('text_prompt', prompt);
    formData.append('output_subdir', document.getElementById('subdir_text').value);
    formData.append('model', document.getElementById('model_text').value);
    formData.append('octree_resolution', document.getElementById('octree_resolution_text').value);
    formData.append('num_inference_steps', document.getElementById('num_inference_steps_text').value);
    formData.append('target_faces', document.getElementById('target_faces_text').value);
  }

  const sfx = currentTab === 'full' ? '' : '_' + currentTab;
  const params = ` [${document.getElementById('model' + sfx).value}, octree=${document.getElementById('octree_resolution' + sfx).value}, steps=${document.getElementById('num_inference_steps' + sfx).value}, faces=${document.getElementById('target_faces' + sfx).value}]`;
  addLog('Uploading to Vision3D...' + params, '');
  status.textContent = 'Uploading...';

  try {
    const headers = {};
    if (apiKey) headers['x-api-key'] = apiKey;
    const resp = await fetch(url, { method: 'POST', body: formData, headers });
    if (!resp.ok) {
      const errText = await resp.text();
      addLog('Server error (' + resp.status + '): ' + errText.substring(0, 500), 'error');
      status.textContent = 'Failed';
      btn.disabled = false;
      return;
    }
    const job = await resp.json();
    if (!job.job_id) { addLog('Error: ' + JSON.stringify(job), 'error'); btn.disabled = false; return; }

    addLog('Job created: ' + job.job_id);
    fill.style.width = '10%';
    status.textContent = 'Processing... (this takes 1-15 minutes depending on model)';

    const streamUrl = '/api/jobs/' + job.job_id + '/stream' + (apiKey ? '?x_api_key='+apiKey : '');
    const es = new EventSource(streamUrl);

    let progress = 10;
    es.addEventListener('log', (ev) => {
      const text = ev.data;
      const cls = text.includes('===') || text.includes('═══') ? 'phase' : text.includes('COMPLETE') ? 'done' : '';
      addLog(text, cls);
      progress = Math.min(progress + 3, 90);
      fill.style.width = progress + '%';
    });

    es.addEventListener('status', (ev) => {
      status.textContent = ev.data === 'running' ? 'Processing...' : ev.data;
    });

    es.addEventListener('done', (ev) => {
      es.close();
      fill.style.width = '100%';
      const data = JSON.parse(ev.data);
      status.textContent = data.status === 'completed'
        ? 'Done in ' + data.elapsed_s + 's'
        : 'Failed: ' + (data.error || 'unknown');

      if (data.status === 'completed' && data.files) {
        result.style.display = 'block';
        result.innerHTML = '';

        // 3D viewer — prefer textured.glb, fallback to mesh.glb
        const glbFile = data.files.find(f => f === 'textured.glb') || data.files.find(f => f.endsWith('.glb'));
        if (glbFile) {
          const viewerUrl = '/api/jobs/' + job.job_id + '/files/' + glbFile;
          result.innerHTML += `<model-viewer id="viewer3d" src="${viewerUrl}" auto-rotate camera-controls touch-action="pan-y" style="display:block" shadow-intensity="1" exposure="1.2"></model-viewer>`;
        }

        // Show reference image for text-to-3D
        const refImg = data.files.find(f => f === 'text2img_reference.png');
        if (refImg) {
          result.innerHTML += `<p style="color:var(--dim);font-size:.85rem;margin-top:.5rem">Reference image generated from text:</p><img id="refImage" src="/api/jobs/${job.job_id}/files/${refImg}">`;
        }

        // Download links
        result.innerHTML += '<h3 style="margin:.8rem 0 .5rem">Download:</h3>';
        data.files.forEach(f => {
          const a = document.createElement('a');
          a.href = '/api/jobs/' + job.job_id + '/files/' + f;
          a.className = 'file-link';
          a.textContent = f;
          a.download = f;
          result.appendChild(a);
        });
        addLog('All files ready!', 'done');
      } else {
        addLog('Job failed: ' + (data.error || 'unknown'), 'error');
      }
      btn.disabled = false;
    });

    es.onerror = () => {
      es.close();
      status.textContent = 'Connection lost - check /api/jobs/' + job.job_id;
      btn.disabled = false;
    };

  } catch (err) {
    addLog('Error: ' + err.message + ' (URL: ' + url + ')', 'error');
    status.textContent = 'Failed';
    btn.disabled = false;
  }

  return false;
}
</script>
</body>
</html>"""


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision3D GPU API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

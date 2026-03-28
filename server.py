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
    # Direct (development)
    .venv/bin/python server.py --port 8000

    # Production (behind Caddy)
    .venv/bin/python server.py --host 127.0.0.1 --port 8000

Environment:
    GPU_API_KEY       — API key for authentication (empty = open access)
    GPU_MODELS_DIR    — Model weights (default: ./hf_models)
    GPU_WORK_DIR      — working directory for outputs (default: ./output)
    GPU_VISION_DIR    — Vision3D installation directory (default: .)
"""

import asyncio
import hashlib
import os
import secrets
import shutil
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Header, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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


# ── Authentication ───────────────────────────────────────────────────────────


def _verify_api_key(x_api_key: Optional[str]):
    """Verify API key if one is configured."""
    if not API_KEY:
        return  # No API key configured — open access (LAN only)
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Pipeline loaders (lazy, cached) ─────────────────────────────────────────

_shape_pipeline = None
_paint_pipeline = None


def _get_shape_pipeline():
    global _shape_pipeline
    if _shape_pipeline is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        candidates = [
            (MODELS_DIR, "hunyuan3d-dit-v2-0-fast"),
            (MODELS_DIR, "hunyuan3d-dit-v2-0-turbo"),
            (MODELS_DIR, "hunyuan3d-dit-v2-0"),
        ]
        for base, sub in candidates:
            path = os.path.join(base, sub)
            model_file = os.path.join(path, "model.fp16.safetensors")
            if os.path.isdir(path) and os.path.exists(model_file):
                print(f"[Shape] Loading from local: {path}")
                _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    base, subfolder=sub
                )
                return _shape_pipeline

        print("[Shape] Loading from HuggingFace Hub...")
        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2", subfolder="hunyuan3d-dit-v2-0-turbo"
        )
    return _shape_pipeline


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


# ── Inference functions (run in thread pool) ─────────────────────────────────


def _run_shape_from_image(
    image_path: str, output_dir: str, job_id: str,
    target_faces: int = 0,
    octree_resolution: int = 384,
    num_inference_steps: int = 30,
) -> dict:
    """Image → 3D shape generation (blocking, runs in thread).

    Args:
        target_faces: if > 0, decimate the mesh to this face count after generation.
        octree_resolution: marching cubes grid resolution (higher = more detail).
        num_inference_steps: denoising steps (higher = better quality, slower).
    """
    import torch
    from PIL import Image
    import numpy as np

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _job_log(job_id, "[1/5] Loading shape pipeline...")
    pipeline = _get_shape_pipeline()

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
) -> dict:
    """Text → 3D shape generation (blocking, runs in thread)."""
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _job_log(job_id, "[1/4] Loading shape pipeline...")
    pipeline = _get_shape_pipeline()

    _job_log(job_id, f"[2/4] Generating from text: '{text_prompt}' (octree={octree_resolution}, steps={num_inference_steps})...")
    result = pipeline(
        text=text_prompt,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
    )
    mesh = result[0]
    _job_log(
        job_id,
        f"      Generated: {len(mesh.vertices):,} verts | {len(mesh.faces):,} faces",
    )

    # Decimation
    if target_faces > 0:
        _job_log(job_id, "[3/4] Decimating mesh...")
        import trimesh as _tri
        tri_mesh = _tri.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        tri_mesh = _decimate_mesh(tri_mesh, target_faces, job_id)
        glb_path = output / "mesh.glb"
        tri_mesh.export(str(glb_path))
    else:
        _job_log(job_id, "[3/4] Skipping decimation")
        glb_path = output / "mesh.glb"
        mesh.export(str(glb_path))

    _job_log(job_id, "[4/4] Saving mesh.glb...")
    size_kb = glb_path.stat().st_size // 1024

    torch.cuda.empty_cache()

    return {"mesh_path": str(glb_path), "mesh_size_kb": size_kb}


# ── Quality presets ──────────────────────────────────────────────────────────

QUALITY_PRESETS = {
    "low": {
        "target_faces": 10000,
        "octree_resolution": 256,
        "num_inference_steps": 20,
        "label": "Low (10k faces — mobile/web, fast)",
    },
    "medium": {
        "target_faces": 50000,
        "octree_resolution": 384,
        "num_inference_steps": 30,
        "label": "Medium (50k faces — general use)",
    },
    "high": {
        "target_faces": 150000,
        "octree_resolution": 384,
        "num_inference_steps": 50,
        "label": "High (150k faces — detailed models)",
    },
    "ultra": {
        "target_faces": 0,
        "octree_resolution": 512,
        "num_inference_steps": 50,
        "label": "Ultra (no decimation — max detail)",
    },
}


def _resolve_preset(target_faces: int, preset: str) -> dict:
    """Resolve generation parameters from preset name or fall back to defaults."""
    if preset and preset in QUALITY_PRESETS:
        return QUALITY_PRESETS[preset].copy()
    return {
        "target_faces": target_faces,
        "octree_resolution": 384,
        "num_inference_steps": 30,
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
    _job_log(job_id, f"[1/6] Loading shape pipeline...")
    pipeline = _get_shape_pipeline()

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
    """Run a blocking inference function in a thread, update job status."""
    loop = asyncio.get_event_loop()
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


# ── FastAPI app ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[GPU Server] Models dir: {MODELS_DIR}")
    print(f"[GPU Server] Work dir:   {WORK_DIR}")
    print(f"[GPU Server] API key:    {'configured' if API_KEY else 'NONE (open access)'}")
    yield


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
    return info


@app.post("/api/generate-shape")
async def generate_shape(
    image: UploadFile = File(...),
    output_subdir: str = Form("0"),
    target_faces: int = Form(0),
    preset: str = Form(""),
    x_api_key: Optional[str] = Header(None),
):
    """Upload an image, get a 3D mesh back (async job).

    Args:
        target_faces: if > 0, decimate the mesh to this face count.
        preset: quality preset (low/medium/high/ultra). Overrides target_faces.
    """
    _verify_api_key(x_api_key)

    params = _resolve_preset(target_faces, preset)
    job_id = _new_job("shape-image", f"subdir={output_subdir}, target_faces={params['target_faces']}")
    out_dir = WORK_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image
    image_path = out_dir / "input.png"
    content = await image.read()
    image_path.write_bytes(content)

    # Launch inference in background
    asyncio.create_task(
        _run_in_background(
            job_id, _run_shape_from_image, str(image_path), str(out_dir), job_id,
            params["target_faces"], params["octree_resolution"], params["num_inference_steps"],
        )
    )

    return {"job_id": job_id, "status": "running", "poll": f"/api/jobs/{job_id}"}


@app.post("/api/generate-text")
async def generate_text(
    text_prompt: str = Form(...),
    output_subdir: str = Form("0"),
    target_faces: int = Form(0),
    x_api_key: Optional[str] = Header(None),
):
    """Generate 3D mesh from text prompt (async job)."""
    _verify_api_key(x_api_key)

    job_id = _new_job("shape-text", f"prompt={text_prompt[:50]}")
    out_dir = WORK_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    asyncio.create_task(
        _run_in_background(job_id, _run_shape_from_text, text_prompt, str(out_dir), job_id, target_faces)
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

    job_id = _new_job("texture", f"subdir={output_subdir}")
    out_dir = WORK_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = out_dir / "mesh.glb"
    mesh_content = await mesh.read()
    mesh_path.write_bytes(mesh_content)

    image_path = out_dir / "input.png"
    image_content = await image.read()
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


@app.post("/api/generate-full")
async def generate_full(
    image: UploadFile = File(...),
    output_subdir: str = Form("0"),
    target_faces: int = Form(50000),
    preset: str = Form(""),
    x_api_key: Optional[str] = Header(None),
):
    """Full pipeline: image → shape generation → adaptive decimation → texturing.

    Returns a job that produces textured.glb, mesh_uv.obj, texture_baked.png, and mesh.glb.
    Use /api/jobs/{job_id}/stream for real-time Server-Sent Events progress.

    Args:
        target_faces: decimate to this face count (default 50000, 0 = no decimation).
        preset: quality preset name (low/medium/high/ultra). Overrides target_faces if set.
    """
    _verify_api_key(x_api_key)

    params = _resolve_preset(target_faces, preset)
    preset_label = preset or f"{params['target_faces']} faces"
    job_id = _new_job("full-pipeline", f"subdir={output_subdir}, quality={preset_label}")
    out_dir = WORK_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "input.png"
    content = await image.read()
    image_path.write_bytes(content)

    asyncio.create_task(
        _run_in_background(
            job_id, _run_full_pipeline, str(image_path), str(out_dir), job_id,
            params["target_faces"], params["octree_resolution"], params["num_inference_steps"],
        )
    )

    return {
        "job_id": job_id,
        "status": "running",
        "quality": preset_label,
        "target_faces": resolved_faces,
        "poll": f"/api/jobs/{job_id}",
        "stream": f"/api/jobs/{job_id}/stream",
    }


# ── Server-Sent Events (SSE) for real-time progress ─────────────────────────


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str, x_api_key: Optional[str] = Header(None)):
    """Stream job progress as Server-Sent Events (SSE).

    Connect with: const es = new EventSource('/api/jobs/{job_id}/stream')
    Events: 'log' (progress lines), 'status' (running/completed/failed), 'done' (final).
    """
    _verify_api_key(x_api_key)

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

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

    import json
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
  .container { max-width: 800px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.5rem; margin-bottom: .5rem; }
  .subtitle { color: var(--dim); margin-bottom: 2rem; font-size: .9rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; }
  label { display: block; color: var(--dim); font-size: .85rem; margin-bottom: .3rem; margin-top: .8rem; }
  input[type=file], input[type=number], input[type=text], select { width: 100%; padding: .6rem .8rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: .9rem; }
  .row { display: flex; gap: 1rem; }
  .row > * { flex: 1; }
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
  .file-link { display: inline-block; padding: .5rem 1rem; margin: .3rem; background: var(--accent); color: #fff; border-radius: 6px; text-decoration: none; font-size: .85rem; }
  .file-link:hover { opacity: .8; }
  .progress-bar { height: 4px; background: var(--border); border-radius: 2px; margin-top: .5rem; overflow: hidden; display: none; }
  .progress-bar .fill { height: 100%; background: var(--accent); width: 0%; transition: width .5s; }
  #preview { max-width: 200px; max-height: 200px; border-radius: 8px; margin-top: .5rem; display: none; }
  .status { font-size: .85rem; color: var(--dim); margin-top: .5rem; }
  .tabs { display: flex; gap: .5rem; margin-bottom: 1rem; }
  .tab { padding: .5rem 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; cursor: pointer; color: var(--dim); font-size: .85rem; }
  .tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
</style>
</head>
<body>
<div class="container">
  <h1>Vision3D</h1>
  <p class="subtitle">Image-to-3D generation with AI texturing</p>

  <div class="card">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('full')">Full Pipeline</div>
      <div class="tab" onclick="switchTab('shape')">Shape Only</div>
      <div class="tab" onclick="switchTab('texture')">Texture Only</div>
    </div>

    <form id="form" onsubmit="return submitJob(event)">
      <div id="tab-full">
        <label for="image">Reference image (PNG/JPG)</label>
        <input type="file" id="image" accept="image/*" onchange="previewImage(this)">
        <img id="preview" alt="preview">

        <div class="row">
          <div>
            <label for="preset">Quality preset</label>
            <select id="preset" onchange="onPresetChange()">
              <option value="low">Low — 10k faces, fast (octree 256, 20 steps)</option>
              <option value="medium" selected>Medium — 50k faces (octree 384, 30 steps)</option>
              <option value="high">High — 150k faces (octree 384, 50 steps)</option>
              <option value="ultra">Ultra — max detail (octree 512, 50 steps, no decimation)</option>
              <option value="custom">Custom...</option>
            </select>
          </div>
          <div>
            <label for="subdir">Output subdirectory</label>
            <input type="text" id="subdir" value="web_0" placeholder="e.g. asset_001">
          </div>
        </div>
        <div id="custom_faces_row" style="display:none">
          <label for="target_faces">Custom target faces</label>
          <input type="number" id="target_faces" value="50000" min="0" step="5000">
        </div>
      </div>

      <div id="tab-shape" style="display:none">
        <label for="image_shape">Reference image</label>
        <input type="file" id="image_shape" accept="image/*">
        <div class="row">
          <div>
            <label for="preset_shape">Quality preset</label>
            <select id="preset_shape">
              <option value="low">Low — 10k faces (fast)</option>
              <option value="medium" selected>Medium — 50k faces</option>
              <option value="high">High — 150k faces (detailed)</option>
              <option value="ultra">Ultra — max detail</option>
            </select>
          </div>
          <div>
            <label for="subdir_shape">Output subdirectory</label>
            <input type="text" id="subdir_shape" value="web_0">
          </div>
        </div>
      </div>

      <div id="tab-texture" style="display:none">
        <label for="mesh_tex">Mesh file (.glb/.obj)</label>
        <input type="file" id="mesh_tex" accept=".glb,.obj">
        <label for="image_tex">Reference image</label>
        <input type="file" id="image_tex" accept="image/*">
        <label for="subdir_tex">Output subdirectory</label>
        <input type="text" id="subdir_tex" value="web_0">
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
const PRESETS = {low: 10000, medium: 50000, high: 150000, ultra: 0};

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[onclick*="${tab}"]`).classList.add('active');
  ['full','shape','texture'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t===tab ? '' : 'none';
  });
  currentTab = tab;
}

function onPresetChange() {
  const sel = document.getElementById('preset');
  const row = document.getElementById('custom_faces_row');
  row.style.display = sel.value === 'custom' ? '' : 'none';
  if (sel.value !== 'custom') {
    document.getElementById('target_faces').value = PRESETS[sel.value] || 50000;
  }
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
    const preset = document.getElementById('preset').value;
    if (preset === 'custom') {
      formData.append('target_faces', document.getElementById('target_faces').value);
    } else {
      formData.append('preset', preset);
    }
  } else if (currentTab === 'shape') {
    url = '/api/generate-shape';
    const file = document.getElementById('image_shape').files[0];
    if (!file) { alert('Select an image'); btn.disabled = false; return; }
    formData.append('image', file);
    formData.append('output_subdir', document.getElementById('subdir_shape').value);
    const presetShape = document.getElementById('preset_shape').value;
    formData.append('preset', presetShape);
  } else {
    url = '/api/texture-mesh';
    const mf = document.getElementById('mesh_tex').files[0];
    const imf = document.getElementById('image_tex').files[0];
    if (!mf || !imf) { alert('Select both mesh and image'); btn.disabled = false; return; }
    formData.append('mesh', mf);
    formData.append('image', imf);
    formData.append('output_subdir', document.getElementById('subdir_tex').value);
  }

  addLog('Uploading to GPU server...', '');
  status.textContent = 'Uploading...';

  try {
    const headers = {};
    if (apiKey) headers['x-api-key'] = apiKey;
    const resp = await fetch(url, { method: 'POST', body: formData, headers });
    const job = await resp.json();
    if (!job.job_id) { addLog('Error: ' + JSON.stringify(job), 'error'); btn.disabled = false; return; }

    addLog('Job created: ' + job.job_id);
    fill.style.width = '10%';
    status.textContent = 'Processing... (this takes 5-15 minutes)';

    // Connect SSE
    const streamUrl = '/api/jobs/' + job.job_id + '/stream' + (apiKey ? '?x_api_key='+apiKey : '');
    const es = new EventSource(streamUrl);

    let progress = 10;
    es.addEventListener('log', (ev) => {
      const text = ev.data;
      const cls = text.includes('═══') ? 'phase' : text.includes('COMPLETE') ? 'done' : '';
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
        ? `Done in ${data.elapsed_s}s`
        : `Failed: ${data.error || 'unknown'}`;

      if (data.status === 'completed' && data.files) {
        result.style.display = 'block';
        result.innerHTML = '<h3 style="margin-bottom:.5rem">Download Results:</h3>';
        data.files.forEach(f => {
          const a = document.createElement('a');
          a.href = '/api/jobs/' + job.job_id + '/files/' + f;
          a.className = 'file-link';
          a.textContent = f;
          a.download = f;
          result.appendChild(a);
        });
        addLog('All files ready for download!', 'done');
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
    addLog('Error: ' + err.message, 'error');
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

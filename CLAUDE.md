# Vision3D — Contexto Crítico para Claude

## 1. Arquitectura

**Vision3D** es un servidor FastAPI que corre en la máquina GPU **glorfindel** (Rocky Linux) y expone una REST API para generación de modelos 3D usando **Hunyuan3D-2** (Tencent).

El servidor implementa tres pipelines principales:
- **image-to-3D** (`/api/generate-shape`): Convierte una imagen en malla 3D
- **text-to-3D** (`/api/generate-text`): Convierte un prompt de texto en malla 3D
- **texture painting** (`/api/texture-mesh`): Agrega textura a una malla existente
- **full-pipeline** (`/api/full-pipeline`): Combina shape + texture en un solo llamado

---

## 2. Entorno de Ejecución

### Máquina GPU
- **Servidor**: `glorfindel` (GPU con CUDA, Rocky Linux)
- **Usuario**: root (vía systemd)
- **Directorio de trabajo**: `/home/flame/ai-studio/vision3d/`

### Estructura de directorios en glorfindel
```
~/ai-studio/vision3d/
├── .venv/                          # Virtual environment
├── server.py                        # Servidor FastAPI
├── hy3dgen/                         # Código Vision3D local
├── Hunyuan3D-2/                     # Clone git de Tencent/Hunyuan3D-2
└── hf_models/                       # Modelos descargados
    ├── hunyuan3d-dit-v2-0-turbo/   # ~400 MB, ~1 min generación
    ├── hunyuan3d-dit-v2-0-fast/    # ~1 GB, ~2-3 min
    ├── hunyuan3d-dit-v2-0/         # ~3 GB, ~5 min (full quality)
    └── hunyuan3d-paint-v2-0-turbo/ # Modelo de textura
```

### Systemd Service
- **Archivo**: `/etc/systemd/system/vision3d.service`
- **Reiniciar**: `sudo systemctl daemon-reload && sudo systemctl restart vision3d`
- **Ver logs**: `sudo journalctl -u vision3d -f`

---

## 3. Pipelines y Modelos

### Shape Models (Generación de Geometría)

La generación de geometría utiliza `Hunyuan3DDiTFlowMatchingPipeline` de Tencent/Hunyuan3D-2.

**Modelos disponibles** (por velocidad/calidad):
| Modelo | Tiempo | Steps | Parámetros | Uso |
|--------|--------|-------|-----------|-----|
| `turbo` | ~1 min | 5-10 | Fast inference | Prototipado rápido |
| `fast` | ~2-3 min | 15-20 | Balance | Iteraciones |
| `full` | ~5 min | 30-50 | Máxima calidad | Renders finales |

### ⚠️ CRÍTICO: Parámetros del Shape Pipeline

```python
# Shape pipeline SOLO acepta image=, NUNCA text=
pipeline(image=pil_image, ...)  # ✅ Correcto
pipeline(text="...", ...)        # ❌ Error: parámetro no soportado
```

**Problema histórico**: `text-to-3D` crasheaba porque `server.py` intentaba pasar `text=` directamente al shape pipeline, que lo ignoraba, dejando `image=None` → crash en `cv2.resize()`.

**Solución actual (3 fases)**: La función `_run_shape_from_text()` implementa un pipeline completo:

**Fase 1/3 — Text→Image**:
1. Prompt enhancement: se añade "isolated object, centered, no floor, no ground, no shadow, white background, studio lighting, photorealistic, clean lines, product photography"
2. `Flux.1-schnell` (Black Forest Labs) genera imagen 1024×1024 con 4 steps (~10s)
3. Flux se descarga de VRAM inmediatamente después (`_unload_t2i_pipeline()`) para liberar memoria para las fases siguientes y otras apps (ComfyUI, FaceSwap, etc.)
4. `BackgroundRemover` (rembg) elimina fondo para limpieza

**Fase 2/3 — Shape Generation**:
4. Imagen limpia → shape pipeline (image-to-3D)
5. Decimación a `target_faces` (default 10k)

**Fase 3/3 — Texturing**:
6. Paint pipeline genera textura usando la imagen de referencia
7. Salida: `textured.glb`, `mesh_uv.obj`, `texture_baked.png`, `mesh.glb`

### Text-to-Image
- **Pipeline**: `FluxPipeline` (de `diffusers`) — **Flux.1-schnell** (Black Forest Labs)
- **Modelo**: `"black-forest-labs/FLUX.1-schnell"` (~12GB en bfloat16)
- **Descarga automática** en primer uso (vía HuggingFace)
- **Dependencias**: `diffusers`, `transformers`, `accelerate`, `sentencepiece`, `protobuf`
- **Gestión de VRAM**: carga bajo demanda (`_load_t2i_pipeline()`), descarga completa después de generar (`_unload_t2i_pipeline()`) — libera VRAM para ComfyUI, FaceSwap, etc.
- **Parámetros**: 4 inference steps, guidance_scale=0.0, 1024×1024
- **Velocidad**: ~10s por imagen en RTX 3090
- **Ventaja sobre HunyuanDiT**: adherencia al prompt muy superior (95% vs 60%), distingue "moderno minimalista" de "clásico ornamentado"

### Paint Model (Textura)
- **Modelo**: `hunyuan3d-paint-v2-0-turbo`
- **Función**: `_run_texture()` en `server.py`
- Se usa tanto en image-to-3D (full-pipeline) como en text-to-3D

---

## 4. Quality Presets

La función `_resolve_preset()` en `server.py` mapea `quality` → configuración:

| Preset | Modelo | Octree | Steps | Caras máx | Uso |
|--------|--------|--------|-------|-----------|-----|
| `low` | turbo | 256 | 10 | 10k | Web preview |
| `medium` | turbo | 384 | 20 | 50k | Uso estándar |
| `high` | full | 384 | 30 | 150k | Alta calidad |
| `ultra` | full | 512 | 50 | sin límite | Producción |

---

## 5. REST API Endpoints

### Generación de Forma
```
POST /api/generate-shape
Content-Type: multipart/form-data

Parámetros:
  - image: archivo PNG/JPG
  - model: "turbo" | "fast" | "full" (default: "turbo")
  - quality: "low" | "medium" | "high" | "ultra"
  - steps: número de pasos (sobrescribe quality)
  - seed: semilla para reproducibilidad (default: aleatorio)

Respuesta:
  {
    "job_id": "uuid-string",
    "status": "processing"
  }
```

```
POST /api/generate-text
Content-Type: application/json

{
  "prompt": "describe the 3D object in detail",
  "model": "turbo" | "fast" | "full",
  "quality": "low" | "medium" | "high" | "ultra",
  "seed": int (optional)
}

Respuesta:
  {
    "job_id": "uuid-string",
    "status": "processing"
  }
```

### Texturizado
```
POST /api/texture-mesh
Content-Type: multipart/form-data

Parámetros:
  - mesh: archivo .glb (malla 3D)
  - image: archivo PNG/JPG (para textura)

Respuesta:
  {
    "job_id": "uuid-string",
    "status": "processing"
  }
```

### Pipeline Completo
```
POST /api/full-pipeline
Content-Type: multipart/form-data

Parámetros:
  - image: archivo PNG/JPG
  - model: "turbo" | "fast" | "full"
  - quality: preset
  - texture_image: imagen para textura (opcional)

Respuesta:
  {
    "job_id": "uuid-string",
    "status": "processing"
  }
```

### Status y Descarga
```
GET /api/jobs/{job_id}
Respuesta:
  {
    "job_id": "uuid",
    "status": "processing" | "completed" | "failed",
    "progress": 0-100,
    "files": ["mesh.glb", "preview.png", ...],
    "error": "mensaje si failed"
  }

GET /api/jobs/{job_id}/files/{filename}
Descarga archivo de salida

GET /api/jobs/{job_id}/stream  (SSE)
Stream de eventos en tiempo real
```

### Información
```
GET /api/health
GET /api/models
GET /api/presets
```

---

## 6. Bugs Conocidos y Notas

### Solucionados
- ✅ **text-to-3D crash (CORREGIDO)**: El pipeline de shape no aceptaba `text=`. Ahora se genera imagen intermedia primero.
- ✅ **HunyuanDiT reemplazado por Flux.1-schnell**: HunyuanDiT tenía mala adherencia al prompt (resultados "toon", ignoraba estilos). Flux.1-schnell ofrece calidad muy superior y libera VRAM al terminar.
- ✅ **pip shebang roto**: El shebang de `.venv/bin/pip` apunta a path antiguo (`vision` en vez de `vision3d`). Usar siempre `.venv/bin/python -m pip`.
- ✅ **Text-to-3D sin textura + suelo no deseado**: Corregido con rembg + enhanced prompt + paint pipeline completo.
- ✅ **Job text-to-3D exitoso**: Job `a145c468` completado: HunyuanDiT → shape (226k verts) → decimación (10k faces) → mesh.glb con textura.

### Pendientes de Verificación
- ⚠️ **Debug print**: `custom_rasterizer/render.py` tiene un `print()` debug pendiente de eliminar en site-packages de glorfindel.

### Notas Operacionales
- El servidor corre como **root** vía systemd
- Código fuente se edita desde Mac local
- Después de `git pull` en glorfindel: `sudo systemctl daemon-reload && sudo systemctl restart vision3d`
- Web UI está embebida en `server.py` (HTML inline en endpoint raíz `/`)
- Web UI tiene 2 tabs: "Image → 3D", "Text → 3D"
- Tab "Text → 3D" incluye input de prompt, presets, controles de modelo/octree/steps/faces
- Resultados GLB se muestran en visor 3D interactivo (`<model-viewer>` de Google, orbit controls)
- Text-to-3D muestra también la imagen de referencia generada

---

## 7. Integración con Otros Proyectos

```
vision3d (API FastAPI en glorfindel)
    ↑
    │ HTTP REST (port 8000)
    ↓
maya-mcp (servidor MCP, cliente de Vision3D API)
    ↑
    │ MCP protocol
    ↓
fpt-mcp (consola Qt, orquesta maya-mcp + ShotGrid)
    ↑
    │ Claude Code CLI
    ↓
Mac local (~/Claude_projects/)
```

**Ubicaciones**:
- `vision3d`: `/home/flame/ai-studio/vision3d/` (glorfindel)
- `maya-mcp`: `~/Claude_projects/maya-mcp-project/` (Mac)
- `fpt-mcp`: `~/Claude_projects/fpt-mcp/` (Mac)

**Flujo típico**:
1. Usuario envía request a `maya-mcp` (MCP server)
2. `maya-mcp` llama `POST /api/generate-shape` (Vision3D)
3. Vision3D procesa en GPU, devuelve `job_id`
4. `maya-mcp` espera a `GET /api/jobs/{job_id}` (polling o SSE)
5. Resultado descargado y cargado en Maya

---

## 8. Desarrollo y Debugging

### Workflows Típicos

**Después de git pull en glorfindel**:
```bash
# SSH en glorfindel
ssh glorfindel

# Reload y restart del servicio
sudo systemctl daemon-reload && sudo systemctl restart vision3d

# Ver logs
sudo journalctl -u vision3d -f
```

**Desarrollo local (en glorfindel)**:
```bash
cd ~/ai-studio/vision3d/
source .venv/bin/activate

# Test directo
.venv/bin/python server.py --port 8000

# Con opciones
.venv/bin/python server.py --host 127.0.0.1 --port 8000 --reload
```

### Edición de Código
- Editar en Mac: `~/Claude_projects/vision3d/`
- Sincronizar: `git push` → `git pull` en glorfindel
- Restart: `sudo systemctl restart vision3d`

### Environment Variables (en glorfindel)
```bash
GPU_API_KEY=""           # API key (vacío = acceso abierto)
GPU_MODELS_DIR="./hf_models"
GPU_WORK_DIR="./output"
GPU_VISION_DIR="."
```

### Estructura del Código (server.py)

**Funciones principales**:
- `_get_shape_pipeline(model)` — Cargar shape model
- `_get_paint_pipeline()` — Cargar paint model
- `_get_t2i_pipeline()` — Cargar text-to-image model
- `_run_shape_from_image()` — Procesar image-to-3D
- `_run_shape_from_text()` — Procesar text-to-3D (con intermediate image)
- `_run_texture()` — Aplicar textura a malla
- `_run_full_pipeline()` — Shape + texture en un paso
- `_decimate_mesh()` — Reducir número de polígonos

**Job Management**:
- `_new_job(type, detail)` — Crear nuevo job con UUID
- `_job_log(job_id, msg)` — Registrar evento
- `_job_done(job_id, output_dir, files)` — Marcar como completado
- `_job_fail(job_id, error)` — Marcar como fallido

---

## 9. Puntos Clave para Recordar

1. **text-to-3D necesita imagen intermedia**: No pases `text=` directamente al shape pipeline.
2. **Shape models son stateful**: Cargarlos es caro. Se cachean en `_SHAPE_PIPELINES`.
3. **GPU tiene límite de memoria**: Preset `ultra` puede requerir 24GB+ VRAM.
4. **Systemd maneja el ciclo de vida**: No usar `kill -9`, usar `systemctl restart`.
5. **Logs son importantes**: `journalctl -u vision3d -f` es tu mejor amigo para debugging.
6. **API key es opcional**: En desarrollo puedes dejar `GPU_API_KEY=""` (acceso abierto).
7. **Output está en `GPU_WORK_DIR`**: Por defecto `./output`, organizado por `job_id`.
8. **Web UI está en raíz**: Acceder a `http://glorfindel:8000/` para UI HTML interactiva.

---

## 10. Referencias Rápidas

### Archivo Principal
- `/home/flame/ai-studio/vision3d/server.py`

### Directorios Críticos
- Modelos: `~/ai-studio/vision3d/hf_models/`
- Código Hunyuan: `~/ai-studio/vision3d/Hunyuan3D-2/`
- Output: `~/ai-studio/vision3d/output/{job_id}/`

### Comandos Frecuentes
```bash
# Restart del servidor
sudo systemctl restart vision3d

# Ver logs en tiempo real
sudo journalctl -u vision3d -f -n 100

# Verificar estado
systemctl status vision3d

# Ver config del servicio
cat /etc/systemd/system/vision3d.service
```

### URLs Locales (en glorfindel)
- **API**: `http://localhost:8000/api/...`
- **Web UI**: `http://localhost:8000/`
- **Health**: `http://localhost:8000/api/health`

---

## 11. Workflow de Despliegue (para Claude)

Después de editar archivos en Cowork/Mac, el usuario debe ejecutar estos comandos manualmente. Claude debe proporcionarlos siempre que haya cambios pendientes de desplegar.

### Mac — Commit y push de los tres repos
```bash
# vision3d
cd ~/Claude_projects/vision3d
git add -A && git commit -m "descripción del cambio" && git push

# maya-mcp
cd ~/Claude_projects/maya-mcp-project
git add -A && git commit -m "descripción del cambio" && git push

# fpt-mcp
cd ~/Claude_projects/fpt-mcp
git add -A && git commit -m "descripción del cambio" && git push
```

### glorfindel — Pull y restart (solo si se cambió vision3d)
```bash
cd ~/ai-studio/vision3d && git pull && sudo systemctl restart vision3d && sudo journalctl -u vision3d -f -n 20
```

**Notas**:
- Claude NO tiene acceso SSH a glorfindel — siempre dar comandos al usuario
- Los repos Mac están en `~/Claude_projects/`, NO `~/Developer/`
- El repo en glorfindel está en `~/ai-studio/vision3d/`, NO `~/ai-studio/vision/`
- Usar `.venv/bin/python -m pip` en glorfindel (el shebang de pip está roto)

---

**Última actualización**: 2026-03-30
**Versión del servidor**: FastAPI + Hunyuan3D-2 (Tencent)

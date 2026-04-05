# HANDOFF — vision3d

**Last updated**: 2026-04-05 — Phase 8.2: robustness fixes in server.py
**Completitud**: ~96% (server funcional, 6 bugs corregidos, sin tests)

---

## 1. Mapa del componente

**Vision3D** es un servidor FastAPI de inferencia GPU para generación 3D.

| Aspecto | Detalle |
|---|---|
| **Tipo** | REST API server (FastAPI + uvicorn) |
| **Lenguaje** | Python 3 (single file: `server.py`, 1646 lines) |
| **Dónde corre** | `glorfindel` (Rocky Linux, NVIDIA RTX 3090 24 GB) |
| **Puerto** | 8000 (configurable via `--port`) |
| **Dependencias ML** | Hunyuan3D-2 (Tencent), SDXL Turbo (Stability AI), pyfqmr, trimesh, rembg |
| **Dependencias Python** | FastAPI, uvicorn, torch, diffusers, transformers, accelerate, Pillow, numpy |
| **Requiere** | NVIDIA GPU con 16+ GB VRAM, CUDA, `custom_rasterizer` compilado |

### Relación con el ecosistema

```
Claude Code / Claude Desktop (Mac)
    │
    │ MCP protocol (stdio)
    ↓
maya-mcp (FastMCP, Mac)
    │
    │ HTTP REST (httpx → port 8000)
    ↓
vision3d (FastAPI, glorfindel GPU)
    │
    │ Hunyuan3D-2 / SDXL Turbo / pyfqmr
    ↓
GPU inference (RTX 3090, 24 GB VRAM)
```

vision3d es el **backend GPU** de maya-mcp. No hay dependencia de código compartido — solo HTTP. maya-mcp actúa como cliente REST vía `httpx`.

---

## 2. Estado actual

### Endpoints (11 REST + Web UI)

| Endpoint | Método | Estado | Notas |
|---|---|---|---|
| `/api/generate-shape` | POST | ✅ Funcional | Image → mesh.glb |
| `/api/generate-text` | POST | ✅ Funcional | Text → image → shape → texture (pipeline completo) |
| `/api/texture-mesh` | POST | ✅ Funcional | Mesh + image → textured mesh |
| `/api/generate-full` | POST | ✅ Funcional | Image → shape → decimate → texture |
| `/api/jobs/{id}` | GET | ✅ Funcional | Job polling con log y download links |
| `/api/jobs/{id}/files/{f}` | GET | ✅ Funcional | File download |
| `/api/jobs/{id}/stream` | GET | ✅ Funcional | SSE real-time progress |
| `/api/health` | GET | ✅ Funcional | GPU info, models, text_to_3d availability |
| `/api/models` | GET | ✅ Funcional | Available models con weights en disco |
| `/api/presets` | GET | ✅ Funcional | Quality preset configurations |
| `/` | GET | ✅ Funcional | Web UI embebida (2 tabs: Image/Text → 3D) |

### Pipelines

| Pipeline | Modelos usados | Output files |
|---|---|---|
| generate-shape | Shape (turbo/fast/full) | `mesh.glb` |
| generate-text | SDXL Turbo → Shape → Paint | `text2img_reference.png`, `textured.glb`, `mesh_uv.obj`, `texture_baked.png`*, `mesh.glb` |
| texture-mesh | Paint | `textured.glb`, `mesh_uv.obj`, `texture_baked.png`* |
| generate-full | Shape → Paint | `textured.glb`, `mesh_uv.obj`, `texture_baked.png`*, `mesh.glb` |

\* `texture_baked.png` es condicional — la extracción de textura puede fallar silenciosamente.

### Tests
**No hay tests** (ni automatizados ni manuales).

### Compliance

| Archivo | Estado |
|---|---|
| LICENSE | ✅ MIT |
| NOTICE.md | ✅ Creado |
| README.md | ✅ Con disclaimer WARNING |
| CLAUDE.md | ✅ Auditado contra server.py (Phase 6) |

### Rutas hardcodeadas

**Sin rutas hardcodeadas** en `server.py`. Todos los paths se derivan de:
- `Path(__file__).resolve().parent` (relativo al script)
- Environment variables: `GPU_MODELS_DIR`, `GPU_WORK_DIR`, `GPU_VISION_DIR`, `GPU_API_KEY`

| Archivo | Ruta | Uso |
|---|---|---|
| `setup.sh` | `/etc/systemd/system/vision3d.service` | systemd service (Linux estándar) |
| `CLAUDE.md` | `/home/flame/ai-studio/vision3d/`, `~/Claude_projects/maya-mcp/`, `~/Claude_projects/fpt-mcp/` | Documentación de referencia |

---

## 3. Hallazgos de la auditoría de server.py (Phase 6)

### 3.1 Discrepancias corregidas en CLAUDE.md

Estas discrepancias entre el CLAUDE.md anterior y el código real de server.py fueron **corregidas** en el commit `fe9a39e`:

**Parámetros incorrectos documentados**:
- `quality` → en realidad se llama `preset` (todos los endpoints)
- `steps` → en realidad se llama `num_inference_steps`
- `seed` → **no existe** en ningún endpoint del servidor
- `texture_image` en generate-full → **no existe**
- `prompt` en generate-text → se llama `text_prompt`

**Content-Type incorrecto**:
- generate-text documentado como `application/json` → en realidad usa `multipart/form-data` (Form fields)

**Parámetros no documentados** (ahora añadidos):
- `output_subdir` (todos los POST endpoints, default `"0"`)
- `octree_resolution` (generate-shape, generate-text, generate-full)
- `target_faces` default de generate-full es `50000` (no `0` como generate-shape)

**Secciones ausentes** (ahora creadas):
- Formatos de response para todos los endpoints
- SSE event types: `log`, `status`, `done`, `error` con formato y poll interval (2s)
- HTTP error codes por endpoint (404, 409, 401)
- Job system: lifecycle in-memory, job types, background execution
- Decimation system: adaptive curvature-aware con fallback chain
- Environment variables table con defaults
- CLI arguments (`--host`, `--port`, `--reload`)
- Authentication details (`secrets.compare_digest`)

**Información incompleta** (ahora expandida):
- Architecture: de "4 endpoints" a "11 endpoints + Web UI"
- Text-to-3D: aclarado que produce output texturizado (no solo shape)
- Paint pipeline: documentado que NO se descarga después de usar
- VRAM cleanup: secuencia exacta documentada
- Output files: listados por pipeline con condicionales

### 3.2 Bugs y problemas potenciales en server.py

Estos problemas se encontraron en el código y están **documentados en CLAUDE.md §12** pero **NO corregidos** (pendientes para Fase 8):

| # | Severidad | Problema | Estado |
|---|---|---|---|
| 1 | **Media** | `_resolve_preset` ignora `target_faces` explícito cuando `preset` está presente | ✅ **FIXED Phase 8.1** — `target_faces > 0` ahora prevalece sobre preset |
| 2 | **Media** | Jobs acumulan en memoria sin TTL/cleanup — memory leak en server long-running | ✅ **FIXED Phase 8.2** — `_cleanup_old_jobs()` cada 5 min, TTL 1h, max 100 jobs |
| 3 | **Alta** | Sin protección de concurrencia GPU — múltiples jobs simultáneos causan OOM | ✅ **FIXED Phase 8.1** — `asyncio.Semaphore(1)` + HTTP 429 |
| 4 | **Media** | SSE auth bypass desde Web UI — EventSource no soporta custom headers | ✅ **FIXED Phase 8.2** — `_verify_api_key()` acepta query param fallback |
| 5 | **Alta** | Sin sanitización de `output_subdir` — path traversal potencial | ✅ **FIXED Phase 8.1** — `_validate_output_subdir()` + HTTP 400. **FIXED Phase 8.2** — `_validate_upload()` MIME type + size limit (50 MB) |
| 6 | **Baja** | Colisión de `output_subdir` — dos jobs con mismo subdir sobrescriben archivos | ❌ Pendiente |
| 7 | **Baja** | `texture_baked.png` falla silenciosamente | ❌ Pendiente |

### 3.3 Qué se corrigió vs. qué queda pendiente

| Categoría | Corregido (CLAUDE.md) | Pendiente (código) |
|---|---|---|
| Documentación de endpoints | ✅ Parámetros, types, defaults, responses | — |
| Documentación de SSE | ✅ Event types, format, poll interval | — |
| Documentación de jobs | ✅ Lifecycle, types, output structure | — |
| Documentación de decimation | ✅ Algorithm, fallback chain | — |
| Bug `_resolve_preset` | ✅ Documentado en §12 | ❌ Fix en server.py |
| Memory leak jobs | ✅ Documentado en §12 | ❌ Implementar TTL/cleanup |
| Concurrencia GPU | ✅ Documentado en §12 | ❌ Implementar job queue/semaphore |
| SSE auth bypass | ✅ Documentado en §12 | ❌ Leer key de query param como fallback |
| Path traversal | ✅ Documentado en §12 | ❌ Sanitizar output_subdir |
| Input validation | ✅ Documentado en §12 | ❌ MIME type check, size limits |
| Tests | — | ❌ No existen |

---

## 4. Decisiones pendientes

1. **¿Implementar job queue con semaphore?** — Actualmente nada previene OOM por jobs concurrentes. Opciones: `asyncio.Semaphore(1)`, queue con Redis, o simplemente rechazar jobs si uno está running.

2. **¿Añadir TTL a jobs?** — Los jobs crecen en memoria indefinidamente. Opciones: cleanup periódico (e.g., 1 hora), max jobs en memoria, persistencia en SQLite.

3. **¿Fix del SSE auth?** — El EventSource del browser no soporta custom headers. Opciones: leer `x_api_key` también de query param, usar token en URL path, o session cookies.

4. **¿Sanitización de output_subdir?** — Path traversal es una vulnerabilidad real. Opciones: whitelist de caracteres, `Path.resolve()` + check que está dentro de WORK_DIR, o UUIDs forzados.

5. **¿Tests automatizados?** — No hay ninguno. Mínimo viable: health endpoint, presets endpoint, unit tests de `_resolve_preset`, mock tests de job lifecycle.

---

## 5. Próximos pasos (candidato a Fase 8)

**Prioridad alta** (bugs con impacto en producción):
- [x] Fix `_resolve_preset` — permitir override de `target_faces` con preset **(Phase 8.1)**
- [x] Sanitizar `output_subdir` contra path traversal **(Phase 8.1)**
- [x] Implementar semaphore para concurrencia GPU (1 job a la vez) **(Phase 8.1)**

**Prioridad media** (robustez):
- [x] Fix SSE auth — leer `x_api_key` de query param como fallback **(Phase 8.2)**
- [x] Job cleanup — TTL o max jobs en memoria **(Phase 8.2)**
- [x] Input validation — MIME type check en uploads **(Phase 8.2)**

**Prioridad baja** (mejoras):
- [ ] Tests automatizados (health, presets, _resolve_preset, job lifecycle)
- [ ] Señalizar ausencia de `texture_baked.png` en response
- [ ] Documentar Web UI query param `?key=` en README

---

## Última actualización: 2026-04-05 — Phase 8.2: fixes de prioridad media (robustez)

### Phase 8.2 — Cambios aplicados
- **Fix 4 (Bug #4)**: SSE auth fallback — `_verify_api_key()` ahora acepta segundo arg `query_api_key`. Endpoint `stream_job()` lee `x_api_key` de Header y de Query param (alias). Web UI ya enviaba `?x_api_key=` — ahora el server lo respeta.
- **Fix 5 (Bug #2)**: Job cleanup — constantes `JOB_TTL_SECONDS=3600`, `JOB_CLEANUP_INTERVAL=300`, `MAX_JOBS=100`. Función `_cleanup_old_jobs()` elimina jobs completed/failed >1h. Background task `_job_cleanup_loop()` registrado en `lifespan()`. `_check_max_jobs()` en los 4 POST endpoints rechaza con HTTP 503 si >100 jobs.
- **Fix 6 (Bug #5 parcial)**: Input validation — helper `_validate_upload()` verifica Content-Type y tamaño (max 50 MB). Tipos permitidos: imágenes (`image/png`, `image/jpeg`, `image/webp`), meshes (`model/gltf-binary`, `application/octet-stream`). Aplicado en `generate-shape`, `texture-mesh`, `generate-full`. Devuelve HTTP 400.
- **CLAUDE.md §12**: Bugs 2, 4, 5 marcados como resueltos con descripción del fix.

### Phase 8.1 — Cambios aplicados
- **Fix 1 (Bug #1)**: `_resolve_preset()` — cambiado `if target_faces >= 0 and not preset:` → `if target_faces > 0:`. Ahora `target_faces` explícito (>0) prevalece sobre el valor del preset.
- **Fix 3 (Bug #3)**: GPU semaphore — `asyncio.Semaphore(1)` a nivel de módulo. `_check_gpu_available()` en los 4 POST endpoints devuelve HTTP 429 si la GPU está ocupada. `_run_in_background()` mantiene el semaphore durante toda la inferencia.
- **Fix 5 (Bug #5)**: Path traversal — nueva función `_validate_output_subdir()` rechaza `..`, `/`, `\` y verifica con `Path.resolve()` que el path queda dentro de WORK_DIR. Aplicada en los 4 POST endpoints. Devuelve HTTP 400.
- **CLAUDE.md §12**: Bugs 1, 3, 5 marcados como resueltos con descripción del fix.

### Historial anterior
- 2026-04-05 — Phase 6: auditoría exhaustiva de server.py, reescritura completa de CLAUDE.md (360 insertions, 111 deletions), 7 bugs documentados

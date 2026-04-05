# HANDOFF — vision3d

**Last updated**: 2026-04-05 — Phase 8.3: final fixes + automated tests
**Completitud**: ~99% (server funcional, 7/7 bugs corregidos, 20 tests)

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
**20 tests automatizados** (Phase 8.3) — `tests/test_server.py` con pytest.
- 12 unit tests: `_validate_output_subdir`, `_resolve_preset`, `_resolve_output_subdir`, `_cleanup_old_jobs`, `_validate_upload`
- 8 endpoint tests: health, presets, models, generate-shape (422/400), GPU busy 429, auth 401, SSE auth query param
- No requieren GPU, CUDA ni modelos ML — usan mocks para torch/trimesh/diffusers
- Ejecutar: `python -m pytest tests/ -v`

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
| 6 | **Baja** | Colisión de `output_subdir` — dos jobs con mismo subdir sobrescriben archivos | ✅ **FIXED Phase 8.3** — `_resolve_output_subdir()` reemplaza default "0" con UUID8 |
| 7 | **Baja** | `texture_baked.png` falla silenciosamente | ✅ **FIXED Phase 8.3** — `_job_log()` advierte al cliente cuando la extracción falla |

### 3.3 Estado de correcciones (actualizado Phase 8.3)

| Categoría | Documentación | Código |
|---|---|---|
| Documentación de endpoints | ✅ Parámetros, types, defaults, responses | — |
| Documentación de SSE | ✅ Event types, format, poll interval | — |
| Documentación de jobs | ✅ Lifecycle, types, output structure | — |
| Documentación de decimation | ✅ Algorithm, fallback chain | — |
| Bug `_resolve_preset` | ✅ §12 | ✅ Phase 8.1 |
| Memory leak jobs | ✅ §12 | ✅ Phase 8.2 |
| Concurrencia GPU | ✅ §12 | ✅ Phase 8.1 |
| SSE auth bypass | ✅ §12 | ✅ Phase 8.2 |
| Path traversal | ✅ §12 | ✅ Phase 8.1 |
| Input validation | ✅ §12 | ✅ Phase 8.2 |
| output_subdir collision | ✅ §12 | ✅ Phase 8.3 |
| texture_baked.png silencioso | ✅ §12 | ✅ Phase 8.3 |
| Tests | — | ✅ Phase 8.3 (20 tests) |

---

## 4. Decisiones ejecutadas (Phase 8)

1. **GPU semaphore** — `asyncio.Semaphore(1)` con HTTP 429 si busy. Sin queue Redis (innecesario para un solo usuario).
2. **Job TTL** — Cleanup periódico cada 5 min, TTL 1h, max 100 jobs en memoria. Sin persistencia SQLite.
3. **SSE auth** — Query param fallback en `_verify_api_key()`. Sin session cookies.
4. **output_subdir** — Path traversal: regex + `Path.resolve()`. Colisión: UUID8 para default `"0"`.
5. **Tests** — 20 tests con pytest, mocks de torch/ML, sin GPU necesaria.

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
- [x] Tests automatizados (20 tests: helpers + endpoints) **(Phase 8.3)**
- [x] Señalizar ausencia de `texture_baked.png` en response **(Phase 8.3)**
- [x] Colisión de `output_subdir` — unique UUID para default **(Phase 8.3)**
- [ ] Documentar Web UI query param `?key=` en README

---

## Changelog

### Phase 8.3 (2026-04-05) — Fixes finales + tests
- **Fix 7 (Bug #6)**: output_subdir collision — `_resolve_output_subdir()` reemplaza default "0" con UUID8.
- **Fix 8 (Bug #7)**: texture_baked.png silencioso — `_job_log()` advierte al cliente cuando la extracción falla.
- **Tests**: 20 tests en `tests/test_server.py` — 12 unit + 8 endpoint. No requieren GPU.
- **CLAUDE.md §12**: Bugs 6 y 7 marcados como resueltos.

### Phase 8.2 (2026-04-05) — Robustez
- **Fix 4 (Bug #4)**: SSE auth fallback — `_verify_api_key()` ahora acepta segundo arg `query_api_key`. Endpoint `stream_job()` lee `x_api_key` de Header y de Query param (alias). Web UI ya enviaba `?x_api_key=` — ahora el server lo respeta.
- **Fix 5 (Bug #2)**: Job cleanup — constantes `JOB_TTL_SECONDS=3600`, `JOB_CLEANUP_INTERVAL=300`, `MAX_JOBS=100`. Función `_cleanup_old_jobs()` elimina jobs completed/failed >1h. Background task `_job_cleanup_loop()` registrado en `lifespan()`. `_check_max_jobs()` en los 4 POST endpoints rechaza con HTTP 503 si >100 jobs.
- **Fix 6 (Bug #5 parcial)**: Input validation — helper `_validate_upload()` verifica Content-Type y tamaño (max 50 MB). Tipos permitidos: imágenes (`image/png`, `image/jpeg`, `image/webp`), meshes (`model/gltf-binary`, `application/octet-stream`). Aplicado en `generate-shape`, `texture-mesh`, `generate-full`. Devuelve HTTP 400.
- **CLAUDE.md §12**: Bugs 2, 4, 5 marcados como resueltos con descripción del fix.

### Phase 8.1 (2026-04-05) — Seguridad + estabilidad
- **Fix 1 (Bug #1)**: `_resolve_preset()` — cambiado `if target_faces >= 0 and not preset:` → `if target_faces > 0:`. Ahora `target_faces` explícito (>0) prevalece sobre el valor del preset.
- **Fix 3 (Bug #3)**: GPU semaphore — `asyncio.Semaphore(1)` a nivel de módulo. `_check_gpu_available()` en los 4 POST endpoints devuelve HTTP 429 si la GPU está ocupada. `_run_in_background()` mantiene el semaphore durante toda la inferencia.
- **Fix 5 (Bug #5)**: Path traversal — nueva función `_validate_output_subdir()` rechaza `..`, `/`, `\` y verifica con `Path.resolve()` que el path queda dentro de WORK_DIR. Aplicada en los 4 POST endpoints. Devuelve HTTP 400.
- **CLAUDE.md §12**: Bugs 1, 3, 5 marcados como resueltos con descripción del fix.

### Phase 6 (2026-04-05) — Auditoría
- Auditoría exhaustiva de server.py, reescritura completa de CLAUDE.md (360 insertions, 111 deletions), 7 bugs documentados

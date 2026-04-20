# Changelog

All notable changes to **Vision3D** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release is also tagged in git and published as a [GitHub Release](https://github.com/abrahamADSK/vision3d/releases).

---

## [Unreleased]

## [v1.6.3] — 2026-04-20

### Added
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is now set via
  `os.environ.setdefault()` at the top of `server.py`, before the first
  torch import — matches the existing `PYTORCH_ENABLE_MPS_FALLBACK` pattern.
  Complements the v1.6.2 paint-unload fix by reducing allocator
  fragmentation so the reclaimed VRAM actually comes back as contiguous
  free memory. `setdefault` lets operators override via the systemd
  `Environment=` directive without code changes. Startup log prints the
  effective value.

## [v1.6.2] — 2026-04-20

### Fixed
- **VRAM leak on idle** — the paint pipeline (~14 GB) stayed cached across
  requests with no release path, so VRAM remained held for days even when
  no jobs were running. On glorfindel this blocked sibling GPU services
  (Ollama, ComfyUI) on the shared 24 GB RTX 3090. Added
  `_unload_paint_pipeline()` mirroring the shape/t2i unload pattern, plus a
  background task `_paint_idle_loop()` that auto-unloads after
  `PAINT_IDLE_SECONDS` of inactivity (default 900 s = 15 min, overridable
  via env). Protected by `_gpu_semaphore` so the unload never races an
  in-flight job. Lifespan also unloads on clean shutdown.
- `_clear_device_cache()` now calls `torch.cuda.ipc_collect()` in addition
  to `empty_cache()` — without this, VRAM cached inside PyTorch's IPC
  segments was not actually returned to the CUDA driver. `ipc_collect` is
  wrapped in try/except because not every PyTorch build exposes it.

### Added
- Startup log hints that `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  be set in the systemd unit to reduce allocator fragmentation across
  load/unload cycles. Not auto-set — operator decision.
- `.concepts.yml` — new `paint_pipeline_unload` concept with three
  `claim_verifies` invariants guarding the unload function, the idle loop,
  and the lifespan registration. Prevents silent regression of the VRAM fix.
- 5 new unit tests in `tests/test_server.py` (TestPaintUnload) covering
  no-op on unloaded state, state clear on loaded, tolerance of pipelines
  that don't implement `.to()`, constants sanity, and env override.
- `scripts/verify_concepts.py` — `--accept-current-as-truth` + `--i-reviewed-diff` double-flag escape hatch (REPORT MODE ONLY). When both flags are passed, the runner inspects every failing invariant and prints a human-readable "would update \<mirror\>" line describing what a hypothetical writer mode would change, then exits 0 without touching any file. Single-flag usage is rejected with exit code 2 by design — the double-flag requirement prevents accidental drift acceptance. Intended for repos that drifted while dormant and need a one-shot review before flipping `strict: true`. Writer mode is deferred to a future pass with explicit user sign-off. Chat 44 ultraplan Q5.

## [v1.6.1] — 2026-04-20

### Added
- `.concepts.yml` — `github_release_per_tag` concept with the
  `every_v1plus_tag_has_github_release` invariant. Enforces that every
  `vX.Y.Z` tag (v1.0.0+) has a matching published GitHub Release;
  `gh release list` is the oracle. Pre-1.0 tags excluded (pre-release
  noise). Ecosystem-wide policy introduced in Chat 45. Backfilled
  GitHub Releases for the three historical tags (`v1.0.0`, `v1.1.0`,
  `v1.2.0`) that were missing one.
- `scripts/invariant_types.py` + `scripts/verify_concepts.py` — synced
  to the ecosystem-canonical version from Chat 45 (additive extensions:
  `ast_decorator_functions.name_kwarg`, `ast_enum_values`,
  `ast_decorator_kwarg` back-compat alias). Byte-identical with
  flame-mcp, maya-mcp, fpt-mcp.

## [v1.6.0] — 2026-04-17

### Added
- **Cross-cutting concept registry** (`.concepts.yml`) with 10 machine-
  checkable invariants protecting the public REST surface, stable preset
  and shape-model names, timing-safe API-key validation, retired-concept
  absence (FlashVDM, attention_slicing, PAINT_MAX_FACES), single-install-
  script rule, and release cadence.
- **`scripts/verify_concepts.py` + `scripts/invariant_types.py`** — the
  invariant engine (8 invariant types, 11 source types, stdlib + PyYAML
  only). Shared surface with flame-mcp.
- **`.pre-commit-config.yaml`** wiring `verify_concepts.py` to every
  commit via the pre-commit framework. Soft-launch mode (`strict: false`)
  for two weeks, then flip to strict.
- **Release-cadence invariants** (ecosystem rule, Chat 44):
  `commits_since_last_tag_under_threshold` (warn at 10, fail at 30 commits
  past last tag; fail if tag > 30 days old with pending work) and
  `changelog_sections_match_tags` (bidirectional between CHANGELOG version
  headings and git tags).
- **`install.sh --check` dry-run flag** that verifies Python version, platform,
  and pip.conf without creating a venv, installing dependencies, or registering
  a service. Useful for CI pre-merge checks to catch installation issues early.
- **`NOTICE.md`** — Third-party component attribution file. Documents the
  non-permissive licenses that users must be aware of (Tencent Hunyuan
  Community License for Hunyuan3D-2, Stability AI Non-Commercial for SDXL
  Turbo, GPL-3.0 for PyMeshLab) and lists permissive components with
  upstream pointers. Not legal advice, but prevents accidental commercial
  misuse by downstream users.
- **Explicit PyTorch installation in `install.sh` (Step 5a)**. The script
  now installs `torch==2.6.0` with a platform-specific index — the
  `+cu124` wheel from `download.pytorch.org` for CUDA, the vanilla PyPI
  wheel for macOS/MPS/CPU — **before** installing `requirements.txt`.
  Idempotent: skips the install when the target version is already
  present. Previously `torch` was pulled transitively by `diffusers`,
  which did not guarantee the 2.6.0 version or the correct CUDA wheel on
  a fresh Linux install. `requirements.txt` now has an explanatory
  comment documenting why `torch` is intentionally absent from it.

### Changed
- **`install.sh` now fails fast on critical dependency installation errors**
  (torch, requirements.txt, Hunyuan3D-2 editable) instead of logging and
  continuing, which previously led to cascading failures in later steps.
  Exit code is now non-zero at the end of the script if any error was
  recorded, so CI and automation can reliably detect failures.
- **`install.sh` mesh extras split into critical and optional groups.**
  Critical (`pymeshlab`, `xatlas` — required for texture baking) abort on
  failure; optional (`pygltflib`, `opencv-python`, `einops`, `omegaconf`)
  print warnings only.
- **`install.sh` STEP 8 import verification is now resilient to Python
  subprocess crashes**: captures the exit code explicitly and validates the
  JSON output before parsing, aborting with a clear message instead of
  failing opaquely.
- **`install.sh` systemd registration now checks exit codes and shows error
  summary before exiting on failure**, ensuring the user sees which step
  failed even when `set -e` would normally suppress output.
- **MPS default shape model is now `fast`** (was `full`). On Apple Silicon
  the `full` model (18.8 GB) is rarely available locally; `fast` (4.93 GB)
  is the only MPS-compatible shape model downloaded by default. Affects
  `_get_shape_pipeline()`, `get_models()`, and the embedded Web UI fallback.
  Re-applies the logic of `894b613` which was silently lost during the
  Chat 34 rollback to `fdc11a8`.
- **`install.sh` systemd unit now includes `--local` in `ExecStart`.**
  Previously relied on `EOFError` fallback when the interactive prompt ran
  without a TTY; the explicit flag eliminates a silent hang window at boot.
- **Embedded Web UI fallback aligned with backend.** When the model list is
  empty, the Web UI now falls back to `fast` on MPS (was `full`).

### Fixed
- **Root cause of texture paint hang on complex subjects identified and
  removed**: `enable_flashvdm()` (Lightning Vecset Decoder, ICCV 2025)
  produced degenerate topology (~570 verts / ~460k faces, 0.0012 verts/face
  ratio) on non-trivial subjects such as multi-part or concave meshes,
  choking the `custom_rasterizer` in the paint stage. FlashVDM was removed
  from `server.py` during the rollback to `fdc11a8` and must not be
  re-introduced without validating against a complex mesh.

### Consolidated earlier this cycle
- **Consolidated `install.sh` and `setup.sh` into a single entry point.**
  `install.sh` now handles both code installation and service registration.
  Flags: `--no-service` (install only), `--service-only` (register service
  only, assumes venv exists), `--uninstall` (remove the registered service,
  keeps code), `--help`. Default behaviour (no flags) runs both phases in
  order. The previous two-script design was easy to forget — `install.sh`
  alone left the machine "installed but not running" as a silent partial
  state. This change makes the happy path a single command.

### Removed
- `setup.sh` — merged into `install.sh`. Use `install.sh --service-only` for
  the equivalent of the old `setup.sh` behaviour.

---

## [v1.5.1] — 2026-04-09

### Changed
- **No hardcoded infrastructure in the repo.** The interactive prompt's
  default remote host now reads from the env var
  `VISION3D_DEFAULT_REMOTE_HOST` (a personal setting in the operator's
  shell). If unset, the prompt has no default and the hostname is
  required. Removes the previously hardcoded default that leaked a
  specific machine name into the public repo.
- All documentation (`README.md`, `CLAUDE.md`, `HANDOFF.md`,
  `CHANGELOG.md`, `install.sh` user-facing strings) genericized to
  `<gpu-host>` placeholders. No machine name, IP address, or concrete
  API key value appears anywhere in the repo.
- SSE docstring placeholder `?x_api_key=KEY` → `?x_api_key=<value>`
  for consistency.

### Added
- New env var `VISION3D_DEFAULT_REMOTE_HOST` documented in `README.md`
  and `CLAUDE.md`.

**Commit**: `c174569`.

---

## [v1.5.0] — 2026-04-09

### Added
- **Interactive backend selection at startup.** `server.py` now asks
  `Run locally? [y/N]:` and, if not, `Remote host [<default>]:`. The
  default for the remote prompt is read from the env var
  `VISION3D_DEFAULT_REMOTE_HOST` (a personal env var, never committed) —
  no hostname is hardcoded in the repo. The prompt validates the remote
  with `GET http://{host}:8000/api/health` (5 s timeout) and loops until
  a healthy host responds.
- **Remote proxy mode.** When a remote host is selected, the local
  process becomes a thin HTTP façade: every `/api/*` endpoint is
  forwarded via `httpx` async to the remote — generation
  (`generate-shape`, `generate-text`, `texture-mesh`, `generate-full`),
  polling (`/api/jobs/{id}`), file downloads, and the SSE progress
  stream (byte-for-byte pass-through). Zero local job state, zero local
  GPU load. Job IDs are owned by the remote.
- **CLI flags** to skip the prompt: `--local` (force local backend) and
  `--remote HOST` (force remote). `--reload` implicitly forces local
  because the interactive prompt is incompatible with uvicorn's reload
  loop.
- **Environment variables** `VISION3D_REMOTE_HOST`, `VISION3D_REMOTE_PORT`
  (default `8000`), and `VISION3D_REMOTE_KEY` (optional API key
  forwarded to the remote — defaults to passing through the inbound
  `x-api-key` from the local client). The selected mode is persisted
  via env var so uvicorn workers and `--reload` inherit it.
- `/api/health` now reports the active mode (`mode: "local" | "remote"`,
  plus `remote_host` / `remote_port` when applicable).

### Changed
- `server.py` header docstring updated to document both modes, the new
  endpoints, the CLI flags, and the `VISION3D_REMOTE_*` env vars.
- `README.md` gained a "Backend selection" section with prompt example,
  CLI flags, façade explanation, and env var table.
- `CLAUDE.md` architecture summary, env var table, and CLI reference
  bumped to reflect dual-mode operation.

### Compatibility
- Local mode is fully backwards-compatible. Existing deployments
  (the dedicated GPU host, Mac MPS) keep working unchanged.
- All 21 existing automated tests pass without modification — the new
  code paths are guarded by the env var, so test runs default to local.

**Commits**: `f77b8e7` (feat), `c5e80e6` (docs).

---

## [v1.4.0] — 2026-04-09

### Added
- **Real SSE progress events.** `server.py` now emits a `progress` event
  on `/api/jobs/{id}/stream` with JSON `{stage, progress, message}`,
  fed by a new `_job_progress()` helper, a diffusion-step callback
  (`_make_diffusion_callback`), and a `_paint_progress_scope()` for the
  paint pipeline. SSE poll cadence dropped from 2 s → 0.5 s.
- **Configurable job TTL with disk cleanup.** New `_delete_job_output()`
  removes the on-disk artifacts of expired jobs in addition to the
  in-memory entry.

### Changed
- SSE event vocabulary: existing `log` / `status` / `done` events keep
  the same format; the new `progress` event is additive.

**Commit**: `a055fce`.

---

## [v1.3.0] — 2026-04-09

### Added
- **Apple Silicon (MPS) texturing support.** Paint pipeline now runs on
  MPS via conditional `from_pretrained` and improved error logging
  (integrates the `hunyuan3d-mac` fork).
- **Cross-platform installer.** `setup.sh` installs a `LaunchAgent` on
  macOS and a `systemd` service on Linux, with `--uninstall` support.
- **Web UI**: localhost auth bypass, default model now `full`, preset
  dropdown fix.

### Documentation
- Apple Silicon section now documents the verified stack: PyTorch 2.6.0
  pinned, `torch >= 2.7` MPS regression warning, fp16-on-MPS caveats
  for the multiview UNet and the SD x4 upscaler, C++ extension rebuild
  reminder, and the M4 Pro texturing benchmark (626 s, 23.8 GB peak).

**Commit**: `ee2e949`.

---

## [v1.2.0] — 2026-04-08

### Documentation
- Apple Silicon (MPS) setup guide added to `README.md`.
- Quick Start section rewritten to use the new `install.sh` (auto-detects
  CUDA / MPS / CPU).

**Commit**: `63721cc`.

---

## [v1.1.0] — 2026-04-08

### Added
- **MPS support for Apple Silicon.** Device auto-detection
  (CUDA → MPS → CPU). On MPS the server uses the base model
  `hunyuan3d-dit-v2-0` with `variant='fp16'` and `use_safetensors=True`
  (turbo unavailable on Mac because the required scheduler is missing).

**Commit**: `4addd40`.

---

## [v1.0.0] — 2026-04-07

### Added
- First tagged release.
- 11 REST endpoints + embedded Web UI (image-to-3D, text-to-3D,
  texture painting, full pipeline, job polling, SSE progress, file
  downloads, health, models, presets).
- Phase 8 security and stability fixes:
  - GPU concurrency protection via `asyncio.Semaphore(1)` + HTTP 429.
  - `output_subdir` path-traversal sanitization.
  - SSE auth fallback via query param (EventSource cannot send headers).
  - Job cleanup loop (TTL 1 h, max 100 jobs in memory).
  - Upload validation (MIME type + 50 MB size limit).
  - `output_subdir` collision avoidance via UUID8 default.
- 20 automated pytest tests (helpers + endpoints, no GPU required).

**Commit**: `567796b`.

---

[v1.5.1]: https://github.com/abrahamADSK/vision3d/releases/tag/v1.5.1
[v1.5.0]: https://github.com/abrahamADSK/vision3d/releases/tag/v1.5.0
[v1.4.0]: https://github.com/abrahamADSK/vision3d/releases/tag/v1.4.0
[v1.3.0]: https://github.com/abrahamADSK/vision3d/releases/tag/v1.3.0
[v1.2.0]: https://github.com/abrahamADSK/vision3d/releases/tag/v1.2.0
[v1.1.0]: https://github.com/abrahamADSK/vision3d/releases/tag/v1.1.0
[v1.0.0]: https://github.com/abrahamADSK/vision3d/releases/tag/v1.0.0

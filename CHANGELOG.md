# Changelog

All notable changes to **Vision3D** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release is also tagged in git and published as a [GitHub Release](https://github.com/abrahamADSK/vision3d/releases).

---

## [Unreleased]

### Changed
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

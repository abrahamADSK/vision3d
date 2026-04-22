# Changelog

All notable changes to **Vision3D** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release is also tagged in git and published as a [GitHub Release](https://github.com/abrahamADSK/vision3d/releases).

---

## [Unreleased]

## [1.6.6] — 2026-04-22

### Added
- `scripts/invariant_types.py` — `_write_subset` handler registered in
  WRITERS (Phase C + D, Chat 48). Covers two shapes:
  - `b_source.type: anchor_list` (no `item_pattern`) → appends missing
    items as `- \`<item>\`` bullets inside the concept block.
  - `b_source.type: file_regex_matches` with YAML opt-in
    `b_source.writer.line_template: "- \`{item}\` — ..."` → appends
    template-formatted lines after the last existing regex match
    (default) or at end_of_file. `{item}` placeholder required.
  Other shapes report `WRITER UNSUPPORTED`. Enables `/propagate-change`
  Path A to auto-fix the common subset-drift pattern (code grows,
  README mirror falls behind) across 4 repos.
- `.github/workflows/ci.yml` — Codecov coverage upload step
  (`codecov/codecov-action@v4`), gated to `matrix.python-version ==
  '3.12'` so the ~10s upload runs once per PR. `fail_ci_if_error:
  false` so Codecov outages don't block merges. Public-repo tokenless
  flow; env block left for future opt-in to authenticated upload.

### Fixed
- `scripts/invariant_types.py` — `version_match` handler now honors
  opt-in `tolerate_release_in_progress: true` parameter. When set, a
  drift of the form `a == CUT_RELEASE_VERSION != b` is tolerated so
  `cut-release.sh` can commit a version bump before the matching git
  tag exists, without weakening strict-mode guarantees for non-release
  commits. vision3d uses `VERSION` file (no pyproject), so the
  opt-in is not currently applied here but the canonical remains
  byte-identical with the other 3 repos.

## [1.6.5] — 2026-04-22

### Added
- `.github/workflows/ci.yml` — GitHub Actions CI workflow. Four blocking
  jobs: pytest (3.9/3.10/3.11/3.12 matrix — glorfindel runs 3.9), ruff
  lint, mypy, verify_concepts on every push + PR. Pytest coverage
  reported inline via `--cov=<pkg> --cov-report=term`. Heavy runtime
  deps (torch, diffusers, trimesh, etc.) are mocked via
  `tests/conftest.py`, so CI only installs the FastAPI stack.
- `.github/workflows/pr-review.yml` — automated Claude PR review
  (`anthropics/claude-code-action@v1`). Byte-identical across the 4
  ecosystem repos; canonical at `~/Projects/pr-review-canonical.yml`.
  Uses `claude_code_oauth_token` (Max/Pro subscription, not API key).
  Requires the Claude Code GitHub App installed on the repo + workflow
  permission `id-token: write` + `--model claude-sonnet-4-6` pin so the
  OAuth token (Sonnet-scoped) works against the default-Opus action.
- `scripts/verify_concepts.py --write` — WRITER MODE (Chat 46). Requires
  the triple flag `--accept-current-as-truth --i-reviewed-diff --write`.
  Dispatches to per-type writers in `invariant_types.py::WRITERS`.
  Currently supports `tool_count` and `review_expiry`; other types
  report `WRITER UNSUPPORTED`. No auto-commit.
- `scripts/cut-release.sh` — ecosystem-shared release orchestrator.
  Validates clean tree + semver arg + non-empty `[Unreleased]`, edits
  CHANGELOG + `VERSION` (vision3d has no `pyproject.toml`; the
  `VERSION` file is the release anchor), commits with
  `CUT_RELEASE_VERSION=X.Y.Z` so the `changelog_tag_sync` invariant
  tolerates the transient pre-commit drift, tags, pushes, and creates
  a GitHub release. Byte-identical across the 4 MCP-ecosystem repos.
- `VERSION` — plain-text version anchor (`1.6.4`) read by
  `scripts/verify_concepts.py` and bumped by `scripts/cut-release.sh`.
- `scripts/invariant_types.py` — new `changelog_tag_sync` handler
  replaces the previous `subset`-based `changelog_tag_coherence`.
  Release-in-progress tolerance anchored to env `CUT_RELEASE_VERSION`
  OR the `VERSION` file content.
- `scripts/invariant_types.py` — `ast_dict_keys` canonical (Chat 47)
  now reads `ast.AnnAssign` in addition to `ast.Assign`, so typed-dict
  declarations resolve correctly. Synced byte-identical across 4 repos.
- `scripts/invariant_types.py` — `version_match` canonical (Chat 48)
  honors opt-in `tolerate_release_in_progress: true`. When set, a
  drift of the form `a == CUT_RELEASE_VERSION != b` is tolerated so
  `cut-release.sh` can commit a version bump before the matching git
  tag exists, without weakening strict-mode guarantees for non-release
  commits.
- `scripts/verify_concepts.py` — `ci_skip: true` flag on individual
  invariants + auto-skip of `review_expiry` under `GITHUB_ACTIONS`
  (Chat 47). Keeps dev-side invariants active via pre-commit while CI
  runs stay green without shipping `~/Projects/.external_versions.yml`
  or broad `gh` auth.

### Changed
- `.concepts.yml` — `strict: false → true`. The pre-commit hook now
  blocks commits on any unresolved invariant drift instead of only
  reporting it. Ecosystem-wide flip on 2026-04-20 (Chat 46), unblocked
  by the `changelog_tag_sync` release-in-progress tolerance.
- CI pipeline cleanup (Chat 47): ruff baseline cleared (all warnings
  fixed, job flipped to blocking), mypy job flipped to blocking (baseline
  already clean). Both jobs now block merge rather than
  `continue-on-error: true`.

### Fixed
- `install.sh` — replaced two unused-counter `for i in 1..5` loops with
  `for _ in 1..5` to silence shellcheck SC2034 (Chat 46).
- `.github/workflows/ci.yml` — install `pytest-asyncio` so the two
  async-decorated tests actually run (Chat 47).
- `.github/workflows/pr-review.yml` — added `id-token: write` workflow
  permission (Chat 48). Without it the action errored with "Unable to
  get ACTIONS_ID_TOKEN_REQUEST_URL env variable" in 3 retries.
- `.github/workflows/pr-review.yml` — pinned `--model claude-sonnet-4-6`
  via `claude_args` (Chat 48). OAuth tokens from `claude setup-token`
  are scoped to Sonnet on Max/Pro; the action's default model (Opus
  after v1.0.100) returned `401 Invalid bearer token` against those
  credentials (see anthropics/claude-code-action#584).

## [v1.6.4] — 2026-04-20

### Fixed
- Python 3.9 compatibility — the `_paint_pipeline_last_use: float | None`
  annotation added in v1.6.2 used PEP 604 union syntax (Python 3.10+),
  which crashed glorfindel (Rocky Linux 9 ships Python 3.9) with
  `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`
  at module load. Replaced with `Optional[float]` (already imported in
  this file) so v1.6.2's paint-unload code actually deploys on glorfindel.

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

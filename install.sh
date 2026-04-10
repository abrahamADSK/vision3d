#!/usr/bin/env bash
# =============================================================================
# install.sh — vision3d installer
# =============================================================================
# Automates the full installation of vision3d from a clean clone.
# Safe to run multiple times (idempotent).
#
# What this script does:
#   1. Verifies Python 3.10+ is available
#   2. Detects platform: CUDA (Linux + nvidia-smi) vs MPS (macOS arm64) vs CPU
#   3. Fixes pip.conf if "user = true" is set (incompatible with venvs)
#   4. Creates a virtual environment in .venv/ if not present
#   5. Installs dependencies from requirements.txt
#   6. Installs the Hunyuan3D-2 fork in editable mode (pip install -e)
#   7. Installs mesh-processing extras (pymeshlab, xatlas, etc.)
#   8. Verifies critical imports and torch backend
#   9. Creates .env from .env.example if not present
#  10. Prints an installation summary
#
# What this script does NOT do (see setup.sh instead):
#   - systemd service creation
#   - API key generation
#   - Deployment to a remote GPU host
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# Tested on: macOS (arm64), Rocky Linux (CUDA)
# Requires:  Python 3.10+
# =============================================================================

# =============================================================================
# IMPORTANT: C++ extension rebuild after torch version changes
# =============================================================================
# PyTorch native extensions (custom_rasterizer, differentiable_renderer) are
# compiled against a specific torch ABI version. After any torch upgrade or
# downgrade, these extensions MUST be rebuilt:
#
#   1. Clean build directories:
#        rm -rf <extension>/build/ <extension>/*.egg-info/
#   2. Reinstall in editable mode:
#        pip install -e <extension-path> --no-build-isolation --force-reinstall
#
# Symptom when not rebuilt: "Symbol not found: __ZNK3c10*" at import time,
# or silent hangs in paint pipeline.
#
# The compiled .so files require torch to be imported first to resolve
# libc10.so and other shared libraries. server.py handles this by importing
# torch before any extension that depends on it. On Linux (systemd), the
# LD_LIBRARY_PATH is set explicitly in setup.sh to include torch/lib/.
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[vision3d]${RESET} $*"; }
success() { echo -e "${GREEN}[vision3d] ✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}[vision3d] ⚠${RESET} $*"; }
error()   { echo -e "${RED}[vision3d] ✗${RESET} $*" >&2; }

# ── Track results for the final summary ──────────────────────────────────────
STEPS_OK=()
STEPS_WARN=()
STEPS_ERR=()

# ── Resolve repo root (works even if script is called from another directory) ─
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
REQUIREMENTS="${REPO_ROOT}/requirements.txt"
ENV_FILE="${REPO_ROOT}/.env"
ENV_EXAMPLE="${REPO_ROOT}/.env.example"

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  vision3d — installation${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
info "Repo root : ${REPO_ROOT}"
info "Venv dir  : ${VENV_DIR}"
echo ""

# =============================================================================
# STEP 1 — Verify Python 3.10+
# =============================================================================
info "Step 1/9 — Checking Python version..."

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver_ok=$("$candidate" -c "
import sys
ok = sys.version_info >= (3, 10)
print('ok' if ok else 'no')
")
        if [[ "$ver_ok" == "ok" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    error "Python 3.10 or newer is required but was not found."
    error "Install it via your package manager or from https://python.org"
    STEPS_ERR+=("Python 3.10+ not found — installation aborted")
    exit 1
fi

PY_VERSION=$("$PYTHON_BIN" --version 2>&1)
success "Found ${PY_VERSION} at $(command -v "$PYTHON_BIN")"
STEPS_OK+=("Python version check passed (${PY_VERSION})")

# =============================================================================
# STEP 2 — Detect platform: CUDA vs MPS vs CPU
# =============================================================================
info "Step 2/9 — Detecting platform..."

PLATFORM="cpu"

if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$GPU_NAME" ]]; then
        PLATFORM="cuda"
        success "CUDA platform detected — ${GPU_NAME}"
        STEPS_OK+=("Platform: CUDA (${GPU_NAME})")
    fi
elif [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    PLATFORM="mps"
    success "MPS platform detected — macOS arm64 (Apple Silicon)"
    STEPS_OK+=("Platform: MPS (macOS arm64)")
fi

if [[ "$PLATFORM" == "cpu" ]]; then
    warn "No GPU detected — falling back to CPU (inference will be slow)"
    STEPS_WARN+=("Platform: CPU only — no CUDA or MPS detected")
fi

# =============================================================================
# STEP 3 — Fix pip.conf if "user = true" (incompatible with venvs)
# =============================================================================
info "Step 3/9 — Checking pip.conf..."

PIP_CONF=""
if [[ "$(uname)" == "Darwin" ]]; then
    PIP_CONF="${HOME}/.config/pip/pip.conf"
else
    PIP_CONF="${HOME}/.config/pip/pip.conf"
fi

if [[ -f "$PIP_CONF" ]] && grep -q "^user.*=.*true" "$PIP_CONF" 2>/dev/null; then
    info "Found 'user = true' in ${PIP_CONF} — commenting it out (breaks venvs)..."
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' 's/^\(user.*=.*true\)/#\1  # disabled by vision3d install.sh/' "$PIP_CONF"
    else
        sed -i 's/^\(user.*=.*true\)/#\1  # disabled by vision3d install.sh/' "$PIP_CONF"
    fi
    success "Commented out 'user = true' in pip.conf"
    STEPS_OK+=("pip.conf fixed — 'user = true' commented out")
else
    success "pip.conf OK (no 'user = true' conflict)"
    STEPS_OK+=("pip.conf OK")
fi

# =============================================================================
# STEP 4 — Create virtual environment
# =============================================================================
info "Step 4/9 — Setting up virtual environment..."

if [[ -d "${VENV_DIR}" && -f "${VENV_DIR}/bin/python" ]]; then
    success "Virtual environment already exists at .venv/ — skipping creation"
    STEPS_OK+=("Venv already present — skipped creation")
else
    info "Creating virtual environment at ${VENV_DIR}..."
    "$PYTHON_BIN" -m venv "${VENV_DIR}"
    success "Virtual environment created"
    STEPS_OK+=("Venv created at .venv/")
fi

# Point to the venv's python/pip from here on
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# Upgrade pip silently first
"${VENV_PIP}" install --quiet --upgrade pip

# =============================================================================
# STEP 5 — Install requirements.txt
# =============================================================================
info "Step 5/9 — Installing dependencies from requirements.txt..."

if [[ ! -f "${REQUIREMENTS}" ]]; then
    error "requirements.txt not found at ${REQUIREMENTS}"
    STEPS_ERR+=("requirements.txt missing — dependencies not installed")
else
    if "${VENV_PIP}" install --quiet -r "${REQUIREMENTS}"; then
        success "requirements.txt installed"
        STEPS_OK+=("Dependencies installed from requirements.txt")
    else
        error "pip install -r requirements.txt failed"
        STEPS_ERR+=("requirements.txt installation failed — check pip output")
    fi
fi

# =============================================================================
# STEP 6 — Install Hunyuan3D-2 fork (editable mode)
# =============================================================================
info "Step 6/9 — Looking for Hunyuan3D-2 fork..."

HY3D_DIR=""

if [[ "$PLATFORM" == "mps" || "$(uname)" == "Darwin" ]]; then
    # macOS: look for the Mac-specific fork
    SEARCH_DIRS=(
        "${REPO_ROOT}/../hunyuan3d-mac"
        "${HOME}/Projects/hunyuan3d-mac"
        "${HOME}/Claude_projects/hunyuan3d-mac"
    )
    FORK_LABEL="hunyuan3d-mac (macOS fork)"
else
    # Linux: look for the upstream Hunyuan3D-2
    SEARCH_DIRS=(
        "${REPO_ROOT}/../Hunyuan3D-2"
        "${HOME}/ai-studio/Hunyuan3D-2"
        "${HOME}/Claude_projects/Hunyuan3D-2"
    )
    FORK_LABEL="Hunyuan3D-2"
fi

for dir in "${SEARCH_DIRS[@]}"; do
    resolved=$(cd "$dir" 2>/dev/null && pwd) || continue
    if [[ -f "${resolved}/setup.py" && -d "${resolved}/hy3dgen" ]]; then
        HY3D_DIR="$resolved"
        break
    fi
done

if [[ -n "$HY3D_DIR" ]]; then
    info "Found ${FORK_LABEL} at ${HY3D_DIR}"
    if "${VENV_PIP}" install --quiet -e "${HY3D_DIR}"; then
        success "Hunyuan3D-2 fork installed (pip install -e ${HY3D_DIR})"
        STEPS_OK+=("Hunyuan3D-2 fork installed from ${HY3D_DIR}")
    else
        error "pip install -e ${HY3D_DIR} failed"
        STEPS_ERR+=("Hunyuan3D-2 fork installation failed")
    fi
else
    warn "${FORK_LABEL} not found in search paths:"
    for dir in "${SEARCH_DIRS[@]}"; do
        warn "  ${dir}"
    done
    warn "Clone it and re-run install.sh, or install manually: pip install -e /path/to/fork"
    STEPS_WARN+=("Hunyuan3D-2 fork not found — install manually")
fi

# =============================================================================
# STEP 7 — Install mesh-processing extras
# =============================================================================
info "Step 7/9 — Installing mesh-processing extras..."

MESH_EXTRAS=(pymeshlab xatlas pygltflib opencv-python einops omegaconf)

if "${VENV_PIP}" install --quiet "${MESH_EXTRAS[@]}"; then
    success "Mesh extras installed: ${MESH_EXTRAS[*]}"
    STEPS_OK+=("Mesh extras installed (${#MESH_EXTRAS[@]} packages)")
else
    warn "Some mesh extras failed to install — non-fatal but some features may be limited"
    STEPS_WARN+=("Mesh extras partially failed — check pip output")
fi

# =============================================================================
# STEP 8 — Verify critical imports
# =============================================================================
info "Step 8/9 — Verifying imports..."

IMPORT_RESULTS=$("${VENV_PYTHON}" - <<'PYEOF'
import json, sys

results = []

# ── torch + backend ──
try:
    import torch
    results.append({"name": "torch", "ok": True, "detail": torch.__version__})

    # Check backend matches platform
    import platform
    uname = platform.system()
    arch = platform.machine()

    if uname == "Linux":
        if torch.cuda.is_available():
            results.append({"name": "torch.cuda", "ok": True, "detail": torch.cuda.get_device_name(0)})
        else:
            results.append({"name": "torch.cuda", "ok": False, "detail": "CUDA not available on Linux"})
    elif uname == "Darwin" and arch == "arm64":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            results.append({"name": "torch.mps", "ok": True, "detail": "MPS available"})
        else:
            results.append({"name": "torch.mps", "ok": False, "detail": "MPS not available on arm64 Mac"})
    else:
        results.append({"name": "torch.backend", "ok": True, "detail": "CPU-only (expected)"})
except ImportError as e:
    results.append({"name": "torch", "ok": False, "detail": str(e)})

# ── Core server deps ──
for mod, label in [
    ("fastapi", "fastapi"),
    ("pymeshlab", "pymeshlab"),
    ("trimesh", "trimesh"),
    ("PIL", "Pillow"),
    ("diffusers", "diffusers"),
]:
    try:
        __import__(mod)
        results.append({"name": label, "ok": True, "detail": ""})
    except ImportError as e:
        results.append({"name": label, "ok": False, "detail": str(e)})

# ── Hunyuan3D fork ──
try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    results.append({"name": "hy3dgen.shapegen", "ok": True, "detail": ""})
except ImportError as e:
    results.append({"name": "hy3dgen.shapegen", "ok": False, "detail": str(e)})
except Exception as e:
    # May fail for CUDA-specific reasons on Mac — import exists but init fails
    results.append({"name": "hy3dgen.shapegen", "ok": True, "detail": f"importable (init warning: {e})"})

json.dump(results, sys.stdout)
PYEOF
)

# Read the summary line for the step tracker
IMPORT_SUMMARY=$(echo "$IMPORT_RESULTS" | "${VENV_PYTHON}" -c "
import json, sys
results = json.load(sys.stdin)
ok = sum(1 for r in results if r['ok'])
fail = sum(1 for r in results if not r['ok'])
print(f'{ok} ok, {fail} failed')
")

# Display individual results with colours
echo "$IMPORT_RESULTS" | "${VENV_PYTHON}" -c "
import json, sys
results = json.load(sys.stdin)
for r in results:
    name = r['name']
    detail = f\" ({r['detail']})\" if r['detail'] else ''
    if r['ok']:
        print(f'  \033[0;32m✓\033[0m {name}{detail}')
    else:
        print(f'  \033[0;31m✗\033[0m {name}{detail}')
"

FAIL_COUNT=$(echo "$IMPORT_RESULTS" | "${VENV_PYTHON}" -c "
import json, sys
results = json.load(sys.stdin)
print(sum(1 for r in results if not r['ok']))
")

if [[ "$FAIL_COUNT" == "0" ]]; then
    success "All imports verified (${IMPORT_SUMMARY})"
    STEPS_OK+=("Import verification passed (${IMPORT_SUMMARY})")
else
    warn "Some imports failed (${IMPORT_SUMMARY})"
    STEPS_WARN+=("Import verification: ${IMPORT_SUMMARY}")
fi

# =============================================================================
# STEP 9 — Create .env from .env.example
# =============================================================================
info "Step 9/9 — Checking .env..."

if [[ -f "${ENV_FILE}" ]]; then
    success ".env already exists — skipping"
    STEPS_OK+=(".env already present")
elif [[ -f "${ENV_EXAMPLE}" ]]; then
    cp "${ENV_EXAMPLE}" "${ENV_FILE}"
    success ".env created from .env.example"
    STEPS_OK+=(".env created from template")
else
    warn ".env.example not found — create .env manually"
    STEPS_WARN+=(".env not created — no template found")
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  Installation summary${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

if [[ ${#STEPS_OK[@]} -gt 0 ]]; then
    for msg in "${STEPS_OK[@]}"; do
        echo -e "  ${GREEN}✓${RESET} ${msg}"
    done
fi

if [[ ${#STEPS_WARN[@]} -gt 0 ]]; then
    echo ""
    for msg in "${STEPS_WARN[@]}"; do
        echo -e "  ${YELLOW}⚠${RESET} ${msg}"
    done
fi

if [[ ${#STEPS_ERR[@]} -gt 0 ]]; then
    echo ""
    for msg in "${STEPS_ERR[@]}"; do
        echo -e "  ${RED}✗${RESET} ${msg}"
    done
fi

echo ""

# ── Next steps ───────────────────────────────────────────────────────────────
if [[ ${#STEPS_ERR[@]} -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}vision3d is ready.${RESET}"
    echo ""
    echo -e "  ${BOLD}Platform:${RESET} ${PLATFORM}"
    echo ""
    echo -e "  ${BOLD}Next steps:${RESET}"
    echo -e "  1. Edit ${CYAN}.env${RESET} if you need to set GPU_API_KEY or custom paths"
    echo -e "  2. Run the server locally:"
    echo -e "     ${CYAN}.venv/bin/python server.py --port 8000${RESET}"
    echo ""
    echo -e "  ${BOLD}For GPU host deployment:${RESET}"
    echo -e "     ${CYAN}bash setup.sh${RESET}  — systemd service, API key, auto-start"
    echo ""
else
    echo -e "${RED}${BOLD}Installation completed with errors.${RESET}"
    echo -e "Review the ${RED}✗${RESET} items above and fix them before running vision3d."
    echo ""
fi

echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

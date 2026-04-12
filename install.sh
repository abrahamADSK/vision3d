#!/usr/bin/env bash
# =============================================================================
# install.sh — vision3d unified installer + service manager
# =============================================================================
# Single entry point for installing vision3d and (optionally) registering it
# as a system service. Safe to run multiple times (idempotent).
#
# Default (no flags): installs dependencies AND registers the service.
# Use flags below to run only one phase.
#
# Usage:
#   bash install.sh                 # install code + register service (full setup)
#   bash install.sh --no-service    # install code only (skip service registration)
#   bash install.sh --service-only  # register service only (assumes venv exists)
#   bash install.sh --uninstall     # remove the registered service (keeps code)
#   bash install.sh --check         # dry-run: verify python+platform+pip.conf, no install
#   bash install.sh --help          # print this help
#
# Install phase does:
#   1. Verifies Python 3.10+ is available
#   2. Detects platform: CUDA (Linux + nvidia-smi) vs MPS (macOS arm64) vs CPU
#   3. Fixes pip.conf if "user = true" is set (incompatible with venvs)
#   4. Creates a virtual environment in .venv/ if not present
#   5. Installs dependencies from requirements.txt
#   6. Installs the Hunyuan3D-2 fork in editable mode (pip install -e)
#   7. Installs mesh-processing extras (pymeshlab, xatlas, etc.)
#   8. Verifies critical imports and torch backend
#   9. Creates .env from .env.example if not present
#
# Service phase does:
#   10. Generates a random API key (saved to .api_key) if not present
#   11. Registers vision3d as a service:
#         - Linux: systemd unit (0.0.0.0:8000, runs as current user, needs sudo)
#         - macOS: LaunchAgent (127.0.0.1:8000, runs at login, no sudo)
#   12. Starts the service and verifies /api/health
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
# LD_LIBRARY_PATH is set explicitly in the unit file generated below to
# include torch/lib/.
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

hr() {
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

# ── Track results for the final summary ──────────────────────────────────────
STEPS_OK=()
STEPS_WARN=()
STEPS_ERR=()

# ── Resolve repo root (works even if script is called from another directory) ─
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${GPU_VENV:-${REPO_ROOT}/.venv}"
REQUIREMENTS="${REPO_ROOT}/requirements.txt"
ENV_FILE="${REPO_ROOT}/.env"
ENV_EXAMPLE="${REPO_ROOT}/.env.example"
WORK_DIR="${GPU_WORK_DIR:-${REPO_ROOT}/output}"
MODELS_DIR="${GPU_MODELS_DIR:-${REPO_ROOT}/hf_models}"
VISION_DIR="${GPU_VISION_DIR:-${REPO_ROOT}}"
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# Service-phase paths
PORT=8000
PLIST_LABEL="com.abraham.vision3d"
PLIST_PATH="${HOME}/Library/LaunchAgents/${PLIST_LABEL}.plist"
LOG_PATH="${HOME}/Library/Logs/vision3d.log"
SERVICE_FILE="/etc/systemd/system/vision3d.service"
API_KEY_FILE="${VISION_DIR}/.api_key"

# Shared state populated by run_install, consumed by run_service
PLATFORM=""
PYTHON_BIN=""

# ── Parse flags ──────────────────────────────────────────────────────────────
SKIP_INSTALL=0
SKIP_SERVICE=0
DO_UNINSTALL=0
DO_CHECK=0

print_help() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-service)    SKIP_SERVICE=1; shift ;;
        --service-only)  SKIP_INSTALL=1; shift ;;
        --uninstall)     DO_UNINSTALL=1; shift ;;
        --check)         DO_CHECK=1; shift ;;
        -h|--help)       print_help; exit 0 ;;
        *)               error "Unknown option: $1"; echo "Run 'bash install.sh --help' for usage."; exit 2 ;;
    esac
done

# =============================================================================
# run_install — the "install code" phase (steps 1-9)
# =============================================================================
run_install() {
    echo ""
    hr
    echo -e "${BOLD}  vision3d — installation${RESET}"
    hr
    echo ""
    info "Repo root : ${REPO_ROOT}"
    info "Venv dir  : ${VENV_DIR}"
    echo ""

    # ── STEP 1 — Verify Python 3.10+ ──────────────────────────────────────────
    info "Step 1/9 — Checking Python version..."

    PYTHON_BIN=""
    for candidate in python3 python; do
        if command -v "$candidate" &>/dev/null; then
            local ver_ok
            ver_ok=$("$candidate" -c "import sys; print('ok' if sys.version_info >= (3, 10) else 'no')")
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

    local PY_VERSION
    PY_VERSION=$("$PYTHON_BIN" --version 2>&1)
    success "Found ${PY_VERSION} at $(command -v "$PYTHON_BIN")"
    STEPS_OK+=("Python version check passed (${PY_VERSION})")

    # ── STEP 2 — Detect platform: CUDA vs MPS vs CPU ─────────────────────────
    info "Step 2/9 — Detecting platform..."

    PLATFORM="cpu"

    if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi &>/dev/null; then
        local GPU_NAME
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

    # ── STEP 3 — Fix pip.conf if "user = true" ───────────────────────────────
    info "Step 3/9 — Checking pip.conf..."

    local PIP_CONF="${HOME}/.config/pip/pip.conf"

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

    # ── STEP 4 — Create virtual environment ──────────────────────────────────
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

    # Upgrade pip silently first
    "${VENV_PIP}" install --quiet --upgrade pip

    # ── STEP 5 — Install PyTorch (platform-specific) + requirements.txt ──────
    info "Step 5/9 — Installing PyTorch and base dependencies..."

    # 5a. PyTorch must be installed FIRST with a platform-specific index.
    # requirements.txt intentionally omits torch because a single pinned
    # version cannot satisfy both CUDA 12.4 (needs +cu124 wheel from
    # download.pytorch.org) and macOS MPS (needs the vanilla PyPI wheel).
    # Transitive installation via diffusers/transformers would fetch
    # whatever torch happens to satisfy the version range — not reliable.
    local TORCH_TARGET="2.6.0"
    local TORCH_CURRENT
    TORCH_CURRENT=$("${VENV_PYTHON}" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")

    local TORCH_EXPECTED
    if [[ "$PLATFORM" == "cuda" ]]; then
        TORCH_EXPECTED="${TORCH_TARGET}+cu124"
    else
        TORCH_EXPECTED="${TORCH_TARGET}"
    fi

    if [[ "$TORCH_CURRENT" == "$TORCH_EXPECTED" ]]; then
        success "torch ${TORCH_EXPECTED} already installed — skipping"
        STEPS_OK+=("torch already at ${TORCH_EXPECTED}")
    else
        if [[ -n "$TORCH_CURRENT" ]]; then
            info "Found torch ${TORCH_CURRENT}, target is ${TORCH_EXPECTED} — reinstalling"
        else
            info "Installing torch ${TORCH_EXPECTED}..."
        fi
        local TORCH_OK=0
        if [[ "$PLATFORM" == "cuda" ]]; then
            if "${VENV_PIP}" install --quiet "torch==${TORCH_TARGET}" \
                --index-url "https://download.pytorch.org/whl/cu124"; then
                TORCH_OK=1
            fi
        else
            if "${VENV_PIP}" install --quiet "torch==${TORCH_TARGET}"; then
                TORCH_OK=1
            fi
        fi
        if (( TORCH_OK == 1 )); then
            success "torch ${TORCH_EXPECTED} installed"
            STEPS_OK+=("torch ${TORCH_EXPECTED} installed")
        else
            error "torch installation failed — Hunyuan3D-2 will not run"
            STEPS_ERR+=("torch installation failed (${TORCH_EXPECTED})")
            print_summary
            exit 1
        fi
    fi

    # 5b. Install the rest from requirements.txt (torch is already in place).
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
            print_summary
            exit 1
        fi
    fi

    # ── STEP 6 — Install Hunyuan3D-2 fork (editable mode) ────────────────────
    info "Step 6/9 — Looking for Hunyuan3D-2 fork..."

    local HY3D_DIR=""
    local FORK_LABEL=""
    local -a SEARCH_DIRS

    if [[ "$PLATFORM" == "mps" || "$(uname)" == "Darwin" ]]; then
        SEARCH_DIRS=(
            "${REPO_ROOT}/../hunyuan3d-mac"
            "${HOME}/Projects/hunyuan3d-mac"
            "${HOME}/Claude_projects/hunyuan3d-mac"
        )
        FORK_LABEL="hunyuan3d-mac (macOS fork)"
    else
        SEARCH_DIRS=(
            "${REPO_ROOT}/../Hunyuan3D-2"
            "${HOME}/ai-studio/Hunyuan3D-2"
            "${HOME}/Claude_projects/Hunyuan3D-2"
        )
        FORK_LABEL="Hunyuan3D-2"
    fi

    local dir resolved
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
            print_summary
            exit 1
        fi
    else
        warn "${FORK_LABEL} not found in search paths:"
        for dir in "${SEARCH_DIRS[@]}"; do
            warn "  ${dir}"
        done
        warn "Clone it and re-run install.sh, or install manually: pip install -e /path/to/fork"
        STEPS_WARN+=("Hunyuan3D-2 fork not found — install manually")
    fi

    # ── STEP 7 — Install mesh-processing extras ──────────────────────────────
    info "Step 7/9 — Installing mesh-processing extras..."

    local CRITICAL_MESH_EXTRAS=(pymeshlab xatlas)
    local OPTIONAL_MESH_EXTRAS=(pygltflib opencv-python einops omegaconf)

    if "${VENV_PIP}" install --quiet "${CRITICAL_MESH_EXTRAS[@]}"; then
        success "Critical mesh extras installed: ${CRITICAL_MESH_EXTRAS[*]}"
        STEPS_OK+=("Critical mesh extras installed (pymeshlab, xatlas)")
    else
        error "Critical mesh extras failed to install — texture baking will not work"
        STEPS_ERR+=("Critical mesh extras failed (pymeshlab/xatlas) — check pip output")
        print_summary
        exit 1
    fi

    if "${VENV_PIP}" install --quiet "${OPTIONAL_MESH_EXTRAS[@]}"; then
        success "Optional mesh extras installed: ${OPTIONAL_MESH_EXTRAS[*]}"
        STEPS_OK+=("Optional mesh extras installed (pygltflib, opencv-python, einops, omegaconf)")
    else
        warn "Some optional mesh extras failed to install — non-fatal but some features may be limited"
        STEPS_WARN+=("Optional mesh extras partially failed (pygltflib/opencv-python/einops/omegaconf) — check pip output")
    fi

    # ── STEP 8 — Verify critical imports ─────────────────────────────────────
    info "Step 8/9 — Verifying imports..."

    local IMPORT_RESULTS
    local import_rc
    set +e
    IMPORT_RESULTS=$("${VENV_PYTHON}" - <<'PYEOF'
import json, sys

results = []

# ── torch + backend ──
try:
    import torch
    results.append({"name": "torch", "ok": True, "detail": torch.__version__})

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

try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    results.append({"name": "hy3dgen.shapegen", "ok": True, "detail": ""})
except ImportError as e:
    results.append({"name": "hy3dgen.shapegen", "ok": False, "detail": str(e)})
except Exception as e:
    results.append({"name": "hy3dgen.shapegen", "ok": True, "detail": f"importable (init warning: {e})"})

json.dump(results, sys.stdout)
PYEOF
)
    import_rc=$?
    set -e

    if [[ $import_rc -ne 0 ]]; then
        error "STEP 8 — Python import verification process exited with code ${import_rc}"
        STEPS_ERR+=("Import verification failed — Python process crashed (exit code ${import_rc})")
        print_summary
        exit 1
    fi

    if ! echo "$IMPORT_RESULTS" | "${VENV_PYTHON}" -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
        error "STEP 8 returned invalid JSON — import verification output is corrupt"
        STEPS_ERR+=("Import verification returned invalid JSON")
        print_summary
        exit 1
    fi

    local IMPORT_SUMMARY
    IMPORT_SUMMARY=$(echo "$IMPORT_RESULTS" | "${VENV_PYTHON}" -c "
import json, sys
results = json.load(sys.stdin)
ok = sum(1 for r in results if r['ok'])
fail = sum(1 for r in results if not r['ok'])
print(f'{ok} ok, {fail} failed')
")

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

    local FAIL_COUNT
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

    # ── STEP 9 — Create .env from .env.example ───────────────────────────────
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
}

# =============================================================================
# run_service — the "register as service" phase (steps 10-12)
# =============================================================================
run_service() {
    echo ""
    hr
    echo -e "${BOLD}  vision3d — service registration${RESET}"
    hr
    echo ""
    info "Vision dir : ${VISION_DIR}"
    info "Venv       : ${VENV_DIR}"
    info "Port       : ${PORT}"
    echo ""

    # Pre-flight: verify venv exists
    if [[ ! -f "${VENV_PYTHON}" ]]; then
        error "Venv not found at ${VENV_DIR}"
        error "Run 'bash install.sh' without --service-only first to create the venv."
        STEPS_ERR+=("Service phase skipped — venv missing")
        return 1
    fi

    # Detect platform if not already set (when --service-only is used)
    local SVC_PLATFORM=""
    local HOST=""
    if [[ "$(uname)" == "Darwin" ]]; then
        SVC_PLATFORM="macos"
        HOST="127.0.0.1"
        info "Step 10/12 — macOS detected — will create LaunchAgent (${HOST}:${PORT})"
        STEPS_OK+=("Platform: macOS — LaunchAgent")
    else
        SVC_PLATFORM="linux"
        HOST="0.0.0.0"
        info "Step 10/12 — Linux detected — will create systemd service (${HOST}:${PORT})"
        STEPS_OK+=("Platform: Linux — systemd")
    fi

    # ── STEP 11 — Generate API key ───────────────────────────────────────────
    info "Step 11/12 — Generating API key..."

    local API_KEY
    if [[ -f "$API_KEY_FILE" ]]; then
        API_KEY=$(cat "$API_KEY_FILE")
        success "Using existing key from ${API_KEY_FILE}"
        STEPS_OK+=("API key already present")
    else
        API_KEY=$("${VENV_PYTHON}" -c "import secrets; print(secrets.token_urlsafe(32))")
        echo "$API_KEY" > "$API_KEY_FILE"
        chmod 600 "$API_KEY_FILE"
        success "New key saved to ${API_KEY_FILE}"
        STEPS_OK+=("API key generated")
    fi

    # ── STEP 12 — Register and start service ─────────────────────────────────
    info "Step 12/12 — Registering and starting service..."

    if [[ "$SVC_PLATFORM" == "macos" ]]; then
        register_launchagent "$HOST" "$API_KEY"
    else
        register_systemd "$HOST" "$API_KEY"
    fi

    # Store values for the final summary
    SVC_HOST_OUT="$HOST"
    SVC_API_KEY_OUT="$API_KEY"
    SVC_PLATFORM_OUT="$SVC_PLATFORM"
}

register_launchagent() {
    local HOST="$1"
    local API_KEY="$2"

    # Build environment variables from .env if it exists
    local ENV_VARS="		<key>PATH</key>
		<string>${VENV_DIR}/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
		<key>GPU_API_KEY</key>
		<string>${API_KEY}</string>"

    if [[ -f "$ENV_FILE" ]]; then
        local key value
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            if [[ "$key" == GPU_MODELS_DIR || "$key" == GPU_WORK_DIR || "$key" == GPU_VISION_DIR ]]; then
                [[ -n "$value" ]] && ENV_VARS="${ENV_VARS}
		<key>${key}</key>
		<string>${value}</string>"
            fi
        done < "$ENV_FILE"
    fi

    mkdir -p "$(dirname "${PLIST_PATH}")"

    cat > "${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>${PLIST_LABEL}</string>

	<key>ProgramArguments</key>
	<array>
		<string>${VENV_PYTHON}</string>
		<string>${VISION_DIR}/server.py</string>
		<string>--host</string>
		<string>${HOST}</string>
		<string>--port</string>
		<string>${PORT}</string>
	</array>

	<key>WorkingDirectory</key>
	<string>${VISION_DIR}</string>

	<key>EnvironmentVariables</key>
	<dict>
${ENV_VARS}
	</dict>

	<key>RunAtLoad</key>
	<true/>

	<key>KeepAlive</key>
	<true/>

	<key>StandardOutPath</key>
	<string>${LOG_PATH}</string>

	<key>StandardErrorPath</key>
	<string>${LOG_PATH}</string>
</dict>
</plist>
PLIST

    success "Created ${PLIST_PATH}"

    if launchctl print "gui/$(id -u)/${PLIST_LABEL}" &>/dev/null; then
        info "Service already loaded — reloading..."
        launchctl bootout "gui/$(id -u)/${PLIST_LABEL}" 2>/dev/null || true
        local i
        for i in 1 2 3 4 5; do
            if ! launchctl print "gui/$(id -u)/${PLIST_LABEL}" &>/dev/null; then
                break
            fi
            sleep 1
        done
    fi

    if launchctl bootstrap "gui/$(id -u)" "${PLIST_PATH}"; then
        success "LaunchAgent loaded"
        STEPS_OK+=("LaunchAgent created and loaded")
    else
        error "launchctl bootstrap failed"
        STEPS_ERR+=("LaunchAgent load failed — check: launchctl print gui/$(id -u)/${PLIST_LABEL}")
        return 1
    fi

    info "Waiting for server to start..."
    local HEALTH_OK=false
    local i
    for i in 1 2 3 4 5; do
        sleep 2
        if curl -sf "http://${HOST}:${PORT}/api/health" > /dev/null 2>&1; then
            HEALTH_OK=true
            break
        fi
    done

    if $HEALTH_OK; then
        local HEALTH_JSON
        HEALTH_JSON=$(curl -sf "http://${HOST}:${PORT}/api/health")
        success "Server is running — ${HEALTH_JSON}"
        STEPS_OK+=("Health check passed")
    else
        warn "Server not responding yet — check logs: tail -f ${LOG_PATH}"
        STEPS_WARN+=("Health check failed — server may still be starting")
    fi
}

register_systemd() {
    local HOST="$1"
    local API_KEY="$2"

    # Try to find torch/lib dynamically for LD_LIBRARY_PATH
    local TORCH_LIB
    TORCH_LIB=$("${VENV_PYTHON}" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")

    if ! sudo tee "$SERVICE_FILE" > /dev/null <<UNIT
[Unit]
Description=Vision3D — AI 3D Generation Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${VISION_DIR}
Environment=GPU_API_KEY=${API_KEY}
Environment=GPU_MODELS_DIR=${MODELS_DIR}
Environment=GPU_WORK_DIR=${WORK_DIR}
Environment=GPU_VISION_DIR=${VISION_DIR}
Environment=LD_LIBRARY_PATH=${TORCH_LIB}
ExecStart=${VENV_PYTHON} ${VISION_DIR}/server.py --host ${HOST} --port ${PORT} --local
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
    then
        error "Failed to write systemd unit (sudo denied or disk full?)"
        STEPS_ERR+=("systemd unit write failed — sudo denied or disk full?")
        print_summary; exit 1
    fi

    success "Created ${SERVICE_FILE}"

    if ! sudo systemctl daemon-reload; then
        error "systemctl daemon-reload failed"
        STEPS_ERR+=("systemctl daemon-reload failed")
        print_summary; exit 1
    fi

    if ! sudo systemctl enable --now vision3d.service; then
        error "systemctl enable --now vision3d.service failed"
        STEPS_ERR+=("systemctl enable --now vision3d.service failed")
        print_summary; exit 1
    fi

    success "systemd service enabled and started"
    STEPS_OK+=("systemd service created and started")

    sleep 2
    if curl -sf "http://localhost:${PORT}/api/health" > /dev/null 2>&1; then
        success "Server is running"
        STEPS_OK+=("Health check passed")
    else
        warn "Server not responding yet — check: sudo journalctl -u vision3d -f"
        STEPS_WARN+=("Health check failed — server may still be starting")
    fi
}

# =============================================================================
# run_check — dry-run: verifies python, platform, and pip.conf only
# Runs steps 1-3 without creating a venv or installing anything.
# =============================================================================
run_check() {
    echo ""
    hr
    echo -e "${BOLD}  vision3d — pre-flight check (dry-run)${RESET}"
    hr
    echo ""
    info "Repo root : ${REPO_ROOT}"
    echo ""

    # ── STEP 1 — Verify Python 3.10+ ──────────────────────────────────────────
    info "Step 1/3 — Checking Python version..."

    PYTHON_BIN=""
    for candidate in python3 python; do
        if command -v "$candidate" &>/dev/null; then
            local ver_ok
            ver_ok=$("$candidate" -c "import sys; print('ok' if sys.version_info >= (3, 10) else 'no')")
            if [[ "$ver_ok" == "ok" ]]; then
                PYTHON_BIN="$candidate"
                break
            fi
        fi
    done

    if [[ -z "$PYTHON_BIN" ]]; then
        error "Python 3.10 or newer is required but was not found."
        STEPS_ERR+=("Python 3.10+ not found")
        return
    fi

    local PY_VERSION
    PY_VERSION=$("$PYTHON_BIN" --version 2>&1)
    success "Found ${PY_VERSION} at $(command -v "$PYTHON_BIN")"
    STEPS_OK+=("Python version check passed (${PY_VERSION})")

    # ── STEP 2 — Detect platform ──────────────────────────────────────────────
    info "Step 2/3 — Detecting platform..."

    PLATFORM="cpu"

    if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi &>/dev/null; then
        local GPU_NAME
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
        warn "No GPU detected — would fall back to CPU"
        STEPS_WARN+=("Platform: CPU only — no CUDA or MPS detected")
    fi

    # ── STEP 3 — Check pip.conf ───────────────────────────────────────────────
    info "Step 3/3 — Checking pip.conf..."

    local PIP_CONF="${HOME}/.config/pip/pip.conf"

    if [[ -f "$PIP_CONF" ]] && grep -q "^user.*=.*true" "$PIP_CONF" 2>/dev/null; then
        warn "Found 'user = true' in ${PIP_CONF} — this would be commented out on full install"
        STEPS_WARN+=("pip.conf has 'user = true' — will be fixed on real install")
    else
        success "pip.conf OK (no 'user = true' conflict)"
        STEPS_OK+=("pip.conf OK")
    fi
}

# =============================================================================
# run_uninstall — remove the registered service only (keeps code/venv)
# =============================================================================
run_uninstall() {
    echo ""
    hr
    echo -e "${BOLD}  vision3d — uninstall service${RESET}"
    hr
    echo ""

    if [[ "$(uname)" == "Darwin" ]]; then
        info "Removing LaunchAgent..."
        if launchctl bootout "gui/$(id -u)/${PLIST_LABEL}" 2>/dev/null; then
            success "Service stopped"
        else
            warn "Service was not loaded (already stopped?)"
        fi
        if [[ -f "${PLIST_PATH}" ]]; then
            rm "${PLIST_PATH}"
            success "Removed ${PLIST_PATH}"
        else
            warn "Plist not found at ${PLIST_PATH}"
        fi
    else
        info "Removing systemd service..."
        if sudo systemctl disable --now vision3d.service 2>/dev/null; then
            success "Service stopped and disabled"
        else
            warn "Service was not active (already removed?)"
        fi
        if [[ -f "${SERVICE_FILE}" ]]; then
            sudo rm "${SERVICE_FILE}"
            sudo systemctl daemon-reload
            success "Removed ${SERVICE_FILE}"
        else
            warn "Unit file not found at ${SERVICE_FILE}"
        fi
    fi

    echo ""
    echo -e "${GREEN}${BOLD}vision3d service removed.${RESET}"
    echo -e "  Code, venv, and models are untouched. Re-register anytime with: bash install.sh --service-only"
    echo ""
}

# =============================================================================
# print_summary — the final "what was done" block, shared by both phases
# =============================================================================
print_summary() {
    echo ""
    hr
    echo -e "${BOLD}  Summary${RESET}"
    hr
    echo ""

    if [[ ${#STEPS_OK[@]} -gt 0 ]]; then
        local msg
        for msg in "${STEPS_OK[@]}"; do
            echo -e "  ${GREEN}✓${RESET} ${msg}"
        done
    fi

    if [[ ${#STEPS_WARN[@]} -gt 0 ]]; then
        echo ""
        local msg
        for msg in "${STEPS_WARN[@]}"; do
            echo -e "  ${YELLOW}⚠${RESET} ${msg}"
        done
    fi

    if [[ ${#STEPS_ERR[@]} -gt 0 ]]; then
        echo ""
        local msg
        for msg in "${STEPS_ERR[@]}"; do
            echo -e "  ${RED}✗${RESET} ${msg}"
        done
    fi

    echo ""

    if [[ ${#STEPS_ERR[@]} -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}vision3d is ready.${RESET}"
        echo ""

        if [[ -n "${SVC_HOST_OUT:-}" ]]; then
            # Service was registered — print service info
            echo -e "  ${BOLD}API Key:${RESET}  ${SVC_API_KEY_OUT}"
            echo ""
            if [[ "${SVC_PLATFORM_OUT}" == "macos" ]]; then
                echo -e "  ${BOLD}Server:${RESET}   http://${SVC_HOST_OUT}:${PORT}"
                echo -e "  ${BOLD}Logs:${RESET}     tail -f ${LOG_PATH}"
                echo ""
                echo -e "  ${BOLD}Manage:${RESET}"
                echo -e "    ${CYAN}launchctl stop ${PLIST_LABEL}${RESET}        — stop"
                echo -e "    ${CYAN}launchctl start ${PLIST_LABEL}${RESET}       — start"
                echo -e "    ${CYAN}bash install.sh --uninstall${RESET}          — remove service"
            else
                local HOSTNAME_VAL
                HOSTNAME_VAL=$(hostname)
                echo -e "  ${BOLD}Server:${RESET}   http://${HOSTNAME_VAL}:${PORT}"
                echo -e "  ${BOLD}Logs:${RESET}     sudo journalctl -u vision3d -f"
                echo ""
                echo -e "  ${BOLD}Manage:${RESET}"
                echo -e "    ${CYAN}sudo systemctl stop vision3d${RESET}         — stop"
                echo -e "    ${CYAN}sudo systemctl restart vision3d${RESET}      — restart"
                echo -e "    ${CYAN}bash install.sh --uninstall${RESET}          — remove service"
                echo ""
                echo -e "  ${BOLD}Maya-mcp config:${RESET}"
                echo -e "    ${CYAN}GPU_API_URL=http://${HOSTNAME_VAL}:${PORT}${RESET}"
                echo -e "    ${CYAN}GPU_API_KEY=${SVC_API_KEY_OUT}${RESET}"
            fi
        else
            # Install-only path — no service registered
            echo -e "  ${BOLD}Platform:${RESET} ${PLATFORM}"
            echo ""
            echo -e "  ${BOLD}Next steps:${RESET}"
            echo -e "  1. Edit ${CYAN}.env${RESET} if you need custom paths"
            echo -e "  2. Run the server locally:"
            echo -e "     ${CYAN}.venv/bin/python server.py --port 8000${RESET}"
            echo ""
            echo -e "  ${BOLD}To register as a service later:${RESET}"
            echo -e "     ${CYAN}bash install.sh --service-only${RESET}"
        fi
        echo ""
    else
        echo -e "${RED}${BOLD}Completed with errors.${RESET}"
        echo -e "Review the ${RED}✗${RESET} items above and fix them before running vision3d."
        echo ""
    fi

    hr
    echo ""
}

# =============================================================================
# Main dispatcher
# =============================================================================

if [[ $DO_UNINSTALL -eq 1 ]]; then
    run_uninstall
    exit 0
fi

if [[ $DO_CHECK -eq 1 ]]; then
    run_check
    print_summary
    exit 0
fi

if [[ $SKIP_INSTALL -eq 0 ]]; then
    run_install
fi

if [[ $SKIP_SERVICE -eq 0 ]]; then
    run_service
fi

print_summary

if [[ ${#STEPS_ERR[@]} -gt 0 ]]; then
    exit 1
fi

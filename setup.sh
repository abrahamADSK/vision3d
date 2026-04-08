#!/usr/bin/env bash
# =============================================================================
# setup.sh — vision3d service registration (systemd / LaunchAgent)
# =============================================================================
# Registers vision3d as a system service so it starts on boot/login.
# Safe to run multiple times (idempotent).
#
# What this script does:
#   1. Detects platform (Linux / macOS)
#   2. Generates a random API key (saved to .api_key) if not present
#   3. Registers vision3d as a service:
#      - Linux: systemd unit (0.0.0.0:8000, runs as current user)
#      - macOS: LaunchAgent (127.0.0.1:8000, runs at login)
#   4. Starts the service and verifies health
#
# What this script does NOT do (see install.sh instead):
#   - Create venv or install dependencies
#   - Install Hunyuan3D-2 fork
#   - Verify imports
#
# Usage:
#   bash setup.sh              # install and start service
#   bash setup.sh --uninstall  # stop and remove service
#
# Tested on: macOS (arm64), Rocky Linux (CUDA)
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

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISION_DIR="${GPU_VISION_DIR:-$SCRIPT_DIR}"
VENV_DIR="${GPU_VENV:-$VISION_DIR/.venv}"
WORK_DIR="${GPU_WORK_DIR:-$VISION_DIR/output}"
MODELS_DIR="${GPU_MODELS_DIR:-$VISION_DIR/hf_models}"
VENV_PYTHON="${VENV_DIR}/bin/python"
PORT=8000

# macOS-specific paths
PLIST_LABEL="com.abraham.vision3d"
PLIST_PATH="${HOME}/Library/LaunchAgents/${PLIST_LABEL}.plist"
LOG_PATH="${HOME}/Library/Logs/vision3d.log"

# Linux-specific paths
SERVICE_FILE="/etc/systemd/system/vision3d.service"

# =============================================================================
# Handle --uninstall
# =============================================================================
if [[ "${1:-}" == "--uninstall" ]]; then
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  vision3d — uninstall service${RESET}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
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
    echo ""
    exit 0
fi

# =============================================================================
# Main installation flow
# =============================================================================
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  vision3d — service setup${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
info "Vision dir  : ${VISION_DIR}"
info "Venv        : ${VENV_DIR}"
info "Port        : ${PORT}"
echo ""

# ── Pre-flight: verify venv exists ───────────────────────────────────────────
if [[ ! -f "${VENV_PYTHON}" ]]; then
    error "Venv not found at ${VENV_DIR}"
    error "Run install.sh first to create the venv and install dependencies."
    exit 1
fi

# =============================================================================
# STEP 1 — Detect platform
# =============================================================================
info "Step 1/3 — Detecting platform..."

PLATFORM=""
if [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM="macos"
    HOST="127.0.0.1"
    success "macOS detected — will create LaunchAgent (${HOST}:${PORT})"
    STEPS_OK+=("Platform: macOS — LaunchAgent")
else
    PLATFORM="linux"
    HOST="0.0.0.0"
    success "Linux detected — will create systemd service (${HOST}:${PORT})"
    STEPS_OK+=("Platform: Linux — systemd")
fi

# =============================================================================
# STEP 2 — Generate API key
# =============================================================================
info "Step 2/3 — Generating API key..."

API_KEY_FILE="${VISION_DIR}/.api_key"
if [[ -f "$API_KEY_FILE" ]]; then
    API_KEY=$(cat "$API_KEY_FILE")
    success "Using existing key from ${API_KEY_FILE}"
    STEPS_OK+=("API key already present")
else
    API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo "$API_KEY" > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
    success "New key saved to ${API_KEY_FILE}"
    STEPS_OK+=("API key generated")
fi

# =============================================================================
# STEP 3 — Register service
# =============================================================================
info "Step 3/3 — Registering service..."

if [[ "$PLATFORM" == "macos" ]]; then
    # ── macOS: LaunchAgent ───────────────────────────────────────────────────

    # Build environment variables from .env if it exists
    ENV_VARS="		<key>PATH</key>
		<string>${VENV_DIR}/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
		<key>GPU_API_KEY</key>
		<string>${API_KEY}</string>"

    ENV_FILE="${VISION_DIR}/.env"
    if [[ -f "$ENV_FILE" ]]; then
        # Read GPU_MODELS_DIR, GPU_WORK_DIR, GPU_VISION_DIR from .env if set
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
            # Trim whitespace
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            # Only include GPU_* vars (skip GPU_API_KEY — already set above)
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

    # Bootout if already loaded, wait for clean shutdown, then bootstrap
    if launchctl print "gui/$(id -u)/${PLIST_LABEL}" &>/dev/null; then
        info "Service already loaded — reloading..."
        launchctl bootout "gui/$(id -u)/${PLIST_LABEL}" 2>/dev/null || true
        # Wait for the old process to fully exit before re-bootstrapping
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
    fi

    # Wait for server to start and verify health
    info "Waiting for server to start..."
    HEALTH_OK=false
    for i in 1 2 3 4 5; do
        sleep 2
        if curl -sf "http://${HOST}:${PORT}/api/health" > /dev/null 2>&1; then
            HEALTH_OK=true
            break
        fi
    done

    if $HEALTH_OK; then
        HEALTH_JSON=$(curl -sf "http://${HOST}:${PORT}/api/health")
        success "Server is running — ${HEALTH_JSON}"
        STEPS_OK+=("Health check passed")
    else
        warn "Server not responding yet — check logs: tail -f ${LOG_PATH}"
        STEPS_WARN+=("Health check failed — server may still be starting")
    fi

else
    # ── Linux: systemd ───────────────────────────────────────────────────────

    HOSTNAME_VAL=$(hostname)

    sudo tee "$SERVICE_FILE" > /dev/null <<UNIT
[Unit]
Description=Vision3D — AI 3D Generation Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$VISION_DIR
Environment=GPU_API_KEY=$API_KEY
Environment=GPU_MODELS_DIR=$MODELS_DIR
Environment=GPU_WORK_DIR=$WORK_DIR
Environment=GPU_VISION_DIR=$VISION_DIR
Environment=LD_LIBRARY_PATH=$VENV_DIR/lib/python3.10/site-packages/torch/lib
ExecStart=$VENV_PYTHON $VISION_DIR/server.py --host $HOST --port $PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

    success "Created ${SERVICE_FILE}"

    sudo systemctl daemon-reload
    sudo systemctl enable --now vision3d.service

    success "systemd service enabled and started"
    STEPS_OK+=("systemd service created and started")

    # Wait for server to start
    sleep 2
    if curl -sf "http://localhost:${PORT}/api/health" > /dev/null 2>&1; then
        success "Server is running"
        STEPS_OK+=("Health check passed")
    else
        warn "Server not responding yet — check: sudo journalctl -u vision3d -f"
        STEPS_WARN+=("Health check failed — server may still be starting")
    fi
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  Setup summary${RESET}"
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

if [[ ${#STEPS_ERR[@]} -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}vision3d service is running.${RESET}"
    echo ""
    echo -e "  ${BOLD}API Key:${RESET}  ${API_KEY}"
    echo ""

    if [[ "$PLATFORM" == "macos" ]]; then
        echo -e "  ${BOLD}Server:${RESET}   http://${HOST}:${PORT}"
        echo -e "  ${BOLD}Logs:${RESET}     tail -f ${LOG_PATH}"
        echo ""
        echo -e "  ${BOLD}Manage:${RESET}"
        echo -e "    ${CYAN}launchctl stop ${PLIST_LABEL}${RESET}        — stop"
        echo -e "    ${CYAN}launchctl start ${PLIST_LABEL}${RESET}       — start"
        echo -e "    ${CYAN}bash setup.sh --uninstall${RESET}                — remove"
    else
        HOSTNAME_VAL=$(hostname)
        echo -e "  ${BOLD}Server:${RESET}   http://${HOSTNAME_VAL}:${PORT}"
        echo -e "  ${BOLD}Logs:${RESET}     sudo journalctl -u vision3d -f"
        echo ""
        echo -e "  ${BOLD}Manage:${RESET}"
        echo -e "    ${CYAN}sudo systemctl stop vision3d${RESET}         — stop"
        echo -e "    ${CYAN}sudo systemctl restart vision3d${RESET}      — restart"
        echo -e "    ${CYAN}bash setup.sh --uninstall${RESET}            — remove"
        echo ""
        echo -e "  ${BOLD}Maya-mcp config:${RESET}"
        echo -e "    ${CYAN}GPU_API_URL=http://${HOSTNAME_VAL}:${PORT}${RESET}"
        echo -e "    ${CYAN}GPU_API_KEY=${API_KEY}${RESET}"
    fi

    echo ""
else
    echo -e "${RED}${BOLD}Setup completed with errors.${RESET}"
    echo -e "Review the ${RED}✗${RESET} items above."
    echo ""
fi

echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

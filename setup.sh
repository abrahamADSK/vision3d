#!/usr/bin/env bash
# setup.sh — Install and configure the GPU inference API server.
#
# Run this on the GPU machine:
#   bash setup.sh
#
# What it does:
#   1. Installs FastAPI + uvicorn into the existing venv
#   2. Generates a random API key (saved to .api_key)
#   3. Creates a systemd service for the Vision3D server
#   4. Prints the config values to use on the Mac

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VISION_DIR="${GPU_VISION_DIR:-$SCRIPT_DIR}"
VENV_DIR="${GPU_VENV:-$VISION_DIR/.venv}"
WORK_DIR="${GPU_WORK_DIR:-$VISION_DIR/output}"
MODELS_DIR="${GPU_MODELS_DIR:-$VISION_DIR/hf_models}"
PORT=8000
HOSTNAME=$(hostname)

echo "═══════════════════════════════════════════════════════════"
echo " Vision3D Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo " Vision dir:  $VISION_DIR"
echo " Venv:        $VENV_DIR"
echo " Work dir:    $WORK_DIR"
echo " Models dir:  $MODELS_DIR"
echo " Hostname:    $HOSTNAME"
echo ""

# ── 1. Install Python dependencies ──────────────────────────────────────────

echo "[1/3] Installing FastAPI + uvicorn..."
if [ -f "$VENV_DIR/bin/pip" ]; then
    "$VENV_DIR/bin/pip" install --quiet fastapi uvicorn python-multipart
    echo "      Done."
else
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Create it first: python3 -m venv $VENV_DIR && $VENV_DIR/bin/pip install -r requirements.txt"
    exit 1
fi

# ── 2. Generate API key ─────────────────────────────────────────────────────

echo "[2/3] Generating API key..."
API_KEY_FILE="$VISION_DIR/.api_key"
if [ -f "$API_KEY_FILE" ]; then
    API_KEY=$(cat "$API_KEY_FILE")
    echo "      Using existing key from $API_KEY_FILE"
else
    API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo "$API_KEY" > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
    echo "      New key saved to $API_KEY_FILE"
fi

# ── 3. Create systemd service ───────────────────────────────────────────────

echo "[3/3] Creating systemd service..."

SERVICE_FILE="/etc/systemd/system/vision3d.service"
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
Environment=LD_LIBRARY_PATH=$VENV_DIR/lib/python3.9/site-packages/torch/lib
ExecStart=$VENV_DIR/bin/python $VISION_DIR/server.py --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

echo "      Created $SERVICE_FILE"

# ── Enable and start ─────────────────────────────────────────────────────────

sudo systemctl daemon-reload
sudo systemctl enable --now vision3d.service

# Wait a moment for service to start
sleep 2

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Setup complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo " Vision3D:  http://$HOSTNAME:$PORT/api/health"
echo ""
echo " API Key:   $API_KEY"
echo ""
echo " ── Mac configuration (~/.claude.json) ──"
echo " Set these env vars in maya-mcp:"
echo ""
echo "   GPU_API_URL=http://$HOSTNAME:$PORT"
echo "   GPU_API_KEY=$API_KEY"
echo ""
echo " ── Quick test ──"
echo "   curl http://$HOSTNAME:$PORT/api/health"
echo ""

# Verify health
echo "Verifying..."
if curl -sf "http://localhost:$PORT/api/health" > /dev/null 2>&1; then
    echo "✓ Vision3D server is running"
else
    echo "⚠ Vision3D server may still be starting — check: systemctl status vision3d"
fi

#!/bin/bash
# Complimentary Machine - RunPod Pod Startup Script
# Usage: Set Pod start command to: bash /workspace/CM_Server/start.sh

# --- Config ---
export HF_MODEL_ID="GRMD/complimentary-machine-vlm"
export HF_TOKEN=""
export PORT=8000
export HF_HOME="/workspace/.cache/huggingface"

# --- Get script directory ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Complimentary Machine Server Starting  "
echo "=========================================="
echo "Model: $HF_MODEL_ID"
echo "Port: $PORT"
echo "Dir: $SCRIPT_DIR"
echo ""

# --- Install dependencies ---
echo "Installing dependencies..."
pip install -r requirements.txt -q

# --- Start server ---
echo "Starting server..."
python server_pod.py

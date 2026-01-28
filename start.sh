#!/bin/bash
# Complimentary Machine - RunPod Pod 启动脚本
# 用法: 在 Pod 启动命令中设置 bash /workspace/CM_Server/start.sh

# --- 配置 ---
export HF_MODEL_ID="GRMD/complimentary-machine-vlm"
export HF_TOKEN=""
export PORT=8000
export HF_HOME="/workspace/.cache/huggingface"

# --- 获取脚本所在目录（支持任意位置运行） ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Complimentary Machine Server Starting  "
echo "=========================================="
echo "Model: $HF_MODEL_ID"
echo "Port: $PORT"
echo "Dir: $SCRIPT_DIR"
echo ""

# --- 安装依赖（静默模式） ---
echo "Installing dependencies..."
pip install -r requirements.txt -q

# --- 启动服务 ---
echo "Starting server..."
python server_pod.py

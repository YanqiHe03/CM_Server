#!/bin/bash
# Complimentary Machine - RunPod Pod 启动脚本
# 用法: 在 Pod 启动命令中设置 bash /workspace/CM_Server/start.sh

# --- 配置 ---
export HF_MODEL_ID="GRMD/cm-gallery-vlm"  # 改成你的模型路径
export HF_TOKEN=""  # 如果是私有仓库，填入你的 HuggingFace Token
export PORT=8000

# --- 设置 HuggingFace 缓存目录（可选，加速重启） ---
export HF_HOME="/workspace/.cache/huggingface"

# --- 启动服务 ---
echo "=========================================="
echo "  Complimentary Machine Server Starting  "
echo "=========================================="
echo "Model: $HF_MODEL_ID"
echo "Port: $PORT"
echo ""

cd /workspace/CM_Server
python server_pod.py

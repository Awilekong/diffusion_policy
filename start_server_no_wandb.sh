#!/bin/bash

# Diffusion Policy 推理服务器启动脚本（不启用 WandB）
# 用于生产环境，无调试开销

set -e

CHECKPOINT="/mlp_vepfs/share/zpw/IL/diffusion_policy/data/outputs/2025.12.15/22.23.14_train_diffusion_unet_franka_image_franka_peg_in_hole_image/checkpoints/epoch=0100-train_loss=0.005.ckpt"
PORT=8000
DEVICE="cuda"

echo "=========================================="
echo "Diffusion Policy 推理服务器（生产模式）"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "端口: $PORT"
echo "设备: $DEVICE"
echo "WandB 调试: 关闭"
echo "=========================================="
echo ""

echo "🚀 启动服务器..."
echo ""

/root/miniconda3/envs/robodiff/bin/python serve_diffusion_policy_single_frame.py \
    -i "$CHECKPOINT" \
    -p "$PORT" \
    -d "$DEVICE"

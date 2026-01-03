#!/bin/bash

# Diffusion Policy 推理服务器启动脚本
# 用法:
#   ./start_server.sh                    # 使用默认配置
#   ./start_server.sh --wandb            # 启用 WandB 调试
#   ./start_server.sh --port 8080        # 指定端口
#   ./start_server.sh --device cuda      # 指定设备

set -e

# 默认配置
CHECKPOINT="/mlp_vepfs/share/zpw/IL/diffusion_policy/data/outputs/2025.12.15/22.23.14_train_diffusion_unet_franka_image_franka_peg_in_hole_image/checkpoints/epoch=0100-train_loss=0.005.ckpt"
PORT=8000
DEVICE="cuda"
WANDB=""
WANDB_PROJECT="diffusion_policy_inference"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        --wandb)
            WANDB="--wandb"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -i, --checkpoint PATH    Checkpoint 文件路径 (默认: 最新的 epoch=0100)"
            echo "  -p, --port PORT          WebSocket 端口 (默认: 8000)"
            echo "  -d, --device DEVICE      设备 cuda/cpu (默认: cuda)"
            echo "  --wandb                  启用 WandB 调试"
            echo "  --wandb-project NAME     WandB 项目名称 (默认: diffusion_policy_inference)"
            echo "  -h, --help               显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                          # 基础启动"
            echo "  $0 --wandb                                  # 启用 WandB 调试"
            echo "  $0 --port 8080 --device cuda                # 指定端口和设备"
            echo "  $0 -i /path/to/checkpoint.ckpt --wandb      # 指定 checkpoint 并启用 WandB"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 打印配置
echo "=========================================="
echo "Diffusion Policy 推理服务器"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "端口: $PORT"
echo "设备: $DEVICE"
echo "WandB 调试: $([ -n "$WANDB" ] && echo "启用 (项目: $WANDB_PROJECT)" || echo "关闭")"
echo "=========================================="
echo ""

# 检查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CHECKPOINT"
    exit 1
fi

# 激活 conda 环境并启动服务器
echo "🚀 启动服务器..."
echo ""

# 构建命令
CMD="/root/miniconda3/envs/robodiff/bin/python serve_diffusion_policy_single_frame.py -i $CHECKPOINT -p $PORT -d $DEVICE"

if [ -n "$WANDB" ]; then
    CMD="$CMD --wandb --wandb-project $WANDB_PROJECT"
fi

# 执行命令
echo "执行命令: $CMD"
echo ""
$CMD

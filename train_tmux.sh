#!/bin/bash

# ===============================================
# Franka 训练启动脚本 - Tmux 版本
# ===============================================
# 基于原命令:
# nohup python train.py \
#     --config-name=train_diffusion_unet_franka_image_workspace \
#     task.dataset_path=/home/zpw/ws_zpw/zpw/data/zarr_dataset/peg_in_hole_zarr \
#     exp_name="baseline_h16_obs1" \
#     training.device="cuda:1" \
#     > logs/train_baseline_h16_obs1.log 2>&1 &
# ===============================================

# 配置
SESSION_NAME="train_franka"
EXP_NAME="baseline_h16_obs1"
DEVICE="cuda:1"
DATASET_PATH="/home/zpw/ws_zpw/zpw/data/zarr_dataset/peg_in_hole_zarr"
PROJECT_DIR="/data0/ws_zpw/zpw/IL/diffusion_policy"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查 tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}Warning: tmux 未安装！${NC}"
    echo "安装: sudo apt install tmux"
    exit 1
fi

# 进入项目目录
cd ${PROJECT_DIR}

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Franka 训练启动器${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Session:  ${SESSION_NAME}"
echo "Exp Name: ${EXP_NAME}"
echo "Device:   ${DEVICE}"
echo "Dataset:  ${DATASET_PATH}"
echo ""

# 检查已存在的 session
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo -e "${YELLOW}Warning: Session '${SESSION_NAME}' 已存在${NC}"
    read -p "是否终止旧 session 并重新启动? (y/N): " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        tmux kill-session -t ${SESSION_NAME}
        echo "  旧 session 已终止"
        sleep 1
    else
        echo "取消启动"
        echo ""
        echo "连接到现有 session:"
        echo "  tmux attach -t ${SESSION_NAME}"
        exit 0
    fi
fi

# 创建 tmux session
echo -e "${GREEN}正在创建 tmux session...${NC}"
tmux new-session -d -s ${SESSION_NAME}

# 设置工作目录
tmux send-keys -t ${SESSION_NAME} "cd ${PROJECT_DIR}" C-m

# 激活 conda 环境
tmux send-keys -t ${SESSION_NAME} "conda activate robodiff" C-m
sleep 1

# 发送训练命令
tmux send-keys -t ${SESSION_NAME} "python train.py \\" C-m
tmux send-keys -t ${SESSION_NAME} "    --config-name=train_diffusion_unet_franka_image_workspace \\" C-m
tmux send-keys -t ${SESSION_NAME} "    task.dataset_path=${DATASET_PATH} \\" C-m
tmux send-keys -t ${SESSION_NAME} "    exp_name=\"${EXP_NAME}\" \\" C-m
tmux send-keys -t ${SESSION_NAME} "    training.device=\"${DEVICE}\"" C-m

echo ""
echo -e "${GREEN}✓ 训练启动成功！${NC}"
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  使用说明${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${GREEN}查看实时输出:${NC}"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo -e "${GREEN}在 tmux session 中:${NC}"
echo "  Ctrl+B, D     - 退出但保持训练运行"
echo "  Ctrl+C        - 中断训练"
echo "  Ctrl+B, [     - 滚动查看历史输出（按 q 退出）"
echo ""
echo -e "${GREEN}终止训练:${NC}"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo ""
echo -e "${GREEN}查看 GPU:${NC}"
echo "  nvidia-smi"
echo "  watch -n 1 nvidia-smi  # 实时监控"
echo ""

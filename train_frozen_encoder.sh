#!/bin/bash

# ===============================================
# 实验1: 冻结Encoder + 增加Epoch
# ===============================================
# 假设: 数据量大(310条)且泛化性强，导致难以收敛
# 策略: 冻结obs encoder，让网络专注于学习动作预测
#      增加训练epoch让网络充分学习
# ===============================================

# 配置
SESSION_NAME="train_frozen_encoder"
EXP_NAME="frozen_encoder_epoch400"
DEVICE="cuda:1"  # 使用空闲GPU
DATASET_PATH="/home/zpw/ws_zpw/zpw/data/zarr_dataset/peg_in_hole_zarr"
PROJECT_DIR="/home/zpw/ws_zpw/zpw/IL/diffusion_policy"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查 tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}Warning: tmux 未安装！${NC}"
    echo "安装: sudo apt install tmux"
    exit 1
fi

# 进入项目目录
cd ${PROJECT_DIR}

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  实验1: 冻结Encoder + 增加Epoch${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Session:        ${SESSION_NAME}"
echo "Exp Name:       ${EXP_NAME}"
echo "Device:         ${DEVICE}"
echo "Dataset:        ${DATASET_PATH}"
echo -e "${RED}freeze_encoder: True${NC}"
echo -e "${RED}num_epochs:     400${NC} (原150)"
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
tmux send-keys -t ${SESSION_NAME} "    training.freeze_encoder=True \\" C-m
tmux send-keys -t ${SESSION_NAME} "    training.num_epochs=400 \\" C-m
tmux send-keys -t ${SESSION_NAME} "    training.device=\"${DEVICE}\"" C-m

echo ""
echo -e "${GREEN}✓ 训练启动成功！${NC}"
echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  实验配置${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}关键参数:${NC}"
echo "  • training.freeze_encoder=True (冻结obs encoder)"
echo "  • training.num_epochs=400 (从150增加到400)"
echo "  • 数据集: 310条高泛化数据"
echo ""
echo -e "${GREEN}实验假设:${NC}"
echo "  数据量大且泛化性强，冻结encoder可以："
echo "  1. 减少需要训练的参数量"
echo "  2. 让网络专注于学习动作预测部分"
echo "  3. 增加epoch确保充分学习"
echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  使用说明${NC}"
echo -e "${BLUE}=========================================${NC}"
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
echo -e "${GREEN}查看 WandB:${NC}"
echo "  训练日志会自动上传到 WandB project: diffusion_policy_franka"
echo "  标签: frozen_encoder_epoch400"
echo ""

#!/bin/bash

# ===============================================
# 实验4: 50条数据 + 冻结Encoder + Delta Action
# ===============================================
# 对比实验2，只改变delta_action=True
# 验证delta action在小数据集上的影响
# ===============================================

# 配置
SESSION_NAME="train_exp4_delta"
EXP_NAME="exp4_50episodes_frozen_encoder_delta"
DEVICE="cuda:5"
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
echo -e "${BLUE}  实验4: Delta Action (50条数据)${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Session:        ${SESSION_NAME}"
echo "Exp Name:       ${EXP_NAME}"
echo "Device:         ${DEVICE}"
echo "Dataset:        ${DATASET_PATH}"
echo "Config:         train_diffusion_unet_exp4_delta.yaml"
echo ""
echo -e "${RED}关键配置:${NC}"
echo -e "  • max_train_episodes: 50 (随机采样)"
echo -e "  • freeze_encoder:     True"
echo -e "  • ${RED}delta_action:        True${NC} ${YELLOW}(相比实验2的唯一差异)${NC}"
echo -e "  • num_epochs:         600"
echo ""
echo -e "${YELLOW}对比实验2:${NC}"
echo "  实验2: 50条 + 冻结encoder + 600 epochs + ${GREEN}absolute action${NC}"
echo "  实验4: 50条 + 冻结encoder + 600 epochs + ${RED}delta action${NC}"
echo ""
echo -e "${YELLOW}完整对齐 train_diffusion_unet_real_image_workspace.yaml${NC}"
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
tmux send-keys -t ${SESSION_NAME} "python train.py --config-name=train_diffusion_unet_exp4_delta" C-m

echo ""
echo -e "${GREEN}✓ 训练启动成功！${NC}"
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
echo ""
echo -e "${GREEN}终止训练:${NC}"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo ""
echo -e "${GREEN}查看 WandB:${NC}"
echo "  训练日志会自动上传到 WandB project: diffusion_policy_franka"
echo "  标签: ${EXP_NAME}"
echo ""

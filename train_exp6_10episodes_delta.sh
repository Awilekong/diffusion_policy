#!/bin/bash

# ===============================================
# 实验6: 10条数据 + 冻结Encoder + Delta Action
# ===============================================
# 极小数据量实验，测试delta action的过拟合能力
# ===============================================

# 配置
SESSION_NAME="train_exp6_10episodes_delta"
EXP_NAME="exp6_10episodes_frozen_encoder_delta"
DEVICE="cuda:6"
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
echo -e "${BLUE}  实验6: Delta Action (10条数据)${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Session:        ${SESSION_NAME}"
echo "Exp Name:       ${EXP_NAME}"
echo "Device:         ${DEVICE}"
echo "Dataset:        ${DATASET_PATH}"
echo "Config:         train_diffusion_unet_exp6_10episodes_delta.yaml"
echo ""
echo -e "${RED}关键配置:${NC}"
echo -e "  • max_train_episodes: 10 ${RED}(极小数据量)${NC}"
echo -e "  • freeze_encoder:     True"
echo -e "  • delta_action:       ${RED}True (delta位姿)${NC}"
echo -e "  • num_epochs:         ${RED}1000${NC} (从600增加)"
echo ""
echo -e "${YELLOW}对比实验5:${NC}"
echo "  实验5: 10条 + 1000 epochs + ${GREEN}absolute action${NC}"
echo "  实验6: 10条 + 1000 epochs + ${RED}delta action${NC}"
echo ""
echo -e "${YELLOW}实验目的:${NC}"
echo "  测试在极小数据量(10条)下，delta action能否过拟合训练集"
echo "  对比absolute vs delta在极小数据下的学习能力"
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
tmux send-keys -t ${SESSION_NAME} "python train.py --config-name=train_diffusion_unet_exp6_10episodes_delta" C-m

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

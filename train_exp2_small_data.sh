#!/bin/bash

# ===============================================
# 实验2: 减小数据量 (50条) + 冻结Encoder
# ===============================================
# 假设: 数据量过大导致难以收敛
# 策略: 使用50条数据 + 冻结encoder + 参数对齐real_image配置
# ===============================================

# 配置
SESSION_NAME="train_exp2_small_data"
EXP_NAME="exp2_50episodes_frozen_encoder"
DEVICE="cuda:2"  # 使用空闲GPU
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
echo -e "${BLUE}  实验2: 减小数据量 + 冻结Encoder${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Session:        ${SESSION_NAME}"
echo "Exp Name:       ${EXP_NAME}"
echo "Device:         ${DEVICE}"
echo "Dataset:        ${DATASET_PATH}"
echo "Config:         train_diffusion_unet_exp2_small_data.yaml"
echo ""
echo -e "${RED}关键配置:${NC}"
echo -e "  • max_train_episodes: 50 ${RED}(从310减少到50)${NC}"
echo -e "  • freeze_encoder:     True"
echo -e "  • use_state_input:    True ${RED}(允许robot state)${NC}"
echo -e "  • n_obs_steps:        1"
echo ""
echo -e "${YELLOW}完整对齐 train_diffusion_unet_real_image_workspace.yaml:${NC}"
echo "  • batch_size:         64"
echo "  • lr:                 1.0e-4"
echo "  • lr_warmup_steps:    500"
echo "  • num_epochs:         600"
echo "  • crop_shape:         [216, 288]"
echo "  • random_crop:        False"
echo "  • val_every:          1"
echo "  • shuffle:            True"
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

# 发送训练命令 - 使用新的配置文件
tmux send-keys -t ${SESSION_NAME} "python train.py --config-name=train_diffusion_unet_exp2_small_data" C-m

echo ""
echo -e "${GREEN}✓ 训练启动成功！${NC}"
echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  实验配置总结${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}实验假设:${NC}"
echo "  数据量过大(310条)导致难以收敛，减小到50条后:"
echo "  1. 减少数据多样性，网络更容易记忆"
echo "  2. 冻结encoder减少参数量"
echo "  3. 使用更aggressive的学习率和更多epochs"
echo ""
echo -e "${GREEN}对比实验1:${NC}"
echo "  实验1: 310条数据 + 冻结encoder + 400 epochs"
echo "  实验2: 50条数据  + 冻结encoder + 600 epochs"
echo ""
echo -e "${GREEN}配置方式:${NC}"
echo "  使用独立的yaml配置文件，完整复制real_image配置"
echo "  避免命令行覆盖参数导致的遗漏问题"
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
echo "  标签: ${EXP_NAME}"
echo ""

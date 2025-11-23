#!/bin/bash

# ===============================================
# Franka Peg-in-Hole 任务训练脚本 - 多实验组
# ===============================================

# 进入项目目录
cd /home/zpw/ws_zpw/megvii/IL/diffusion_policy

# ===============================================
# Baseline: horizon=16 (默认配置)
# ===============================================
nohup python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/home/zpw/ws_zpw/megvii/data/2025_11_18/zarr_dataset/peg_in_hole_zarr \
    exp_name="baseline_h16" \
    training.device="cuda:3" \
    > logs/train_baseline_h16.log 2>&1 &

echo "Baseline (horizon=16) started on cuda:3, log: logs/train_baseline_h16.log"

# ===============================================
# 实验组5: 关闭Random Crop
# ===============================================
nohup python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/home/zpw/ws_zpw/megvii/data/2025_11_18/zarr_dataset/peg_in_hole_zarr \
    policy.obs_encoder.random_crop=False \
    exp_name="no_random_crop" \
    training.device="cuda:4" \
    > logs/train_no_random_crop.log 2>&1 &

echo "Exp5 (no random crop) started on cuda:4, log: logs/train_no_random_crop.log"

# ===============================================
# 实验组: horizon=32 (更长预测)
# ===============================================
nohup python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/home/zpw/ws_zpw/megvii/data/2025_11_18/zarr_dataset/peg_in_hole_zarr \
    horizon=32 \
    n_action_steps=16 \
    exp_name="horizon_32" \
    training.device="cuda:5" \
    > logs/train_horizon_32.log 2>&1 &

echo "Exp (horizon=32) started on cuda:5, log: logs/train_horizon_32.log"

# ===============================================
# 实验组: horizon=8 (更短预测)
# ===============================================
nohup python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/home/zpw/ws_zpw/megvii/data/2025_11_18/zarr_dataset/peg_in_hole_zarr \
    horizon=8 \
    n_action_steps=4 \
    exp_name="horizon_8" \
    training.device="cuda:6" \
    > train_horizon_8.log 2>&1 &

echo "Exp (horizon=8) started on cuda:6, log: logs/train_horizon_8.log"

# ===============================================
# 查看所有训练进程
# ===============================================
sleep 2
echo ""
echo "All training jobs started! Check status with:"
echo "  ps aux | grep train.py"
echo "  tail -f logs/train_*.log"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/train_baseline_h16.log"
echo "  tail -f logs/train_no_random_crop.log"
echo "  tail -f logs/train_horizon_32.log"
echo "  tail -f logs/train_horizon_8.log"

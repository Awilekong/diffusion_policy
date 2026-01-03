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
    task.dataset_path=/mlp_vepfs/share/zpw/data/zarr_dataset/peg_in_hole_zarr \
    exp_name="baseline_h16_obs1" \
    training.device="cuda:1" \
    > logs/train_baseline_h16_obs1.log 2>&1 &

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
    exp_name="horizon_8_new" \
    training.device="cuda:6" \
    > train_horizon_8_new.log 2>&1 &

echo "Exp (horizon=8) started on cuda:6, log: logs/train_horizon_8.log"

# ===============================================
# 实验组: n_obs_steps=1 (单帧观测)
# ===============================================
nohup python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/home/zpw/ws_zpw/megvii/data/2025_11_18/zarr_dataset/peg_in_hole_zarr \
    horizon=8 \
    n_action_steps=4 \
    n_obs_steps=1 \
    exp_name="n_obs_1" \
    training.device="cuda:7" \
    > train_n_obs_1.log 2>&1 &

echo "Exp (n_obs_steps=1) started on cuda:7, log: logs/train_n_obs_1.log"

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
echo "  tail -f logs/train_n_obs_1.log"


conda run -n robodiff pip install --upgrade diffusers

conda run -n robodiff pip install huggingface_hub==0.19.4 -i https://pypi.org/simple

# ===============================================
# Resume训练命令 - 从checkpoint继续训练
# ===============================================
# 说明: checkpoint中已保存所有配置，只需指定输出目录和resume=True即可

# Resume方式1: 恢复 baseline_h16_obs1 实验
python train.py \
    training.resume=True \
    hydra.run.dir=/mlp_vepfs/share/zpw/IL/diffusion_policy/data/outputs/2025.12.13/15.31.25_train_diffusion_unet_franka_image_franka_peg_in_hole_image \
    training.device="cuda:1"

# Resume方式2: 恢复 horizon_32 实验
# nohup python train.py \
#     training.resume=True \
#     hydra.run.dir=<your_output_dir> \
#     training.device="cuda:5" \
#     > logs/train_horizon32_resume.log 2>&1 &

# Resume方式3: 恢复 horizon_8_new 实验
# nohup python train.py \
#     training.resume=True \
#     hydra.run.dir=<your_output_dir> \
#     training.device="cuda:6" \
#     > logs/train_horizon8_resume.log 2>&1 &

# Resume方式4: 恢复 n_obs_1 实验
# nohup python train.py \
#     training.resume=True \
#     hydra.run.dir=<your_output_dir> \
#     training.device="cuda:7" \
#     > logs/train_nobs1_resume.log 2>&1 &

# Resume方式5: 恢复 no_random_crop 实验
# nohup python train.py \
#     training.resume=True \
#     hydra.run.dir=<your_output_dir> \
#     training.device="cuda:4" \
#     > logs/train_nocrop_resume.log 2>&1 &

# 使用步骤:
# 1. 找到输出目录: cd /home/zpw/ws_zpw/megvii/IL/diffusion_policy && ls -lt outputs/ | head -20
# 2. 确认有checkpoint: ls <output_dir>/checkpoints/latest.ckpt
# 3. 替换 <your_output_dir> 为实际路径 (如: outputs/2025-12-15/14-30-00)
# 4. 删除 # 号运行即可

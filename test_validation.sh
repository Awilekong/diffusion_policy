#!/bin/bash
# 快速测试validation是否能正常运行

cd /data0/ws_zpw/zpw/IL/diffusion_policy

timeout 120 /home/zpw/ws_zpw/miniconda3/envs/robodiff/bin/python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/home/zpw/ws_zpw/zpw/data/zarr_dataset/peg_in_hole_zarr \
    exp_name="val_test" \
    training.device="cuda:1" \
    training.num_epochs=1 \
    training.val_every=0 \
    training.checkpoint_every=999 \
    training.sample_every=999

echo "退出码: $?"

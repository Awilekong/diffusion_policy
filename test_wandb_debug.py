#!/usr/bin/env python3
"""
测试 WandB 调试功能
不启动 WebSocket 服务器，直接测试推理流程
"""

import sys
import os
import numpy as np
import torch

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from serve_diffusion_policy_single_frame import DiffusionPolicySingleFrameWrapper

def main():
    print("=" * 60)
    print("测试 WandB 调试功能")
    print("=" * 60)

    # Checkpoint 路径
    checkpoint_path = "/mlp_vepfs/share/zpw/IL/diffusion_policy/data/outputs/2025.12.15/22.23.14_train_diffusion_unet_franka_image_franka_peg_in_hole_image/checkpoints/epoch=0100-train_loss=0.005.ckpt"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        return

    print(f"\n✅ 找到 checkpoint: {checkpoint_path}")

    # 创建 wrapper（不启用 WandB，仅测试推理流程）
    print("\n🔄 加载模型...")
    try:
        wrapper = DiffusionPolicySingleFrameWrapper(
            ckpt_path=checkpoint_path,
            device='cpu',  # 使用 CPU 测试
            use_wandb=False  # 不启用 WandB
        )
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建测试数据
    print("\n🔄 准备测试数据...")
    obs = {
        'observation/image': np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
        'observation/image_1': np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
        'observation/image_2': np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
        'observation/state': np.random.randn(7).astype(np.float32)  # 模型训练时用的是 7 维
    }
    print("✅ 测试数据准备完成")

    # 测试推理
    print("\n🚀 开始推理测试...")
    try:
        result = wrapper.infer(obs)
        print(f"✅ 推理成功！")
        print(f"   动作 shape: {result['actions'].shape}")
        print(f"   动作范围: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试调试回调是否被正确设置和清理
    print("\n🔍 检查调试回调...")
    if wrapper.policy.debug_callback is None:
        print("✅ Policy debug_callback 已正确清理")
    else:
        print("⚠️  Policy debug_callback 未清理")

    if wrapper.policy.obs_encoder.debug_callback is None:
        print("✅ ObsEncoder debug_callback 已正确清理")
    else:
        print("⚠️  ObsEncoder debug_callback 未清理")

    print("\n" + "=" * 60)
    print("基础功能测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

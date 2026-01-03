#!/usr/bin/env python3
"""
完整测试 WandB 调试功能（启用 WandB）
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
    print("完整测试 WandB 调试功能（启用 WandB）")
    print("=" * 60)

    # Checkpoint 路径
    checkpoint_path = "/mlp_vepfs/share/zpw/IL/diffusion_policy/data/outputs/2025.12.15/22.23.14_train_diffusion_unet_franka_image_franka_peg_in_hole_image/checkpoints/epoch=0100-train_loss=0.005.ckpt"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        return

    print(f"\n✅ 找到 checkpoint: {checkpoint_path}")

    # 创建 wrapper（启用 WandB）
    print("\n🔄 加载模型（启用 WandB）...")
    try:
        wrapper = DiffusionPolicySingleFrameWrapper(
            ckpt_path=checkpoint_path,
            device='cpu',  # 使用 CPU 测试
            use_wandb=True,  # 启用 WandB
            wandb_project="diffusion_policy_debug_test"
        )
        print("✅ 模型加载成功")
        print(f"   WandB enabled: {wrapper.wandb_debugger.enabled}")
        if wrapper.wandb_debugger.enabled:
            print(f"   WandB run: {wrapper.wandb_debugger.wandb.run.name}")
            print(f"   WandB URL: {wrapper.wandb_debugger.wandb.run.url}")
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
        'observation/state': np.random.randn(7).astype(np.float32)
    }
    print("✅ 测试数据准备完成")

    # 测试多步推理
    print("\n🚀 开始推理测试（5 步）...")
    try:
        for i in range(5):
            # 每次生成新的随机数据
            obs = {
                'observation/image': np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
                'observation/image_1': np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
                'observation/image_2': np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
                'observation/state': np.random.randn(7).astype(np.float32)
            }
            result = wrapper.infer(obs)
            print(f"   Step {i+1}: 动作 shape={result['actions'].shape}, "
                  f"范围=[{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

        print(f"✅ 推理成功！共执行 5 步")
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 检查调试回调
    print("\n🔍 检查调试回调...")
    if wrapper.policy.debug_callback is None:
        print("✅ Policy debug_callback 已正确清理")
    else:
        print("⚠️  Policy debug_callback 未清理")

    if wrapper.policy.obs_encoder.debug_callback is None:
        print("✅ ObsEncoder debug_callback 已正确清理")
    else:
        print("⚠️  ObsEncoder debug_callback 未清理")

    # 显示 WandB 信息
    if wrapper.wandb_debugger.enabled:
        print("\n" + "=" * 60)
        print("📊 WandB 调试信息")
        print("=" * 60)
        print(f"项目: {wrapper.wandb_debugger.wandb.run.project}")
        print(f"运行名称: {wrapper.wandb_debugger.wandb.run.name}")
        print(f"URL: {wrapper.wandb_debugger.wandb.run.url}")
        print("\n请在浏览器中打开上述 URL 查看调试数据：")
        print("  - images/stage1_raw/*")
        print("  - images/stage2_processed/*")
        print("  - images/stage3_normalized/*")
        print("  - images/stage4_final_to_unet/*")
        print("  - debug/camera_mapping")
        print("  - actions/stage1_normalized")
        print("  - actions/stage2_pred_full")
        print("  - actions/stage3_exec")
        print("  - actions/stage4_final")

    print("\n" + "=" * 60)
    print("完整测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

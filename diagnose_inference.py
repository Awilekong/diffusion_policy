#!/usr/bin/env python
"""
推理问题诊断脚本

帮助定位为什么训练和推理数据流一致，但推理效果不好的问题
"""

import torch
import dill
import numpy as np
import sys
from pathlib import Path

def diagnose_checkpoint(ckpt_path: str):
    """诊断 checkpoint"""

    print("=" * 80)
    print("🔍 Diffusion Policy 推理问题诊断")
    print("=" * 80)

    if not Path(ckpt_path).exists():
        print(f"❌ Checkpoint 不存在: {ckpt_path}")
        return

    print(f"\n📂 Checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    # ========== 1. 检查配置 ==========
    print("\n" + "=" * 80)
    print("1️⃣  配置检查")
    print("=" * 80)

    cfg = payload['cfg']
    print(f"\n📝 基本配置:")
    print(f"   模型名称: {cfg.name}")
    print(f"   horizon (预测长度): {cfg.horizon}")
    print(f"   n_obs_steps (观测步数): {cfg.n_obs_steps}")
    print(f"   n_action_steps (执行步数): {cfg.n_action_steps}")
    print(f"   obs_as_global_cond: {cfg.obs_as_global_cond}")

    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'obs_encoder'):
        print(f"\n📸 图像编码器配置:")
        enc_cfg = cfg.policy.obs_encoder
        print(f"   crop_shape: {enc_cfg.get('crop_shape', 'N/A')}")
        print(f"   resize_shape: {enc_cfg.get('resize_shape', 'N/A')}")
        print(f"   random_crop: {enc_cfg.get('random_crop', 'N/A')}")
        print(f"   imagenet_norm: {enc_cfg.get('imagenet_norm', 'N/A')}")

    # ========== 2. 检查 Normalizer 参数 ==========
    print("\n" + "=" * 80)
    print("2️⃣  Normalizer 参数检查（最关键！）")
    print("=" * 80)

    if 'state_dicts' in payload and 'model' in payload['state_dicts']:
        model_state = payload['state_dicts']['model']

        # 找到所有 normalizer 相关的参数
        normalizer_params = {}
        for key in sorted(model_state.keys()):
            if 'normalizer' in key:
                normalizer_params[key] = model_state[key]

        if normalizer_params:
            print(f"\n找到 {len(normalizer_params)} 个 normalizer 参数:")
            for key, value in normalizer_params.items():
                print(f"\n   📊 {key}:")
                print(f"      shape: {value.shape}")
                print(f"      dtype: {value.dtype}")
                print(f"      device: {value.device}")

                # 转成 numpy 方便查看
                val_np = value.cpu().numpy()
                print(f"      mean: {val_np.mean():.6f}")
                print(f"      std: {val_np.std():.6f}")
                print(f"      min: {val_np.min():.6f}")
                print(f"      max: {val_np.max():.6f}")

                # 如果参数不多，打印具体值
                if value.numel() <= 20:
                    print(f"      值: {val_np.flatten()}")
                else:
                    print(f"      前10个值: {val_np.flatten()[:10]}")
        else:
            print("   ⚠️  未找到 normalizer 参数（可能是旧版本 checkpoint）")
    else:
        print("   ❌ 未找到 state_dicts")

    # ========== 3. 检查 shape_meta ==========
    print("\n" + "=" * 80)
    print("3️⃣  Shape Meta 检查")
    print("=" * 80)

    shape_meta = cfg.shape_meta
    print(f"\n📐 观测空间 (obs):")
    for key, attr in shape_meta['obs'].items():
        obs_type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        print(f"   {key}:")
        print(f"      type: {obs_type}")
        print(f"      shape: {shape}")
        if obs_type == 'rgb':
            c, h, w = shape
            print(f"      分辨率: {w}x{h} (宽x高), 通道={c}")

    print(f"\n🎯 动作空间 (action):")
    action_shape = shape_meta['action']['shape']
    print(f"   shape: {action_shape}")
    print(f"   维度: {action_shape[0]}")

    # ========== 4. 训练状态检查 ==========
    print("\n" + "=" * 80)
    print("4️⃣  训练状态检查")
    print("=" * 80)

    if 'epoch' in payload:
        print(f"   训练轮数: {payload['epoch']}")
    if 'global_step' in payload:
        print(f"   全局步数: {payload['global_step']}")

    # ========== 5. 常见问题检查 ==========
    print("\n" + "=" * 80)
    print("5️⃣  常见问题自动检查")
    print("=" * 80)

    issues = []

    # 检查 1: n_obs_steps 是否为 1
    if cfg.n_obs_steps != 1:
        issues.append(f"⚠️  n_obs_steps={cfg.n_obs_steps}（推理脚本假设为1）")

    # 检查 2: random_crop 在推理时应该关闭
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'obs_encoder'):
        if cfg.policy.obs_encoder.get('random_crop', False):
            issues.append("⚠️  训练时启用了 random_crop，推理时会使用中心裁剪")

    # 检查 3: 图像分辨率
    if 'obs' in shape_meta:
        for key, attr in shape_meta['obs'].items():
            if attr.get('type') == 'rgb':
                c, h, w = attr['shape']
                if (h, w) != (240, 320):
                    issues.append(f"⚠️  {key} 分辨率为 {w}x{h}，不是标准的 320x240")

    if issues:
        print("\n发现以下潜在问题:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n   ✅ 未发现明显问题")

    # ========== 6. 推理建议 ==========
    print("\n" + "=" * 80)
    print("6️⃣  推理问题排查建议")
    print("=" * 80)

    print("""
推理效果不好可能的原因：

1. 🎯 动作空间问题
   - 检查客户端接收到的动作范围是否合理
   - 检查动作是否被正确 unnormalize
   - 确认动作维度和机器人控制接口一致

2. 📸 相机输入问题
   - 真机相机视角和训练数据是否一致
   - 检查 WandB 中的 debug/camera_mapping 表
   - 确认客户端发送的相机键名和训练时一致

3. ⏱️  时序问题
   - n_action_steps: 每次执行多少个动作
   - 确认客户端是否正确执行动作序列

4. 🔧 环境差异
   - 训练数据的场景和真机环境是否匹配
   - 光照、背景、物体位置是否一致

5. 📊 模型训练质量
   - 训练 loss 是否收敛
   - 在 WandB 上检查训练曲线
   - 模型是否训练足够的 epoch

推荐操作：
1. 启用 WandB 调试: --wandb
2. 对比训练和推理的 stage4_final_to_unet 图像
3. 检查推理时的动作输出范围
4. 打印 normalizer 参数验证加载正确
""")

    print("=" * 80)
    print("✅ 诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_inference.py <checkpoint_path>")
        print("\n示例:")
        print("  python diagnose_inference.py data/outputs/2025.12.30/xxx/checkpoints/latest.ckpt")
        sys.exit(1)

    diagnose_checkpoint(sys.argv[1])

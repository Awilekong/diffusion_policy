#!/usr/bin/env python3
"""
诊断脚本：检查训练过程中action的数值范围和归一化状态
"""

import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path

# 设置路径
import sys
import os
sys.path.insert(0, str(Path(__file__).parent))

@hydra.main(
    version_base=None,
    config_path="diffusion_policy/config",
    config_name="train_diffusion_unet_franka_image_workspace"
)
def main(cfg):
    print("=" * 80)
    print("诊断: 训练过程中action的归一化状态")
    print("=" * 80)

    # 加载数据集
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"\n数据集类型: {type(dataset).__name__}")
    print(f"数据集大小: {len(dataset)}")

    # 获取一个样本
    sample = dataset[0]
    action_from_dataset = sample['action'].numpy()

    print(f"\n【Dataset返回的action】")
    print(f"  Shape: {action_from_dataset.shape}")
    print(f"  Dtype: {action_from_dataset.dtype}")
    print(f"  Min: {action_from_dataset.min():.6f}")
    print(f"  Max: {action_from_dataset.max():.6f}")
    print(f"  Mean: {action_from_dataset.mean():.6f}")
    print(f"  Std: {action_from_dataset.std():.6f}")
    print(f"  前3个样本的第一步action (xyz):")
    for i in range(min(3, action_from_dataset.shape[0])):
        print(f"    t={i}: {action_from_dataset[i, :3]}")

    # 获取normalizer
    normalizer = dataset.get_normalizer()
    action_normalizer = normalizer['action']

    print(f"\n【Action Normalizer参数】")
    print(f"  Input stats:")
    if hasattr(action_normalizer, 'params_dict'):
        params = action_normalizer.params_dict()
        for key, val in params.items():
            if isinstance(val, (np.ndarray, torch.Tensor)):
                val_np = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
                print(f"    {key}: {val_np[:7]}")  # 只打印前7维

    # 测试归一化
    action_tensor = torch.from_numpy(action_from_dataset).unsqueeze(0)  # (1, T, 7)
    naction = action_normalizer.normalize(action_tensor)
    denorm_action = action_normalizer.unnormalize(naction)

    print(f"\n【归一化测试】")
    print(f"  原始action范围: [{action_from_dataset.min():.6f}, {action_from_dataset.max():.6f}]")
    print(f"  归一化后范围: [{naction.numpy().min():.6f}, {naction.numpy().max():.6f}]")
    print(f"  反归一化后范围: [{denorm_action.numpy().min():.6f}, {denorm_action.numpy().max():.6f}]")
    print(f"  反归一化误差: {torch.abs(denorm_action - action_tensor).max().item():.10f}")

    # 加载模型并测试predict_action
    print(f"\n【加载模型测试predict_action】")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.eval()

    # 准备obs
    obs_dict = {k: v.unsqueeze(0) for k, v in sample['obs'].items()}  # 添加batch维度

    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        action_pred = result['action_pred'][0].numpy()  # (T, 7)

    print(f"  Prediction shape: {action_pred.shape}")
    print(f"  Prediction range: [{action_pred.min():.6f}, {action_pred.max():.6f}]")
    print(f"  Prediction mean: {action_pred.mean():.6f}")
    print(f"  Prediction std: {action_pred.std():.6f}")
    print(f"  前3步predicted action (xyz):")
    for i in range(min(3, action_pred.shape[0])):
        print(f"    t={i}: {action_pred[i, :3]}")

    # 对比GT和Pred
    print(f"\n【GT vs Pred 数值对比】")
    print(f"  GT action[0:3, :3]:")
    print(action_from_dataset[:3, :3])
    print(f"\n  Pred action[0:3, :3]:")
    print(action_pred[:3, :3])

    gt_xyz = action_from_dataset[:, :3]
    pred_xyz = action_pred[:, :3]

    gt_distances = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
    pred_distances = np.linalg.norm(np.diff(pred_xyz, axis=0), axis=1)

    print(f"\n【轨迹点之间距离】")
    print(f"  GT相邻点距离:")
    print(f"    Mean: {gt_distances.mean():.6f}")
    print(f"    Min: {gt_distances.min():.6f}")
    print(f"    Max: {gt_distances.max():.6f}")
    print(f"    First 5: {gt_distances[:5]}")

    print(f"\n  Pred相邻点距离:")
    print(f"    Mean: {pred_distances.mean():.6f}")
    print(f"    Min: {pred_distances.min():.6f}")
    print(f"    Max: {pred_distances.max():.6f}")
    print(f"    First 5: {pred_distances[:5]}")

    print(f"\n  距离比例 (Pred/GT): {pred_distances.mean() / gt_distances.mean():.2f}x")

    # 检查ReplayBuffer中的原始数据
    print(f"\n【ReplayBuffer中的原始action数据】")
    raw_actions = dataset.replay_buffer['action'][:]
    print(f"  Shape: {raw_actions.shape}")
    print(f"  Range: [{raw_actions.min():.6f}, {raw_actions.max():.6f}]")
    print(f"  Mean: {raw_actions.mean():.6f}")
    print(f"  Std: {raw_actions.std():.6f}")
    print(f"  First sample (first 3 steps, xyz):")
    for i in range(3):
        print(f"    t={i}: {raw_actions[i, :3]}")

    print("\n" + "=" * 80)
    print("诊断完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

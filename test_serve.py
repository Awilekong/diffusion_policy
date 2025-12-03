#!/usr/bin/env python3
"""
测试 serve_diffusion_policy.py 脚本
验证 checkpoint 加载和推理功能
"""

import sys
import os
import numpy as np
import torch

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web_policy'))

from serve_diffusion_policy import DiffusionPolicyWrapper


def test_checkpoint_loading(ckpt_path: str, device: str = 'cuda'):
    """测试 checkpoint 加载"""
    print("=" * 70)
    print("🧪 测试 1: Checkpoint 加载")
    print("=" * 70)
    
    try:
        policy = DiffusionPolicyWrapper(
            ckpt_path=ckpt_path,
            device=device
        )
        print("\n✅ Checkpoint 加载成功！")
        return policy
    except Exception as e:
        print(f"\n❌ Checkpoint 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_metadata(policy: DiffusionPolicyWrapper):
    """测试元数据"""
    print("\n" + "=" * 70)
    print("🧪 测试 2: Policy 元数据")
    print("=" * 70)
    
    metadata = policy.metadata
    print("\n📊 元数据信息:")
    for key, value in metadata.items():
        if key == 'shape_meta':
            print(f"   {key}:")
            print(f"      obs keys: {list(value['obs'].keys())}")
            print(f"      action keys: {list(value['action'].keys())}")
        else:
            print(f"   {key}: {value}")
    
    return metadata


def test_dummy_inference(policy: DiffusionPolicyWrapper):
    """测试虚拟数据推理"""
    print("\n" + "=" * 70)
    print("🧪 测试 3: 虚拟数据推理")
    print("=" * 70)
    
    shape_meta = policy.shape_meta
    n_obs_steps = policy.n_obs_steps
    
    # 创建虚拟观测数据
    print(f"\n🔨 创建虚拟观测数据 (n_obs_steps={n_obs_steps})...")
    dummy_obs = {}
    
    for key, attr in shape_meta['obs'].items():
        obs_type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        
        if obs_type == 'rgb':
            # 图像: shape_meta 中是 (C, H, W)，输入需要 (n_obs_steps, H, W, C)
            c, h, w = shape
            dummy_obs[key] = np.random.randint(
                0, 255, 
                size=(n_obs_steps, h, w, c), 
                dtype=np.uint8
            )
            print(f"   {key} (图像): shape={dummy_obs[key].shape}, dtype={dummy_obs[key].dtype}")
        
        elif obs_type == 'low_dim':
            # 低维数据: (n_obs_steps, *shape)
            dummy_obs[key] = np.random.randn(n_obs_steps, *shape).astype(np.float32)
            print(f"   {key} (低维): shape={dummy_obs[key].shape}, dtype={dummy_obs[key].dtype}")
    
    # 执行推理
    print(f"\n🔮 执行推理...")
    try:
        result = policy.infer(dummy_obs)
        
        print(f"\n✅ 推理成功！")
        print(f"\n📤 输出结果:")
        for key, value in result.items():
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            if value.size <= 20:  # 如果数据量不大，打印一下
                print(f"      值: {value.flatten()}")
            else:
                print(f"      前5个值: {value.flatten()[:5]}")
                print(f"      后5个值: {value.flatten()[-5:]}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_inferences(policy: DiffusionPolicyWrapper, n_runs: int = 5):
    """测试多次推理"""
    print("\n" + "=" * 70)
    print(f"🧪 测试 4: 多次推理 (n={n_runs})")
    print("=" * 70)
    
    shape_meta = policy.shape_meta
    n_obs_steps = policy.n_obs_steps
    
    # 创建虚拟观测数据
    dummy_obs = {}
    for key, attr in shape_meta['obs'].items():
        obs_type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        
        if obs_type == 'rgb':
            c, h, w = shape
            dummy_obs[key] = np.random.randint(
                0, 255, 
                size=(n_obs_steps, h, w, c), 
                dtype=np.uint8
            )
        elif obs_type == 'low_dim':
            dummy_obs[key] = np.random.randn(n_obs_steps, *shape).astype(np.float32)
    
    # 多次推理
    print(f"\n🔄 执行 {n_runs} 次推理...")
    import time
    
    inference_times = []
    actions_list = []
    
    for i in range(n_runs):
        start_time = time.time()
        result = policy.infer(dummy_obs)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        inference_times.append(inference_time)
        actions_list.append(result['actions'])
        
        print(f"   Run {i+1}: {inference_time:.2f} ms")
    
    # 统计
    print(f"\n📊 推理性能统计:")
    print(f"   平均时间: {np.mean(inference_times):.2f} ms")
    print(f"   最快: {np.min(inference_times):.2f} ms")
    print(f"   最慢: {np.max(inference_times):.2f} ms")
    print(f"   标准差: {np.std(inference_times):.2f} ms")
    
    # 检查输出是否稳定（相同输入应该产生相同输出）
    print(f"\n🔍 检查输出确定性:")
    if n_runs >= 2:
        diff = np.abs(actions_list[0] - actions_list[1])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"   前两次推理差异 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        if max_diff < 1e-5:
            print(f"   ✅ 输出确定性良好（差异 < 1e-5）")
        else:
            print(f"   ⚠️ 输出存在差异（可能由于随机采样）")


def test_reset(policy: DiffusionPolicyWrapper):
    """测试 reset 功能"""
    print("\n" + "=" * 70)
    print("🧪 测试 5: Reset 功能")
    print("=" * 70)
    
    try:
        policy.reset()
        print("\n✅ Reset 成功！")
    except Exception as e:
        print(f"\n❌ Reset 失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Checkpoint 路径
    ckpt_path = "/home/zpw/ws_zpw/megvii/IL/diffusion_policy/data/outputs/2025.11.23/23.27.07_train_diffusion_unet_franka_image_franka_peg_in_hole_image/checkpoints/latest.ckpt"
    
    print("\n🎯 测试目标:")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 测试 1: 加载 checkpoint
    policy = test_checkpoint_loading(ckpt_path)
    if policy is None:
        print("\n❌ 测试终止：无法加载 checkpoint")
        return
    
    # 测试 2: 元数据
    metadata = test_metadata(policy)
    
    # 测试 3: 虚拟数据推理
    result = test_dummy_inference(policy)
    if result is None:
        print("\n⚠️ 跳过后续测试")
        return
    
    # 测试 4: 多次推理
    test_multiple_inferences(policy, n_runs=5)
    
    # 测试 5: Reset
    test_reset(policy)
    
    # 总结
    print("\n" + "=" * 70)
    print("🎉 所有测试完成！")
    print("=" * 70)
    print("\n✅ 下一步:")
    print("   1. 启动服务器: python serve_diffusion_policy.py -i <checkpoint_path>")
    print("   2. 测试客户端: python test_remote_inference.py")
    print()


if __name__ == '__main__':
    main()

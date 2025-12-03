#!/usr/bin/env python3
"""
Diffusion Policy 远程推理客户端示例

演示如何连接到 Diffusion Policy 服务器并获取动作
"""

import numpy as np
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web_policy'))

from web_policy import WebSocketClientPolicy


def create_dummy_observation(metadata, n_obs_steps=2):
    """
    根据 shape_meta 创建虚拟观测数据
    
    Args:
        metadata: 服务器返回的元数据
        n_obs_steps: 观测步数
    
    Returns:
        obs: 观测数据字典
    """
    obs = {}
    shape_meta = metadata['shape_meta']
    
    for key, attr in shape_meta['obs'].items():
        obs_type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        
        if obs_type == 'rgb':
            # 图像数据: (n_obs_steps, H, W, C)
            c, h, w = shape
            obs[key] = np.random.randint(
                0, 255, 
                size=(n_obs_steps, h, w, c), 
                dtype=np.uint8
            )
            print(f"   {key}: shape={obs[key].shape}, dtype={obs[key].dtype} (图像)")
            
        elif obs_type == 'low_dim':
            # 低维数据: (n_obs_steps, ...)
            obs[key] = np.random.randn(n_obs_steps, *shape).astype(np.float32)
            print(f"   {key}: shape={obs[key].shape}, dtype={obs[key].dtype} (低维)")
    
    # 添加时间戳（可选）
    obs['timestamp'] = np.arange(n_obs_steps, dtype=np.float64)
    
    return obs


def main():
    print("=" * 60)
    print("Diffusion Policy 远程推理客户端")
    print("=" * 60)
    
    # 连接到服务器
    print("\n🔌 连接到服务器 localhost:8000...")
    client = WebSocketClientPolicy(
        host="localhost",
        port=8000
    )
    
    # 获取服务器元数据
    print("\n📊 服务器元数据:")
    metadata = client.get_server_metadata()
    for key, value in metadata.items():
        if key != 'shape_meta':  # shape_meta 太长，不打印全部
            print(f"   {key}: {value}")
    
    n_obs_steps = metadata['n_obs_steps']
    print(f"\n📸 创建虚拟观测数据 (n_obs_steps={n_obs_steps}):")
    obs = create_dummy_observation(metadata, n_obs_steps)
    
    # 推理测试
    print(f"\n🚀 执行推理测试...")
    num_tests = 5
    
    for i in range(num_tests):
        print(f"\n--- 推理 {i+1}/{num_tests} ---")
        
        start_time = time.time()
        result = client.infer(obs)
        inference_time = time.time() - start_time
        
        actions = result['actions']
        server_timing = result.get('server_timing', {})
        
        print(f"✅ 推理成功!")
        print(f"   动作 shape: {actions.shape}")
        print(f"   动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"   服务器推理耗时: {server_timing.get('infer_ms', 0):.2f} ms")
        print(f"   客户端总耗时: {inference_time*1000:.2f} ms")
        
        # 模拟控制频率
        time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

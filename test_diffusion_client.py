#!/usr/bin/env python3
"""
测试 DiffusionPolicyClient
连接到 serve_diffusion_policy.py 服务器并进行推理测试

Usage:
    # 先启动服务器
    python serve_diffusion_policy.py -i <checkpoint_path> -p 8000
    
    # 然后运行此测试脚本
    python test_diffusion_client.py
"""

import sys
import os
import numpy as np
import cv2

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web_policy'))

from web_policy.utils import DiffusionPolicyClient


def main():
    print("=" * 60)
    print("测试 DiffusionPolicyClient")
    print("=" * 60)
    
    # 创建客户端
    print("\n🔄 连接到服务器...")
    client = DiffusionPolicyClient(base_url='http://localhost:8000')
    
    # 创建测试数据
    print("\n🔄 准备测试数据...")
    
    # 图像数据 (模拟三个相机)
    img_0 = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    img_1 = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    img_2 = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    
    # 状态数据 (模拟机器人状态，例如末端位姿)
    state = np.random.randn(14).astype(np.float32)  # 假设 state_dim=14
    
    # 将图像编码为 bytes (模拟从网络接收)
    _, img_0_bytes = cv2.imencode('.jpg', img_0)
    img_0_bytes = img_0_bytes.tobytes()
    _, img_1_bytes = cv2.imencode('.jpg', img_1)
    img_1_bytes = img_1_bytes.tobytes()
    _, img_2_bytes = cv2.imencode('.jpg', img_2)
    img_2_bytes = img_2_bytes.tobytes()
    
    print(f"✅ 测试数据准备完成")
    print(f"   图像 0 shape: {img_0.shape}")
    print(f"   图像 1 shape: {img_1.shape}")
    print(f"   图像 2 shape: {img_2.shape}")
    print(f"   状态 shape: {state.shape}")
    
    # 推理测试
    print("\n🚀 开始推理...")
    try:
        actions = client.process_frame(
            image_0=img_0_bytes,
            image_1=img_1_bytes,
            image_2=img_2_bytes,
            state=state
        )
        
        print(f"✅ 推理成功！")
        print(f"   动作 shape: {actions.shape}")
        print(f"   动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"   动作前 5 步:\n{actions[:5]}")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 多次推理测试
    print("\n🔄 进行 10 次推理测试...")
    import time
    start_time = time.time()
    
    for i in range(10):
        # 每次生成新的随机数据
        img_0 = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        img_1 = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        img_2 = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        state = np.random.randn(14).astype(np.float32)
        
        _, img_0_bytes = cv2.imencode('.jpg', img_0)
        img_0_bytes = img_0_bytes.tobytes()
        _, img_1_bytes = cv2.imencode('.jpg', img_1)
        img_1_bytes = img_1_bytes.tobytes()
        _, img_2_bytes = cv2.imencode('.jpg', img_2)
        img_2_bytes = img_2_bytes.tobytes()
        
        actions = client.process_frame(
            image_0=img_0_bytes,
            image_1=img_1_bytes,
            image_2=img_2_bytes,
            state=state
        )
    
    elapsed = time.time() - start_time
    print(f"✅ 完成 10 次推理")
    print(f"   总时间: {elapsed:.3f} 秒")
    print(f"   平均每次: {elapsed/10:.3f} 秒")
    print(f"   FPS: {10/elapsed:.1f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

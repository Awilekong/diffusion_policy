#!/usr/bin/env python3
"""
分析推理调试日志，找出数据流问题
"""

import numpy as np
import sys
from pathlib import Path

def analyze_npz(npz_path):
    """分析单个npz文件"""
    print(f"\n{'='*80}")
    print(f"📂 分析文件: {npz_path.name}")
    print(f"{'='*80}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    print("\n📦 文件内容:")
    for key in data.files:
        print(f"   - {key}")
    
    # 1. 分析原始输入
    if 'input_obs_raw' in data:
        print("\n1️⃣ 原始输入 (input_obs_raw):")
        obs_raw = data['input_obs_raw'].item()
        for k, v in obs_raw.items():
            if isinstance(v, np.ndarray):
                print(f"   {k}:")
                print(f"      shape: {v.shape}, dtype: {v.dtype}")
                if 'image' in k:
                    print(f"      像素范围: [{v.min()}, {v.max()}], 均值: {v.mean():.2f}")
                    if v.max() == 0:
                        print(f"      ⚠️  图像全黑！")
                else:
                    print(f"      数值范围: [{v.min():.4f}, {v.max():.4f}]")
                    print(f"      前5个值: {v.ravel()[:5]}")
            else:
                print(f"   {k}: {type(v)}")
    
    # 2. 分析历史队列组装
    if 'env_obs' in data:
        print("\n2️⃣ 历史队列组装 (env_obs):")
        env_obs = data['env_obs'].item()
        for k, v in env_obs.items():
            print(f"   {k}:")
            print(f"      shape: {v.shape}, dtype: {v.dtype}")
            if 'camera' in k:
                print(f"      像素范围: [{v.min()}, {v.max()}], 均值: {v.mean():.2f}")
            else:
                print(f"      数值范围: [{v.min():.4f}, {v.max():.4f}]")
                # 打印帧数据
                for i in range(min(len(v), 2)):
                    print(f"      第{i+1}帧: {v[i]}")
    
    # 3. 分析预处理后数据
    if 'obs_dict_np' in data:
        print("\n3️⃣ 预处理后 (obs_dict_np):")
        obs_dict_np = data['obs_dict_np'].item()
        for k, v in obs_dict_np.items():
            print(f"   {k}:")
            print(f"      shape: {v.shape}, dtype: {v.dtype}")
            if len(v.shape) == 4:  # 图像
                print(f"      像素范围: [{v.min():.4f}, {v.max():.4f}], 均值: {v.mean():.4f}")
            else:
                print(f"      数值范围: [{v.min():.4f}, {v.max():.4f}]")
                for i in range(min(len(v), 2)):
                    print(f"      第{i+1}帧: {v[i]}")
    
    # 4. 分析模型输入
    if 'obs_dict_tensor' in data:
        print("\n4️⃣ 模型输入 (obs_dict_tensor):")
        obs_dict_tensor = data['obs_dict_tensor'].item()
        for k, v in obs_dict_tensor.items():
            print(f"   {k}:")
            print(f"      shape: {v.shape}, dtype: {v.dtype}")
            if len(v.shape) == 5:  # 图像
                print(f"      像素范围: [{v.min():.4f}, {v.max():.4f}], 均值: {v.mean():.4f}")
            else:
                print(f"      数值范围: [{v.min():.4f}, {v.max():.4f}]")
                # batch=1, 去掉batch维度
                for i in range(min(v.shape[1], 2)):
                    print(f"      第{i+1}帧: {v[0, i]}")
    
    # 5. 分析输出动作
    if 'action' in data:
        print("\n5️⃣ 输出动作 (action):")
        action = data['action']
        print(f"   shape: {action.shape}, dtype: {action.dtype}")
        print(f"   数值范围: [{action.min():.4f}, {action.max():.4f}]")
        print(f"   均值: {action.mean():.4f}, 标准差: {action.std():.4f}")
        print(f"\n   前3个动作:")
        for i in range(min(3, len(action))):
            print(f"      动作{i}: {action[i]}")
        
        # 检查动作是否异常
        if np.allclose(action, 0):
            print(f"   ⚠️  动作全为0！")
        if np.isnan(action).any():
            print(f"   ⚠️  动作包含NaN！")
        if np.isinf(action).any():
            print(f"   ⚠️  动作包含Inf！")
    
    # 6. 总结问题
    print("\n" + "="*80)
    print("📋 问题总结:")
    print("="*80)
    
    issues = []
    
    # 检查图像问题
    if 'input_obs_raw' in data:
        obs_raw = data['input_obs_raw'].item()
        for k, v in obs_raw.items():
            if isinstance(v, np.ndarray) and 'image' in k:
                if v.max() == 0:
                    issues.append(f"❌ {k}: 图像全黑（全0）")
    
    # 检查状态数据一致性
    if 'env_obs' in data:
        env_obs = data['env_obs'].item()
        if 'robot_eef_pose' in env_obs:
            state = env_obs['robot_eef_pose']
            if len(state) > 1 and np.allclose(state[0], state[1]):
                issues.append(f"⚠️  状态历史队列中两帧完全相同")
    
    # 检查动作异常
    if 'action' in data:
        action = data['action']
        if np.allclose(action, 0):
            issues.append(f"❌ 输出动作全为0")
        if np.isnan(action).any():
            issues.append(f"❌ 输出动作包含NaN")
        if action.std() < 0.001:
            issues.append(f"⚠️  输出动作标准差过小 ({action.std():.6f})，可能缺乏多样性")
    
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ 未发现明显问题")
    
    print()

def main():
    if len(sys.argv) > 1:
        npz_path = Path(sys.argv[1])
        if not npz_path.exists():
            print(f"❌ 文件不存在: {npz_path}")
            return
        analyze_npz(npz_path)
    else:
        # 分析最新的文件
        debug_dir = Path(__file__).parent / "debug_logs"
        if not debug_dir.exists():
            print(f"❌ 调试日志目录不存在: {debug_dir}")
            return
        
        npz_files = sorted(debug_dir.glob("*.npz"))
        if not npz_files:
            print(f"❌ 没有找到调试日志文件")
            return
        
        # 分析最新的一个
        latest = npz_files[-1]
        print(f"📊 分析最新的调试日志...")
        analyze_npz(latest)

if __name__ == '__main__':
    main()

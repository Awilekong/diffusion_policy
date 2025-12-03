#!/usr/bin/env python3
"""
对比推理日志和训练数据

用于诊断训练和推理时的 obs 处理是否对齐

Usage:
    # 对比推理日志与训练数据
    python compare_debug.py \
        --inference-log debug_logs/20231201_120000_step_0000.npz \
        --train-dataset /path/to/dataset.zarr \
        --episode 0 --step 10
    
    # 只查看推理日志
    python compare_debug.py \
        --inference-log debug_logs/20231201_120000_step_0000.npz
"""

import click
import numpy as np
from pathlib import Path
import zarr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_data_stats(name, data, indent=2):
    """打印数据统计信息"""
    prefix = " " * indent
    if isinstance(data, dict):
        print(f"{prefix}{name}: (dict with {len(data)} keys)")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"{prefix}  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{prefix}  {key}: {type(value).__name__}")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}{name}: shape={data.shape}, dtype={data.dtype}")
    else:
        print(f"{prefix}{name}: {type(data).__name__} = {data}")


def visualize_image_processing_pipeline(inference_data, train_data=None, output_dir=None):
    """
    可视化图像处理流水线的每一步
    
    对比推理和训练时的图像处理过程：
    1. input_obs_raw: 原始输入图像
    2. env_obs: 历史队列组装后
    3. obs_dict_np: get_real_obs_dict 输出（resize + 归一化）
    4. obs_dict_tensor: 模型输入
    """
    print_section("📸 可视化图像处理流水线")
    
    if output_dir is None:
        output_dir = Path('debug_comparison')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 提取推理数据
    input_raw = inference_data.get('input_obs_raw', {})
    env_obs = inference_data.get('env_obs', {})
    obs_dict_np = inference_data.get('obs_dict_np', {})
    obs_dict_tensor = inference_data.get('obs_dict_tensor', {})
    
    # 找到所有相机
    camera_keys = [k for k in env_obs.keys() if 'camera' in k]
    
    if not camera_keys:
        print("   ⚠️  未找到图像数据")
        return
    
    print(f"   找到 {len(camera_keys)} 个相机: {camera_keys}")
    
    for cam_key in camera_keys:
        print(f"\n   🎥 处理 {cam_key}:")
        
        # 提取各阶段数据
        # 1. 原始输入 (如果有)
        raw_img = None
        for obs_key in input_raw.keys():
            if 'image' in obs_key:
                # 找到对应的相机
                if cam_key == 'camera_0' and obs_key == 'observation/image':
                    raw_img = input_raw[obs_key]
                    break
                elif cam_key == 'camera_1' and obs_key == 'observation/image_1':
                    raw_img = input_raw[obs_key]
                    break
                elif cam_key == 'camera_2' and obs_key == 'observation/image_2':
                    raw_img = input_raw[obs_key]
                    break
        
        # 2. 历史队列 (n_obs_steps, H, W, C)
        env_imgs = env_obs.get(cam_key)
        if env_imgs is None:
            print(f"      ⚠️  在 env_obs 中未找到 {cam_key}")
            continue
        
        n_obs_steps = env_imgs.shape[0]
        
        # 3. get_real_obs_dict 输出 (n_obs_steps, C, H, W)
        processed_imgs = obs_dict_np.get(cam_key)
        if processed_imgs is None:
            print(f"      ⚠️  在 obs_dict_np 中未找到 {cam_key}")
            continue
        
        # 4. 模型输入 (1, n_obs_steps, C, H, W)
        tensor_imgs = obs_dict_tensor.get(cam_key)
        if tensor_imgs is not None:
            tensor_imgs = tensor_imgs[0]  # 去掉 batch 维度
        
        # 创建可视化
        # 每一行显示一个历史帧，每一列显示一个处理阶段
        n_stages = 4 if raw_img is not None else 3
        fig = plt.figure(figsize=(5*n_stages, 4*n_obs_steps))
        gs = GridSpec(n_obs_steps, n_stages, figure=fig)
        
        for t in range(n_obs_steps):
            col = 0
            
            # 阶段1: 原始输入（如果是第一帧且有原始数据）
            if raw_img is not None:
                ax = fig.add_subplot(gs[t, col])
                if t == 0:
                    img_show = raw_img
                    if img_show.dtype == np.uint8:
                        ax.imshow(img_show)
                    else:
                        ax.imshow(np.clip(img_show, 0, 1))
                    ax.set_title(f'1. 原始输入 (t={t})\nshape={img_show.shape}\ndtype={img_show.dtype}')
                else:
                    ax.text(0.5, 0.5, f'历史帧 (t={t})\n使用第0帧填充', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'1. 原始输入 (t={t})')
                ax.axis('off')
                col += 1
            
            # 阶段2: 历史队列 (n_obs_steps, H, W, C)
            ax = fig.add_subplot(gs[t, col])
            img_show = env_imgs[t]  # (H, W, C)
            if img_show.dtype == np.uint8:
                ax.imshow(img_show)
            else:
                ax.imshow(np.clip(img_show, 0, 1))
            ax.set_title(f'2. 历史队列 (t={t})\nshape={img_show.shape}\ndtype={img_show.dtype}')
            ax.axis('off')
            col += 1
            
            # 阶段3: get_real_obs_dict 输出 (n_obs_steps, C, H, W)
            ax = fig.add_subplot(gs[t, col])
            img_show = processed_imgs[t]  # (C, H, W)
            img_show = np.transpose(img_show, (1, 2, 0))  # -> (H, W, C)
            ax.imshow(np.clip(img_show, 0, 1))
            ax.set_title(f'3. get_real_obs_dict (t={t})\n处理: resize+归一化\nshape={processed_imgs[t].shape}\nrange=[{processed_imgs[t].min():.2f}, {processed_imgs[t].max():.2f}]')
            ax.axis('off')
            col += 1
            
            # 阶段4: 模型输入 (n_obs_steps, C, H, W)
            if tensor_imgs is not None:
                ax = fig.add_subplot(gs[t, col])
                img_show = tensor_imgs[t]  # (C, H, W)
                img_show = np.transpose(img_show, (1, 2, 0))  # -> (H, W, C)
                ax.imshow(np.clip(img_show, 0, 1))
                ax.set_title(f'4. 模型输入 (t={t})\n添加batch维度\nshape={tensor_imgs[t].shape}\nrange=[{tensor_imgs[t].min():.2f}, {tensor_imgs[t].max():.2f}]')
                ax.axis('off')
        
        plt.suptitle(f'推理图像处理流水线: {cam_key}\n从左到右展示各处理阶段，从上到下展示历史帧', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_file = output_dir / f'{cam_key}_pipeline.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ 已保存: {output_file}")
    
    print(f"\n   📁 所有可视化图片已保存到: {output_dir}")


def compare_arrays(name, arr1, arr2, rtol=1e-5, atol=1e-8):
    """对比两个数组是否接近"""
    print(f"\n  🔍 对比 {name}:")
    print(f"     推理: shape={arr1.shape}, dtype={arr1.dtype}")
    print(f"     训练: shape={arr2.shape}, dtype={arr2.dtype}")
    
    if arr1.shape != arr2.shape:
        print(f"     ❌ Shape 不匹配!")
        return False
    
    if arr1.dtype != arr2.dtype:
        print(f"     ⚠️  Dtype 不同，尝试转换...")
        arr2 = arr2.astype(arr1.dtype)
    
    # 统计信息对比
    print(f"     推理: range=[{arr1.min():.4f}, {arr1.max():.4f}], mean={arr1.mean():.4f}, std={arr1.std():.4f}")
    print(f"     训练: range=[{arr2.min():.4f}, {arr2.max():.4f}], mean={arr2.mean():.4f}, std={arr2.std():.4f}")
    
    # 数值对比
    if np.allclose(arr1, arr2, rtol=rtol, atol=atol):
        print(f"     ✅ 数值完全对齐 (rtol={rtol}, atol={atol})")
        return True
    else:
        diff = np.abs(arr1 - arr2)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"     ❌ 数值不对齐: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        # 找到最大差异的位置
        max_idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"     最大差异位置 {max_idx}: 推理={arr1[max_idx]:.6f}, 训练={arr2[max_idx]:.6f}")
        return False


def load_inference_log(log_path):
    """加载推理日志"""
    print_section("📥 加载推理日志")
    print(f"   文件: {log_path}")
    
    data = np.load(log_path, allow_pickle=True)
    
    # 提取数据
    result = {}
    for key in data.files:
        if key == 'metadata':
            result[key] = data[key].item()  # dict
        else:
            result[key] = data[key].item() if data[key].shape == () else data[key]
    
    print(f"   ✅ 加载成功，包含 {len(result)} 个字段:")
    for key in result.keys():
        print(f"      - {key}")
    
    return result


def load_train_sample(dataset_path, episode_idx, step_idx):
    """从训练数据集加载一个样本"""
    print_section("📥 加载训练数据")
    print(f"   数据集: {dataset_path}")
    print(f"   Episode: {episode_idx}, Step: {step_idx}")
    
    root = zarr.open(dataset_path, 'r')
    
    # 获取样本数据
    sample = {}
    
    # 读取数据键
    data_group = root['data']
    print(f"   可用数据键: {list(data_group.keys())}")
    
    for key in data_group.keys():
        data = data_group[key]
        # 从 episode 的特定 step 提取数据
        if hasattr(data, 'shape') and len(data.shape) > 0:
            # 假设数据格式是 (episode_len, ...) 按 episode 拼接
            # 需要根据你的数据集格式调整
            sample[key] = data[episode_idx, step_idx]
    
    print(f"   ✅ 加载成功")
    return sample


@click.command()
@click.option('--inference-log', '-i', required=True, 
              help='推理日志文件 (.npz)')
@click.option('--train-dataset', '-t', default=None,
              help='训练数据集路径 (.zarr)')
@click.option('--episode', '-e', default=0, type=int,
              help='训练数据集的 episode 索引')
@click.option('--step', '-s', default=0, type=int,
              help='训练数据集的 step 索引')
@click.option('--output-dir', '-o', default='debug_comparison',
              help='可视化输出目录')
def main(inference_log, train_dataset, episode, step, output_dir):
    """对比推理日志和训练数据"""
    
    print("="*80)
    print("  Diffusion Policy 训练推理数据对比工具")
    print("="*80)
    
    # 1. 加载推理日志
    inference_data = load_inference_log(inference_log)
    
    # 2. 打印推理数据流
    print_section("📊 推理数据流")
    
    print("\n1️⃣  input_obs_raw (原始输入):")
    if 'input_obs_raw' in inference_data:
        print_data_stats('input_obs_raw', inference_data['input_obs_raw'])
    
    print("\n2️⃣  env_obs (历史队列组装后):")
    if 'env_obs' in inference_data:
        print_data_stats('env_obs', inference_data['env_obs'])
    
    print("\n3️⃣  obs_dict_np (get_real_obs_dict 输出):")
    if 'obs_dict_np' in inference_data:
        print_data_stats('obs_dict_np', inference_data['obs_dict_np'])
    
    print("\n4️⃣  obs_dict_tensor (模型输入):")
    if 'obs_dict_tensor' in inference_data:
        print_data_stats('obs_dict_tensor', inference_data['obs_dict_tensor'])
    
    print("\n5️⃣  action (模型输出):")
    if 'action' in inference_data:
        print_data_stats('action', inference_data['action'])
    
    print("\n📝 Metadata:")
    if 'metadata' in inference_data:
        metadata = inference_data['metadata']
        for key, value in metadata.items():
            if key != 'shape_meta':  # shape_meta 太长
                print(f"   {key}: {value}")
    
    # 2. 可视化图像处理流水线
    visualize_image_processing_pipeline(inference_data, output_dir=output_dir)
    
    # 3. 如果提供了训练数据集，进行对比
    if train_dataset:
        try:
            train_sample = load_train_sample(train_dataset, episode, step)
            
            print_section("🔄 训练数据对比")
            print("注意：需要根据你的数据集格式调整对比逻辑")
            print("以下是示例对比，可能需要修改:")
            
            # 示例：对比 obs_dict_np 的某些字段
            # 具体对比逻辑需要根据你的数据集格式调整
            print("\n提示：请根据实际数据集结构修改此脚本的对比逻辑")
            
        except Exception as e:
            print(f"\n⚠️  加载训练数据失败: {e}")
            print("   提示：请检查数据集路径和格式")
    
    print_section("✅ 对比完成")
    print(f"\n💡 使用建议:")
    print(f"   1. 查看可视化图片: {output_dir}/")
    print(f"   2. 对比各处理阶段的图像是否符合预期")
    print(f"   3. 检查 resize、归一化等操作是否正确")
    print(f"   4. 重点关注图像的分辨率、颜色范围、内容是否清晰")
    print(f"   5. 如果有训练数据，可以对比训练和推理的图像处理是否一致")


if __name__ == '__main__':
    main()

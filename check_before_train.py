#!/usr/bin/env python
"""
训练前验证脚本 - 检查配置和数据是否正确
"""

import os
import sys
from pathlib import Path

print("="*60)
print("Diffusion Policy 训练前检查")
print("="*60)

# ==================== 1. 检查项目路径 ====================
print("\n[1/7] 检查项目路径...")
project_root = Path("/home/zpw/ws_zpw/megvii/IL/diffusion_policy")
if not project_root.exists():
    print(f"❌ 项目路径不存在: {project_root}")
    sys.exit(1)
print(f"✅ 项目路径存在: {project_root}")

os.chdir(project_root)
sys.path.insert(0, str(project_root))

# ==================== 2. 检查数据集 ====================
print("\n[2/7] 检查数据集...")
dataset_path = Path("/home/zpw/ws_zpw/megvii/data/2025_11_18/zarr_dataset/peg_in_hole_zarr")
zarr_path = dataset_path / "replay_buffer.zarr"
videos_path = dataset_path / "videos"

if not dataset_path.exists():
    print(f"❌ 数据集路径不存在: {dataset_path}")
    sys.exit(1)
print(f"✅ 数据集路径存在: {dataset_path}")

if not zarr_path.exists():
    print(f"❌ Zarr 数据不存在: {zarr_path}")
    sys.exit(1)
print(f"✅ Zarr 数据存在: {zarr_path}")

if not videos_path.exists():
    print(f"❌ 视频目录不存在: {videos_path}")
    sys.exit(1)
print(f"✅ 视频目录存在: {videos_path}")

# ==================== 3. 检查 Zarr 数据结构 ====================
print("\n[3/7] 检查 Zarr 数据结构...")
try:
    import zarr
    root = zarr.open(str(zarr_path), mode='r')
    
    # 检查数据组
    required_keys = ['action', 'robot_eef_pose', 'robot_eef_pose_vel', 
                     'robot_joint', 'robot_joint_vel', 'stage', 'timestamp']
    
    for key in required_keys:
        if key not in root['data']:
            print(f"❌ 缺少数据字段: {key}")
            sys.exit(1)
        data_shape = root['data'][key].shape
        print(f"  ✓ {key}: {data_shape}")
    
    # 检查元数据
    if 'episode_ends' not in root['meta']:
        print("❌ 缺少 episode_ends")
        sys.exit(1)
    
    episode_ends = root['meta']['episode_ends'][:]
    print(f"  ✓ episode_ends: {episode_ends.shape}, {len(episode_ends)} episodes")
    
    # 验证数据维度
    action_shape = root['data']['action'].shape
    eef_shape = root['data']['robot_eef_pose'].shape
    
    if action_shape[1] != 7:
        print(f"❌ action 维度错误: {action_shape[1]}, 应该是 7")
        sys.exit(1)
    
    if eef_shape[1] != 7:
        print(f"❌ robot_eef_pose 维度错误: {eef_shape[1]}, 应该是 7")
        sys.exit(1)
    
    print(f"✅ 数据维度正确: action={action_shape[1]}D, robot_eef_pose={eef_shape[1]}D")
    
except Exception as e:
    print(f"❌ 读取 Zarr 数据失败: {e}")
    sys.exit(1)

# ==================== 4. 检查视频文件 ====================
print("\n[4/7] 检查视频文件...")
episode_dirs = sorted([d for d in videos_path.iterdir() if d.is_dir()])
if len(episode_dirs) == 0:
    print(f"❌ 没有找到视频文件")
    sys.exit(1)

print(f"✅ 找到 {len(episode_dirs)} 个 episode 的视频")

# 检查第一个 episode 的视频
first_episode = episode_dirs[0]
video_files = list(first_episode.glob("*.mp4"))
print(f"  第一个 episode ({first_episode.name}) 有 {len(video_files)} 个相机视频:")
for vf in sorted(video_files):
    print(f"    - {vf.name}")

expected_cameras = {0, 1, 2}  # camera_0, camera_1, camera_2
actual_cameras = {int(vf.stem) for vf in video_files}
if actual_cameras != expected_cameras:
    print(f"⚠️  相机编号不完整: 期望 {expected_cameras}, 实际 {actual_cameras}")
else:
    print(f"  ✅ 相机配置正确: {actual_cameras}")

# ==================== 5. 检查配置文件 ====================
print("\n[5/7] 检查配置文件...")
config_file = project_root / "diffusion_policy" / "config" / "train_diffusion_unet_franka_image_workspace.yaml"
task_config_file = project_root / "diffusion_policy" / "config" / "task" / "franka_peg_in_hole_image.yaml"

if not config_file.exists():
    print(f"❌ 训练配置文件不存在: {config_file}")
    sys.exit(1)
print(f"✅ 训练配置文件存在: {config_file.name}")

if not task_config_file.exists():
    print(f"❌ 任务配置文件不存在: {task_config_file}")
    sys.exit(1)
print(f"✅ 任务配置文件存在: {task_config_file.name}")

# ==================== 6. 尝试加载配置 ====================
print("\n[6/7] 尝试加载配置...")
try:
    import hydra
    from omegaconf import OmegaConf
    
    # 注册 eval resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    with hydra.initialize(config_path="diffusion_policy/config", version_base=None):
        cfg = hydra.compose(
            config_name="train_diffusion_unet_franka_image_workspace",
            overrides=[f"task.dataset_path={dataset_path}"]
        )
        OmegaConf.resolve(cfg)
    
    print(f"✅ 配置加载成功")
    print(f"  - Task: {cfg.task_name}")
    print(f"  - Dataset: {cfg.task.dataset_path}")
    print(f"  - Batch size: {cfg.dataloader.batch_size}")
    print(f"  - Epochs: {cfg.training.num_epochs}")
    print(f"  - Horizon: {cfg.horizon}")
    print(f"  - Obs steps: {cfg.n_obs_steps}")
    print(f"  - Action steps: {cfg.n_action_steps}")
    
    # 检查 shape_meta
    print(f"\n  Shape Meta:")
    for key, value in cfg.shape_meta.obs.items():
        print(f"    obs.{key}: shape={value.shape}, type={value.type}")
    print(f"    action: shape={cfg.shape_meta.action.shape}")
    
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 7. 尝试实例化数据集 ====================
print("\n[7/7] 尝试实例化数据集...")
try:
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"✅ 数据集实例化成功")
    print(f"  - 训练集大小: {len(dataset)}")
    print(f"  - Replay buffer episodes: {dataset.replay_buffer.n_episodes}")
    print(f"  - Replay buffer steps: {dataset.replay_buffer.n_steps}")
    
    # 尝试获取一个样本
    print(f"\n  尝试获取第一个样本...")
    sample = dataset[0]
    print(f"  ✅ 样本获取成功")
    print(f"    obs keys: {list(sample['obs'].keys())}")
    for key, value in sample['obs'].items():
        print(f"      {key}: {value.shape}, dtype={value.dtype}")
    print(f"    action: {sample['action'].shape}, dtype={sample['action'].dtype}")
    
    # 检查 normalizer
    print(f"\n  检查 Normalizer...")
    normalizer = dataset.get_normalizer()
    print(f"  ✅ Normalizer 创建成功")
    for key in normalizer.keys():
        print(f"    {key}: {type(normalizer[key]).__name__}")
    
except Exception as e:
    print(f"❌ 数据集实例化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 完成 ====================
print("\n" + "="*60)
print("✅ 所有检查通过！可以开始训练了")
print("="*60)
print("\n训练命令:")
print(f"cd {project_root}")
print("python train.py --config-name=train_diffusion_unet_franka_image_workspace")
print("\n或者使用脚本:")
print("bash train_franka.sh")
print("="*60)

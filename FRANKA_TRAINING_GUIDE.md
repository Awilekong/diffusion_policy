# Franka 机器人 Diffusion Policy 训练配置指南

## 📋 概述

本指南说明如何使用你采集的 Franka 机器人数据训练 Diffusion Policy。

## 🗂️ 数据格式

你的数据转换脚本 `franka_to_zarr.py` 生成的数据结构：

```
peg_in_hole_zarr/
├── replay_buffer.zarr/
│   ├── data/
│   │   ├── action              # (N, 7) - 6维末端位姿 + 1维夹爪
│   │   ├── robot_eef_pose      # (N, 7) - xyz + rotation_vector + gripper
│   │   ├── robot_eef_pose_vel  # (N, 7)
│   │   ├── robot_joint         # (N, 8) - 7个关节 + 1个夹爪
│   │   ├── robot_joint_vel     # (N, 8)
│   │   ├── stage               # (N,)
│   │   └── timestamp           # (N,)
│   └── meta/
│       └── episode_ends        # (num_episodes,)
└── videos/
    ├── 0/  # episode_0
    │   ├── 0.mp4  # camera_0 (main_realsense)
    │   ├── 1.mp4  # camera_1 (handeye_realsense)
    │   └── 2.mp4  # camera_2 (side_realsense)
    └── 1/  # episode_1
        └── ...
```

## 📝 配置文件说明

我为你创建了以下配置文件：

### 1. Task 配置 (任务定义)

#### `franka_peg_in_hole_image.yaml` - 完整版（3个相机）
```yaml
obs:
  camera_0: [3, 480, 640]  # main_realsense
  camera_1: [3, 480, 640]  # handeye_realsense
  camera_2: [3, 480, 640]  # side_realsense
  robot_eef_pose: [7]      # 末端位姿 + 夹爪
action: [7]                # 6维末端 + 1维夹爪
```

#### `franka_peg_in_hole_image_minimal.yaml` - 最小版（1个相机）
```yaml
obs:
  camera_0: [3, 480, 640]  # 只用主相机
  robot_eef_pose: [7]      # 末端位姿 + 夹爪
action: [7]
```

### 2. Training 配置

#### `train_diffusion_unet_franka_image_workspace.yaml`
主要训练配置文件，包含：
- 网络架构参数
- 训练超参数
- 数据加载配置

## 🔧 需要修改的关键参数

### 1. **数据路径** (必须修改)
在 `franka_peg_in_hole_image.yaml` 中：
```yaml
dataset_path: /home/zpw/ws_zpw/megvii/data/2025_11_18/zarr_dataset/peg_in_hole_zarr
```

### 2. **相机选择** (根据需要修改)
如果不想使用全部3个相机，可以注释掉不需要的：
```yaml
obs:
  camera_0:  # 保留
    shape: ${task.image_shape}
    type: rgb
  # camera_1:  # 注释掉不用的
  #   shape: ${task.image_shape}
  #   type: rgb
  camera_2:  # 保留
    shape: ${task.image_shape}
    type: rgb
```

### 3. **训练超参数** (可选调整)
在 `train_diffusion_unet_franka_image_workspace.yaml` 中：

```yaml
# 预测相关
horizon: 16          # 预测未来16步
n_obs_steps: 2       # 使用2帧历史观测
n_action_steps: 8    # 执行前8步

# 训练相关
batch_size: 64       # 根据GPU显存调整（建议 16-64）
num_epochs: 3000     # 训练轮数
lr: 1.0e-4           # 学习率

# 验证集
val_ratio: 0.02      # 2%数据作为验证集
```

### 4. **图像预处理** (可选调整)
```yaml
obs_encoder:
  resize_shape: [480, 640]  # 输入分辨率
  crop_shape: [432, 576]    # 随机裁剪大小（90%）
  random_crop: True         # 训练时随机裁剪
```

## 🚀 使用方法

### 方法1: 使用完整配置（3相机）
```bash
cd /home/zpw/ws_zpw/megvii/IL/diffusion_policy

python train.py --config-name=train_diffusion_unet_franka_image_workspace
```

### 方法2: 使用最小配置（1相机）
修改 `train_diffusion_unet_franka_image_workspace.yaml` 第3行：
```yaml
- task: franka_peg_in_hole_image_minimal
```
然后运行：
```bash
python train.py --config-name=train_diffusion_unet_franka_image_workspace
```

### 方法3: 命令行覆盖参数
```bash
python train.py --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/path/to/your/data \
    dataloader.batch_size=32 \
    training.num_epochs=1000
```

## 📊 训练监控

训练过程会自动记录到 Weights & Biases：
- Project name: `diffusion_policy_franka`
- 可以在 wandb 网页查看训练曲线

输出目录：
```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_franka_image_franka_peg_in_hole_image/
├── checkpoints/  # 模型检查点
├── logs/         # 训练日志
└── videos/       # 验证时生成的视频（如果有）
```

## ⚙️ 常见问题

### 1. 内存不足 (OOM)
减小 `batch_size`：
```yaml
dataloader:
  batch_size: 16  # 从 64 降到 16
```

### 2. 训练太慢
- 减少 `num_workers`
- 关闭 `use_cache` (如果数据集很大)
- 使用更小的网络：将 `resnet18` 改为更小的模型

### 3. 验证集太小
增加 `val_ratio`：
```yaml
dataset:
  val_ratio: 0.1  # 10% 作为验证集
```

### 4. 想使用预训练权重
修改 `rgb_model` 的 `weights`：
```yaml
rgb_model:
  weights: 'IMAGENET1K_V1'  # 使用 ImageNet 预训练
```

## 🎯 关键配置选项对比

| 配置项 | pusht (原始) | 你的 Franka 数据 |
|--------|--------------|------------------|
| 图像分辨率 | 240x320 | 480x640 |
| 动作维度 | 2D (x, y) | 7D (6D pose + gripper) |
| 状态维度 | 2D (x, y) | 7D (6D eef pose + gripper) |
| 相机数量 | 1-2 | 3 |
| 动作类型 | 绝对位置 | 绝对位姿 |

## 📚 下一步

1. **验证数据加载**：
   ```python
   # 在 Python 中测试数据加载
   import hydra
   from omegaconf import OmegaConf
   
   OmegaConf.register_new_resolver("eval", eval, replace=True)
   
   with hydra.initialize(config_path="diffusion_policy/config"):
       cfg = hydra.compose(config_name="train_diffusion_unet_franka_image_workspace")
       dataset = hydra.utils.instantiate(cfg.task.dataset)
       print(f"Dataset size: {len(dataset)}")
       print(f"Sample: {dataset[0].keys()}")
   ```

2. **开始训练**：使用上述命令开始训练

3. **监控训练**：在 wandb 上查看训练进度

4. **调整超参数**：根据训练效果调整学习率、batch size 等

## 💡 提示

- 第一次运行会创建缓存，可能需要一些时间
- 建议先用少量数据测试配置是否正确
- 多相机会显著增加显存占用，可以先从单相机开始
- `delta_action: False` 表示使用绝对位姿，与你的数据一致

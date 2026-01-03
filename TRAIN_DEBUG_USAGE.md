# 训练数据调试功能使用指南

## 功能概述

新增的训练数据调试功能允许你在训练过程中记录 batch 数据的完整处理流程到 WandB，格式与推理时完全一致。这样可以方便对比训练/推理的数据一致性，快速发现问题。

## 快速开始

### 1. 启用调试模式训练

```bash
python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/path/to/dataset.zarr \
    training.enable_data_debug=True \
    training.debug_every=5 \
    exp_name="debug_training"
```

**参数说明**:
- `training.enable_data_debug=True`: 启用数据调试功能
- `training.debug_every=5`: 每 5 个 epoch 记录一次数据（默认 10）
- 建议在调试时设置较小的 `debug_every`，正常训练时设置为 10 或更大

### 2. 查看 WandB 记录

训练启动后，WandB 会记录以下数据：

#### 图像处理流程（3个阶段）

```
train_images/
├── stage1_batch_raw/
│   ├── camera_0      # 原始 batch（已经是 [0,1] float32）
│   ├── camera_1
│   └── camera_2
├── stage3_normalized/
│   └── camera_*      # LinearNormalizer 后（[-1,1]）
└── stage4_final_to_unet/
    └── camera_* ⭐   # ImageNet normalize 后（最终输入）
```

**注意**: Stage 2 (get_real_obs_dict) 在训练时被跳过，因为数据集已经处理好了。

#### 动作对比

```
train_actions/
├── ground_truth      # 数据集中的 GT action
└── predicted         # 当前 epoch 模型预测的 action
```

#### 调试信息

```
train_debug/
└── batch_info        # Batch 的 shape 和 dtype 信息
```

### 3. 对比训练/推理数据

在 WandB 界面中：

1. **打开训练 run**: 找到包含 `train_images/` 的数据
2. **打开推理 run**: 找到包含 `images/` 的数据
3. **并排对比**:
   - 训练的 `train_images/stage3_normalized/camera_0`
   - 推理的 `images/stage3_normalized/camera_0`
   - 它们的 shape、数值范围、图像内容应该一致

## 配置选项

### 配置文件（train_diffusion_unet_franka_image_workspace.yaml）

```yaml
training:
  # ... 其他配置 ...

  # 调试数据记录
  enable_data_debug: False  # 默认关闭
  debug_every: 10           # 每 N 个 epoch 记录一次
```

### 命令行覆盖

你可以在命令行覆盖任何配置：

```bash
# 启用调试，每个 epoch 都记录
python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    training.enable_data_debug=True \
    training.debug_every=1

# 只在特定 epoch 记录（例如每 20 个）
python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    training.enable_data_debug=True \
    training.debug_every=20
```

## 推理时的对应配置

### 启用推理调试

```bash
./start_server.sh --wandb --wandb-project my_debug_project
```

或手动：

```bash
python serve_diffusion_policy_single_frame.py \
    -i checkpoint.ckpt \
    -p 8000 \
    -d cuda \
    --wandb \
    --wandb-project my_debug_project
```

### 推理时记录的数据结构

```
images/
├── stage1_raw/           # 客户端原始输入
├── stage2_processed/     # get_real_obs_dict 处理后
├── stage3_normalized/    # LinearNormalizer 后
└── stage4_final_to_unet/ # ImageNet normalize 后 ⭐

actions/
├── stage1_normalized     # 归一化空间
├── stage2_pred_full      # 完整预测
├── stage3_exec           # 可执行动作
└── stage4_final          # 发送给客户端

debug/
└── camera_mapping        # 相机映射表
```

## 实际使用场景

### 场景 1：验证训练数据预处理正确

**问题**: 怀疑数据集的图像预处理有问题

**步骤**:
1. 启用训练调试：`training.enable_data_debug=True training.debug_every=1`
2. 训练 1 个 epoch
3. 查看 WandB 中的 `train_images/stage1_batch_raw/camera_0`
4. 检查图像是否正确、shape 是否匹配预期

### 场景 2：对比训练/推理的归一化一致性

**问题**: 推理效果差，怀疑归一化参数不一致

**步骤**:
1. 训练时启用调试并记录数据
2. 推理时也启用 WandB：`--wandb`
3. 在 WandB 中对比：
   - `train_images/stage3_normalized/camera_0` 的数值范围
   - `images/stage3_normalized/camera_0` 的数值范围
4. 它们应该都在 [-1, 1] 范围，图像亮度应该相似

### 场景 3：检查最终送入 UNet 的图像

**问题**: 模型预测不准，想看实际输入是什么

**步骤**:
1. 查看训练时的 `train_images/stage4_final_to_unet/camera_0 ⭐`
2. 查看推理时的 `images/stage4_final_to_unet/camera_0 ⭐`
3. 对比两者的：
   - Shape（应该一致）
   - 图像内容（训练集 vs 真机图像的差异）
   - 亮度分布（ImageNet normalize 后应该相似）

### 场景 4：分析动作预测质量

**问题**: 想知道模型在训练集上预测准不准

**步骤**:
1. 启用训练调试
2. 查看 WandB 中的：
   - `train_actions/ground_truth`: 数据集中的真实动作
   - `train_actions/predicted`: 模型预测的动作
3. 对比两个表格，看 MSE 误差大不大

## 性能影响

- **存储开销**: 每个 epoch 记录一次会增加 WandB 存储
- **计算开销**: 捕获和记录数据有轻微计算开销（~几十 ms）
- **建议**:
  - 调试时：`debug_every=1` 或 `debug_every=5`
  - 正常训练：`debug_every=10` 或 `debug_every=20`
  - 不需要时：`enable_data_debug=False`（默认）

## 与推理调试功能的对应关系

| 训练阶段 | 推理阶段 | 说明 |
|---------|---------|-----|
| `train_images/stage1_batch_raw/` | `images/stage1_raw/` | 原始输入 |
| (跳过) | `images/stage2_processed/` | 训练时数据集已处理 |
| `train_images/stage3_normalized/` | `images/stage3_normalized/` | LinearNormalizer ⭐ |
| `train_images/stage4_final_to_unet/` | `images/stage4_final_to_unet/` | 最终输入 ⭐⭐ |
| `train_actions/ground_truth` | `actions/stage4_final` | 真实动作 vs 推理输出 |
| `train_actions/predicted` | `actions/stage2_pred_full` | 训练预测 vs 推理预测 |

**关键对比点** (⭐):
- **Stage 3**: 检查 LinearNormalizer 是否一致
- **Stage 4**: 检查最终送入模型的图像是否一致

## 故障排查

### Q1: 训练时没有看到调试数据

**检查**:
1. `training.enable_data_debug=True` 是否生效？
2. 是否到了记录的 epoch？（例如 `debug_every=10`，只在 epoch 0, 10, 20... 记录）
3. 查看训练日志，是否有 "✅ 训练数据调试已启用" 的提示？

### Q2: 训练和推理的图像看起来不一样

**可能原因**:
1. 相机映射不一致（训练时 camera_0 vs 推理时 camera_1）
2. LinearNormalizer 参数不同（检查 checkpoint 中的 normalizer）
3. ImageNet normalize 应用顺序不同

**解决**:
- 对比 `debug/camera_mapping` 表格
- 检查 Stage 3 和 Stage 4 的数值范围

### Q3: 报错 "captured_data is empty"

**原因**: 调试回调没有被正确触发

**解决**:
1. 确认 Policy 和 ObsEncoder 的 `debug_callback` 属性存在
2. 检查是否在正确的位置设置了回调
3. 查看训练日志中的错误信息

## 总结

新的训练数据调试功能让你可以：
- ✅ 可视化训练 batch 的完整数据流
- ✅ 对比训练/推理的数据一致性
- ✅ 快速定位归一化、图像处理等问题
- ✅ 最小化对训练性能的影响

建议在遇到推理效果问题时，第一时间启用此功能进行对比分析！

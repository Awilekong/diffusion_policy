# WandB 调试指南 - Diffusion Policy 真机推理

本指南介绍如何使用 WandB 优雅地调试真机推理过程，追踪完整的数据流。

## 🎯 为什么需要 WandB 调试？

在真机推理过程中，数据从客户端→服务器→模型的流转过程中可能出现各种问题：
- 图像是否正确传输？（是否全黑、分辨率是否正确）
- 数据预处理是否正确？（resize、归一化）
- 模型输入是否符合预期？（维度、数值范围）
- 动作输出是否合理？（数值范围、波动情况）

传统的 `print` 和保存 `.npz` 文件的方式不够优雅，难以：
- ✅ 实时查看数据流
- ✅ 可视化图像处理过程
- ✅ 分析动作序列趋势
- ✅ 对比不同推理步骤
- ✅ 追踪性能瓶颈

**WandB 提供了完美的解决方案！**

## 🚀 快速开始

### 1. 安装 WandB

```bash
pip install wandb
wandb login  # 首次使用需要登录
```

### 2. 启动服务器（启用 WandB）

```bash
# 基础用法（不启用 WandB）
python serve_diffusion_policy_single_frame.py \
    -i /path/to/checkpoint.ckpt \
    -p 8000 \
    -d cuda

# 启用 WandB 调试模式
python serve_diffusion_policy_single_frame.py \
    -i /path/to/checkpoint.ckpt \
    -p 8000 \
    -d cuda \
    --wandb

# 自定义 WandB 项目名称
python serve_diffusion_policy_single_frame.py \
    -i /path/to/checkpoint.ckpt \
    -p 8000 \
    -d cuda \
    --wandb \
    --wandb-project my_robot_debug
```

### 3. 启动后的输出

```
============================================================
Diffusion Policy 远程推理服务器 (单帧版本)
专用于 n_obs_steps=1 的模型
🔍 WandB 调试模式已启用
============================================================
✅ WandB 初始化成功: diffusion_policy_inference/inference_20251230_143025
   查看调试信息: https://wandb.ai/your-team/diffusion_policy_inference/runs/abc123

🚀 启动 WebSocket 服务器...
   地址: 0.0.0.0:8000

🔍 WandB 调试:
   实时查看: https://wandb.ai/your-team/diffusion_policy_inference/runs/abc123
   - 观测输入统计和图像
   - 数据处理流程可视化
   - 动作输出分析
   - 推理时间统计
============================================================
```

**点击链接即可在浏览器中实时查看调试信息！**

## 📊 WandB 记录的内容

### 1. 图像处理流程（images/）

WandB 会追踪图像从客户端原始输入到最终送入 UNet 的完整处理流程，分为 4 个关键阶段：

#### 阶段 1: 原始输入图像（images/stage1_raw/）
- `images/stage1_raw/observation_image`: 客户端发送的原始图像（主相机）
- `images/stage1_raw/observation_image_1`: 第二相机原始图像
- `images/stage1_raw/observation_image_2`: 第三相机原始图像

**用途**：
- 检查图像是否全黑/全白
- 验证图像传输是否正常
- 检查曝光和对比度

#### 阶段 2: get_real_obs_dict 处理后（images/stage2_processed/）
- `images/stage2_processed/camera_0`: 经过 resize + crop 后的图像，归一化到 [0,1]
- `images/stage2_processed/camera_1`
- `images/stage2_processed/camera_2`

**用途**：
- 确认图像 resize 到正确尺寸（如 240×320）
- 检查裁剪是否合理
- 验证 [0,1] 归一化是否正确

#### 阶段 3: LinearNormalizer 归一化后（images/stage3_normalized/）
- `images/stage3_normalized/camera_0`: 应用训练时的归一化参数后，通常在 [-1,1] 范围
- `images/stage3_normalized/camera_1`
- `images/stage3_normalized/camera_2`

**用途**：
- 验证 LinearNormalizer 是否正确应用
- 检查归一化后的图像亮度/对比度
- 确认没有异常的亮度偏移

#### 阶段 4: 最终送入 UNet 的图像 ⭐（images/stage4_final_to_unet/）
- `images/stage4_final_to_unet/camera_0`: 经过 ImageNet normalize 的最终图像
- `images/stage4_final_to_unet/camera_1`
- `images/stage4_final_to_unet/camera_2`

**用途**：**这是最重要的阶段！**
- 验证模型实际看到的图像
- 检查 ImageNet normalize 是否正确
- 如果模型预测异常，从这里开始调试

**记录频率**：
- 前 10 步：每步都记录图像
- 之后：每 20 步记录一次（节省存储空间）

### 2. 相机映射表（debug/camera_mapping）

使用 WandB Table 显示客户端相机到模型输入的完整映射关系：

| 客户端键 | env_obs键 | shape_meta键 | 训练形状(C,H,W) | 实际形状(H,W,C) |
|---------|-----------|-------------|----------------|----------------|
| observation/image | camera_0 | camera_0 | (3, 240, 320) | (480, 640, 3) |
| observation/image_1 | camera_1 | camera_1 | (3, 240, 320) | (480, 640, 3) |
| observation/image_2 | camera_2 | camera_2 | (3, 240, 320) | (480, 640, 3) |

**用途**：
- ⭐ **快速发现相机对应错误**（训练/推理时相机不匹配是最常见的错误）
- 验证训练时的分辨率和实际输入是否一致
- 检查相机索引是否正确

### 3. 动作处理流程（actions/）

WandB 会追踪动作从模型输出到发送给客户端的完整转换过程，分为 4 个阶段：

#### 阶段 1: 归一化空间的动作（actions/stage1_normalized）
- 模型在归一化空间输出的原始动作（DDIM 采样结果）
- 通常在 [-1, 1] 或其他归一化范围
- WandB Table 格式，包含 time_step 和每个动作维度

**用途**：
- 检查模型在归一化空间的输出是否合理
- 验证扩散模型的采样是否正常

#### 阶段 2: 完整预测序列（actions/stage2_pred_full）
- 反归一化后的完整动作预测（action_pred）
- 长度 = horizon（如 16）
- 真实物理单位（如弧度、米等）

**用途**：
- 检查动作反归一化是否正确
- 查看完整的预测轨迹
- 分析动作的长期趋势

#### 阶段 3: 可执行动作（actions/stage3_exec）
- 从完整预测中提取的可执行部分（action）
- 长度 = n_action_steps（如 8）
- 实际会被发送给机器人执行的动作

**用途**：
- 检查动作提取是否正确
- 验证执行的动作数量
- 分析短期动作平滑性

#### 阶段 4: 最终发送给客户端（actions/stage4_final）
- 最终通过 WebSocket 发送给客户端的动作
- 与阶段 3 相同（numpy 格式）

**用途**：
- 确认客户端收到的动作
- 对比不同阶段的动作是否一致

### 4. 状态数据（state/）

- `state/dim_0` ~ `state/dim_13`: 每个状态维度的值（如机器人关节角度、末端位姿等）

**用途**：
- 检查机器人状态是否合理
- 验证状态数据传输是否正确
- 分析状态与动作的关系

### 5. 错误信息（error/）

- `error/message`: 错误消息
- `error/timestamp`: 错误时间戳
- `error/step`: 错误发生的步数
- `error/input/*/shape`: 导致错误的输入形状
- `error/input/*/dtype`: 导致错误的输入类型

**用途**：记录并分析错误原因

## 🔍 实际调试流程

### 场景 1: 图像全黑问题

**症状**：模型输出的动作全是零或不合理

**调试步骤**：
1. 打开 WandB 链接
2. 查看 `images/stage1_raw/observation_image`
   - 检查原始图像是否全黑/全白
   - 如果全黑：检查相机硬件和连接
3. 逐步检查后续阶段：
   - `images/stage2_processed/camera_0`: resize 后是否正常
   - `images/stage3_normalized/camera_0`: 归一化后是否合理
   - `images/stage4_final_to_unet/camera_0`: 最终送入模型的图像
4. 如果只有最后阶段异常，检查 ImageNet normalize 参数

### 场景 2: 相机对应错误

**症状**：模型预测结果很差，但图像看起来正常

**调试步骤**：
1. 查看 `debug/camera_mapping` 表格
2. 检查每个相机的映射：
   - 客户端发送的 `observation/image` 对应模型的哪个相机？
   - 训练时是否用的是同样的相机顺序？
3. 对比不同阶段的图像：
   - `images/stage1_raw/observation_image` vs `images/stage2_processed/camera_0`
   - 确认它们确实是同一个相机
4. 如果发现错误映射：
   - 修改客户端代码调整相机顺序
   - 或重新训练模型使用正确的相机映射

### 场景 3: 动作异常跳变

**症状**：机器人运动不平滑，有突然跳变

**调试步骤**：
1. 查看 `actions/stage3_exec` 表格
   - 找出哪个时间步和哪个维度有跳变
2. 回溯动作处理流程：
   - `actions/stage1_normalized`: 归一化空间是否平滑？
   - `actions/stage2_pred_full`: 反归一化是否引入跳变？
3. 对比输入图像：
   - 查看跳变发生时的 4 个图像阶段
   - 确认是否因为观测异常导致
4. 检查动作归一化参数：
   - 打印 `policy.normalizer['action'].params_dict`
   - 确认 scale 和 offset 是否合理

### 场景 4: 模型输出全是相同的动作

**症状**：所有预测的动作都一样，没有根据观测变化

**调试步骤**：
1. 检查图像处理流程（4 个阶段）：
   - 是否每步的图像都正确变化？
   - 特别关注 `images/stage4_final_to_unet/camera_0`
2. 查看 `actions/stage1_normalized`：
   - 归一化空间的输出是否变化？
   - 如果不变化，说明模型本身有问题
3. 检查 LinearNormalizer：
   - 确认训练和推理使用相同的 normalizer
   - 验证 `images/stage3_normalized` 是否合理
4. 可能的原因：
   - 图像归一化参数不匹配
   - 模型未充分训练
   - 相机对应错误

### 场景 5: 理解完整的数据流

**目标**：验证从图像到动作的完整流程

**步骤**：
1. **图像流**：逐步查看 4 个阶段
   ```
   stage1_raw → stage2_processed → stage3_normalized → stage4_final_to_unet
   ```
   - 每个阶段都应该是合理的变换
   - 最重要的是 stage4，这是模型实际输入

2. **相机映射**：检查 `debug/camera_mapping`
   - 确认所有相机索引正确对应

3. **动作流**：逐步查看 4 个阶段
   ```
   stage1_normalized → stage2_pred_full → stage3_exec → stage4_final
   ```
   - stage1: 模型原始输出（归一化空间）
   - stage2: 反归一化后的完整预测
   - stage3: 提取的可执行动作
   - stage4: 发送给客户端

4. **状态数据**：查看 `state/dim_*`
   - 确认状态值合理
   - 与动作输出做对比分析

## 📈 WandB 可视化技巧

### 1. 创建自定义图表

在 WandB 界面中，你可以：
- 拖拽指标创建多维度对比图
- 使用 "Add Panel" 添加图像轮播
- 创建动作序列的折线图
- 对比不同推理步骤的数据

### 2. 过滤和筛选

- 使用 Step 滑块查看特定时间步
- 使用搜索框快速找到指标
- 使用分组功能（Group by）对比不同运行

### 3. 导出数据

- 点击 "Download" 下载原始数据
- 使用 wandb API 批量分析：

```python
import wandb
api = wandb.Api()
run = api.run("your-team/diffusion_policy_inference/abc123")
history = run.history()
print(history["output/action/mean"])
```

## 🎨 最佳实践

### 1. 分阶段调试

**第一阶段：验证图像处理流**
- 启动服务器并发送测试数据
- 在 WandB 中按顺序检查 4 个图像阶段
- 重点关注 `images/stage4_final_to_unet`（最重要）
- 确认每个阶段的变换都合理

**第二阶段：验证相机映射**
- 查看 `debug/camera_mapping` 表格
- 确认客户端相机与模型输入的对应关系
- 对比训练配置和推理配置
- 如有问题，调整客户端代码或重新训练

**第三阶段：验证动作转换流**
- 按顺序检查 4 个动作阶段
- 从归一化空间 → 物理空间 → 可执行部分 → 客户端
- 确认反归一化参数正确
- 验证动作范围和平滑性

**第四阶段：端到端测试**
- 使用真机数据测试完整流程
- 观察状态、图像、动作的联动关系
- 识别和修复任何异常

### 2. 图像记录频率

默认配置：
- 前10步：每步都记录图像
- 之后：每20步记录一次

理由：
- 避免 WandB 存储过大
- 前期记录密集用于快速定位问题
- 后期稀疏记录用于长期监控

如需修改，编辑代码：
```python
# 在 log_inference_step 中
if self.step_counter < 10 or self.step_counter % 20 == 0:
    # 改为: if self.step_counter % 5 == 0:  # 每5步记录
```

### 3. 错误处理

启用 WandB 后，所有错误会自动记录：
- `get_real_obs_dict` 失败会记录错误信息
- 模型推理失败会记录错误信息
- 同时记录导致错误的输入数据

查看方式：
- 在 WandB 中搜索 "error"
- 查看错误发生的步数
- 查看对应的输入数据

## 🔧 配置选项

### 修改 WandB 项目名称

```bash
python serve_diffusion_policy_single_frame.py \
    -i checkpoint.ckpt \
    --wandb \
    --wandb-project franka_peg_in_hole_debug
```

### 添加自定义标签

修改代码中的 `WandbDebugger.__init__`：
```python
self.wandb.init(
    project=project,
    name=run_name,
    tags=["inference", "real_robot", "single_frame", "franka", "peg_in_hole"]  # 添加自定义标签
)
```

### 记录额外的配置信息

在 `DiffusionPolicySingleFrameWrapper.__init__` 中添加：
```python
if self.wandb_debugger.enabled:
    self.wandb_debugger.wandb.config.update({
        "robot_type": "franka",
        "task": "peg_in_hole",
        "camera_count": 3,
        # 其他配置...
    })
```

## 🐛 常见问题

### Q1: WandB 初始化失败

**症状**：
```
⚠️  WandB 初始化失败: ...
```

**解决方案**：
1. 检查网络连接
2. 运行 `wandb login` 重新登录
3. 检查 wandb 版本：`pip install --upgrade wandb`

### Q2: 图像无法显示

**症状**：WandB 中看不到图像

**可能原因**：
1. 图像格式问题（需要 RGB uint8）
2. 图像维度问题（需要 HWC 格式）
3. 数值范围问题（需要 0-255）

**解决**：代码中已自动处理这些问题

### Q3: 某个阶段的图像异常

**症状**：某个处理阶段的图像显示异常（全黑、全白、色彩错乱）

**解决方案**：
1. 检查该阶段的数值范围：
   - stage1 (raw): 应该是 [0, 255] uint8
   - stage2 (processed): 应该是 [0, 1] float32
   - stage3 (normalized): 通常是 [-1, 1] float32
   - stage4 (final): ImageNet normalize，范围约 [-2, 2]
2. 如果是代码问题，检查 WandbDebugger 的图像转换逻辑
3. 对比前后阶段，确定问题发生在哪个变换

### Q4: 存储空间占用大

**症状**：WandB 项目占用空间过大

**解决方案**：
1. 定期清理旧的 runs
2. 减少图像记录频率（修改代码中的采样频率）
3. 使用 WandB 的 retention policy

## 📝 总结

### 新版 WandB 调试系统的优势

- ✅ **完整的图像处理流可视化**：从客户端到 UNet 的 4 个关键阶段
- ✅ **最重要的是 stage4**：可以看到模型实际输入的图像
- ✅ **相机映射表**：快速发现训练/推理时相机不匹配的问题
- ✅ **完整的动作转换流**：从归一化空间到客户端的 4 个阶段
- ✅ **实时可视化**：无需等待推理结束
- ✅ **历史记录**：可以回看任意时刻的数据
- ✅ **团队协作**：分享链接给团队成员

### 与旧版本的区别

**移除的内容**：
- ❌ 图像统计数据（mean/std/min/max）
- ❌ 状态统计数据
- ❌ 性能监控（timing）
- ❌ 复杂的数据形状表格

**新增的内容**：
- ✅ 图像 4 阶段可视化（特别是 stage4 最终输入）
- ✅ 相机映射表
- ✅ 动作 4 阶段追踪

**理念变化**：
- **旧版**：记录大量统计数据，用户需要自己分析
- **新版**：专注于关键的可视化，让问题一目了然

### 建议工作流

1. **开发/调试阶段**：始终启用 `--wandb`
2. **首次运行**：
   - 先检查图像 4 阶段是否正常
   - 重点关注 `images/stage4_final_to_unet`
   - 检查 `debug/camera_mapping` 表格
3. **发现问题时**：
   - 图像问题：逐步回溯 4 个阶段
   - 动作问题：逐步回溯 4 个动作阶段
   - 相机问题：查看映射表
4. **问题解决后**：对比修复前后的数据
5. **生产部署**：可选择关闭以获得最佳性能

祝调试顺利！🚀

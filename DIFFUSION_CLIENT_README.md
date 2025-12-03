# Diffusion Policy 客户端使用说明

## 概述

本文档介绍如何使用 `DiffusionPolicyClient` 与 Diffusion Policy 服务器进行通信。

## 主要特性

### 1. 统一接口格式
- **输入格式**：与 PIClient 和 LLaVAClient 保持一致
  - `image_0`, `image_1`, `image_2`: 三个相机的图像（bytes 格式）
  - `state`: 机器人状态（numpy array）

- **输出格式**：
  - `actions`: 动作序列，shape `(action_horizon, action_dim)`

### 2. 历史帧管理
- **问题**：Diffusion Policy 需要 `n_obs_steps` 个历史帧（例如 2 帧），但客户端只能传单帧
- **解决方案**：在服务器端维护历史帧队列
  - 首次推理：用第一帧填充整个队列
  - 后续推理：自动维护滑动窗口
  - 调用 `reset()` 可清空历史队列

### 3. 多相机支持
- 固定支持 **3 个相机**
- 自动映射：
  - `image_0` → `observation/image`
  - `image_1` → `observation/image_1`
  - `image_2` → `observation/image_2`

## 使用方法

### 1. 启动服务器

```bash
cd /home/zpw/ws_zpw/megvii/IL/diffusion_policy

python serve_diffusion_policy.py \
    -i /home/zpw/ws_zpw/megvii/IL/diffusion_policy/data/outputs/2025.11.23/23.48.38_train_diffusion_unet_franka_image_franka_peg_in_hole_image/checkpoints/epoch=0150-train_loss=0.003.ckpt \
    -p 8000
```

**参数说明**：
- `-i, --input`: checkpoint 文件路径（必需）
- `-p, --port`: 服务器端口（默认 8000）
- `-h, --host`: 服务器地址（默认 0.0.0.0）
- `-d, --device`: 计算设备（默认 cuda）

### 2. 健康检查

```bash
curl http://localhost:8000/healthz
```

### 3. 使用客户端

#### 方式 1：直接导入使用

```python
import sys
import os
sys.path.insert(0, '/home/zpw/ws_zpw/megvii/web_policy')

from web_policy.utils import DiffusionPolicyClient
import numpy as np
import cv2

# 创建客户端
client = DiffusionPolicyClient(base_url='http://localhost:8000')

# 准备数据
img_0 = cv2.imread('camera_0.jpg')
img_1 = cv2.imread('camera_1.jpg')
img_2 = cv2.imread('camera_2.jpg')
state = np.array([...])  # 机器人状态

# 编码图像
_, img_0_bytes = cv2.imencode('.jpg', img_0)
_, img_1_bytes = cv2.imencode('.jpg', img_1)
_, img_2_bytes = cv2.imencode('.jpg', img_2)

# 推理
actions = client.process_frame(
    image_0=img_0_bytes.tobytes(),
    image_1=img_1_bytes.tobytes(),
    image_2=img_2_bytes.tobytes(),
    state=state
)

print(f"动作: {actions.shape}")  # (action_horizon, action_dim)
```

#### 方式 2：运行测试脚本

```bash
cd /home/zpw/ws_zpw/megvii/IL/diffusion_policy
python test_diffusion_client.py
```

## 接口对比

### 输入格式（统一）

| 客户端 | 图像 | 状态 |
|--------|------|------|
| PIClient | `image_0`, `image_1`, `image_2` (bytes) | `state` (numpy) |
| LLaVAClient | `image_0`, `image_1`, `image_2` (bytes) | `state` (numpy) |
| **DiffusionPolicyClient** | `image_0`, `image_1`, `image_2` (bytes) | `state` (numpy) |

### 输出格式（统一）

| 客户端 | 返回值 |
|--------|--------|
| PIClient | `actions` (numpy array) |
| **DiffusionPolicyClient** | `actions` (numpy array) |

## 关键实现细节

### 服务器端 (serve_diffusion_policy.py)

1. **历史帧队列**：
   ```python
   self.obs_history = {
       'images': {0: deque(...), 1: deque(...), 2: deque(...)},
       'state': deque(...)
   }
   ```

2. **自动填充**：
   - 第一帧：用相同的帧填充整个队列
   - 后续帧：自动滑动窗口（`maxlen=n_obs_steps`）

3. **格式转换**：
   - 输入：单帧 `(H, W, C)`
   - 队列：`(n_obs_steps, H, W, C)`
   - 输出：动作 `(action_horizon, action_dim)`

### 客户端端 (utils.py)

1. **固定三相机**：
   ```python
   for camera_idx in range(3):
       if f'image_{camera_idx}' in kwargs:
           # 处理图像
   ```

2. **格式映射**：
   ```python
   obs_key = "observation/image" if camera_idx == 0 else f"observation/image_{camera_idx}"
   observation[obs_key] = img_array
   ```

## 注意事项

1. **历史帧初始化**：
   - 第一次推理会用单帧填充整个历史队列
   - 可能导致第一次推理结果不准确
   - 建议：实际使用时推理几次"预热"

2. **Reset 操作**：
   - 每个 episode 开始前建议调用 `client.reset()`
   - 会清空服务器端的历史帧队列

3. **性能优化**：
   - 服务器使用 DDIM 采样（16 步），速度较快
   - WebSocket 连接，低延迟
   - GPU 推理，高吞吐

4. **相机数量**：
   - 当前固定支持 3 个相机
   - 如需修改，需同时修改客户端和服务器代码

## 文件清单

- `serve_diffusion_policy.py`: 服务器实现
- `test_diffusion_client.py`: 测试脚本
- `/home/zpw/ws_zpw/megvii/web_policy/utils.py`: 客户端实现（DiffusionPolicyClient 类）

## 故障排查

### 1. 连接失败
```
ConnectionRefusedError: [Errno 111] Connection refused
```
**解决**：检查服务器是否启动，端口是否正确

### 2. 形状不匹配
```
RuntimeError: Expected tensor with shape ...
```
**解决**：检查图像尺寸和状态维度是否与训练时一致

### 3. 推理慢
**解决**：
- 检查是否使用 GPU（`device='cuda'`）
- 减少 `num_inference_steps`（修改服务器代码）
- 检查网络延迟

## 开发指南

### 修改相机数量

如果需要支持不同数量的相机，需要修改：

1. **客户端** (`utils.py`)：
   ```python
   for camera_idx in range(N_CAMERAS):  # 修改这里
       ...
   ```

2. **服务器端**会自动适配（使用 `while True` 循环检测）

### 修改历史帧数

历史帧数由模型训练时的 `n_obs_steps` 决定，无需手动修改。

## 总结

✅ 统一接口格式，与 PIClient/LLaVAClient 一致  
✅ 自动管理历史帧队列，客户端只需传单帧  
✅ 支持三个相机  
✅ 高性能 WebSocket 通信  
✅ 完整的测试脚本

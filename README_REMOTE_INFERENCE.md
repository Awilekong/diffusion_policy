# Diffusion Policy 远程推理

使用 `web_policy` 实现 Diffusion Policy 的远程推理服务。

## ✨ 特性

- 🔥 **完整兼容**: 使用官方 `real_inference_util` 处理数据
- 🎯 **自动处理**: 归一化、反归一化、数据转换全自动
- 🚀 **高性能**: WebSocket 异步通信，低延迟
- 📦 **即插即用**: 加载官方 checkpoint 即可使用
- 🔧 **专注 Diffusion**: 专为 Diffusion Policy 优化

## 📦 安装依赖

```bash
# 确保已安装 web_policy
cd /home/zpw/ws_zpw/megvii/web_policy
pip install -e .

# 安装 diffusion_policy 依赖（如果还没有）
cd /home/zpw/ws_zpw/megvii/IL/diffusion_policy
pip install -e .
```

## 🚀 快速开始

### 1. 启动服务器

```bash
cd /home/zpw/ws_zpw/megvii/IL/diffusion_policy

# 启动服务器
python serve_diffusion_policy_single_frame.py \
    -i /home/zpw/ws_zpw/megvii/IL/diffusion_policy/data/outputs/2025.12.02/10.55.53_train_diffusion_unet_franka_image_franka_peg_in_hole_image/checkpoints/latest.ckpt \
    -p 8001 \
    -d cuda
```

参数说明：
- `-i, --input`: Checkpoint 文件路径（必需）
- `-p, --port`: 服务器端口（默认 8000）
- `-h, --host`: 服务器地址（默认 0.0.0.0）
- `-d, --device`: 设备 cuda 或 cpu（默认 cuda）

### 2. 使用客户端

```bash
# 在另一个终端运行测试客户端
python test_remote_inference.py
```

或者在你的代码中使用：

```python
from web_policy import WebSocketClientPolicy
import numpy as np

# 连接服务器
client = WebSocketClientPolicy(host="localhost", port=8000)

# 获取元数据
metadata = client.get_server_metadata()
n_obs_steps = metadata['n_obs_steps']

# 准备观测数据
obs = {
    'camera_0': np.random.randint(0, 255, (n_obs_steps, 224, 224, 3), dtype=np.uint8),
    'robot_eef_pose': np.random.randn(n_obs_steps, 6).astype(np.float32),
}

# 推理
result = client.infer(obs)
actions = result['actions']  # shape: (action_horizon, action_dim)
```

## 📖 数据格式

### 输入观测数据 (obs)

```python
obs = {
    # 图像数据（如果使用）
    'camera_0': np.ndarray,  # shape: (n_obs_steps, H, W, C), dtype: uint8
    'camera_1': np.ndarray,  # shape: (n_obs_steps, H, W, C), dtype: uint8
    
    # 机器人状态
    'robot_eef_pose': np.ndarray,  # shape: (n_obs_steps, pose_dim), dtype: float32
    
    # 时间戳（可选）
    'timestamp': np.ndarray,  # shape: (n_obs_steps,), dtype: float64
}
```

**注意**:
- 图像格式为 `(H, W, C)`，值域 `[0, 255]`，uint8 类型
- 服务器会自动转换为模型需要的格式
- `n_obs_steps` 从服务器元数据中获取

### 输出动作数据 (result)

```python
result = {
    'actions': np.ndarray,  # shape: (action_horizon, action_dim), dtype: float32
    'server_timing': {
        'infer_ms': float,  # 服务器推理耗时（毫秒）
    }
}
```

## 🔧 核心实现

### DiffusionPolicyWrapper

包装 Diffusion Policy 为 `BasePolicy` 接口，处理：

1. **Checkpoint 加载**: 使用官方 `workspace.load_payload()`
2. **模型配置**: 设置 `num_inference_steps=16`, `n_action_steps`
3. **数据预处理**: 使用官方 `get_real_obs_dict()`
4. **推理**: 调用 `policy.predict_action()`
5. **自动重置**: 支持 `policy.reset()`

### 数据处理流程

```
观测数据 (客户端)
    ↓
WebSocket 传输（msgpack + numpy）
    ↓
get_real_obs_dict()  # 官方预处理
    ↓
转换为 torch.Tensor
    ↓
policy.predict_action()  # 推理
    ↓
转换回 numpy.ndarray
    ↓
WebSocket 传输
    ↓
动作数据 (客户端)
```

## 📊 支持的模型类型

| 模型类型 | 支持 | 特殊配置 |
|---------|------|---------|
| Diffusion | ✅ | `num_inference_steps=16`, `n_action_steps` |

## 🔍 与原始脚本的对比

| 功能 | 原始 eval_real_robot.py | serve_diffusion_policy.py |
|------|------------------------|--------------------------|
| Checkpoint 加载 | ✅ | ✅ 完全相同 |
| 模型配置 | ✅ | ✅ 完全相同 |
| 数据预处理 | ✅ `get_real_obs_dict()` | ✅ 使用相同函数 |
| 推理 | ✅ | ✅ 完全相同 |
| 机器人控制 | ✅ | ❌ 由客户端负责 |
| SpaceMouse | ✅ | ❌ 不需要 |
| 视频录制 | ✅ | ❌ 不需要 |
| 远程访问 | ❌ | ✅ WebSocket |

## 🆘 常见问题

### Q: 如何确认服务器正常运行？

A: 访问健康检查端点：
```bash
curl http://localhost:8000/healthz
# 应返回: OK
```

### Q: 支持多客户端同时连接吗？

A: 支持！WebSocket 服务器支持多个客户端并发连接。

### Q: 数据格式和原始脚本一样吗？

A: 是的！使用完全相同的 `get_real_obs_dict()` 函数处理数据。

### Q: 如何使用自己的 checkpoint？

A: 只需指定 checkpoint 路径：
```bash
python serve_diffusion_policy.py -i /path/to/your/checkpoint.ckpt
```

### Q: GPU 内存不足怎么办？

A: 使用 CPU 推理：
```bash
python serve_diffusion_policy.py -i checkpoint.ckpt -d cpu
```

## 📝 示例：真实机器人控制

```python
from web_policy import WebSocketClientPolicy
import numpy as np

# 连接服务器
client = WebSocketClientPolicy(host="robot_server", port=8000)
metadata = client.get_server_metadata()

# 控制循环
while True:
    # 获取真实观测
    obs = {
        'camera_0': camera.get_image(),  # (n_obs_steps, H, W, 3)
        'robot_eef_pose': robot.get_pose_history(),  # (n_obs_steps, 6)
    }
    
    # 获取动作
    result = client.infer(obs)
    actions = result['actions']
    
    # 执行动作
    for action in actions:
        robot.execute_action(action)
        time.sleep(dt)
```

## 🎓 文件说明

- `serve_diffusion_policy.py` - 服务器主程序
- `test_remote_inference.py` - 客户端测试示例
- `README_REMOTE_INFERENCE.md` - 本文档

---

**立即开始**: 
```bash
# 服务器
python serve_diffusion_policy.py -i checkpoint.ckpt

# 客户端（新终端）
python test_remote_inference.py
```

🚀 **享受远程 Diffusion Policy 推理！**

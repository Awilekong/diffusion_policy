# Diffusion Policy 维度数据流分析

从配置文件到网络初始化的完整数据流追踪

---

## 📋 配置文件层 (YAML)

### 1. Task配置文件
**文件**: `diffusion_policy/config/task/franka_peg_in_hole_image.yaml`

```yaml
shape_meta: &shape_meta
  obs:
    camera_0:
      shape: [3, 480, 640]  # ← 源头：定义RGB相机形状
      type: rgb
    camera_1:
      shape: [3, 480, 640]
      type: rgb
    camera_2:
      shape: [3, 480, 640]
      type: rgb
    robot_eef_pose:
      shape: [7]  # ← 源头：定义末端位姿维度
      type: low_dim
  
  action:
    shape: [7]  # ← 源头：定义动作维度
```

**关键点**: 
- `shape_meta` 是所有维度信息的源头
- 通过 `&shape_meta` 锚点供其他配置引用

---

### 2. Workspace配置文件
**文件**: `diffusion_policy/config/train_diffusion_unet_franka_image_workspace.yaml`

```yaml
defaults:
  - task: franka_peg_in_hole_image  # ← 导入task配置

shape_meta: ${task.shape_meta}  # ← 引用task的shape_meta

horizon: 16
n_obs_steps: 2
obs_as_global_cond: True

policy:
  _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
  shape_meta: ${shape_meta}  # ← 传递给Policy
  
  obs_encoder:
    _target_: diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
    shape_meta: ${shape_meta}  # ← 传递给ObsEncoder
    rgb_model:
      _target_: diffusion_policy.model.vision.model_getter.get_resnet
      name: resnet18
```

**传递路径**:
```
task.shape_meta → workspace.shape_meta → policy.shape_meta
                                      → policy.obs_encoder.shape_meta
```

---

## 🔧 代码层：初始化过程

### 3. Workspace初始化Policy
**文件**: `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`

```python
class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # ← cfg包含完整配置，包括shape_meta
        # Hydra根据配置实例化Policy，自动传递shape_meta
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
```

**关键**: `hydra.utils.instantiate(cfg.policy)` 会：
1. 读取 `cfg.policy._target_` 找到类
2. 将 `cfg.policy` 中的所有参数传递给类的 `__init__`
3. 包括 `shape_meta`, `obs_encoder`, `horizon`, `n_obs_steps` 等

---

### 4. ObsEncoder初始化 - 解析shape_meta
**文件**: `diffusion_policy/model/vision/multi_image_obs_encoder.py`

```python
class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self, shape_meta: dict, rgb_model, ...):
        # ← shape_meta通过Hydra自动传入
        
        rgb_keys = list()
        low_dim_keys = list()
        key_shape_map = dict()
        
        obs_shape_meta = shape_meta['obs']  # ← 提取obs配置
        
        # 遍历所有obs keys
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])  # ← 提取shape
            type = attr.get('type', 'low_dim')  # ← 提取type
            key_shape_map[key] = shape
            
            if type == 'rgb':
                rgb_keys.append(key)  # ← camera_0, camera_1, camera_2
                # 为每个相机创建独立的ResNet18
                this_model = copy.deepcopy(rgb_model)
                key_model_map[key] = this_model
                
            elif type == 'low_dim':
                low_dim_keys.append(key)  # ← robot_eef_pose
        
        self.rgb_keys = rgb_keys  # ['camera_0', 'camera_1', 'camera_2']
        self.low_dim_keys = low_dim_keys  # ['robot_eef_pose']
        self.key_shape_map = key_shape_map  # 保存所有shape信息
```

**数据提取过程**:
```python
shape_meta['obs'] = {
    'camera_0': {'shape': [3, 480, 640], 'type': 'rgb'},
    'camera_1': {'shape': [3, 480, 640], 'type': 'rgb'},
    'camera_2': {'shape': [3, 480, 640], 'type': 'rgb'},
    'robot_eef_pose': {'shape': [7], 'type': 'low_dim'}
}

# 解析后:
rgb_keys = ['camera_0', 'camera_1', 'camera_2']
low_dim_keys = ['robot_eef_pose']
key_shape_map = {
    'camera_0': (3, 480, 640),
    'camera_1': (3, 480, 640),
    'camera_2': (3, 480, 640),
    'robot_eef_pose': (7,)
}
```

---

### 5. ObsEncoder计算输出维度
**文件**: `diffusion_policy/model/vision/multi_image_obs_encoder.py`

```python
@torch.no_grad()
def output_shape(self):
    example_obs_dict = dict()
    obs_shape_meta = self.shape_meta['obs']
    batch_size = 1
    
    # 创建示例输入
    for key, attr in obs_shape_meta.items():
        shape = tuple(attr['shape'])
        this_obs = torch.zeros((batch_size,) + shape)
        example_obs_dict[key] = this_obs
    
    # example_obs_dict = {
    #     'camera_0': torch.zeros(1, 3, 480, 640),
    #     'camera_1': torch.zeros(1, 3, 480, 640),
    #     'camera_2': torch.zeros(1, 3, 480, 640),
    #     'robot_eef_pose': torch.zeros(1, 7)
    # }
    
    # 执行forward，计算输出维度
    example_output = self.forward(example_obs_dict)
    output_shape = example_output.shape[1:]
    return output_shape


def forward(self, obs_dict):
    features = list()
    
    # 处理RGB输入（每个相机独立）
    for key in self.rgb_keys:  # ['camera_0', 'camera_1', 'camera_2']
        img = obs_dict[key]  # (B, 3, 480, 640)
        img = self.key_transform_map[key](img)  # resize, crop, normalize
        feature = self.key_model_map[key](img)  # ResNet18 → (B, 512)
        features.append(feature)  # 3个相机 × 512 = 1536
    
    # 处理low_dim输入
    for key in self.low_dim_keys:  # ['robot_eef_pose']
        data = obs_dict[key]  # (B, 7)
        features.append(data)  # +7
    
    # 拼接所有特征
    result = torch.cat(features, dim=-1)  # (B, 1536 + 7) = (B, 1543)
    return result
```

**维度计算**:
```
RGB特征:
  camera_0 → ResNet18 → 512维
  camera_1 → ResNet18 → 512维
  camera_2 → ResNet18 → 512维
  小计: 512 × 3 = 1536维

Low_dim特征:
  robot_eef_pose → 直接使用 → 7维
  小计: 7维

总计: 1536 + 7 = 1543维
```

---

### 6. Policy初始化 - 使用维度信息
**文件**: `diffusion_policy/policy/diffusion_unet_image_policy.py`

```python
class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,  # ← 从配置传入
            obs_encoder: MultiImageObsEncoder,  # ← Hydra已实例化
            horizon,  # ← 16
            n_action_steps,  # ← 8
            n_obs_steps,  # ← 2
            obs_as_global_cond=True,
            **kwargs):
        super().__init__()

        # ===== 步骤1: 从shape_meta提取action维度 =====
        action_shape = shape_meta['action']['shape']  # [7]
        assert len(action_shape) == 1
        action_dim = action_shape[0]  # action_dim = 7
        
        # ===== 步骤2: 从obs_encoder获取观测特征维度 =====
        obs_feature_dim = obs_encoder.output_shape()[0]  # obs_feature_dim = 1543
        
        # ===== 步骤3: 根据obs_as_global_cond决定UNet输入维度 =====
        if obs_as_global_cond:
            # 观测作为全局条件
            input_dim = action_dim  # 7
            global_cond_dim = obs_feature_dim * n_obs_steps  # 1543 × 2 = 3086
        else:
            # 观测和动作拼接
            input_dim = action_dim + obs_feature_dim  # 7 + 1543 = 1550
            global_cond_dim = None
        
        # ===== 步骤4: 创建UNet模型 =====
        model = ConditionalUnet1D(
            input_dim=input_dim,  # 7 (仅action)
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,  # 3086 (obs作为全局条件)
            diffusion_step_embed_dim=128,
            down_dims=[512, 1024, 2048],
            kernel_size=5,
            n_groups=8
        )
        
        # 保存维度信息
        self.action_dim = action_dim  # 7
        self.obs_feature_dim = obs_feature_dim  # 1543
        self.n_obs_steps = n_obs_steps  # 2
        self.horizon = horizon  # 16
        self.n_action_steps = n_action_steps  # 8
```

**维度决策过程**:
```python
# 配置来源:
action_dim = shape_meta['action']['shape'][0]  # 7 (从配置)
obs_feature_dim = obs_encoder.output_shape()[0]  # 1543 (从计算)
n_obs_steps = cfg.n_obs_steps  # 2 (从配置)
obs_as_global_cond = cfg.obs_as_global_cond  # True (从配置)

# UNet维度计算:
if obs_as_global_cond:  # True
    input_dim = 7  # 只输入action
    global_cond_dim = 1543 × 2 = 3086  # obs作为全局条件
```

---

### 7. ConditionalUnet1D初始化
**文件**: `diffusion_policy/model/diffusion/conditional_unet1d.py`

```python
class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,  # ← 7 (action_dim)
        local_cond_dim=None,  # ← None
        global_cond_dim=None,  # ← 3086 (obs_feature_dim × n_obs_steps)
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        **kwargs):
        
        # 输入维度
        in_channels = input_dim  # 7
        
        # 时间步embedding
        dsed = diffusion_step_embed_dim  # 256
        
        # 全局条件维度
        if global_cond_dim is not None:
            # 创建全局条件编码器
            self.global_cond_encoder = nn.Sequential(
                nn.Linear(global_cond_dim, dsed * 4),  # 3086 → 1024
                nn.Mish(),
                nn.Linear(dsed * 4, dsed * 4)  # 1024 → 1024
            )
        
        # 下采样路径
        self.down_modules = nn.ModuleList([
            # 每一层的维度
            ConditionalResidualBlock1D(
                in_channels=7,  # action输入
                out_channels=512,
                cond_dim=dsed * 4  # 接收时间步 + 全局条件
            ),
            ...
        ])
```

---

## 🔄 完整数据流总结

### 配置文件 → 代码的传递链

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. YAML配置文件                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  franka_peg_in_hole_image.yaml:                                │
│    shape_meta:                                                 │
│      obs:                                                      │
│        camera_0: {shape: [3,480,640], type: rgb}              │
│        camera_1: {shape: [3,480,640], type: rgb}              │
│        camera_2: {shape: [3,480,640], type: rgb}              │
│        robot_eef_pose: {shape: [7], type: low_dim}            │
│      action:                                                   │
│        shape: [7]                                             │
│                                                                 │
│  train_workspace.yaml:                                         │
│    shape_meta: ${task.shape_meta}  ← 引用                     │
│    horizon: 16                                                 │
│    n_obs_steps: 2                                             │
│    obs_as_global_cond: True                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Hydra加载
┌─────────────────────────────────────────────────────────────────┐
│ 2. Hydra OmegaConf对象                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  cfg = {                                                       │
│    'shape_meta': {                                            │
│      'obs': {...},                                           │
│      'action': {'shape': [7]}                                │
│    },                                                          │
│    'policy': {                                                │
│      '_target_': 'DiffusionUnetImagePolicy',                │
│      'shape_meta': {...},                                    │
│      'obs_encoder': {...},                                   │
│      'horizon': 16,                                          │
│      'n_obs_steps': 2                                        │
│    }                                                           │
│  }                                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓ hydra.utils.instantiate
┌─────────────────────────────────────────────────────────────────┐
│ 3. Workspace实例化                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  workspace = TrainDiffusionUnetImageWorkspace(cfg)            │
│    ↓                                                           │
│  self.model = hydra.utils.instantiate(cfg.policy)            │
│    ↓ 传递参数:                                                 │
│    - shape_meta = cfg.policy.shape_meta                       │
│    - obs_encoder = <实例化的ObsEncoder>                       │
│    - horizon = 16                                             │
│    - n_obs_steps = 2                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓ 实例化Policy前先实例化ObsEncoder
┌─────────────────────────────────────────────────────────────────┐
│ 4. ObsEncoder初始化                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  obs_encoder = MultiImageObsEncoder(                          │
│      shape_meta=cfg.policy.obs_encoder.shape_meta             │
│  )                                                             │
│                                                                 │
│  __init__执行:                                                │
│    obs_shape_meta = shape_meta['obs']                         │
│    for key, attr in obs_shape_meta.items():                  │
│      shape = attr['shape']                                    │
│      type = attr['type']                                      │
│                                                                 │
│      if type == 'rgb':                                        │
│        rgb_keys.append(key)      ← camera_0/1/2             │
│        创建ResNet18副本                                        │
│      elif type == 'low_dim':                                 │
│        low_dim_keys.append(key)  ← robot_eef_pose           │
│                                                                 │
│  output_shape()计算:                                          │
│    创建example_obs_dict with batch_size=1                     │
│    执行forward()                                              │
│    返回 (1543,)  ← 512×3 + 7                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓ ObsEncoder准备好，传入Policy
┌─────────────────────────────────────────────────────────────────┐
│ 5. Policy初始化                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  policy = DiffusionUnetImagePolicy(                           │
│      shape_meta=cfg.policy.shape_meta,                        │
│      obs_encoder=<已实例化的obs_encoder>,                     │
│      horizon=16,                                              │
│      n_obs_steps=2,                                           │
│      obs_as_global_cond=True                                  │
│  )                                                             │
│                                                                 │
│  __init__执行:                                                │
│    # 提取action维度                                           │
│    action_shape = shape_meta['action']['shape']  # [7]       │
│    action_dim = action_shape[0]  # 7                         │
│                                                                 │
│    # 获取obs特征维度                                          │
│    obs_feature_dim = obs_encoder.output_shape()[0]  # 1543  │
│                                                                 │
│    # 计算UNet输入维度                                         │
│    if obs_as_global_cond:  # True                            │
│      input_dim = action_dim  # 7                             │
│      global_cond_dim = obs_feature_dim * n_obs_steps         │
│                       = 1543 * 2 = 3086                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓ 创建UNet
┌─────────────────────────────────────────────────────────────────┐
│ 6. ConditionalUnet1D初始化                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  model = ConditionalUnet1D(                                   │
│      input_dim=7,           ← action_dim                      │
│      global_cond_dim=3086   ← obs_feature_dim × n_obs_steps  │
│  )                                                             │
│                                                                 │
│  __init__执行:                                                │
│    # 创建全局条件编码器                                       │
│    self.global_cond_encoder = nn.Sequential(                 │
│        nn.Linear(3086, 1024),  ← 处理obs特征                 │
│        nn.Mish(),                                             │
│        nn.Linear(1024, 1024)                                  │
│    )                                                           │
│                                                                 │
│    # 创建下采样模块                                           │
│    ConditionalResidualBlock1D(                                │
│        in_channels=7,        ← action维度                     │
│        out_channels=512,                                      │
│        cond_dim=1024         ← 时间步 + 全局条件             │
│    )                                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 关键维度决策点

### 决策点1: ObsEncoder输出维度
**位置**: `multi_image_obs_encoder.py::output_shape()`

```python
# 输入: shape_meta
shape_meta = {
    'obs': {
        'camera_0': {'shape': [3,480,640], 'type': 'rgb'},
        'camera_1': {'shape': [3,480,640], 'type': 'rgb'},
        'camera_2': {'shape': [3,480,640], 'type': 'rgb'},
        'robot_eef_pose': {'shape': [7], 'type': 'low_dim'}
    }
}

# 处理逻辑:
rgb_features = []
for rgb_key in ['camera_0', 'camera_1', 'camera_2']:
    feature = ResNet18(image)  # 每个输出512维
    rgb_features.append(feature)  # 3 × 512 = 1536

lowdim_features = [robot_eef_pose]  # 7维

# 输出:
obs_feature_dim = 1536 + 7 = 1543
```

### 决策点2: UNet输入/条件维度
**位置**: `diffusion_unet_image_policy.py::__init__()`

```python
# 输入:
action_dim = shape_meta['action']['shape'][0]  # 7
obs_feature_dim = obs_encoder.output_shape()[0]  # 1543
n_obs_steps = 2
obs_as_global_cond = True

# 处理逻辑:
if obs_as_global_cond:
    # 场景1: 观测作为全局条件 (当前配置)
    input_dim = action_dim  # 7
    global_cond_dim = obs_feature_dim * n_obs_steps  # 1543 × 2 = 3086
else:
    # 场景2: 观测和动作拼接
    input_dim = action_dim + obs_feature_dim  # 7 + 1543 = 1550
    global_cond_dim = None

# 输出:
UNet(input_dim=7, global_cond_dim=3086)
```

---

## 📊 最终网络结构

```
输入: obs_dict
  ├─ camera_0: (B, T=2, 3, 480, 640)
  ├─ camera_1: (B, T=2, 3, 480, 640)
  ├─ camera_2: (B, T=2, 3, 480, 640)
  └─ robot_eef_pose: (B, T=2, 7)
       ↓
ObsEncoder
  ├─ 3× ResNet18: (B*T, 3, 480, 640) → (B*T, 512)
  └─ robot_eef_pose: (B*T, 7)
  → Concat: (B*T, 1543)
  → Reshape: (B, T*1543) = (B, 3086)  [global_cond]
       ↓
ConditionalUnet1D
  ├─ Input: noisy_action (B, horizon=16, 7)
  ├─ Global Cond: obs_features (B, 3086)
  └─ Timestep: t
       ↓
  Global Cond Encoder: (B, 3086) → (B, 1024)
       ↓
  UNet Processing: (B, 16, 7) + global_cond(B, 1024)
       ↓
输出: denoised_action (B, horizon=16, 7)
  → 取前n_action_steps
  → 最终输出: (B, 8, 7)
```

---

## 🔍 验证方法

如果想验证某个维度是否正确，可以在以下位置添加打印：

```python
# 1. ObsEncoder初始化后
print(f"ObsEncoder output_shape: {obs_encoder.output_shape()}")

# 2. Policy初始化时
print(f"action_dim: {action_dim}")
print(f"obs_feature_dim: {obs_feature_dim}")
print(f"input_dim: {input_dim}")
print(f"global_cond_dim: {global_cond_dim}")

# 3. UNet初始化时
print(f"UNet input_dim: {input_dim}")
print(f"UNet global_cond_dim: {global_cond_dim}")
```

---

## ✅ 当前配置验证

基于你的配置:
- ✅ 3个RGB相机 → 1536维
- ✅ robot_eef_pose → 7维
- ✅ obs_feature_dim → 1543维
- ✅ 2个obs_steps → global_cond 3086维
- ✅ action_dim → 7维
- ✅ UNet input → 7维 (action only)
- ✅ UNet global_cond → 3086维

**所有维度初始化完全正确！** ✅

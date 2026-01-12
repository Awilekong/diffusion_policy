from typing import Dict, Callable, Tuple, List, Optional
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform

# 维度映射表：语义化维度符号 -> 索引
DIM_MAP = {
    'x': 0,  # X 位置
    'y': 1,  # Y 位置
    'z': 2,  # Z 位置
    'r': 3,  # Roll
    'p': 4,  # Pitch
    'w': 5,  # Yaw (用 w 避免与 y 冲突)
    'g': 6,  # Gripper
}

def parse_dims(dims_str: Optional[str]) -> Optional[List[int]]:
    """解析语义化的维度字符串为索引列表"""
    if dims_str is None:
        return None
    return [DIM_MAP[c] for c in dims_str.lower()]

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    """
    将真实环境/客户端的观测数据转换为模型输入格式
    
    主要处理：
    1. 图像数据（rgb）：
       - 调整分辨率（如果需要）
       - 归一化到 [0, 1]（如果是 uint8）
       - 转换维度顺序：THWC -> TCHW
    2. 低维数据（low_dim）：
       - 提取特定维度（如 pose 只取 XY）
       - 直接传递
    
    ⚠️ 注意：这里只做基本预处理，真正的归一化（normalize）在 policy.predict_action() 内部进行
    
    Args:
        env_obs: 环境/客户端发送的原始观测数据，格式要求：
            {
                # 图像数据（例如 'camera_0', 'wrist_camera' 等）
                '<camera_key>': np.ndarray,  
                    # shape: (n_obs_steps, H, W, C)
                    # dtype: np.uint8 (0-255) 或 np.float32 (0-1)
                    # 通道顺序: RGB（不是BGR）
                    # 例如: (2, 480, 640, 3) 表示 2 帧历史图像
                
                # 低维数据（例如 'robot_eef_pose', 'joint_positions' 等）
                '<state_key>': np.ndarray,
                    # shape: (n_obs_steps, state_dim)
                    # dtype: np.float32 或 np.float64
                    # 例如: (2, 6) 表示 2 步的 6D 末端位姿（x,y,z,rx,ry,rz）
                
                # 可选：时间戳
                'timestamp': np.ndarray,  # shape: (n_obs_steps,)
            }
        
        shape_meta: 模型训练时的 shape 元数据（从 checkpoint 加载）
            {
                'obs': {
                    '<key>': {
                        'type': 'rgb' 或 'low_dim',
                        'shape': (C, H, W) for rgb, (dim,) for low_dim
                    }
                }
            }
    
    Returns:
        obs_dict_np: 模型输入格式的观测数据
            {
                # 图像数据
                '<camera_key>': np.ndarray,  
                    # shape: (n_obs_steps, C, H, W)  # 注意维度顺序变化
                    # dtype: np.float32, 范围 [0, 1]
                
                # 低维数据
                '<state_key>': np.ndarray,
                    # shape: (n_obs_steps, state_dim) 或 裁剪后的维度
                    # dtype: 保持原始 dtype
            }
    """
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')  # 模型期望的 shape
        
        if type == 'rgb':
            # ========== 图像数据处理 ==========
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape  # (n_obs_steps, H_in, W_in, C)
            co,ho,wo = shape  # 模型期望的 (C, H_out, W_out)
            assert ci == co, f"通道数不匹配: 输入 {ci}, 期望 {co}"
            out_imgs = this_imgs_in
            
            # 步骤1: 调整分辨率 或 归一化像素值
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                # 创建图像变换器（resize + crop）
                tf = get_image_transform(
                    input_res=(wi,hi),    # 输入分辨率
                    output_res=(wo,ho),   # 输出分辨率
                    bgr_to_rgb=True)     # False=输入已是RGB，不转换；True=输入是BGR，需转RGB
                # 对每一帧应用变换
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                
                # 步骤2: uint8 (0-255) -> float32 (0-1)
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            
            # 步骤3: 转换维度顺序 THWC -> TCHW
            # (n_obs_steps, H, W, C) -> (n_obs_steps, C, H, W)
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
            
        elif type == 'low_dim':
            # ========== 低维数据处理 ==========
            this_data_in = env_obs[key]

            # 通过 dims 参数选择特定维度
            dims = attr.get('dims', None)
            if dims is not None:
                dim_indices = parse_dims(dims)
                this_data_in = this_data_in[..., dim_indices]

            # 直接传递（保持原始 shape 和 dtype）
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res

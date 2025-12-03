#!/usr/bin/env python3
"""
Diffusion Policy 远程推理服务器 (单帧版本 n_obs_steps=1)
使用 web_policy 提供 WebSocket 推理服务

Usage:
    python serve_diffusion_policy_single_frame.py -i <checkpoint_path> -p 8000
"""

import sys
import os
import click
import torch
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
import copy

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web_policy'))

from web_policy import BasePolicy, WebSocketPolicyServer
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict


class DiffusionPolicySingleFrameWrapper(BasePolicy):
    """
    包装 Diffusion Policy 为 BasePolicy 接口 (单帧版本)
    处理所有归一化、反归一化和数据转换
    专门用于 n_obs_steps=1 的模型，不维护历史队列
    """
    
    def __init__(self, ckpt_path: str, device: str = 'cuda', debug: bool = True):
        """
        Args:
            ckpt_path: checkpoint 文件路径
            device: 'cuda' 或 'cpu'
            debug: 是否启用调试模式（保存中间数据）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        self.step_counter = 0
        
        # 创建调试日志目录
        if self.debug:
            self.debug_dir = Path('debug_logs_single_frame')
            self.debug_dir.mkdir(exist_ok=True)
            print(f"🐛 调试模式已启用，日志保存到: {self.debug_dir}")
        
        # 注册 OmegaConf resolver
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        
        # 加载 checkpoint
        print(f"🔄 加载 checkpoint: {ckpt_path}")
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        
        # 创建 workspace
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # 获取 policy（仅支持 Diffusion 模型）
        if 'diffusion' not in self.cfg.name:
            raise RuntimeError(f"仅支持 Diffusion 模型，当前模型: {self.cfg.name}")
        
        # Diffusion model
        policy: BaseImagePolicy = workspace.model
        if self.cfg.training.use_ema:
            policy = workspace.ema_model
            print("✅ 使用 EMA 模型")
        
        # 设置推理参数
        # num_inference_steps: DDIM 去噪采样步数
        #   - 训练时使用 100 步（DDPM 完整采样）
        #   - 推理时使用 16 步（DDIM 快速采样，速度提升 6 倍）
        #   - 权衡：步数越多质量越好但速度越慢
        policy.num_inference_steps = 16
        
        # 打印 horizon 信息
        print(f"📊 模型配置:")
        print(f"   horizon (预测序列长度): {policy.horizon}")
        print(f"   n_obs_steps (观测步数): {policy.n_obs_steps}")
        
        # 验证模型确实是 n_obs_steps=1
        if policy.n_obs_steps != 1:
            raise ValueError(
                f"❌ 此脚本仅支持 n_obs_steps=1 的模型！\n"
                f"   当前模型 n_obs_steps={policy.n_obs_steps}\n"
                f"   请使用 serve_diffusion_policy.py 来运行此模型"
            )
        
        # n_action_steps: 实际使用的动作数
        # 公式解释: max_action_steps = horizon - n_obs_steps + 1
        #   - 例如 horizon=16, n_obs_steps=1
        #   - 输入 [obs_t]，预测 [act_t, act_t+1, ..., act_t+15]
        #   - 可用动作数 = 16 - 1 + 1 = 16 = horizon
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        
        # 检查 n_action_steps 是否合理
        if policy.n_action_steps > policy.horizon:
            raise ValueError(
                f"❌ n_action_steps ({policy.n_action_steps}) 不能大于 horizon ({policy.horizon})"
            )
        elif policy.n_action_steps <= 0:
            raise ValueError(
                f"❌ n_action_steps ({policy.n_action_steps}) 必须为正数"
            )
        
        print(f"   n_action_steps (可用动作数): {policy.n_action_steps}")
        print(f"   ✅ 配置检查通过 (单帧模式)")
        
        self.policy = policy.eval().to(self.device)
        self.shape_meta = self.cfg.task.shape_meta
        self.n_obs_steps = self.cfg.n_obs_steps
        
        print(f"✅ Policy 加载成功")
        print(f"   类型: {self.cfg.name}")
        print(f"   设备: {self.device}")
        print(f"   观测步数: {self.n_obs_steps} (单帧模式)")
        
        # 打印训练时的图像分辨率配置
        print(f"\n📸 训练时图像配置:")
        for key, attr in self.shape_meta['obs'].items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                c, h, w = attr.get('shape')
                print(f"   {key}: {w}x{h} (WxH), 通道数={c}")
        print(f"\n💡 推理时图像处理流程 (单帧模式):")
        print(f"   1. 客户端发送: 单张图像 (H, W, 3) uint8")
        print(f"   2. 直接处理: 无需历史队列，直接添加时间维度")
        print(f"   3. get_real_obs_dict: resize到训练时分辨率 (见上方配置)")
        print(f"   4. policy内部: 进一步处理(可能crop)并归一化")
    
    def reset(self):
        """重置 policy 状态"""
        self.policy.reset()
        self.step_counter = 0
        print(f"🔄 重置 policy 状态 (单帧模式无需清空历史)")
    
    def infer(self, obs: dict) -> dict:
        """
        推理方法 (单帧版本)
        
        Args:
            obs: 观测数据字典，统一格式:
                {
                    # 图像数据 (支持多相机)
                    'observation/image': np.ndarray shape (H, W, C) uint8,  # 主相机
                    'observation/image_1': np.ndarray shape (H, W, C) uint8,  # 第二相机
                    'observation/image_2': np.ndarray shape (H, W, C) uint8,  # 第三相机
                    
                    # 状态数据
                    'observation/state': np.ndarray shape (state_dim,),  # 机器人状态
                }
                
                注意：单帧输入，直接添加时间维度 (1, H, W, C) 而非使用历史队列
        
        Returns:
            结果字典:
                {
                    'actions': np.ndarray shape (action_horizon, action_dim),  # 动作序列
                }
        """
        # ========== 调试: 保存原始输入 ==========
        if self.debug:
            input_obs_raw = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    input_obs_raw[k] = v.copy()
                else:
                    input_obs_raw[k] = v
            
            print(f"\n🔍 原始输入观测:")
            for k, v in input_obs_raw.items():
                if isinstance(v, np.ndarray):
                    print(f"   {k}: shape={v.shape}, dtype={v.dtype}")
                    # 检查图像统计信息
                    if 'image' in k and v.dtype == np.uint8:
                        pixel_mean = v.mean()
                        pixel_max = v.max()
                        pixel_min = v.min()
                        if pixel_max == 0:
                            print(f"      ⚠️  警告：图像全黑（全0）！")
                        else:
                            print(f"      像素统计：min={pixel_min}, max={pixel_max}, mean={pixel_mean:.1f}")
                    elif 'state' in k:
                        print(f"      数值范围：[{v.min():.4f}, {v.max():.4f}]")
                else:
                    print(f"   {k}: type={type(v)}")
            
            # 打印 shape_meta 配置（只在第一次打印）
            if self.step_counter == 0:
                print(f"\n📋 模型训练时的 shape_meta 配置:")
                for key, attr in self.shape_meta['obs'].items():
                    obs_type = attr.get('type', 'low_dim')
                    shape = attr.get('shape')
                    print(f"   {key}: type={obs_type}, shape={shape}")
                print()
        
        # 转换为 diffusion policy 内部格式 (单帧模式)
        env_obs = {}
        
        # 处理图像数据 - 直接添加时间维度
        # 映射：observation/image -> camera_0, observation/image_1 -> camera_1, observation/image_2 -> camera_2
        image_mapping = {
            'observation/image': 0,
            'observation/image_1': 1,
            'observation/image_2': 2,
        }
        
        for obs_key, camera_idx in image_mapping.items():
            if obs_key in obs:
                img = obs[obs_key]
                
                # 确保图像是numpy数组
                if not isinstance(img, np.ndarray):
                    img = np.array(img, dtype=np.uint8)
                
                # 单帧模式：直接添加时间维度 (1, H, W, C)
                img_array = np.expand_dims(img, axis=0)  # (H, W, C) -> (1, H, W, C)
                env_obs[f'camera_{camera_idx}'] = img_array
                
                if self.debug and self.step_counter == 0:
                    print(f"📸 相机 {camera_idx} ({obs_key}): shape={img.shape} -> {img_array.shape}")
        
        # 处理状态数据 - 直接添加时间维度
        if 'observation/state' in obs:
            state = obs['observation/state']
            
            # 确保状态是numpy数组，且是float32类型
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            else:
                state = state.astype(np.float32)
            
            # 单帧模式：直接添加时间维度 (1, state_dim)
            state_array = np.expand_dims(state, axis=0)  # (state_dim,) -> (1, state_dim)
            env_obs['robot_eef_pose'] = state_array
            
            if self.debug and self.step_counter == 0:
                print(f"🤖 状态: shape={state.shape} -> {state_array.shape}")
        
        # ========== 调试: 打印 env_obs 维度 ==========
        if self.debug:
            print(f"\n{'='*60}")
            print(f"🔍 Step {self.step_counter} - 推理数据流追踪 (单帧模式)")
            print(f"{'='*60}")
            print(f"\n1️⃣  env_obs (添加时间维度后):")
            print(f"   说明: n_obs_steps=1, 直接将单帧扩展为 (1, ...)")
            for key, value in env_obs.items():
                if 'camera' in key:
                    print(f"   {key}: shape={value.shape} (1, H, W, C), dtype={value.dtype}")
                else:
                    print(f"   {key}: shape={value.shape} (1, state_dim), dtype={value.dtype}")
        
        # 数据预处理：使用官方的 real_inference_util
        obs_dict_np = get_real_obs_dict(
            env_obs=env_obs, 
            shape_meta=self.shape_meta
        )
        
        # ========== 调试: 打印 obs_dict_np 维度 ==========
        if self.debug:
            print(f"\n2️⃣  obs_dict_np (get_real_obs_dict 输出):")
            print(f"   说明: 图像 resize、归一化到[0,1]、转为(1,C,H,W)")
            for key, value in obs_dict_np.items():
                if len(value.shape) == 4:  # 图像 (T,C,H,W)
                    print(f"   {key}: shape={value.shape} (1, C, H, W), dtype={value.dtype}")
                else:  # 低维数据
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 转换为 torch tensor 并移到设备
        obs_dict = dict_apply(
            obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )
        
        # ========== 调试: 保存模型输入（转为numpy） ==========
        if self.debug:
            obs_dict_tensor = {}
            for key, value in obs_dict.items():
                obs_dict_tensor[key] = value.detach().cpu().numpy()
            print(f"\n3️⃣  obs_dict_tensor (模型输入 - 送入policy前):")
            print(f"   说明: 添加 batch 维度 (1, 1, ...), 在policy内部会进行归一化")
            for key, value in obs_dict_tensor.items():
                if len(value.shape) == 5:  # 图像 (B,T,C,H,W)
                    print(f"   {key}: shape={value.shape} (batch, 1, C, H, W), dtype={value.dtype}")
                else:  # 低维数据
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 推理
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            # action 从第一个 obs step 开始
            action = result['action'][0].detach().to('cpu').numpy()
        
        # ========== 调试: 打印和保存输出动作 ==========
        if self.debug:
            print(f"\n4️⃣  action (模型输出 - 反归一化后):")
            print(f"   shape={action.shape} (action_horizon={self.policy.n_action_steps}, action_dim), dtype={action.dtype}")
            
            # 保存所有数据到 npz 文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.debug_dir / f"step_{self.step_counter:04d}_{timestamp}.npz"
            
            try:
                save_data = {
                    'input_obs_raw': input_obs_raw,
                    'env_obs': env_obs,
                    'obs_dict_np': obs_dict_np,
                    'obs_dict_tensor': obs_dict_tensor,
                    'action': action,
                }
                
                # 使用 allow_pickle=True 保存字典
                np.savez(log_file, **save_data)
                print(f"\n💾 调试数据已保存: {log_file}")
                print(f"   包含: input_obs_raw, env_obs, obs_dict_np, obs_dict_tensor, action")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"⚠️  保存调试数据失败: {e}")
            
            self.step_counter += 1
        
        # 返回结果（统一格式）
        return {
            'actions': action,
        }
    
    @property
    def metadata(self) -> dict:
        """返回 policy 元数据"""
        return {
            'model': self.cfg.name,
            'n_obs_steps': self.n_obs_steps,
            'device': str(self.device),
            'shape_meta': self.shape_meta,
            'mode': 'single_frame',
        }


@click.command()
@click.option('--input', '-i', required=True, help='Checkpoint 文件路径')
@click.option('--port', '-p', default=8000, type=int, help='服务器端口')
@click.option('--host', '-h', default='0.0.0.0', help='服务器地址')
@click.option('--device', '-d', default='cuda', help='设备: cuda 或 cpu')
@click.option('--debug', is_flag=True, help='启用调试模式（保存推理数据流）')
def main(input, port, host, device, debug):
    """启动 Diffusion Policy 远程推理服务器 (单帧版本)"""
    
    print("=" * 60)
    print("Diffusion Policy 远程推理服务器 (单帧版本)")
    print("专用于 n_obs_steps=1 的模型")
    print("=" * 60)
    
    # 创建 policy wrapper
    policy = DiffusionPolicySingleFrameWrapper(
        ckpt_path=input,
        device=device,
        debug=debug
    )
    
    print(f"\n📊 Policy 元数据:")
    for key, value in policy.metadata.items():
        if key != 'shape_meta':  # shape_meta 太长，不打印
            print(f"   {key}: {value}")
    
    print(f"\n📸 训练时图像配置:")
    for key, attr in policy.shape_meta['obs'].items():
        obs_type = attr.get('type', 'low_dim')
        if obs_type == 'rgb':
            c, h, w = attr.get('shape')
            print(f"   {key}: {w}x{h} (宽x高), 通道数={c}")
    
    # 启动服务器
    print(f"\n🚀 启动 WebSocket 服务器...")
    print(f"   地址: {host}:{port}")
    print(f"   健康检查: http://localhost:{port}/healthz")
    print(f"\n💡 提示:")
    print(f"   - 使用 WebSocketClientPolicy 连接此服务器")
    print(f"   - 此版本专为 n_obs_steps=1 的模型优化")
    print(f"   - 无需维护历史帧队列，性能更优")
    print(f"   - 按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    server = WebSocketPolicyServer(
        policy=policy,
        host=host,
        port=port
    )
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Diffusion Policy 远程推理服务器 (单帧版本 n_obs_steps=1)
使用 web_policy 提供 WebSocket 推理服务

Usage:
    python serve_diffusion_policy_single_frame.py -i <checkpoint_path> -p 8000

    # 启用 WandB 调试模式
    python serve_diffusion_policy_single_frame.py -i <checkpoint_path> -p 8000 --wandb
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
import time
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image

# 添加路径
sys.path.insert(0, '/root/code/zpw/IL/web_policy/src')

from web_policy import BasePolicy, WebSocketPolicyServer
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict


class WandbDebugger:
    """WandB 调试器 - 优雅地记录和可视化推理数据流"""

    def __init__(self, enabled: bool = True, project: str = "diffusion_policy_inference"):
        """
        Args:
            enabled: 是否启用 WandB
            project: WandB 项目名称
        """
        self.enabled = enabled
        self.step_counter = 0

        if self.enabled:
            try:
                import wandb
                self.wandb = wandb

                # 初始化 wandb
                run_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.wandb.init(
                    project=project,
                    name=run_name,
                    config={
                        "mode": "real_robot_inference",
                        "timestamp": datetime.now().isoformat(),
                    },
                    tags=["inference", "real_robot", "single_frame"]
                )
                print(f"✅ WandB 初始化成功: {project}/{run_name}")
                print(f"   查看调试信息: {self.wandb.run.url}")
            except ImportError:
                print("⚠️  WandB 未安装，调试功能将被禁用")
                print("   安装: pip install wandb")
                self.enabled = False
            except Exception as e:
                print(f"⚠️  WandB 初始化失败: {e}")
                self.enabled = False

    @staticmethod
    def _tensor_to_image(tensor: torch.Tensor, value_range=(-1, 1)) -> np.ndarray:
        """
        将 PyTorch tensor 转换为可视化的 uint8 图像

        Args:
            tensor: (C, H, W) 或 (H, W, C) 格式
            value_range: 张量的数值范围，用于映射到 [0, 255]

        Returns:
            (H, W, C) uint8 图像
        """
        img = tensor.detach().cpu().numpy()

        # 如果是 (C, H, W)，转为 (H, W, C)
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = img.transpose(1, 2, 0)

        # 归一化到 [0, 255]
        vmin, vmax = value_range
        img = ((img - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)

        return img

    def _create_action_table(self, actions: np.ndarray, stage_name: str) -> 'wandb.Table':
        """为动作序列创建表格"""
        if len(actions.shape) == 1:
            # 单步动作，添加时间维度
            actions = actions.reshape(1, -1)

        horizon, action_dim = actions.shape

        # 构建数据
        table_data = []
        for t in range(horizon):
            row = [t] + actions[t].tolist()
            table_data.append(row)

        # 列名
        columns = ["time_step"] + [f"dim_{i}" for i in range(action_dim)]

        return self.wandb.Table(
            columns=columns,
            data=table_data
        )

    def _create_3d_trajectory_plot(self, actions: np.ndarray, stage_name: str, step_counter: int) -> 'wandb.Image':
        """
        创建动作轨迹的 3D 可视化（前3个维度：x, y, z）

        Args:
            actions: (horizon, action_dim) 动作序列
            stage_name: 阶段名称
            step_counter: 步数计数器

        Returns:
            wandb.Image: 3D 轨迹图
        """
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)

        horizon, action_dim = actions.shape

        # 只取前3个维度（x, y, z）
        if action_dim < 3:
            # 如果维度不足3，补零
            actions_3d = np.zeros((horizon, 3))
            actions_3d[:, :action_dim] = actions
        else:
            actions_3d = actions[:, :3]

        # 创建 3D 图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 提取动作的前3维
        # 原始数据：dim_0=x(法向), dim_1=y(左右), dim_2=z(上下)
        # matplotlib 3D: x轴-右, y轴-前, z轴-上
        # 映射：动作的y->matplotlib的x, 动作的x->matplotlib的y, 动作的z->matplotlib的z
        data_x = actions_3d[:, 0]  # 法向（里外）
        data_y = actions_3d[:, 1]  # 左右
        data_z = actions_3d[:, 2]  # 上下

        # 映射到 matplotlib 坐标系
        plot_x = data_y  # 左右 -> matplotlib x轴
        plot_y = data_x  # 法向 -> matplotlib y轴
        plot_z = data_z  # 上下 -> matplotlib z轴

        # 绘制轨迹线
        ax.plot(plot_x, plot_y, plot_z, 'b-', linewidth=2, label='Trajectory', alpha=0.7)

        # 绘制起点（绿色）
        ax.scatter(plot_x[0], plot_y[0], plot_z[0], c='green', s=100, marker='o', label='Start', zorder=5)

        # 绘制终点（红色）
        ax.scatter(plot_x[-1], plot_y[-1], plot_z[-1], c='red', s=100, marker='s', label='End', zorder=5)

        # 绘制中间点（蓝色，带时间标注）
        for t in range(horizon):
            ax.scatter(plot_x[t], plot_y[t], plot_z[t], c='blue', s=30, alpha=0.5, zorder=3)
            # 每隔几个点标注时间步
            if t % max(1, horizon // 5) == 0:
                ax.text(plot_x[t], plot_y[t], plot_z[t], f't={t}', fontsize=8, alpha=0.6)

        # 设置标签
        # 显示的坐标轴对应原始动作的维度：
        # X轴 (横向) = dim_1 (左右)
        # Y轴 (纵向) = dim_0 (法向，里外)
        # Z轴 (竖向) = dim_2 (上下)
        ax.set_xlabel('Left-Right (dim_1) →', fontsize=10, fontweight='bold')
        ax.set_ylabel('Front-Back (dim_0) ⊙', fontsize=10, fontweight='bold')
        ax.set_zlabel('Up-Down (dim_2) ↑', fontsize=10, fontweight='bold')

        # 设置标题
        ax.set_title(f'3D Action Trajectory - {stage_name}\nStep {step_counter} | Horizon {horizon}',
                     fontsize=12, fontweight='bold')

        # 添加图例
        ax.legend(loc='upper right', fontsize=9)

        # 添加网格
        ax.grid(True, alpha=0.3)

        # 设置视角（调整为更符合真实视角）
        # elev: 仰角（从上往下看的角度，0=水平，90=俯视）
        # azim: 方位角（旋转角度，-90 使得 y 轴朝右）
        ax.view_init(elev=25, azim=-60)

        # 调整布局
        plt.tight_layout()

        # 转换为图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)

        # 关闭图形以释放内存
        plt.close(fig)

        # 转换为 numpy array
        img_array = np.array(img)

        return self.wandb.Image(img_array, caption=f"{stage_name} | Step {step_counter}")

    def log_inference_step(self,
                          # 图像处理流程 (4个阶段)
                          input_images: dict,          # 原始输入图像
                          processed_images: dict,      # get_real_obs_dict 后
                          normalized_images: dict,     # LinearNormalizer 后
                          final_images: dict,          # ImageNet normalize 后 (最重要)
                          camera_mapping: list,        # 相机映射表

                          # 动作处理流程 (4个阶段)
                          action_normalized: np.ndarray,    # 归一化空间
                          action_pred: np.ndarray,          # 完整预测
                          action_exec: np.ndarray,          # 可执行部分
                          action_final: np.ndarray,         # 发送给客户端

                          # 状态（可选）
                          input_state: np.ndarray = None):
        """
        记录单次推理的完整数据流（改进版）

        专注于：
        1. 图像处理的4个关键阶段可视化
        2. 相机到模型输入的映射关系
        3. 动作的4个处理阶段
        4. 移除性能监控和过多统计数据
        """
        if not self.enabled:
            return

        log_data = {}

        # ========== 1. 图像可视化（4个阶段）==========
        # 只在前10步或每20步记录一次图像（避免存储爆炸）
        if self.step_counter < 10 or self.step_counter % 20 == 0:

            # 阶段1: 客户端原始输入
            for key, img in input_images.items():
                log_data[f"images/stage1_raw/{key}"] = self.wandb.Image(
                    img,
                    caption=f"Step {self.step_counter} | {key} | Shape: {img.shape} | [0,255] uint8"
                )

            # 阶段2: get_real_obs_dict 处理后 (resize + crop + [0,1])
            for key, img in processed_images.items():
                log_data[f"images/stage2_processed/{key}"] = self.wandb.Image(
                    img,
                    caption=f"Step {self.step_counter} | {key} | Shape: {img.shape} | [0,1] float32"
                )

            # 阶段3: LinearNormalizer 归一化后 (通常[-1,1])
            for key, img in normalized_images.items():
                log_data[f"images/stage3_normalized/{key}"] = self.wandb.Image(
                    img,
                    caption=f"Step {self.step_counter} | {key} | Shape: {img.shape} | [-1,1] float32"
                )

            # 阶段4: 最终送入 UNet 的图像 (ImageNet normalize, 最重要!)
            for key, img in final_images.items():
                log_data[f"images/stage4_final_to_unet/{key}"] = self.wandb.Image(
                    img,
                    caption=f"Step {self.step_counter} | {key} ⭐ | Shape: {img.shape} | ImageNet norm"
                )

        # ========== 2. 相机映射表（每次都记录）==========
        if camera_mapping:
            log_data["debug/camera_mapping"] = self.wandb.Table(
                columns=["客户端键", "env_obs键", "shape_meta键", "训练形状(C,H,W)", "实际形状(H,W,C)"],
                data=camera_mapping
            )

        # ========== 3. 动作序列可视化（4个阶段）==========
        # 每次都记录动作表格
        log_data["actions/stage1_normalized"] = self._create_action_table(
            action_normalized,
            "归一化空间 (模型原始输出)"
        )
        log_data["actions/stage2_pred_full"] = self._create_action_table(
            action_pred,
            "反归一化后完整预测 (horizon长度)"
        )
        log_data["actions/stage3_exec"] = self._create_action_table(
            action_exec,
            "提取的可执行动作 (n_action_steps长度)"
        )
        log_data["actions/stage4_final"] = self._create_action_table(
            action_final,
            "发送给客户端的最终动作"
        )

        # ========== 3.1 动作轨迹 3D 可视化 ==========
        # 只在前10步或每20步记录一次（避免生成过多图片）
        if self.step_counter < 10 or self.step_counter % 20 == 0:
            # Stage 2: 完整预测轨迹（horizon 长度）
            log_data["trajectory_3d/stage2_pred_full"] = self._create_3d_trajectory_plot(
                action_pred,
                "Full Prediction",
                self.step_counter
            )

            # Stage 4: 最终发送轨迹
            log_data["trajectory_3d/stage4_final"] = self._create_3d_trajectory_plot(
                action_final,
                "Final Output",
                self.step_counter
            )

        # ========== 4. 状态数据（简化版）==========
        if input_state is not None:
            # 只记录每个维度的值，不记录统计
            # 确保是 numpy array
            if isinstance(input_state, list):
                state_flat = np.array(input_state).flatten()
            elif isinstance(input_state, np.ndarray):
                state_flat = input_state.flatten()
            else:
                state_flat = np.array([input_state]).flatten()

            for i, val in enumerate(state_flat[:min(14, len(state_flat))]):
                log_data[f"state/dim_{i}"] = float(val)

        # ========== 提交日志 ==========
        self.wandb.log(log_data, step=self.step_counter)
        self.step_counter += 1

    def log_error(self, error_msg: str, obs: dict = None):
        """记录错误信息"""
        if not self.enabled:
            return

        log_data = {
            "error/message": error_msg,
            "error/timestamp": datetime.now().isoformat(),
            "error/step": self.step_counter
        }

        if obs:
            # 记录导致错误的输入
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    log_data[f"error/input/{key}/shape"] = str(value.shape)
                    log_data[f"error/input/{key}/dtype"] = str(value.dtype)

        self.wandb.log(log_data, step=self.step_counter)

    def finish(self):
        """结束 WandB 运行"""
        if self.enabled:
            self.wandb.finish()
            print("✅ WandB 运行已结束")


class DiffusionPolicySingleFrameWrapper(BasePolicy):
    """
    包装 Diffusion Policy 为 BasePolicy 接口 (单帧版本)
    处理所有归一化、反归一化和数据转换
    专门用于 n_obs_steps=1 的模型，不维护历史队列
    """

    def __init__(self, ckpt_path: str, device: str = 'cuda',
                 use_wandb: bool = False, wandb_project: str = "diffusion_policy_inference"):
        """
        Args:
            ckpt_path: checkpoint 文件路径
            device: 'cuda' 或 'cpu'
            use_wandb: 是否使用 WandB 调试
            wandb_project: WandB 项目名称
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.step_counter = 0

        # 初始化 WandB 调试器
        self.wandb_debugger = WandbDebugger(enabled=use_wandb, project=wandb_project)

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

        # 记录模型配置到 WandB
        if self.wandb_debugger.enabled:
            self.wandb_debugger.wandb.config.update({
                "model_name": self.cfg.name,
                "horizon": policy.horizon,
                "n_obs_steps": policy.n_obs_steps,
                "n_action_steps": policy.n_action_steps,
                "device": str(self.device),
                "num_inference_steps": policy.num_inference_steps,
                "shape_meta": dict(self.shape_meta),
            })

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
        # ========== 阶段1: 保存原始输入图像 ==========
        input_images = {}
        for key, value in obs.items():
            if 'image' in key and isinstance(value, np.ndarray):
                input_images[key] = value.copy()

        # 保存原始状态
        input_state = obs.get('observation/state', None)

        # ========== 转换为 diffusion policy 内部格式 (单帧模式) ==========
        env_obs = {}

        # 处理图像数据 - 直接添加时间维度
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

        # ========== 阶段2: 保存 get_real_obs_dict 处理后的图像 ==========
        try:
            obs_dict_np = get_real_obs_dict(
                env_obs=env_obs,
                shape_meta=self.shape_meta
            )
        except Exception as e:
            error_msg = f"get_real_obs_dict 失败: {str(e)}"
            print(f"❌ {error_msg}")
            self.wandb_debugger.log_error(error_msg, obs)
            raise

        processed_images = {}
        for key, value in obs_dict_np.items():
            if 'camera' in key and len(value.shape) == 4:  # (T,C,H,W)
                # 转回 HWC 用于可视化
                img = value[0].transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
                # 如果是 [0,1] 范围，转回 [0,255] uint8
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                processed_images[key] = img

        # ========== 阶段3: 转为 Tensor ==========
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        # ========== 设置调试回调来捕获中间数据 ==========
        normalized_images = {}
        final_images = {}
        action_normalized = None
        action_pred = None

        def debug_callback(stage_name, data):
            nonlocal normalized_images, final_images, action_normalized, action_pred

            if stage_name == 'stage3_normalized_obs':
                # Stage 3: 归一化后的观测
                for key, value in data.items():
                    if 'camera' in key:
                        img = value[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
                        img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        normalized_images[key] = img

            elif stage_name == 'stage4_final_to_unet':
                # Stage 4: 最终送入 UNet 的图像
                for key, value in data.items():
                    img = value[0].detach().cpu().numpy().transpose(1, 2, 0)
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                    final_images[key] = img

            elif stage_name == 'action_stage1_normalized':
                # Action Stage 1: 归一化空间的动作
                action_normalized = data[0].detach().cpu().numpy()

            elif stage_name == 'action_stage2_pred_full':
                # Action Stage 2: 完整预测序列
                action_pred = data[0].detach().cpu().numpy()

        # 设置回调
        self.policy.debug_callback = debug_callback
        self.policy.obs_encoder.debug_callback = debug_callback

        # ========== 阶段5: 推理（回调会在内部被触发）==========
        try:
            with torch.no_grad():
                result = self.policy.predict_action(obs_dict)

                # 提取动作
                action_exec = result['action'][0].detach().cpu().numpy()       # (n_action_steps, Da)
                action_final = action_exec  # 最终发送给客户端的就是 action_exec

        except Exception as e:
            error_msg = f"模型推理失败: {str(e)}"
            print(f"❌ {error_msg}")
            self.wandb_debugger.log_error(error_msg, obs)
            raise
        finally:
            # 清理调试回调
            self.policy.debug_callback = None
            self.policy.obs_encoder.debug_callback = None

        # ========== 阶段6: 构建相机映射表 ==========
        camera_mapping = []
        for client_key, cam_idx in image_mapping.items():
            if client_key in obs:
                env_key = f'camera_{cam_idx}'
                # 从 shape_meta 查找对应的键
                shape_meta_key = env_key
                if env_key in self.shape_meta['obs']:
                    train_shape = self.shape_meta['obs'][env_key]['shape']
                    actual_shape = obs[client_key].shape
                    camera_mapping.append([
                        client_key,
                        env_key,
                        shape_meta_key,
                        str(train_shape),
                        str(actual_shape)
                    ])

        # ========== 阶段7: 记录到 WandB ==========
        if self.wandb_debugger.enabled:
            self.wandb_debugger.log_inference_step(
                input_images=input_images,
                processed_images=processed_images,
                normalized_images=normalized_images,
                final_images=final_images,
                camera_mapping=camera_mapping,
                action_normalized=action_normalized,
                action_pred=action_pred,
                action_exec=action_exec,
                action_final=action_final,
                input_state=input_state
            )

        self.step_counter += 1

        # 打印简要信息
        if self.step_counter % 10 == 0:
            print(f"📊 Step {self.step_counter}: 动作范围 [{action_final.min():.3f}, {action_final.max():.3f}]")

        # 返回结果（统一格式）
        return {
            'actions': action_final,
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

    def __del__(self):
        """析构函数：确保 WandB 运行正常结束"""
        if hasattr(self, 'wandb_debugger'):
            self.wandb_debugger.finish()


@click.command()
@click.option('--input', '-i', required=True, help='Checkpoint 文件路径')
@click.option('--port', '-p', default=8000, type=int, help='服务器端口')
@click.option('--host', '-h', default='0.0.0.0', help='服务器地址')
@click.option('--device', '-d', default='cuda', help='设备: cuda 或 cpu')
@click.option('--wandb', is_flag=True, help='启用 WandB 调试模式（记录数据流和可视化）')
@click.option('--wandb-project', default='diffusion_policy_inference', help='WandB 项目名称')
def main(input, port, host, device, wandb, wandb_project):
    """启动 Diffusion Policy 远程推理服务器 (单帧版本)"""

    print("=" * 60)
    print("Diffusion Policy 远程推理服务器 (单帧版本)")
    print("专用于 n_obs_steps=1 的模型")
    if wandb:
        print("🔍 WandB 调试模式已启用")
    print("=" * 60)

    # 创建 policy wrapper
    policy = DiffusionPolicySingleFrameWrapper(
        ckpt_path=input,
        device=device,
        use_wandb=wandb,
        wandb_project=wandb_project
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
    if wandb:
        print(f"\n🔍 WandB 调试:")
        print(f"   实时查看: {policy.wandb_debugger.wandb.run.url}")
        print(f"   - 观测输入统计和图像")
        print(f"   - 数据处理流程可视化")
        print(f"   - 动作输出分析")
        print(f"   - 推理时间统计")
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
        # 确保 WandB 运行结束
        if wandb:
            policy.wandb_debugger.finish()


if __name__ == '__main__':
    main()

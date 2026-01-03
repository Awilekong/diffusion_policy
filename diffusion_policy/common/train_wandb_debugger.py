"""
训练专用的 WandB 调试器

用途：在训练过程中记录 batch 数据的完整处理流程，
     格式与推理时的 WandB 记录完全一致，
     方便对比训练/推理的数据一致性。
"""

import numpy as np
import torch
import wandb
from typing import Dict, Optional
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image


class TrainWandbDebugger:
    """
    训练数据调试器

    记录训练 batch 的完整数据流：
    1. 原始 batch 数据（从 DataLoader）
    2. 归一化后的数据（LinearNormalizer）
    3. 最终送入 UNet 的数据（ImageNet normalize）
    4. Ground truth vs predicted actions
    """

    def __init__(self, wandb_run, enabled: bool = True):
        """
        Args:
            wandb_run: 现有的 wandb run 对象（复用训练的 run）
            enabled: 是否启用调试
        """
        self.wandb_run = wandb_run
        self.enabled = enabled
        self.step_counter = 0

    def _create_3d_trajectory_plot(self, actions: np.ndarray, stage_name: str, epoch: int, is_normalized: bool = True) -> 'wandb.Image':
        """
        创建动作轨迹的 3D 可视化（前3个维度：x, y, z）

        Args:
            actions: (T, action_dim) 动作序列（归一化或未归一化）
            stage_name: 阶段名称（如 "Ground Truth", "Prediction"）
            epoch: 当前 epoch
            is_normalized: 是否是归一化后的动作

        Returns:
            wandb.Image: 3D 轨迹图
        """
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)

        horizon, action_dim = actions.shape

        # 只取前3个维度（x, y, z）
        if action_dim < 3:
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
        ax.set_xlabel('Left-Right (dim_1) →', fontsize=10, fontweight='bold')
        ax.set_ylabel('Front-Back (dim_0) ⊙', fontsize=10, fontweight='bold')
        ax.set_zlabel('Up-Down (dim_2) ↑', fontsize=10, fontweight='bold')

        # 设置标题
        norm_text = "Normalized" if is_normalized else "Denormalized"
        ax.set_title(f'3D Action Trajectory - {stage_name}\nEpoch {epoch} | Horizon {horizon} | {norm_text}',
                     fontsize=12, fontweight='bold')

        # 添加图例
        ax.legend(loc='upper right', fontsize=9)

        # 添加网格
        ax.grid(True, alpha=0.3)

        # 设置视角
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

        return wandb.Image(img_array, caption=f"{stage_name} | Epoch {epoch} | {norm_text}")

    def log_training_batch(self,
                          captured_data: dict,  # 从回调中捕获的中间数据
                          batch_raw: dict,      # 原始 batch
                          epoch: int,
                          step: int,
                          sample_idx: int = 0):  # 记录batch中的第几个样本
        """
        记录一个训练 batch 的数据（格式与推理一致）

        Args:
            captured_data: 通过 debug_callback 捕获的中间数据
                {
                    'train_stage3_normalized_obs': nobs (归一化后的观测),
                    'stage4_final_to_unet': final_images (ImageNet norm后),
                    'train_action_pred': action_pred (预测的动作)
                }
            batch_raw: 原始 batch 数据（从 DataLoader）
                {
                    'obs': {'camera_0': ..., 'camera_1': ..., 'robot_eef_pose': ...},
                    'action': ...
                }
            epoch: 当前 epoch
            step: 全局步数
            sample_idx: 记录 batch 中的第几个样本（默认第一个）
        """
        if not self.enabled:
            return

        log_data = {}

        # ========== 1. 图像可视化（对应推理的 4 个阶段）==========

        # 阶段 1: 原始 batch 数据（对应推理的 stage1_raw）
        if 'obs' in batch_raw:
            for key, value in batch_raw['obs'].items():
                if 'camera' in key and isinstance(value, (np.ndarray, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()

                    # 取第一个样本: (B, T, C, H, W) -> (T, C, H, W)
                    img_data = value[sample_idx]

                    # 取第一帧: (T, C, H, W) -> (C, H, W) -> (H, W, C)
                    img = img_data[0].transpose(1, 2, 0)

                    # 转换到 [0,255] uint8
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    log_data[f"train_images/stage1_batch_raw/{key}"] = wandb.Image(
                        img,
                        caption=f"Epoch {epoch} | {key} | Shape: {img.shape} | [0,1] float32"
                    )

        # 阶段 3: LinearNormalizer 归一化后（对应推理的 stage3_normalized）
        if 'train_stage3_normalized_obs' in captured_data:
            nobs = captured_data['train_stage3_normalized_obs']
            for key, value in nobs.items():
                if 'camera' in key:
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()

                    # (B, T, C, H, W) -> 取第一个样本和第一帧
                    img_data = value[sample_idx, 0]  # (C, H, W)
                    img = img_data.transpose(1, 2, 0)  # (H, W, C)

                    # 从 [-1,1] 映射到 [0,255]
                    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                    log_data[f"train_images/stage3_normalized/{key}"] = wandb.Image(
                        img,
                        caption=f"Epoch {epoch} | {key} | Shape: {img.shape} | [-1,1] float32"
                    )

        # 阶段 4: 最终送入 UNet（对应推理的 stage4_final_to_unet）
        if 'stage4_final_to_unet' in captured_data:
            final_images = captured_data['stage4_final_to_unet']
            for key, value in final_images.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()

                # (B*T, C, H, W) -> 取第一个
                img_data = value[sample_idx]  # (C, H, W)
                img = img_data.transpose(1, 2, 0)  # (H, W, C)

                # ImageNet normalize 后的范围约 [-2, 2]，映射到 [0,255]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

                log_data[f"train_images/stage4_final_to_unet/{key} ⭐"] = wandb.Image(
                    img,
                    caption=f"Epoch {epoch} | {key} | Shape: {img.shape} | ImageNet norm"
                )

        # ========== 2. 动作对比：Ground Truth vs Predicted ==========

        # Ground truth action
        action_gt_sample = None
        if 'action' in batch_raw:
            action_gt = batch_raw['action']
            if isinstance(action_gt, torch.Tensor):
                action_gt = action_gt.cpu().numpy()

            # 取第一个样本: (B, T, Da) -> (T, Da)
            action_gt_sample = action_gt[sample_idx]

            log_data["train_actions/ground_truth"] = self._create_action_table(
                action_gt_sample,
                f"Ground Truth (Epoch {epoch})"
            )

        # Normalized action (from model)
        action_norm_sample = None
        if 'train_action_normalized' in captured_data:
            action_norm = captured_data['train_action_normalized']
            if isinstance(action_norm, torch.Tensor):
                action_norm = action_norm.cpu().numpy()

            # 取第一个样本: (B, T, Da) -> (T, Da)
            action_norm_sample = action_norm[sample_idx]

            log_data["train_actions/normalized"] = self._create_action_table(
                action_norm_sample,
                f"Normalized (Epoch {epoch})"
            )

        # ========== 2.1 动作轨迹 3D 可视化 ==========
        # 对比 Ground Truth (原始batch) 和 Normalized (归一化后) 的轨迹
        # 注意：训练时我们记录的都是归一化前后的 GT，因为训练时没有做完整推理

        # Ground Truth 3D 轨迹（原始动作，未归一化）
        if action_gt_sample is not None:
            log_data["train_trajectory_3d/ground_truth_raw"] = self._create_3d_trajectory_plot(
                action_gt_sample,
                "Ground Truth (Raw)",
                epoch,
                is_normalized=False
            )

        # Normalized 3D 轨迹（归一化后的动作）
        if action_norm_sample is not None:
            log_data["train_trajectory_3d/ground_truth_normalized"] = self._create_3d_trajectory_plot(
                action_norm_sample,
                "Ground Truth (Normalized)",
                epoch,
                is_normalized=True
            )

        # ========== 3. Batch 元信息 ==========
        batch_info = []
        if 'obs' in batch_raw:
            for key, value in batch_raw['obs'].items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                batch_info.append([key, str(value.shape), str(value.dtype)])

        if batch_info:
            log_data["train_debug/batch_info"] = wandb.Table(
                columns=["Key", "Shape", "Dtype"],
                data=batch_info
            )

        # ========== 提交日志 ==========
        self.wandb_run.log(log_data, step=step)
        self.step_counter += 1

    @staticmethod
    def _create_action_table(actions: np.ndarray, description: str) -> wandb.Table:
        """
        创建动作序列的 WandB 表格

        Args:
            actions: (T, Da) 动作数组
            description: 描述文字

        Returns:
            wandb.Table
        """
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)

        T, Da = actions.shape

        # 创建列名: time_step, dim_0, dim_1, ...
        columns = ["time_step"] + [f"dim_{i}" for i in range(Da)]

        # 创建数据行
        data = []
        for t in range(T):
            row = [t] + actions[t].tolist()
            data.append(row)

        return wandb.Table(columns=columns, data=data)

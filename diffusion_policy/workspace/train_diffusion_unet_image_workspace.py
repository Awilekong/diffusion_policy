if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner = None
        if 'env_runner' in cfg.task and cfg.task.env_runner is not None:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure training data debugger (handles both training and validation)
        train_debugger = None
        if cfg.training.get('enable_data_debug', False) or cfg.training.get('enable_val_debug', False):
            from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger
            # 获取 delta_action 配置
            delta_action = cfg.task.dataset.get('delta_action', False)
            train_debugger = TrainWandbDebugger(
                wandb_run=wandb_run,
                enabled=True,
                delta_action=delta_action
            )
            if delta_action:
                print(f"✅ Delta Action 模式已启用，3D轨迹可视化将从初始状态重建绝对轨迹")
            if cfg.training.get('enable_data_debug', False):
                print(f"✅ 训练数据调试已启用，每 {cfg.training.get('debug_every', 10)} 个 epoch 记录一次")
            if cfg.training.get('enable_val_debug', False):
                print(f"✅ 验证数据调试已启用，每 {cfg.training.get('val_debug_every', 10)} 个 epoch 记录 {cfg.training.get('val_debug_num_samples', 5)} 个样本")

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # 是否需要记录调试数据（每 N 个 epoch 记录一次，在第一个 batch）
                        # epoch 0 也要记录，所以 epoch % debug_every == 0
                        # 或者在第一个step (global_step==0) 时也记录
                        should_debug = (train_debugger is not None and
                                       batch_idx == 0 and
                                       (self.global_step == 0 or
                                        self.epoch % cfg.training.get('debug_every', 10) == 0))

                        # 保存原始 batch 用于调试
                        batch_raw = None
                        captured_data = {}
                        if should_debug:
                            batch_raw = dict_apply(batch, lambda x: x.detach().cpu())

                            # 设置调试回调来捕获中间数据
                            def debug_callback(stage_name, data):
                                # 保存数据的副本
                                if isinstance(data, dict):
                                    captured_data[stage_name] = {
                                        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                        for k, v in data.items()
                                    }
                                else:
                                    captured_data[stage_name] = data.detach().cpu() if isinstance(data, torch.Tensor) else data

                            self.model.debug_callback = debug_callback
                            self.model.obs_encoder.debug_callback = debug_callback

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # 记录调试数据
                        if should_debug and batch_raw is not None:
                            # 清理回调
                            self.model.debug_callback = None
                            self.model.obs_encoder.debug_callback = None

                            # 对于首次或需要3D轨迹对比的epoch，获取预测动作
                            action_pred_cpu = None
                            if self.global_step == 0 or (cfg.training.get('enable_data_debug', False) and
                                                          self.epoch % cfg.training.get('debug_every', 10) == 0):
                                with torch.no_grad():
                                    # 使用EMA模型预测（如果启用）
                                    debug_policy = self.ema_model if cfg.training.use_ema else self.model
                                    debug_policy.eval()

                                    # 前向推理获取预测动作
                                    obs_dict_single = dict_apply(batch['obs'], lambda x: x[:1])  # 取第一个样本
                                    result = debug_policy.predict_action(obs_dict_single)
                                    action_pred_cpu = result['action_pred'][0].cpu().numpy()  # (Ta, 7)

                            # 记录完整数据流（首次）或仅3D轨迹对比（后续）到 WandB
                            train_debugger.log_training_batch(
                                captured_data=captured_data,
                                batch_raw=batch_raw,
                                epoch=self.epoch,
                                step=self.global_step,
                                sample_idx=0,  # 只记录第一个样本
                                action_pred=action_pred_cpu  # 提供预测动作用于3D对比
                            )

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and env_runner is not None:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()

                        # 判断是否需要收集调试样本（每隔N个epoch）
                        should_debug_val = (train_debugger is not None and
                                           cfg.training.get('enable_val_debug', False) and
                                           self.epoch % cfg.training.get('val_debug_every', 10) == 0)
                        val_debug_samples = [] if should_debug_val else None

                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                                # 计算loss（始终需要）
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)

                                # === 收集验证样本用于3D轨迹可视化 ===
                                if should_debug_val and len(val_debug_samples) < cfg.training.get('val_debug_num_samples', 5):
                                    # 使用 EMA 模型（如果启用）进行预测
                                    debug_policy = self.ema_model if cfg.training.use_ema else self.model

                                    obs_dict = batch['obs']
                                    action_gt = batch['action']  # (B, Ta, 7) - denormalized

                                    # 前向推理获取预测
                                    result = debug_policy.predict_action(obs_dict)
                                    action_pred = result['action_pred']  # (B, Ta, 7) - denormalized

                                    # 收集样本（取 batch 中的前几个样本）
                                    B = action_gt.shape[0]
                                    num_to_collect = min(B, cfg.training.get('val_debug_num_samples', 5) - len(val_debug_samples))

                                    for sample_idx in range(num_to_collect):
                                        sample_data = {
                                            'action_gt': action_gt[sample_idx].cpu().numpy(),  # (Ta, 7)
                                            'action_pred': action_pred[sample_idx].cpu().numpy(),  # (Ta, 7)
                                            'loss': loss.item(),
                                        }
                                        # Delta action 模式下，添加初始状态（用于重建绝对轨迹）
                                        if 'robot_eef_pose' in obs_dict:
                                            robot_state = obs_dict['robot_eef_pose']
                                            if isinstance(robot_state, torch.Tensor):
                                                robot_state = robot_state.cpu().numpy()
                                            # 取最后一个观测时间步作为初始状态
                                            sample_data['initial_state'] = robot_state[sample_idx, -1]  # (7,)
                                        val_debug_samples.append(sample_data)

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                        # === 记录3D轨迹可视化（仅轨迹，不含图像） ===
                        if should_debug_val and val_debug_samples:
                            train_debugger.log_validation_trajectories(
                                val_samples=val_debug_samples,
                                epoch=self.epoch,
                                global_step=self.global_step
                            )

                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

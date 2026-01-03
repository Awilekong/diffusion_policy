# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research implementation of **Diffusion Policy**, a framework for robot learning through behavior cloning using diffusion models. The codebase supports training and evaluating diffusion-based policies on both simulated and real robot tasks with low-dimensional state or visual observations.

**Key Paper**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/)

## Common Commands

### Training

```bash
# Basic training with Hydra config
python train.py --config-name=train_diffusion_unet_image_workspace

# Training with overrides
python train.py \
    --config-name=train_diffusion_unet_franka_image_workspace \
    task.dataset_path=/path/to/dataset.zarr \
    training.device=cuda:0 \
    training.seed=42 \
    exp_name=my_experiment

# Resume training from checkpoint
python train.py \
    training.resume=True \
    hydra.run.dir=/path/to/output/dir \
    training.device=cuda:0
```

### Evaluation

```bash
# Evaluate a checkpoint
python eval.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --output_dir /path/to/output \
    --device cuda:0
```

### Multi-seed Training

```bash
# Start ray cluster
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3

# Run multi-seed training
python ray_train_multirun.py \
    --config-dir=. \
    --config-name=train_diffusion_unet_image_workspace \
    --seeds=42,43,44 \
    --monitor_key=test/mean_score
```

### Remote Inference (Custom Addition)

```bash
# Serve policy over WebSocket (single frame mode, n_obs_steps=1)
python serve_diffusion_policy_single_frame.py \
    -i /path/to/checkpoint.ckpt \
    -p 8000 \
    -d cuda
```

## High-Level Architecture

### Design Philosophy: O(N+M) vs O(N*M)

The codebase is designed so implementing N tasks and M methods requires O(N+M) code instead of O(N*M). This is achieved through:

1. **Unified interface** between tasks and methods
2. **Independent implementations** of tasks and methods
3. **Trade-off**: Code repetition within tasks/methods for independence and linear readability

### Core Abstractions

#### Task Side
- **Dataset**: Adapts raw data to the unified interface, returns samples with proper temporal structure
- **EnvRunner**: Executes a Policy on an environment, returns logs and metrics
- **Env** (optional): gym-compatible environment class
- **Config**: `config/task/<task_name>.yaml` specifies Dataset and EnvRunner

#### Policy Side
- **Policy**: Implements `predict_action()` interface and training logic
- **Workspace**: Manages full training/evaluation lifecycle
- **Config**: `config/<workspace_name>.yaml` specifies Policy and Workspace

### The Interface Contract

#### Low-Dimensional Observations

**LowdimPolicy** interface:
- Input: `obs_dict` with `"obs"`: Tensor `(B, To, Do)`
- Output: `action_dict` with `"action"`: Tensor `(B, Ta, Da)`

**LowdimDataset** interface:
- Returns: `{"obs": (To, Do), "action": (Ta, Da)}`
- Provides: `get_normalizer()` returning `LinearNormalizer` with keys `["obs", "action"]`

#### Image Observations

**ImagePolicy** interface:
- Input: `obs_dict` with image keys: Tensor `(B, To, H, W, 3)` in [0,1] float32
- Output: `action_dict` with `"action"`: Tensor `(B, Ta, Da)`

**ImageDataset** interface:
- Returns: `{"obs": {"camera_0": (To, H, W, 3), ...}, "action": (Ta, Da)}`
- Provides: `get_normalizer()` with keys for each observation and action

### Key Temporal Parameters

```
To = n_obs_steps      # Observation horizon (how many past observations)
Ta = n_action_steps   # Action horizon (how many future actions to execute)
T = horizon           # Prediction horizon (how many future actions to predict)
```

**Example**: `To=3, Ta=4, T=6`
```
|o|o|o|              # 3 observations
| | |a|a|a|a|        # predict 6 actions, execute first 4
|o|o|                # slide window forward
| |a|a|a|a|a|        # execute next 4 actions
| | | | |a|a|        # etc.
```

**Important**: The classical single-step MDP formulation is a special case with `To=1, Ta=1`.

### Critical Components

#### Workspace
- Inherits from `BaseWorkspace`
- Encapsulates all experiment state and training/eval pipeline
- `run()` method contains full experiment logic
- Checkpoints save all object attributes with `state_dict()` method
- Constructed entirely from a single Hydra config

#### Dataset
- Inherits from `torch.utils.data.Dataset`
- Uses `ReplayBuffer` + `SequenceSampler` for efficient episode sampling
- Handles temporal padding at episode boundaries (critical for performance)
- `get_normalizer()` returns normalization parameters (common source of bugs!)

#### Policy
- Inherits from `BaseLowdimPolicy` or `BaseImagePolicy`
- `predict_action(obs_dict)` → action conforming to interface
- `set_normalizer(normalizer)` handles normalization internally
- `compute_loss(batch)` for training (optional)
- Often paired 1:1 with a corresponding Workspace class

#### EnvRunner
- `run(policy)` → dict of logs/metrics (wandb-compatible)
- Uses vectorized environments via modified `gym.vector.AsyncVectorEnv`
- Each env runs in separate process (workaround for Python GIL)
- **Warning**: Environments with OpenGL contexts need special handling due to fork() issues

#### ReplayBuffer
- Key data structure for dataset storage (in-memory and on-disk)
- Uses `zarr` format with chunking and compression
- Can use Jpeg2000 compression for image datasets to fit in RAM
- Structure:
  ```
  dataset.zarr/
    data/
      action: (N, action_dim)
      camera_0: (N, H, W, 3)
      state: (N, state_dim)
    meta/
      episode_ends: (num_episodes,)  # cumulative indices
  ```

## Hydra Configuration System

The codebase uses [Hydra](https://hydra.cc/) for configuration management:

- **Entry point**: `train.py` uses `@hydra.main` decorator
- **Config location**: `diffusion_policy/config/`
- **Override syntax**: `python train.py task=pusht_image training.device=cuda:1`
- **Special resolver**: `${eval:'expression'}` allows arbitrary Python code execution
- **Composition**: Configs use `defaults` to compose from task and method configs

### Important Config Locations

- **Task configs**: `diffusion_policy/config/task/*.yaml`
- **Workspace configs**: `diffusion_policy/config/train_*.yaml`
- **Runtime overrides**: Pass as command-line arguments

### Output Structure

Training outputs go to: `data/outputs/YYYY.MM.DD/HH.MM.SS_<method>_<task>/`
```
output_dir/
  checkpoints/
    latest.ckpt
    epoch=0100-train_loss=0.123.ckpt
  .hydra/
    config.yaml        # Full resolved config
    overrides.yaml     # Command-line overrides
  logs.json.txt        # Training metrics
  train.log            # Console output
```

## Dataset Format

Datasets use the zarr format (directory or .zip file):

```python
# Structure
data/
  <field_name>: (total_steps, *shape)  # All episodes concatenated
meta/
  episode_ends: (num_episodes,)         # End indices for each episode
```

### Creating Datasets

Use `ReplayBuffer` with `SequenceSampler`:
```python
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler

replay_buffer = ReplayBuffer.create_from_path(
    'data/my_dataset.zarr', mode='r'
)
sampler = SequenceSampler(
    replay_buffer=replay_buffer,
    sequence_length=horizon,
    pad_before=n_obs_steps-1,
    pad_after=n_action_steps-1
)
```

## Real Robot Integration

The codebase includes infrastructure for real robot deployment:

### Key Components

- **SharedMemoryRingBuffer**: Lock-free FILO queue for multi-camera image capture
- **RealEnv**: Async environment interface (`get_obs()` + `exec_actions()`)
- **RTDEInterpolationController**: UR5 robot control via RTDE interface
- **real_inference_util**: Official utilities for preprocessing real robot observations

### Real Robot Scripts

- `demo_real_robot.py`: Collect demonstrations with SpaceMouse teleoperation
- `eval_real_robot.py`: Deploy trained policy on real robot
- `serve_diffusion_policy_single_frame.py`: Remote inference server (custom addition)

## Normalization

**Critical**: Normalization bugs are common. The `LinearNormalizer`:
- Is created by Dataset's `get_normalizer()` method
- Has separate `scale` and `bias` for each observation/action key
- Is stored inside Policy checkpoint
- Handles normalization/denormalization on GPU inside Policy

**Debug tip**: Print `normalizer.params_dict` to inspect scale/bias vectors.

## Adding New Tasks

1. Create dataset class: `diffusion_policy/dataset/<task>_dataset.py`
   - Inherit from `BaseDataset`
   - Implement `__getitem__` to return properly shaped dict
   - Implement `get_normalizer()`

2. Create env runner: `diffusion_policy/env_runner/<task>_runner.py`
   - Implement `run(policy)` method
   - Return wandb-compatible metrics dict

3. Create task config: `diffusion_policy/config/task/<task>.yaml`
   - Set `dataset._target_` and `env_runner._target_`
   - Define `shape_meta` with observation/action shapes

## Adding New Methods

1. Create policy: `diffusion_policy/policy/<method>_policy.py`
   - Inherit from `BaseImagePolicy` or `BaseLowdimPolicy`
   - Implement `predict_action(obs_dict)`
   - Implement `compute_loss(batch)` for training

2. Create workspace: `diffusion_policy/workspace/train_<method>_workspace.py`
   - Inherit from `BaseWorkspace`
   - Implement `run()` with training/eval loop

3. Create config: `diffusion_policy/config/train_<method>_workspace.yaml`
   - Set `_target_` to workspace class
   - Configure policy, optimizer, dataloader

## Diffusion Model Specifics

When working with the diffusion models (`DiffusionUnetImagePolicy`):

### Key Parameters

- `num_inference_steps`: DDIM denoising steps
  - Training uses 100 steps (full DDPM)
  - Inference uses 16 steps (6x speedup with DDIM)

- `noise_scheduler`: Controls diffusion process
  - Best: `beta_schedule: squaredcos_cap_v2`
  - Use `DDIMScheduler` for fast inference

- `n_action_steps`: Actions to execute from prediction
  - Formula: `max_steps = horizon - n_obs_steps + 1`
  - Example: `horizon=16, n_obs_steps=1` → can use up to 16 actions

### Single-Frame Mode (n_obs_steps=1)

For models trained with `n_obs_steps=1`:
- No history queue needed
- Direct single-frame processing
- Use `serve_diffusion_policy_single_frame.py` for remote inference
- Simpler and faster than multi-frame models

## Custom Extensions in This Repository

### Remote Inference Server

This repository includes WebSocket-based remote inference:

**Files**:
- `serve_diffusion_policy_single_frame.py`: Server for n_obs_steps=1 models
- `README_REMOTE_INFERENCE.md`: Documentation for remote inference

**Usage**: See README_REMOTE_INFERENCE.md for client integration details.

### Franka Peg-in-Hole Task

Custom task configuration for Franka robot peg-in-hole:
- Config: `diffusion_policy/config/task/franka_peg_in_hole_image.yaml`
- Training script: `train_franka.sh` (multi-experiment launcher)

## Troubleshooting

### Common Issues

1. **Normalization bugs**: Print `normalizer.params_dict` to verify scale/bias
2. **Episode boundary padding**: Check `SequenceSampler` pad_before/pad_after
3. **OpenGL segfaults**: Provide `dummy_env_fn` to `AsyncVectorEnv` for envs with rendering
4. **Checkpoint device mismatch**: Use `map_location` when loading across devices

### Debugging Tools

- Set `training.debug=True` for verbose output
- Use `--debug` flag on inference servers to save intermediate data
- Check wandb logs for training curves
- Inspect `.hydra/config.yaml` in output dir for full resolved config

## Environment Setup

### Simulation
```bash
# Ubuntu 20.04 prerequisites
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# Create conda environment
mamba env create -f conda_environment.yaml
conda activate robodiff
```

### Real Robot (additional)
```bash
# RealSense SDK (see official Intel docs)
# SpaceMouse support
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd

# Real robot environment
mamba env create -f conda_environment_real.yaml
```

## WandB Integration

The codebase uses Weights & Biases for experiment tracking:

```bash
# First time setup
wandb login

# Metrics logged
# - train_loss: Training loss
# - test/mean_score: Evaluation success rate
# - Rollout videos every N epochs
```

## Checkpointing

- **Automatic**: Checkpoints saved every `checkpoint_every` epochs
- **TopK**: Keeps best K checkpoints by `monitor_key` metric
- **Latest**: Always maintains `latest.ckpt`
- **Resume**: Use `training.resume=True` with `hydra.run.dir=<output_dir>`

## Performance Tips

1. **Batch size**: Adjust based on GPU memory (typically 64-256)
2. **Workers**: Set `dataloader.num_workers` to CPU cores (typically 8-16)
3. **Image compression**: Use Jpeg2000 in ReplayBuffer for large image datasets
4. **Vectorized envs**: Use multiple parallel envs for faster evaluation
5. **EMA**: Use `training.use_ema=True` with GroupNorm (not BatchNorm)

## Code Style Notes

- Prefer code duplication over premature abstraction for task/method implementations
- Keep task and method code independent and linearly readable
- Use Hydra for all configuration (avoid hardcoded parameters)
- Normalize inside Policy on GPU (not in Dataset)

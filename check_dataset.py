import zarr
import numpy as np
import os

dataset_path = '/home/zpw/ws_zpw/zpw/data/zarr_dataset/peg_in_hole_zarr'

# 尝试打开 replay_buffer.zarr
zarr_path = os.path.join(dataset_path, 'replay_buffer.zarr')
print(f"尝试打开: {zarr_path}")
print(f"路径存在: {os.path.exists(zarr_path)}")

replay_buffer = zarr.open(zarr_path, mode='r')

print("=" * 60)
print("数据集结构检查")
print("=" * 60)

# 检查所有keys
print("\n可用的数据字段:")
for key in replay_buffer['data'].keys():
    shape = replay_buffer['data'][key].shape
    dtype = replay_buffer['data'][key].dtype
    print(f"  {key:20s}: shape={shape}, dtype={dtype}")

# 获取episode信息
episode_ends = replay_buffer['meta/episode_ends'][:]
num_episodes = len(episode_ends)
print(f"\nEpisode信息:")
print(f"  总episodes数: {num_episodes}")
print(f"  总timesteps数: {episode_ends[-1]}")

# 检查robot_eef_pose维度
if 'robot_eef_pose' in replay_buffer['data']:
    robot_pose = replay_buffer['data']['robot_eef_pose']
    print(f"\n✓ robot_eef_pose存在")
    print(f"  形状: {robot_pose.shape}")
    print(f"  维度: {robot_pose.shape[1]} 维")
    if robot_pose.shape[1] == 7:
        print(f"  ✅ 确认是7维")
    else:
        print(f"  ❌ 不是7维！")
else:
    print(f"\n❌ robot_eef_pose字段不存在！")

# 检查action维度
if 'action' in replay_buffer['data']:
    action = replay_buffer['data']['action']
    print(f"\n✓ action存在")
    print(f"  形状: {action.shape}")
    print(f"  维度: {action.shape[1]} 维")
    if action.shape[1] == 7:
        print(f"  ✅ 确认是7维")
    else:
        print(f"  ❌ 不是7维！")
else:
    print(f"\n❌ action字段不存在！")

print("\n" + "=" * 60)
print("检查 action[t] 是否等于 robot_eef_pose[t+1]")
print("=" * 60)

if 'action' in replay_buffer['data'] and 'robot_eef_pose' in replay_buffer['data']:
    robot_pose = replay_buffer['data']['robot_eef_pose'][:]
    action = replay_buffer['data']['action'][:]

    # 检查前3个episode
    num_check_episodes = min(3, num_episodes)
    print(f"\n检查前 {num_check_episodes} 个episodes:")

    all_match = True
    for ep_idx in range(num_check_episodes):
        start = 0 if ep_idx == 0 else episode_ends[ep_idx-1]
        end = episode_ends[ep_idx]

        print(f"\n  Episode {ep_idx} (timesteps {start}~{end-1}):")

        # 检查 action[t] vs robot_eef_pose[t+1]
        # 最后一帧没有下一帧，所以检查到end-1
        for t in range(start, min(start+3, end-1)):  # 检查前3帧
            action_t = action[t]
            pose_t_plus_1 = robot_pose[t+1]
            diff = np.abs(action_t - pose_t_plus_1)
            max_diff = np.max(diff)

            match = max_diff < 1e-6
            status = "✅" if match else "❌"
            print(f"    t={t}: action[{t}] vs pose[{t+1}], max_diff={max_diff:.10f} {status}")

            if not match:
                all_match = False
                print(f"      action[{t}]  = {action_t}")
                print(f"      pose[{t+1}]  = {pose_t_plus_1}")
                print(f"      difference   = {diff}")

    # 全局检查
    print(f"\n全局统计检查:")
    total_checked = 0
    total_matched = 0

    for ep_idx in range(num_episodes):
        start = 0 if ep_idx == 0 else episode_ends[ep_idx-1]
        end = episode_ends[ep_idx]

        for t in range(start, end-1):
            action_t = action[t]
            pose_t_plus_1 = robot_pose[t+1]
            diff = np.abs(action_t - pose_t_plus_1)
            max_diff = np.max(diff)

            total_checked += 1
            if max_diff < 1e-6:
                total_matched += 1

    match_rate = 100.0 * total_matched / total_checked if total_checked > 0 else 0
    print(f"  总检查帧数: {total_checked}")
    print(f"  匹配帧数: {total_matched}")
    print(f"  匹配率: {match_rate:.2f}%")

    if match_rate > 99.9:
        print(f"  ✅ action[t] == robot_eef_pose[t+1] (绝对位姿)")
    else:
        print(f"  ❌ action[t] != robot_eef_pose[t+1]")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)

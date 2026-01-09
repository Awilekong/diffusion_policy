"""
Delta Action 功能单元测试

测试内容：
1. TrainWandbDebugger.reconstruct_trajectory_from_delta 方法
2. 数据集 delta_action 处理逻辑
3. 轨迹重建的正确性
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestReconstructTrajectoryFromDelta:
    """测试 reconstruct_trajectory_from_delta 方法"""

    def test_basic_reconstruction(self):
        """基本轨迹重建测试"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        # 创建测试数据
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        # Delta actions: 每步移动 (0.1, 0.1, 0.1, 0, 0, 0, 0)
        T = 5
        delta_actions = np.zeros((T, 7))
        delta_actions[:, 0] = 0.1
        delta_actions[:, 1] = 0.1
        delta_actions[:, 2] = 0.1

        reconstructed = TrainWandbDebugger.reconstruct_trajectory_from_delta(
            delta_actions, initial_state
        )

        assert reconstructed.shape == (T, 7), f"形状错误: {reconstructed.shape}"

        expected = np.zeros((T, 7))
        for t in range(T):
            expected[t, 0] = 0.1 * (t + 1)
            expected[t, 1] = 0.1 * (t + 1)
            expected[t, 2] = 0.1 * (t + 1)
            expected[t, 6] = 1.0

        np.testing.assert_array_almost_equal(reconstructed, expected, decimal=6)
        print("✅ test_basic_reconstruction 通过")

    def test_7d_full_action(self):
        """测试完整 7 维动作"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        initial_state = np.array([0.5, 0.0, 0.3, 0.1, 0.0, 0.0, 0.5])
        T = 3
        delta_actions = np.array([
            [0.01, 0.02, -0.01, 0.001, 0.0, 0.0, 0.1],
            [0.02, -0.01, 0.01, 0.0, 0.001, 0.0, -0.05],
            [-0.01, 0.01, 0.0, 0.0, 0.0, 0.001, 0.0],
        ])

        reconstructed = TrainWandbDebugger.reconstruct_trajectory_from_delta(
            delta_actions, initial_state
        )

        expected = np.zeros((T, 7))
        current = initial_state.copy()
        for t in range(T):
            current = current + delta_actions[t]
            expected[t] = current

        np.testing.assert_array_almost_equal(reconstructed, expected, decimal=10)
        print("✅ test_7d_full_action 通过")

    def test_zero_delta(self):
        """测试零 delta"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        initial_state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.8])
        T = 4
        delta_actions = np.zeros((T, 7))

        reconstructed = TrainWandbDebugger.reconstruct_trajectory_from_delta(
            delta_actions, initial_state
        )

        for t in range(T):
            np.testing.assert_array_almost_equal(reconstructed[t], initial_state, decimal=10)
        print("✅ test_zero_delta 通过")

    def test_round_trip(self):
        """测试往返重建"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        T = 10
        absolute_trajectory = np.random.randn(T, 7) * 0.1
        absolute_trajectory = np.cumsum(absolute_trajectory, axis=0)

        delta_actions = np.zeros_like(absolute_trajectory)
        delta_actions[0] = absolute_trajectory[0]
        delta_actions[1:] = np.diff(absolute_trajectory, axis=0)

        initial_state = np.zeros(7)
        reconstructed = TrainWandbDebugger.reconstruct_trajectory_from_delta(
            delta_actions, initial_state
        )

        np.testing.assert_array_almost_equal(reconstructed, absolute_trajectory, decimal=10)
        print("✅ test_round_trip 通过")


class TestDeltaActionDataset:
    """测试数据集的 delta_action 处理"""

    def test_delta_conversion_logic(self):
        """测试 delta 转换逻辑"""
        actions = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.6],
            [0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.6],
        ])

        episode_ends = np.array([5])

        actions_diff = np.zeros_like(actions)
        for i in range(len(episode_ends)):
            start = 0 if i == 0 else episode_ends[i-1]
            end = episode_ends[i]
            actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)

        np.testing.assert_array_almost_equal(actions_diff[0], np.zeros(7), decimal=10)

        expected_deltas = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
        ])

        np.testing.assert_array_almost_equal(actions_diff, expected_deltas, decimal=10)
        print("✅ test_delta_conversion_logic 通过")

    def test_multi_episode_delta(self):
        """测试多 episode 的 delta 转换"""
        actions = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.8],
            [1.1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.8],
            [1.2, 1.1, 1.0, 0.0, 0.0, 0.0, 0.8],
        ])

        episode_ends = np.array([3, 6])

        actions_diff = np.zeros_like(actions)
        for i in range(len(episode_ends)):
            start = 0 if i == 0 else episode_ends[i-1]
            end = episode_ends[i]
            actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)

        assert np.allclose(actions_diff[0], 0), "Episode 1 第一帧 delta 应为零"
        assert np.allclose(actions_diff[3], 0), "Episode 2 第一帧 delta 应为零"

        wrong_delta = actions[3] - actions[2]
        assert not np.allclose(actions_diff[3], wrong_delta), "Episode 边界处理错误"

        print("✅ test_multi_episode_delta 通过")


class TestTrainWandbDebuggerIntegration:
    """集成测试"""

    def test_debugger_init_with_delta_action(self):
        """测试 debugger 初始化"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        debugger1 = TrainWandbDebugger(wandb_run=None, enabled=False)
        assert debugger1.delta_action == False

        debugger2 = TrainWandbDebugger(wandb_run=None, enabled=False, delta_action=True)
        assert debugger2.delta_action == True

        print("✅ test_debugger_init_with_delta_action 通过")

    def test_trajectory_reconstruction_in_context(self):
        """测试实际场景中的轨迹重建"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        debugger = TrainWandbDebugger(wandb_run=None, enabled=False, delta_action=True)

        initial_state = np.array([0.4, 0.1, 0.2, 0.1, 0.0, 0.0, 0.5])
        T = 8
        pred_delta = np.random.randn(T, 7) * 0.01
        gt_delta = np.random.randn(T, 7) * 0.01

        pred_absolute = debugger.reconstruct_trajectory_from_delta(pred_delta, initial_state)
        gt_absolute = debugger.reconstruct_trajectory_from_delta(gt_delta, initial_state)

        assert pred_absolute.shape == (T, 7)
        assert gt_absolute.shape == (T, 7)

        np.testing.assert_array_almost_equal(
            pred_absolute[0], initial_state + pred_delta[0], decimal=10
        )

        print("✅ test_trajectory_reconstruction_in_context 通过")


class TestEdgeCases:
    """边界情况测试"""

    def test_single_step_trajectory(self):
        """测试单步轨迹"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        delta_actions = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.1]])

        reconstructed = TrainWandbDebugger.reconstruct_trajectory_from_delta(
            delta_actions, initial_state
        )

        expected = initial_state + delta_actions[0]
        np.testing.assert_array_almost_equal(reconstructed[0], expected, decimal=10)
        print("✅ test_single_step_trajectory 通过")

    def test_large_trajectory(self):
        """测试长轨迹"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        initial_state = np.zeros(7)
        T = 1000
        delta_actions = np.ones((T, 7)) * 0.001

        reconstructed = TrainWandbDebugger.reconstruct_trajectory_from_delta(
            delta_actions, initial_state
        )

        expected_final = np.ones(7) * T * 0.001
        np.testing.assert_array_almost_equal(reconstructed[-1], expected_final, decimal=6)
        print("✅ test_large_trajectory 通过")

    def test_negative_deltas(self):
        """测试负 delta"""
        from diffusion_policy.common.train_wandb_debugger import TrainWandbDebugger

        initial_state = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        T = 5
        delta_actions = np.ones((T, 7)) * -0.1

        reconstructed = TrainWandbDebugger.reconstruct_trajectory_from_delta(
            delta_actions, initial_state
        )

        expected_final = initial_state + T * (-0.1)
        np.testing.assert_array_almost_equal(reconstructed[-1], expected_final, decimal=10)
        print("✅ test_negative_deltas 通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Delta Action 功能单元测试")
    print("=" * 60)

    print("\n--- TestReconstructTrajectoryFromDelta ---")
    test1 = TestReconstructTrajectoryFromDelta()
    test1.test_basic_reconstruction()
    test1.test_7d_full_action()
    test1.test_zero_delta()
    test1.test_round_trip()

    print("\n--- TestDeltaActionDataset ---")
    test2 = TestDeltaActionDataset()
    test2.test_delta_conversion_logic()
    test2.test_multi_episode_delta()

    print("\n--- TestTrainWandbDebuggerIntegration ---")
    test3 = TestTrainWandbDebuggerIntegration()
    test3.test_debugger_init_with_delta_action()
    test3.test_trajectory_reconstruction_in_context()

    print("\n--- TestEdgeCases ---")
    test4 = TestEdgeCases()
    test4.test_single_step_trajectory()
    test4.test_large_trajectory()
    test4.test_negative_deltas()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

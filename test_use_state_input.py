#!/usr/bin/env python3
"""
Test script to verify use_state_input parameter works correctly
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet

def test_use_state_input():
    """Test that use_state_input parameter correctly controls state inclusion"""

    # Define shape_meta similar to Franka task
    shape_meta = {
        'obs': {
            'camera_0': {
                'shape': [3, 240, 320],
                'type': 'rgb'
            },
            'robot_eef_pose': {
                'shape': [7],
                'type': 'low_dim'
            }
        }
    }

    # Create RGB model
    rgb_model = get_resnet('resnet18', weights=None)

    print("=" * 60)
    print("Testing MultiImageObsEncoder with use_state_input parameter")
    print("=" * 60)

    # Test 1: use_state_input=True (default, includes state)
    print("\n[Test 1] use_state_input=True (default behavior)")
    encoder_with_state = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
        use_group_norm=False,
        share_rgb_model=False,
        imagenet_norm=False,
        use_state_input=True  # Include state
    )

    # Create dummy input
    batch_size = 2
    obs_dict_with_state = {
        'camera_0': torch.randn(batch_size, 3, 240, 320),
        'robot_eef_pose': torch.randn(batch_size, 7)
    }

    output_with_state = encoder_with_state(obs_dict_with_state)
    print(f"  Input: camera_0 shape = {obs_dict_with_state['camera_0'].shape}")
    print(f"  Input: robot_eef_pose shape = {obs_dict_with_state['robot_eef_pose'].shape}")
    print(f"  Output shape: {output_with_state.shape}")
    print(f"  Expected: [batch_size, 512 (ResNet18) + 7 (state)] = [{batch_size}, 519]")
    assert output_with_state.shape == (batch_size, 519), f"Expected shape ({batch_size}, 519), got {output_with_state.shape}"
    print("  ✅ PASSED: Output includes state features")

    # Test 2: use_state_input=False (excludes state)
    print("\n[Test 2] use_state_input=False (image-only mode)")
    encoder_without_state = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet('resnet18', weights=None),  # New model instance
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
        use_group_norm=False,
        share_rgb_model=False,
        imagenet_norm=False,
        use_state_input=False  # Exclude state
    )

    # Same input (state will be ignored)
    obs_dict_without_state = {
        'camera_0': torch.randn(batch_size, 3, 240, 320),
        'robot_eef_pose': torch.randn(batch_size, 7)  # Present but will be ignored
    }

    output_without_state = encoder_without_state(obs_dict_without_state)
    print(f"  Input: camera_0 shape = {obs_dict_without_state['camera_0'].shape}")
    print(f"  Input: robot_eef_pose shape = {obs_dict_without_state['robot_eef_pose'].shape} (ignored)")
    print(f"  Output shape: {output_without_state.shape}")
    print(f"  Expected: [batch_size, 512 (ResNet18 only)] = [{batch_size}, 512]")
    assert output_without_state.shape == (batch_size, 512), f"Expected shape ({batch_size}, 512), got {output_without_state.shape}"
    print("  ✅ PASSED: Output excludes state features")

    # Test 3: Verify dimension difference
    print("\n[Test 3] Verify dimension difference")
    dim_with_state = output_with_state.shape[1]
    dim_without_state = output_without_state.shape[1]
    state_dim = 7
    print(f"  Dimension with state: {dim_with_state}")
    print(f"  Dimension without state: {dim_without_state}")
    print(f"  Difference: {dim_with_state - dim_without_state} (should equal state_dim = {state_dim})")
    assert dim_with_state - dim_without_state == state_dim, "Dimension difference should equal state dimension"
    print("  ✅ PASSED: Dimension difference matches state dimension")

    # Test 4: Multi-camera case (3 cameras like Franka config)
    print("\n[Test 4] Multi-camera case (3 cameras)")
    shape_meta_multi_cam = {
        'obs': {
            'camera_0': {'shape': [3, 240, 320], 'type': 'rgb'},
            'camera_1': {'shape': [3, 240, 320], 'type': 'rgb'},
            'camera_2': {'shape': [3, 240, 320], 'type': 'rgb'},
            'robot_eef_pose': {'shape': [7], 'type': 'low_dim'}
        }
    }

    encoder_multi_with_state = MultiImageObsEncoder(
        shape_meta=shape_meta_multi_cam,
        rgb_model=get_resnet('resnet18', weights=None),
        use_state_input=True
    )

    encoder_multi_without_state = MultiImageObsEncoder(
        shape_meta=shape_meta_multi_cam,
        rgb_model=get_resnet('resnet18', weights=None),
        use_state_input=False
    )

    obs_dict_multi = {
        'camera_0': torch.randn(batch_size, 3, 240, 320),
        'camera_1': torch.randn(batch_size, 3, 240, 320),
        'camera_2': torch.randn(batch_size, 3, 240, 320),
        'robot_eef_pose': torch.randn(batch_size, 7)
    }

    output_multi_with = encoder_multi_with_state(obs_dict_multi)
    output_multi_without = encoder_multi_without_state(obs_dict_multi)

    print(f"  With state: {output_multi_with.shape} (expected: [{batch_size}, {3*512 + 7}])")
    print(f"  Without state: {output_multi_without.shape} (expected: [{batch_size}, {3*512}])")

    assert output_multi_with.shape == (batch_size, 3*512 + 7), f"Expected ({batch_size}, {3*512 + 7})"
    assert output_multi_without.shape == (batch_size, 3*512), f"Expected ({batch_size}, {3*512})"
    print("  ✅ PASSED: Multi-camera mode works correctly")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print("  • use_state_input=True: Encoder includes state features (backward compatible)")
    print("  • use_state_input=False: Encoder uses only image features")
    print("  • Feature dimension changes as expected")
    print("\nUsage example:")
    print("  python train.py \\")
    print("      --config-name=train_diffusion_unet_franka_image_workspace \\")
    print("      policy.obs_encoder.use_state_input=False")


if __name__ == '__main__':
    test_use_state_input()

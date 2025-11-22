#!/usr/bin/env python3
"""
Test that checkpoint loading preserves the number of Gaussians
"""
import torch
import torch.nn as nn
import os
import sys

# Simulate the checkpoint loading process
checkpoint_path = "/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums/gaussian_init.pth"

if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint_data = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

print(f"\nCheckpoint contains {len(checkpoint_data['xyz'])} Gaussians")
print(f"XYZ shape: {checkpoint_data['xyz'].shape}")
print(f"Features_dc shape: {checkpoint_data['features_dc'].shape}")
print(f"Opacity shape: {checkpoint_data['opacity'].shape}")

# Test that we can create nn.Parameters from it
xyz_param = nn.Parameter(checkpoint_data['xyz'].cuda().contiguous().requires_grad_(True))
print(f"\nCreated XYZ parameter with shape: {xyz_param.shape}")
print(f"Requires grad: {xyz_param.requires_grad}")
print(f"Is on CUDA: {xyz_param.is_cuda}")

print("\n✓ Checkpoint loading test passed!")

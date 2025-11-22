#!/usr/bin/env python3
"""
Export Gaussian checkpoint to PLY file for visualization
"""
import torch
import numpy as np
import os
from plyfile import PlyData, PlyElement

def export_gaussians_to_ply(checkpoint_path, output_path):
    """
    Load gaussian_init.pth and export the point cloud to PLY format

    Args:
        checkpoint_path: Path to gaussian_init.pth
        output_path: Path where to save the PLY file
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    if 'xyz' not in checkpoint_data:
        print("Error: Checkpoint doesn't contain 'xyz' key")
        return

    # Extract data
    xyz = checkpoint_data['xyz'].numpy()  # [N, 3]
    features_dc = checkpoint_data['features_dc'].numpy()  # [N, 1, 3] (RGB in first SH coefficient)
    opacity = checkpoint_data['opacity'].numpy()  # [N, 1]
    scaling = checkpoint_data['scaling'].numpy()  # [N, 3]
    rotation = checkpoint_data['rotation'].numpy()  # [N, 4]

    num_points = len(xyz)
    print(f"Number of Gaussians: {num_points}")
    print(f"XYZ range: [{xyz.min(axis=0)}, {xyz.max(axis=0)}]")

    # Convert features_dc to RGB (first SH coefficient represents base color)
    # SH coefficient 0 is related to RGB by: RGB = 0.28209479177387814 * SH_0 + 0.5
    SH_C0 = 0.28209479177387814
    rgb = features_dc.squeeze(1) * SH_C0 + 0.5  # [N, 3]
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    print(f"RGB range: [{rgb.min()}, {rgb.max()}]")

    # Create structured array for PLY
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # normals (set to 0)
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]

    # Create vertex array
    vertices = np.empty(num_points, dtype=dtype)
    vertices['x'] = xyz[:, 0]
    vertices['y'] = xyz[:, 1]
    vertices['z'] = xyz[:, 2]
    vertices['nx'] = 0
    vertices['ny'] = 0
    vertices['nz'] = 0
    vertices['red'] = rgb[:, 0]
    vertices['green'] = rgb[:, 1]
    vertices['blue'] = rgb[:, 2]

    # Create PLY element
    vertex_element = PlyElement.describe(vertices, 'vertex')

    # Write to file
    print(f"Saving to: {output_path}")
    PlyData([vertex_element]).write(output_path)
    print(f"✓ Successfully exported {num_points} points to PLY")

    # Print checkpoint info
    print(f"\nCheckpoint info:")
    print(f"  Iteration: {checkpoint_data.get('iteration', 'unknown')}")
    print(f"  Active SH degree: {checkpoint_data.get('active_sh_degree', 'unknown')}")
    print(f"  Spatial LR scale: {checkpoint_data.get('spatial_lr_scale', 'unknown')}")

if __name__ == "__main__":
    import sys

    # Default paths
    checkpoint_path = "/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums/gaussian_init.pth"
    output_path = "/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums/gaussian_init.ply"

    # Allow command line override
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    export_gaussians_to_ply(checkpoint_path, output_path)

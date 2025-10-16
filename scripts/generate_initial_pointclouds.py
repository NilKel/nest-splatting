#!/usr/bin/env python3
"""
Generate initial point clouds for NeRF Synthetic scenes.
Creates a random point cloud within the scene bounds.
"""
import numpy as np
import os
from plyfile import PlyData, PlyElement

def create_random_point_cloud(num_points=100_000, bounds=1.5):
    """Create a random point cloud within given bounds."""
    # Random points in a cube
    xyz = np.random.uniform(-bounds, bounds, (num_points, 3)).astype(np.float32)
    
    # Random colors
    rgb = (np.random.rand(num_points, 3) * 255).astype(np.uint8)
    
    # Zero normals (not really needed for initial points)
    normals = np.zeros((num_points, 3), dtype=np.float32)
    
    return xyz, rgb, normals

def save_ply(path, xyz, rgb, normals):
    """Save point cloud to PLY file."""
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    print(f"✓ Created {path} with {len(xyz)} points")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate initial point clouds for NeRF Synthetic scenes")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to nerf_synthetic directory")
    parser.add_argument("--num_points", type=int, default=100_000,
                        help="Number of points to generate (default: 100,000)")
    parser.add_argument("--bounds", type=float, default=1.5,
                        help="Bounds for random point generation (default: 1.5)")
    args = parser.parse_args()
    
    scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
    
    print(f"Generating initial point clouds for {len(scenes)} scenes...")
    print(f"  Points per cloud: {args.num_points:,}")
    print(f"  Bounds: ±{args.bounds}")
    print()
    
    for scene in scenes:
        scene_dir = os.path.join(args.dataset_dir, scene)
        ply_path = os.path.join(scene_dir, "points3d.ply")
        
        if not os.path.exists(scene_dir):
            print(f"✗ Scene directory not found: {scene_dir}")
            continue
        
        if os.path.exists(ply_path):
            print(f"⊙ Skipping {scene} (points3d.ply already exists)")
            continue
        
        # Generate and save
        xyz, rgb, normals = create_random_point_cloud(args.num_points, args.bounds)
        save_ply(ply_path, xyz, rgb, normals)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()


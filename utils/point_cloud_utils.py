#
# Point cloud utilities for FPS subsampling and caching
#

import os
import numpy as np
from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import SH2RGB

# Standard cap_max values that get cached
STANDARD_CAP_VALUES = [40000, 100000, 400000, 1000000]


def farthest_point_subsample(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Iteratively select farthest points from point cloud using FPS algorithm.

    Args:
        points: (N, 3) numpy array of xyz coordinates
        num_samples: Target number of points to select

    Returns:
        indices: (num_samples,) array of selected point indices
    """
    N = points.shape[0]

    if num_samples >= N:
        return np.arange(N)

    selected = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(N, np.inf)

    # Start with random point
    selected[0] = np.random.randint(N)

    for i in range(1, num_samples):
        # Update distances to nearest selected point
        last_selected = points[selected[i-1]]
        dist_to_last = np.linalg.norm(points - last_selected, axis=1)
        distances = np.minimum(distances, dist_to_last)

        # Select farthest point
        selected[i] = np.argmax(distances)

        # Progress logging
        if (i + 1) % 10000 == 0:
            print(f"  FPS progress: {i+1}/{num_samples}")

    return selected


def get_cached_fps_path(data_dir: str, cap_max: int) -> str:
    """
    Get path for cached FPS point cloud.

    Args:
        data_dir: Path to dataset directory (contains sparse/0/)
        cap_max: Target number of points

    Returns:
        Path to cached PLY file: {data_dir}/sparse/0/points3D_fps_{cap_max}.ply
    """
    return os.path.join(data_dir, "sparse", "0", f"points3D_fps_{cap_max}.ply")


def _fetch_ply(path: str) -> BasicPointCloud:
    """
    Load point cloud from PLY file.
    Matches behavior of scene/dataset_readers.py fetchPly - regenerates random colors.
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # Generate random SH coefficients (matches fetchPly behavior)
    num_pts = positions.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    colors = SH2RGB(shs)
    normals = np.zeros((num_pts, 3))

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def _store_ply(path: str, xyz: np.ndarray, rgb: np.ndarray = None):
    """
    Store point cloud to PLY file.
    """
    if rgb is None:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)

    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def load_or_create_fps_pointcloud(data_dir: str, original_pcd: BasicPointCloud, cap_max: int) -> BasicPointCloud:
    """
    Load cached FPS point cloud if exists, otherwise create and cache it.

    Args:
        data_dir: Path to dataset directory (contains sparse/0/)
        original_pcd: Full point cloud from COLMAP
        cap_max: Target number of points

    Returns:
        Subsampled BasicPointCloud

    Notes:
        - Only caches for standard cap_max values: 40k, 100k, 400k, 1M
        - Other values compute FPS on-the-fly without caching
        - Cache stores xyz only; colors regenerated on load (matches fetchPly behavior)
    """
    cache_path = get_cached_fps_path(data_dir, cap_max)
    should_cache = cap_max in STANDARD_CAP_VALUES

    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"[FPS] Loading cached FPS point cloud: {cache_path}")
        return _fetch_ply(cache_path)

    # Compute FPS
    num_original = len(original_pcd.points)
    print(f"[FPS] Computing farthest point subsampling ({num_original} -> {cap_max})...")
    indices = farthest_point_subsample(original_pcd.points, cap_max)
    subsampled_points = original_pcd.points[indices]

    # Cache if standard value
    if should_cache:
        print(f"[FPS] Caching subsampled point cloud: {cache_path}")
        _store_ply(cache_path, subsampled_points)

    # Return as BasicPointCloud (colors will be regenerated like fetchPly does)
    num_pts = len(subsampled_points)
    shs = np.random.random((num_pts, 3)) / 255.0
    colors = SH2RGB(shs)

    return BasicPointCloud(
        points=subsampled_points,
        colors=colors,
        normals=np.zeros_like(subsampled_points)
    )

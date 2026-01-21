#
# Point cloud utilities for FPS subsampling and caching
#

import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import SH2RGB

# Standard cap_max values that get cached
STANDARD_CAP_VALUES = [40000, 100000, 400000, 1000000]

# Check if fpsample is available for fast FPS (bucket-based QuickFPS algorithm)
try:
    import fpsample
    FPSAMPLE_AVAILABLE = True
except ImportError:
    FPSAMPLE_AVAILABLE = False


def farthest_point_subsample_cuda(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    CUDA-accelerated farthest point sampling (manual implementation).
    Note: fpsample's QuickFPS is preferred and used by default in farthest_point_subsample().

    Args:
        points: (N, 3) torch tensor on GPU
        num_samples: Target number of points to select

    Returns:
        indices: (num_samples,) tensor of selected point indices
    """
    N = points.shape[0]
    device = points.device

    if num_samples >= N:
        return torch.arange(N, device=device)

    # Manual CUDA implementation (vectorized distance computation on GPU)
    selected = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)

    # Start with random point
    selected[0] = torch.randint(N, (1,), device=device)

    for i in range(1, num_samples):
        # Update distances to nearest selected point (vectorized on GPU)
        last_selected = points[selected[i-1]]  # (3,)
        dist_to_last = torch.norm(points - last_selected, dim=1)  # (N,)
        distances = torch.minimum(distances, dist_to_last)

        # Select farthest point
        selected[i] = torch.argmax(distances)

        # Progress logging
        if (i + 1) % 50000 == 0:
            print(f"  [FPS-CUDA] Progress: {i+1}/{num_samples}")

    return selected


def farthest_point_subsample(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest point sampling - uses fpsample's QuickFPS if available (very fast),
    falls back to CUDA or CPU implementation.

    Args:
        points: (N, 3) numpy array of xyz coordinates
        num_samples: Target number of points to select

    Returns:
        indices: (num_samples,) array of selected point indices
    """
    N = points.shape[0]

    if num_samples >= N:
        return np.arange(N)

    # Use fpsample's QuickFPS (bucket-based, very fast)
    if FPSAMPLE_AVAILABLE:
        # Determine h parameter based on workload
        # h=3 for small, h=5-7 for medium, h=9 for large
        if N > 500000:
            h = 9
        elif N > 100000:
            h = 7
        else:
            h = 5
        print(f"  [FPS] Using fpsample QuickFPS (h={h}) ({N:,} -> {num_samples:,} points)")
        indices = fpsample.bucket_fps_kdline_sampling(points.astype(np.float64), num_samples, h=h)
        return indices.astype(np.int64)

    # CUDA fallback
    if torch.cuda.is_available():
        print(f"  [FPS] Using manual CUDA ({N:,} -> {num_samples:,} points)")
        points_cuda = torch.from_numpy(points).float().cuda()
        indices_cuda = farthest_point_subsample_cuda(points_cuda, num_samples)
        return indices_cuda.cpu().numpy()

    # CPU fallback
    print(f"  [FPS] Using CPU (CUDA not available)")
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

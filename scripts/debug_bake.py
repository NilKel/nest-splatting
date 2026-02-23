#!/usr/bin/env python3
"""
Diagnostic script to verify bake MLP outputs match PyTorch MLP evaluation.

Picks a few Gaussians, evaluates the MLP at their UV grid points in BOTH:
  1. CUDA bake kernel (bakeSHKernel)
  2. Python (PyTorch MLP with Python hash encoding)

Then compares the 48D SH outputs to find discrepancies.
"""

import os
import sys
import math
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams
from diff_surfel_bake import set_mlp_weights, bake_gaussians


SH_C0 = 0.28209479177387814


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--uv_extent", type=float, default=4.0)
    parser.add_argument("--num_test", type=int, default=100, help="Number of Gaussians to test")
    args_debug = parser.parse_args()

    # Load training config
    args_pkl_path = os.path.join(args_debug.model_path, "args.pkl")
    with open(args_pkl_path, 'rb') as f:
        args = pickle.load(f)
    args.model_path = args_debug.model_path
    args.eval = True

    config_yaml_path = os.path.join(args_debug.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(args.yaml)

    # Auto-detect iteration
    iteration = args_debug.iteration
    if iteration == -1:
        import glob
        ngp_files = glob.glob(os.path.join(args_debug.model_path, "ngp_*.pth"))
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)

    # Load model
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)

    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.load_model(args_debug.model_path, iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    ingp.set_active_levels(iteration)

    N = len(gaussians.get_xyz)
    print(f"\n[DEBUG] {N:,} Gaussians, iteration {iteration}")

    # =========================================================================
    # Step 1: CUDA bake
    # =========================================================================
    mlp_weights = ingp.get_fused_mlp_weights()
    W1, W2, W3 = mlp_weights
    set_mlp_weights(W1, W2, W3, is_sh_mode=True)

    hash_encoding = ingp.hash_encoding
    embeddings, offsets, num_levels, per_level_scale, base_resolution, align_corners, interp_id = hash_encoding.get_params()
    hash_features = embeddings.half()
    voxel_min = ingp.voxel_range[0]
    voxel_max = ingp.voxel_range[1]
    l_scale = math.log2(per_level_scale)
    contract = ingp.contract
    active_hashgrid_levels = ingp.levels - ingp.hybrid_levels

    grid_size = args_debug.grid_size
    uv_extent = args_debug.uv_extent

    # Pick test Gaussians: a mix of small and large
    scales = gaussians.get_scaling  # [N, 2] activated
    max_scales = scales.max(dim=1).values
    sorted_idx = max_scales.argsort()
    n_test = min(args_debug.num_test, N)

    # Sample from different scale ranges
    test_indices = torch.cat([
        sorted_idx[:n_test//4],          # smallest
        sorted_idx[N//3:N//3+n_test//4], # medium-small
        sorted_idx[2*N//3:2*N//3+n_test//4], # medium-large
        sorted_idx[-n_test//4:],         # largest
    ])
    test_indices = test_indices[:n_test]

    print(f"[DEBUG] Testing {n_test} Gaussians (scale range: {max_scales[test_indices].min():.6f} to {max_scales[test_indices].max():.6f})")

    # Bake using CUDA
    with torch.no_grad():
        sh_grid_cuda = bake_gaussians(
            centers=gaussians.get_xyz[test_indices].contiguous(),
            quats=gaussians.get_rotation[test_indices].contiguous(),
            scales=gaussians.get_scaling[test_indices].contiguous(),
            gauss_features=gaussians.get_gaussian_features[test_indices].contiguous(),
            hash_features=hash_features.contiguous(),
            level_offsets=offsets.int().contiguous(),
            voxel_min=voxel_min, voxel_max=voxel_max,
            l_scale=l_scale, Base=base_resolution,
            align_corners=align_corners, interp=interp_id,
            if_contract=contract,
            active_hashgrid_levels=active_hashgrid_levels,
            appearance_levels=gaussians._appearance_level[test_indices].int().contiguous(),
            grid_size=grid_size, uv_extent=uv_extent,
        )
    print(f"[DEBUG] CUDA bake output: {list(sh_grid_cuda.shape)}")

    # =========================================================================
    # Step 2: PyTorch MLP evaluation at same points
    # =========================================================================
    # Generate UV grid points
    u_vals = torch.linspace(-uv_extent, uv_extent, grid_size, device='cuda')
    v_vals = torch.linspace(-uv_extent, uv_extent, grid_size, device='cuda')
    uv_grid = torch.stack(torch.meshgrid(v_vals, u_vals, indexing='ij'), dim=-1)  # [gs, gs, 2] (v, u)
    uv_flat = uv_grid.reshape(-1, 2)  # [gs*gs, 2] (v, u)
    n_pts = grid_size * grid_size

    # For each test Gaussian, compute xyz on UV grid
    centers = gaussians.get_xyz[test_indices]  # [n_test, 3]
    quats = gaussians.get_rotation[test_indices]  # [n_test, 4]
    test_scales = gaussians.get_scaling[test_indices]  # [n_test, 2]
    gauss_feats = gaussians.get_gaussian_features[test_indices]  # [n_test, 20]
    ap_levels = gaussians._appearance_level[test_indices].int()  # [n_test, 1]

    # Normalize quaternions
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    # Build rotation matrix columns (tangent frame)
    r00 = 1 - 2*(qy*qy + qz*qz)
    r10 = 2*(qx*qy + qw*qz)
    r20 = 2*(qx*qz - qw*qy)
    r01 = 2*(qx*qy - qw*qz)
    r11 = 1 - 2*(qx*qx + qz*qz)
    r21 = 2*(qy*qz + qw*qx)

    R_col0 = torch.stack([r00, r10, r20], dim=-1)  # [n_test, 3]
    R_col1 = torch.stack([r01, r11, r21], dim=-1)  # [n_test, 3]

    sx = test_scales[:, 0:1]  # [n_test, 1]
    sy = test_scales[:, 1:2]  # [n_test, 1]

    # xyz = center + u*sx*R_col0 + v*sy*R_col1
    # uv_flat[:, 1] = u, uv_flat[:, 0] = v
    u_coords = uv_flat[:, 1]  # [n_pts]
    v_coords = uv_flat[:, 0]  # [n_pts]

    # Expand for broadcasting: [n_test, n_pts, 3]
    xyz_all = (centers.unsqueeze(1)
               + u_coords.unsqueeze(0).unsqueeze(-1) * sx.unsqueeze(1) * R_col0.unsqueeze(1)
               + v_coords.unsqueeze(0).unsqueeze(-1) * sy.unsqueeze(1) * R_col1.unsqueeze(1))

    print(f"[DEBUG] Generated {n_test} × {n_pts} = {n_test * n_pts} xyz points")

    # Query hash features at these xyz positions using Python _encode_3D
    # CRITICAL: must use _encode_3D which normalizes xyz from world space to [0,1]
    # before passing to the GridEncoder. Calling hash_encoding() directly would
    # use un-normalized coordinates and give wrong results!
    xyz_flat = xyz_all.reshape(-1, 3)  # [n_test*n_pts, 3]

    with torch.no_grad():
        hash_out_py = ingp._encode_3D(xyz_flat)  # [n_test*n_pts, 4]

    hash_out_py = hash_out_py.reshape(n_test, n_pts, -1)  # [n_test, n_pts, 4]
    print(f"[DEBUG] Python hash output: {list(hash_out_py.shape)}, range: [{hash_out_py.min():.4f}, {hash_out_py.max():.4f}]")

    # Build MLP input: [gauss(20) | hash(4) | 1.0]
    gauss_expanded = gauss_feats.unsqueeze(1).expand(-1, n_pts, -1)  # [n_test, n_pts, 20]
    bias = torch.ones(n_test, n_pts, 1, device='cuda')
    mlp_input = torch.cat([gauss_expanded, hash_out_py, bias], dim=-1)  # [n_test, n_pts, 25]
    mlp_input_flat = mlp_input.reshape(-1, 25)  # [n_test*n_pts, 25]

    print(f"[DEBUG] MLP input shape: {list(mlp_input_flat.shape)}")

    # Run through PyTorch MLP (mlp_fused)
    with torch.no_grad():
        sh_out_py = ingp.mlp_fused(mlp_input_flat)  # [n_test*n_pts, 48]
    sh_grid_py = sh_out_py.reshape(n_test, n_pts, 48)

    print(f"[DEBUG] PyTorch MLP output: {list(sh_grid_py.shape)}")

    # =========================================================================
    # Step 3: Compare
    # =========================================================================
    diff = (sh_grid_cuda - sh_grid_py).abs()
    print(f"\n{'='*60}")
    print(f"CUDA vs PyTorch SH comparison ({n_test} Gaussians × {n_pts} pts × 48 SH)")
    print(f"{'='*60}")
    print(f"Mean abs diff:  {diff.mean():.6f}")
    print(f"Max abs diff:   {diff.max():.6f}")
    print(f"Median abs diff: {diff.median():.6f}")
    print(f"Std of diff:    {diff.std():.6f}")

    # Per-Gaussian analysis
    per_gauss_mean_diff = diff.mean(dim=(1, 2))  # [n_test]
    per_gauss_max_diff = diff.amax(dim=(1, 2))   # [n_test]
    print(f"\nPer-Gaussian mean diff: min={per_gauss_mean_diff.min():.6f}, max={per_gauss_mean_diff.max():.6f}")
    print(f"Per-Gaussian max diff:  min={per_gauss_max_diff.min():.6f}, max={per_gauss_max_diff.max():.6f}")

    # Check if the SH values themselves vary across the UV grid
    sh_std_cuda = sh_grid_cuda.std(dim=1)  # [n_test, 48]
    sh_std_py = sh_grid_py.std(dim=1)      # [n_test, 48]

    print(f"\n{'='*60}")
    print(f"Spatial variation of SH across UV grid (std over grid points)")
    print(f"{'='*60}")
    print(f"CUDA: mean std={sh_std_cuda.mean():.6f}, max std={sh_std_cuda.max():.6f}")
    print(f"PyTorch: mean std={sh_std_py.mean():.6f}, max std={sh_std_py.max():.6f}")

    # DC coefficient analysis (indices 0, 16, 32 for R, G, B)
    dc_indices = [0, 16, 32]
    for ch_idx, ch_name in zip(dc_indices, ['R', 'G', 'B']):
        dc_cuda = sh_grid_cuda[:, :, ch_idx]  # [n_test, n_pts]
        dc_py = sh_grid_py[:, :, ch_idx]

        dc_std_cuda = dc_cuda.std(dim=1)  # [n_test]
        dc_std_py = dc_py.std(dim=1)

        print(f"\n  DC {ch_name} (coef {ch_idx}):")
        print(f"    CUDA:    mean={dc_cuda.mean():.4f}, std_across_pts={dc_std_cuda.mean():.6f}")
        print(f"    PyTorch: mean={dc_py.mean():.4f}, std_across_pts={dc_std_py.mean():.6f}")
        print(f"    Diff: mean={abs(dc_cuda - dc_py).mean():.6f}")

    # Hash feature comparison: CUDA internal vs Python GridEncoder
    # The CUDA kernel queries hash_features via query_feature<false, 4, 4>()
    # The Python GridEncoder may produce different results
    # Let's check the hash feature variation
    print(f"\n{'='*60}")
    print(f"Hash feature variation analysis")
    print(f"{'='*60}")
    hash_std = hash_out_py.std(dim=1)  # [n_test, 4]
    print(f"Mean hash std across UV grid: {hash_std.mean():.6f}")
    print(f"Max hash std across UV grid:  {hash_std.max():.6f}")

    # Scale analysis
    print(f"\n{'='*60}")
    print(f"Gaussian scale analysis")
    print(f"{'='*60}")
    max_scale_test = max_scales[test_indices]
    cell_size = (voxel_max - voxel_min) / (base_resolution * (per_level_scale ** (num_levels - 1)))
    print(f"Cell size: {cell_size:.6f}")
    print(f"Scale range: [{max_scale_test.min():.6f}, {max_scale_test.max():.6f}]")
    print(f"UV extent in world: [{(uv_extent * max_scale_test).min():.6f}, {(uv_extent * max_scale_test).max():.6f}]")
    print(f"UV extent / cell_size: [{(uv_extent * max_scale_test / cell_size).min():.2f}, {(uv_extent * max_scale_test / cell_size).max():.2f}]")

    # Detailed look at a few Gaussians
    print(f"\n{'='*60}")
    print(f"Detailed per-Gaussian breakdown")
    print(f"{'='*60}")
    for i in [0, n_test//4, n_test//2, 3*n_test//4, n_test-1]:
        gauss_id = test_indices[i].item()
        scale = max_scale_test[i].item()
        uv_world = uv_extent * scale
        cells = uv_world / cell_size

        cuda_sh = sh_grid_cuda[i]  # [n_pts, 48]
        py_sh = sh_grid_py[i]
        d = (cuda_sh - py_sh).abs()

        cuda_std = cuda_sh.std(dim=0).mean().item()
        py_std = py_sh.std(dim=0).mean().item()

        # DC variation
        dc_r_std = cuda_sh[:, 0].std().item()
        dc_g_std = cuda_sh[:, 16].std().item()
        dc_b_std = cuda_sh[:, 32].std().item()

        print(f"\n  Gaussian {gauss_id} (scale={scale:.6f}, UV_world={uv_world:.4f}, cells={cells:.1f}):")
        print(f"    CUDA vs PyTorch: mean_diff={d.mean():.6f}, max_diff={d.max():.6f}")
        print(f"    SH spatial std: CUDA={cuda_std:.6f}, PyTorch={py_std:.6f}")
        print(f"    DC std (R,G,B): {dc_r_std:.6f}, {dc_g_std:.6f}, {dc_b_std:.6f}")

        # Show actual DC values at corners and center
        center_pt = n_pts // 2
        corner_pts = [0, grid_size-1, n_pts-grid_size, n_pts-1]
        print(f"    DC_R values: center={cuda_sh[center_pt, 0]:.4f}, "
              f"corners=[{cuda_sh[corner_pts[0], 0]:.4f}, {cuda_sh[corner_pts[1], 0]:.4f}, "
              f"{cuda_sh[corner_pts[2], 0]:.4f}, {cuda_sh[corner_pts[3], 0]:.4f}]")


if __name__ == "__main__":
    main()

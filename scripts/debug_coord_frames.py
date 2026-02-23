#!/usr/bin/env python3
"""
Verify coordinate frame consistency: CUDA homotrans vs Python quat_to_rotcols.

Two tests:
  Test 1 (Pure Python): Compare build_H columns vs quat_to_rotcols + scale.
  Test 2 (CUDA ground truth): Render with training renderer (render_mode=3) to get
      actual CUDA-computed (s.x, s.y, xyz) per intersection. Recompute xyz from
      (s.x, s.y) using the bake script's Python formula and compare.

This is the definitive test of whether the bake pipeline's coordinate frame
matches the training renderer's.
"""

import os, sys, pickle, glob, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.general_utils import build_rotation, build_H


def quat_to_rotcols(quats):
    """From bake_sh_res.py — the formula used for baking."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm
    r00 = 1 - 2*(y*y + z*z); r10 = 2*(x*y + w*z); r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z); r11 = 1 - 2*(x*x + z*z); r21 = 2*(y*z + w*x)
    return torch.stack([r00, r10, r20], dim=-1), torch.stack([r01, r11, r21], dim=-1)


def load_model(model_path, iteration=-1):
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True
    config_yaml_path = os.path.join(model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)
    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel
    return args, cfg, gaussians, ingp, scene, pipe, iteration


def test1_pure_python(gaussians):
    """Compare build_H homotrans columns vs quat_to_rotcols + scale."""
    print(f"\n{'='*60}")
    print(f"TEST 1: build_H (training) vs quat_to_rotcols (bake)")
    print(f"{'='*60}")

    quats = gaussians.get_rotation    # [N, 4] normalized
    scales = gaussians.get_scaling    # [N, 2]
    centers = gaussians.get_xyz       # [N, 3]
    N = quats.shape[0]

    # --- Training renderer path ---
    rots = build_rotation(quats)        # [N, 3, 3]
    H = build_H(rots, scales, centers)  # [N, 4, 4]

    # Extract SuTu, SvTv, pk using same indexing as CUDA kernel
    # H is stored row-major: H[i, row, col]
    # CUDA reads: SuTu = {H[0], H[4], H[8]} = {H[0,0], H[1,0], H[2,0]} = column 0
    #             SvTv = {H[1], H[5], H[9]} = {H[0,1], H[1,1], H[2,1]} = column 1
    #             pk   = {H[3], H[7], H[11]} = {H[0,3], H[1,3], H[2,3]} = column 3
    SuTu = H[:, :3, 0]  # [N, 3] = column 0 of upper-left 3x3
    SvTv = H[:, :3, 1]  # [N, 3] = column 1
    pk   = H[:, :3, 3]  # [N, 3] = translation column

    # --- Bake script path ---
    R0, R1 = quat_to_rotcols(quats)  # [N, 3] each
    sx = scales[:, 0:1]  # [N, 1]
    sy = scales[:, 1:2]
    bake_SuTu = sx * R0  # [N, 3]
    bake_SvTv = sy * R1  # [N, 3]
    bake_pk   = centers   # [N, 3]

    # Compare
    diff_sutu = (SuTu - bake_SuTu).abs()
    diff_svtv = (SvTv - bake_SvTv).abs()
    diff_pk   = (pk - bake_pk).abs()

    print(f"\n  N = {N} Gaussians")
    print(f"  SuTu diff: mean={diff_sutu.mean():.2e}, max={diff_sutu.max():.2e}")
    print(f"  SvTv diff: mean={diff_svtv.mean():.2e}, max={diff_svtv.max():.2e}")
    print(f"  pk   diff: mean={diff_pk.mean():.2e}, max={diff_pk.max():.2e}")

    # Test at specific UV points
    test_uvs = [(0, 0), (1, 0), (0, 1), (-3.5, -3.5), (3.5, 3.5)]
    print(f"\n  XYZ comparison at test UV points (Gaussian 0):")
    for u, v in test_uvs:
        xyz_train = pk[0] + u * SuTu[0] + v * SvTv[0]
        xyz_bake  = bake_pk[0] + u * bake_SuTu[0] + v * bake_SvTv[0]
        diff = (xyz_train - xyz_bake).abs().max().item()
        print(f"    (u={u:5.1f}, v={v:5.1f}): diff={diff:.2e}")

    max_diff = max(diff_sutu.max().item(), diff_svtv.max().item(), diff_pk.max().item())
    passed = max_diff < 1e-5
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (max_diff={max_diff:.2e}, threshold=1e-5)")
    return passed


def test2_cuda_intersection_buffer(gaussians, ingp, scene, cfg, args, pipe, iteration, cam_idx=0, scale_mult=3.0):
    """Render with training CUDA kernel (render_mode=3), compare CUDA xyz to bake formula."""
    print(f"\n{'='*60}")
    print(f"TEST 2: CUDA intersection buffer xyz vs bake Python xyz")
    print(f"{'='*60}")

    from diff_surfel_3D_sh_res import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings

    N_orig = len(gaussians.get_xyz)

    # Pick and enlarge single Gaussian
    centers = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    centroid = centers.mean(dim=0)
    dists = (centers - centroid).norm(dim=1)
    area = scales[:, 0] * scales[:, 1]
    score = -dists + 0.1 * area.clamp(min=1e-8).log()
    idx = score.argmax().item()

    for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                 '_scaling', '_rotation', '_appearance_level']:
        tensor = getattr(gaussians, attr)
        setattr(gaussians, attr, torch.nn.Parameter(tensor[idx:idx+1].clone()))
    if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        gaussians._shape = torch.nn.Parameter(gaussians._shape[idx:idx+1].clone())
    gaussians._scaling.data += math.log(scale_mult)
    gaussians._opacity.data.fill_(10.0)

    print(f"\n  Single Gaussian {idx}/{N_orig}")
    print(f"  center: {gaussians.get_xyz[0].tolist()}")
    print(f"  scales: {gaussians.get_scaling[0].tolist()}")

    cam = scene.getTestCameras()[cam_idx]
    bg = torch.zeros(3, device="cuda")

    # Build homotrans (same as training renderer)
    homotrans = gaussians.get_homotrans()  # [1, 4, 4]

    # Set up hash grid params (dummy — render_mode=3 doesn't query hash)
    hash_features, hash_offsets, hash_levels, per_level_scale, base_resolution, align_corners, interpolation \
        = ingp.hash_encoding.get_params()
    gridrange = ingp.gridrange
    total_levels = ingp.levels
    active_hashgrid_levels = ingp.hashgrid_levels
    levels = (total_levels << 16) | (active_hashgrid_levels << 8) | 0
    if hash_offsets.shape[0] < 17:
        padded_offsets = torch.zeros(17, dtype=hash_offsets.dtype, device=hash_offsets.device)
        padded_offsets[:hash_offsets.shape[0]] = hash_offsets
        hash_offsets = padded_offsets

    hash_dim = active_hashgrid_levels * ingp.level_dim
    shape_dims = torch.tensor([0, hash_dim, 3], dtype=torch.int32, device="cuda")

    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(getattr(gaussians, 'kernel_type', 'gaussian'), 0)
    shapes = None
    if kernel_type > 0 and hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape

    max_per_pixel = 64

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=gaussians.active_sh_degree, campos=cam.camera_center,
        prefiltered=False, debug=False, beta=cfg.surfel.tg_beta,
        if_contract=ingp.contract, record_transmittance=False,
        max_intersections=0, max_intersections_per_pixel=max_per_pixel,
    )
    hashgrid_settings = HashGridSettings(
        L=levels, S=math.log2(ingp.growth_rate), H=ingp.resolutions[0],
        align_corners=False, interpolation=0, shape_dims=shape_dims,
        aa=0.0, aa_threshold=0.01,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)

    print(f"\n  Calling diff_surfel_3D_sh_res with render_mode=3 (intersection buffer)...")

    with torch.no_grad():
        color, radii, depth, transmittance_avg, pixels, ibuf, icnt, geomBuffer = rasterizer(
            means3D=gaussians.get_xyz,
            means2D=torch.zeros_like(gaussians.get_xyz[:, :2]),
            opacities=gaussians.get_opacity,
            shs=gaussians.get_features,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            homotrans=homotrans,
            ap_level=gaussians.get_appearance_level,
            features=hash_features, offsets=hash_offsets, gridrange=gridrange,
            render_mode=3,  # Intersection buffer mode!
            shapes=shapes, kernel_type=kernel_type,
        )

    print(f"  Intersection buffer shape: {ibuf.shape}")
    print(f"  Intersection count shape: {icnt.shape}")

    H, W = int(cam.image_height), int(cam.image_width)
    max_per_pixel = 64

    # Extract valid intersections
    total_slots = ibuf.shape[0]
    slot_indices = torch.arange(total_slots, device='cuda')
    local_indices = slot_indices % max_per_pixel
    pixel_ids_from_slot = slot_indices // max_per_pixel
    valid_mask = local_indices < icnt[pixel_ids_from_slot]
    valid = ibuf[valid_mask]
    M = valid.shape[0]
    print(f"  Valid intersections: {M}")

    if M == 0:
        print(f"  ERROR: No valid intersections found")
        return False

    # Unpack CUDA-computed data (12-field layout)
    cuda_xyz = valid[:, 3:6].float()    # [M, 3] — CUDA world XYZ
    cuda_sx = valid[:, 6].float()       # [M] — parametric s.x
    cuda_sy = valid[:, 7].float()       # [M] — parametric s.y
    rho_flag = valid[:, 8].float()      # [M] — 1.0=disk, 0.0=center

    # Only compare disk intersections (rho_flag=1), not center fallbacks
    disk_mask = rho_flag > 0.5
    n_disk = disk_mask.sum().item()
    n_center = M - n_disk
    print(f"  Disk intersections: {n_disk}, Center fallbacks: {n_center}")

    if n_disk == 0:
        print(f"  ERROR: No disk intersections")
        return False

    # Recompute xyz using bake formula: center + s.x * sx * R0 + s.y * sy * R1
    quats = gaussians.get_rotation    # [1, 4]
    scales_g = gaussians.get_scaling  # [1, 2]
    center = gaussians.get_xyz[0]     # [3]
    R0, R1 = quat_to_rotcols(quats)
    sx_g = scales_g[0, 0]
    sy_g = scales_g[0, 1]
    r0 = R0[0]  # [3]
    r1 = R1[0]  # [3]

    bake_xyz = (center.unsqueeze(0)
                + cuda_sx[disk_mask].unsqueeze(-1) * (sx_g * r0).unsqueeze(0)
                + cuda_sy[disk_mask].unsqueeze(-1) * (sy_g * r1).unsqueeze(0))

    # Also recompute using build_H (training renderer formula)
    rots = build_rotation(quats)
    H_mat = build_H(rots, scales_g, gaussians.get_xyz)
    SuTu = H_mat[0, :3, 0]
    SvTv = H_mat[0, :3, 1]
    pk = H_mat[0, :3, 3]

    train_xyz = (pk.unsqueeze(0)
                 + cuda_sx[disk_mask].unsqueeze(-1) * SuTu.unsqueeze(0)
                 + cuda_sy[disk_mask].unsqueeze(-1) * SvTv.unsqueeze(0))

    # Compare
    cuda_disk_xyz = cuda_xyz[disk_mask]

    diff_bake = (cuda_disk_xyz - bake_xyz).abs()
    diff_train = (cuda_disk_xyz - train_xyz).abs()
    diff_bake_train = (bake_xyz - train_xyz).abs()

    print(f"\n  CUDA xyz vs bake_sh_res.py xyz (quat_to_rotcols):")
    print(f"    mean diff: {diff_bake.mean():.2e}")
    print(f"    max  diff: {diff_bake.max():.2e}")
    print(f"    per-axis max: x={diff_bake[:,0].max():.2e}, y={diff_bake[:,1].max():.2e}, z={diff_bake[:,2].max():.2e}")

    print(f"\n  CUDA xyz vs build_H xyz (training Python path):")
    print(f"    mean diff: {diff_train.mean():.2e}")
    print(f"    max  diff: {diff_train.max():.2e}")

    print(f"\n  bake xyz vs build_H xyz (Python vs Python):")
    print(f"    mean diff: {diff_bake_train.mean():.2e}")
    print(f"    max  diff: {diff_bake_train.max():.2e}")

    # Show s.x, s.y range
    print(f"\n  s.x range: [{cuda_sx[disk_mask].min():.3f}, {cuda_sx[disk_mask].max():.3f}]")
    print(f"  s.y range: [{cuda_sy[disk_mask].min():.3f}, {cuda_sy[disk_mask].max():.3f}]")

    # Show a few example intersections
    print(f"\n  Sample intersections (first 5 disk):")
    for i in range(min(5, n_disk)):
        idx_d = disk_mask.nonzero(as_tuple=True)[0][i]
        sx_v = cuda_sx[idx_d].item()
        sy_v = cuda_sy[idx_d].item()
        cx = cuda_xyz[idx_d].tolist()
        bx = bake_xyz[i].tolist()
        d = diff_bake[i].max().item()
        print(f"    s=({sx_v:6.3f},{sy_v:6.3f}) cuda=[{cx[0]:8.5f},{cx[1]:8.5f},{cx[2]:8.5f}] "
              f"bake=[{bx[0]:8.5f},{bx[1]:8.5f},{bx[2]:8.5f}] diff={d:.2e}")

    max_diff = diff_bake.max().item()
    passed = max_diff < 1e-3
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (max_diff={max_diff:.2e}, threshold=1e-3)")
    return passed


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--scale_mult", type=float, default=3.0)
    a = parser.parse_args()

    args, cfg, gaussians, ingp, scene, pipe, iteration = load_model(a.model_path, a.iteration)

    # Test 1: Pure Python comparison (uses all Gaussians)
    t1 = test1_pure_python(gaussians)

    # Test 2: CUDA intersection buffer (single enlarged Gaussian)
    t2 = test2_cuda_intersection_buffer(gaussians, ingp, scene, cfg, args, pipe, iteration,
                                        cam_idx=a.cam_idx, scale_mult=a.scale_mult)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Test 1 (build_H vs quat_to_rotcols): {'PASS' if t1 else 'FAIL'}")
    print(f"  Test 2 (CUDA xyz vs bake Python xyz): {'PASS' if t2 else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

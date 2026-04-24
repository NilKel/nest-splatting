#!/usr/bin/env python3
"""
Bake 3D_SH_res or 3D_SH_cat model with adaptive per-Gaussian UV resolution into a packed atlas.

Resolution per Gaussian = next power of 2 above (hash cells across Gaussian).
This ensures every texel is finer than the finest hashgrid cell.

3D_SH_res MLP input: [hash(4) | bias(1) | pad(11)] = 16D
3D_SH_cat MLP input: [hash(4) | DC_SH(3) | bias(1) | pad(8)] = 16D

Output:
  - baked.ply: Gaussians with original SH coefficients (unchanged)
  - atlas_texture.pt: [H, W, 3] FP16 packed atlas of RGB residuals
  - atlas_rects.pt: [N, 4] float32 (u0_px, v0_px, u_span, v_span)
  - bake_meta.json: metadata

Usage:
    python scripts/bake_sh_res_atlas.py --model_path outputs/.../3D_SH_res/run_name
    python scripts/bake_sh_res_atlas.py --model_path outputs/.../3D_SH_cat/run_name
"""

import os, sys, json, pickle, glob, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def quat_to_rotcols(quats):
    """Quaternion [N, 4] (w,x,y,z) -> R_col0 [N, 3], R_col1 [N, 3]."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm
    r00 = 1 - 2*(y*y + z*z); r10 = 2*(x*y + w*z); r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z); r11 = 1 - 2*(x*x + z*z); r21 = 2*(y*z + w*x)
    return torch.stack([r00, r10, r20], dim=-1), torch.stack([r01, r11, r21], dim=-1)


# ---------------------------------------------------------------------------
# Adaptive resolution
# ---------------------------------------------------------------------------
def compute_adaptive_resolution(scales, cell_size, uv_extent=4.0, max_res=64, min_res=4):
    """Per-axis Nyquist resolution → [N, 2] of (res_x, res_y) powers of two.

    Each surfel gets its own res per tangent axis so anisotropic surfels
    (long and thin) don't waste the short axis. Both axes satisfy:
        res = next_pow2(2 · 2 · uv_extent · scale_axis / cell_size)
    clamped to [min_res, max_res].
    """
    n_cells = 2.0 * uv_extent * scales / cell_size           # [N, 2]
    nyquist_samples = 2.0 * n_cells
    log2_res = torch.ceil(torch.log2(nyquist_samples.clamp(min=1.0)))
    resolutions = (2.0 ** log2_res).int()
    return resolutions.clamp(min=min_res, max=max_res)       # [N, 2]


# ---------------------------------------------------------------------------
# Atlas packing (shelf-first-fit-decreasing, rectangular)
# ---------------------------------------------------------------------------
def shelf_pack_atlas(resolutions, atlas_width=4096):
    """Shelf-first-fit-decreasing for rectangular (res_x, res_y) primitives.
    `resolutions` is [N, 2]. Returns (rects [N, 4] = (u0, v0, w, h), height,
    used_rows, utilization %)."""
    res_cpu = resolutions.cpu().numpy().astype(np.int64)    # [N, 2]
    N = res_cpu.shape[0]
    rx = res_cpu[:, 0]
    ry = res_cpu[:, 1]

    # Estimate atlas height by summing shelf heights per distinct ry.
    height = 0
    for sz in sorted(set(int(y) for y in ry), reverse=True):
        sel = (ry == sz)
        total_w = int(rx[sel].sum())
        rows_for_sz = (total_w + atlas_width - 1) // atlas_width
        height += rows_for_sz * sz
    atlas_height = max(((height + 63) // 64) * 64, 64)

    order = np.lexsort((-rx, -ry))     # taller first; wider first within a row
    shelves = []                        # [y_start, shelf_height, next_x]
    rects = np.zeros((N, 4), dtype=np.float32)

    for idx in order:
        ix = int(rx[idx])
        iy = int(ry[idx])
        placed = False
        for shelf in shelves:
            if shelf[1] >= iy and shelf[2] + ix <= atlas_width:
                rects[idx] = [shelf[2], shelf[0], ix, iy]
                shelf[2] += ix
                placed = True
                break
        if not placed:
            y_start = max((s[0] + s[1] for s in shelves), default=0)
            if y_start + iy > atlas_height:
                rects[idx] = [0, 0, 2, 2]   # fallback
                continue
            shelves.append([y_start, iy, ix])
            rects[idx] = [0, y_start, ix, iy]

    used_rows = max((s[0] + s[1] for s in shelves), default=0)
    total_area = float((rx * ry).sum())
    utilization = total_area / (atlas_width * atlas_height) * 100
    return torch.from_numpy(rects).float(), atlas_height, used_rows, utilization


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="Bake 3D_SH_res/3D_SH_cat → adaptive atlas (3D RGB residual)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--uv_extent", type=float, default=4.0)
    parser.add_argument("--max_res", type=int, default=128, help="Max per-Gaussian resolution")
    parser.add_argument("--min_res", type=int, default=4, help="Min per-Gaussian resolution")
    parser.add_argument("--atlas_width", type=int, default=4096)
    parser.add_argument("--atlas_budget_mb", type=float, default=0,
                        help="Max atlas size in MB. Progressively downgrades resolutions to fit. 0=unlimited.")
    parser.add_argument("--ss", type=int, default=1, help="Supersample factor")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--atlas_format", type=str, default="uint8",
                        choices=["uint8", "fp16"],
                        help="Disk atlas format. 'uint8' (default) = ±6σ quantized RGBA, "
                             "4 B/texel, 33%% smaller than FP16 RGB. Benchmarks show uint8 "
                             "matches or slightly beats FP16 due to heavy-tail clipping "
                             "acting as a regularizer. 'fp16' preserves the legacy [H,W,3] "
                             "FP16 format for tools that haven't been updated.")
    parser.add_argument("--atlas_sigma", type=float, default=6.0,
                        help="uint8 quantization range in std-devs. Atlas range = mean ± "
                             "sigma·std. Default 6.0 matches the CUDA runtime path.")
    bake_args = parser.parse_args()

    # Load training config
    print(f"\n[BAKE] Loading config from: {bake_args.model_path}")
    with open(os.path.join(bake_args.model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = bake_args.model_path
    args.eval = True

    config_yaml_path = os.path.join(bake_args.model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    assert args.method in ("3D_SH_res", "3D_SH_cat"), f"Expected 3D_SH_res or 3D_SH_cat, got {args.method}"
    is_cat_mode = (args.method == "3D_SH_cat")

    # Auto-detect iteration
    iteration = bake_args.iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(bake_args.model_path, "ngp_*.pth"))
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)
        print(f"[CONFIG] Latest iteration: {iteration}")

    # Setup model
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(bake_args.model_path, iteration)
    ingp.set_active_levels(iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead Gaussians
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    n_dead = dead_mask.sum().item()
    if n_dead > 0:
        valid_mask = ~dead_mask
        # Prune every per-Gaussian tensor so save_ply doesn't hit a size mismatch.
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                     '_scaling', '_rotation', '_appearance_level',
                     '_gaussian_features',
                     '_shape', '_flex_beta',
                     '_sb_params',
                     '_sg_directions', '_sg_sharpness_sg', '_sg_rgb',
                     '_sv_sites', '_sv_colors',
                     '_gamma', '_adaptive_features',
                     '_adaptive_cat_weight', '_adaptive_zero_weight',
                     '_gate_logits']:
            tensor = getattr(gaussians, attr, None)
            if tensor is not None and tensor.numel() > 0 and tensor.shape[0] == valid_mask.shape[0]:
                setattr(gaussians, attr, tensor[valid_mask.to(tensor.device)])

    N = len(gaussians.get_xyz)
    print(f"\n[BAKE] Gaussians after pruning: {N:,}")

    # ---- Capture training-time activation + kernel configuration ----
    _act_bias = getattr(args, 'activation_bias', [0.5, 0.0])
    if isinstance(_act_bias, (list, tuple)) and len(_act_bias) >= 2:
        bake_sh_bias, bake_res_bias = float(_act_bias[0]), float(_act_bias[1])
    else:
        bake_sh_bias, bake_res_bias = 0.5, 0.0
    bake_feature_mode = getattr(args, 'feature', 'sh')
    bake_fastgs_mult = float(getattr(args, 'fastgs_mult', 1.0)) if getattr(args, 'fastgs', False) else 1.0
    # Kernel_type int mapping matches the training-time path.
    _k_str = getattr(args, 'kernel', 'gaussian')
    _kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4, 'nexel': 5}
    bake_kernel_type = int(_kernel_map.get(_k_str, 0))
    print(f"[BAKE] Activation bias: sh_bias={bake_sh_bias}, res_bias={bake_res_bias}")
    print(f"[BAKE] Feature mode: {bake_feature_mode}")
    print(f"[BAKE] Kernel: {_k_str} (int={bake_kernel_type})")
    print(f"[BAKE] Compact Box mult: {bake_fastgs_mult}")

    # =========================================================================
    # Step 1: Compute adaptive resolution per Gaussian
    # =========================================================================
    uv_extent = bake_args.uv_extent
    hash_encoding = ingp.hash_encoding
    embeddings, offsets, num_levels, per_level_scale, base_resolution, align_corners, interp_id = hash_encoding.get_params()
    voxel_min = ingp.voxel_range[0]
    voxel_max = ingp.voxel_range[1]
    finest_resolution = base_resolution * (per_level_scale ** (num_levels - 1))
    cell_size = (voxel_max - voxel_min) / finest_resolution

    print(f"[BAKE] Hash grid: {num_levels} levels, finest_res={finest_resolution:.0f}, "
          f"cell_size={cell_size:.6f}, range=[{voxel_min}, {voxel_max}]")

    resolutions = compute_adaptive_resolution(
        gaussians.get_scaling, cell_size, uv_extent=uv_extent,
        max_res=bake_args.max_res, min_res=bake_args.min_res)

    # Budget-fit: progressively halve the largest resolution bin (on either axis)
    # until atlas fits. Texel cost per Gaussian is res_x * res_y.
    atlas_width = bake_args.atlas_width
    if bake_args.atlas_budget_mb > 0:
        budget_texels = int(bake_args.atlas_budget_mb * 1024 * 1024 / 6)  # 3ch × FP16 = 6 bytes/texel
        while True:
            total_texels = int((resolutions[:, 0].long() * resolutions[:, 1].long()).sum().item())
            if total_texels <= budget_texels:
                break
            cur_max = int(resolutions.max().item())
            if cur_max <= bake_args.min_res:
                print(f"[BUDGET] Cannot fit within {bake_args.atlas_budget_mb} MB even at min_res={bake_args.min_res}")
                break
            mask = (resolutions == cur_max)
            resolutions[mask] = cur_max // 2
            n_downgraded = int(mask.any(dim=1).sum().item())
            new_mb = total_texels * 6 / 1024 / 1024
            print(f"[BUDGET] Downgraded {n_downgraded:,} Gaussians' oversized axes: "
                  f"{cur_max}→{cur_max//2} (~{new_mb:.0f} MB → target {bake_args.atlas_budget_mb:.0f} MB)")

    # =========================================================================
    # Step 2: Pack atlas
    # =========================================================================
    centers = gaussians.get_xyz        # [N, 3]
    quats = gaussians.get_rotation     # [N, 4]
    scales = gaussians.get_scaling     # [N, 2]
    R0, R1 = quat_to_rotcols(quats)   # [N, 3] each

    # Cast the training MLP to FP16 to match the in-kernel __half2 math the
    # training-time forward uses (float2half_kernel uploads + FP16 GEMM).
    # Without this, Python evaluates in FP32 and the residual numerically drifts
    # from what the training render kernel produced.
    mlp = ingp.mlp_fused.half().eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_input_padded = mlp[0].weight.shape[1]

    if is_cat_mode:
        # 3D_SH_cat: input layout is [hash(hash_dim) | DC_SH(3) | bias(1) | pad].
        # The bias column = 1 is intentional — cat-mode MLP HAS an implicit bias via
        # that column (see hash_encoder/modules.py ~L265-L290).
        dc_start = hash_dim       # 4
        bias_col = hash_dim + 3   # 7
        SH_C0 = 0.28209479177387814
        dc_sh = SH_C0 * gaussians.get_features[:, 0, :] + 0.5  # [N, 3]
        print(f"[BAKE] 3D_SH_cat: DC_SH at cols {dc_start}-{dc_start+2}, bias at col {bias_col}")
        print(f"[BAKE] DC_SH stats: mean={dc_sh.mean():.4f}, min={dc_sh.min():.4f}, max={dc_sh.max():.4f}")
    else:
        # 3D_SH_res: input is [hash(hash_dim) | pad(16 - hash_dim)] = 16D, NO BIAS.
        # hash_encoder/modules.py ~L328 documents: "No bias: all layers bias=False, no
        # implicit bias in input." Writing 1 anywhere would give the MLP an extra
        # input feature it wasn't trained with and shift the residual systematically.
        bias_col = None

    # Resolution distribution grouped by (res_x, res_y) pair.
    print(f"\n[BAKE] Adaptive resolution distribution (res_x × res_y):")
    res_cpu = resolutions.cpu()
    pair_keys = (res_cpu[:, 0].long() * 10000 + res_cpu[:, 1].long())
    unique_keys, counts = torch.unique(pair_keys, return_counts=True)
    areas = (unique_keys // 10000) * (unique_keys % 10000)
    sort_idx = torch.argsort(-areas)
    for k_idx in sort_idx:
        key = unique_keys[k_idx].item()
        cnt = counts[k_idx].item()
        rx_v, ry_v = key // 10000, key % 10000
        print(f"  {rx_v:>4}x{ry_v:<4}: {cnt:>7,} Gaussians")

    atlas_rects, atlas_height, used_rows, utilization = shelf_pack_atlas(
        resolutions, atlas_width=atlas_width)
    atlas_rects = atlas_rects.cuda()

    atlas_mb = atlas_height * atlas_width * 3 * 2 / 1024 / 1024
    print(f"\n[ATLAS] Packed: {atlas_width}×{atlas_height}, "
          f"used {used_rows}/{atlas_height} rows, {utilization:.1f}% util, {atlas_mb:.1f} MB")

    # Allocate atlas (FP32 for accumulation, convert to FP16 at end)
    atlas = torch.zeros(atlas_height, atlas_width, 3, device='cuda', dtype=torch.float32)

    # =========================================================================
    # Step 3: Bake hash+MLP per resolution group
    # =========================================================================
    ss = bake_args.ss

    # Iterate per unique (res_x, res_y) pair; Gaussians within the same pair
    # share the UV lattice shape, so the MLP eval can be batched together.
    unique_pairs = set(
        (int(resolutions[i, 0].item()), int(resolutions[i, 1].item()))
        for i in range(resolutions.shape[0])
    )
    for (res_x, res_y) in sorted(unique_pairs, key=lambda p: -(p[0] * p[1])):
        bake_res_x = res_x * ss
        bake_res_y = res_y * ss
        mask = (resolutions[:, 0] == res_x) & (resolutions[:, 1] == res_y)
        indices = mask.nonzero(as_tuple=True)[0]
        n_group = len(indices)

        print(f"\n[BAKE] {res_x}x{res_y} (ss={ss}x -> {bake_res_x}x{bake_res_y}): "
              f"{n_group:,} Gaussians")

        step_x = 2.0 * uv_extent / bake_res_x
        step_y = 2.0 * uv_extent / bake_res_y
        u_coords = (torch.arange(bake_res_x, dtype=torch.float32, device='cuda') + 0.5) * step_x - uv_extent
        v_coords = (torch.arange(bake_res_y, dtype=torch.float32, device='cuda') + 0.5) * step_y - uv_extent
        uu, vv = torch.meshgrid(u_coords, v_coords, indexing='ij')
        u_flat = uu.reshape(-1)
        v_flat = vv.reshape(-1)
        n_pts = bake_res_x * bake_res_y

        max_samples = 2 * (1024**3) // 160
        max_batch = max(1, max_samples // n_pts)
        chunk_size = min(n_group, max_batch)

        for ci_start in range(0, n_group, chunk_size):
            ci_end = min(ci_start + chunk_size, n_group)
            batch_indices = indices[ci_start:ci_end]
            n_batch = len(batch_indices)

            c = centers[batch_indices]
            sx = scales[batch_indices, 0:1]
            sy = scales[batch_indices, 1:2]
            r0 = R0[batch_indices]
            r1 = R1[batch_indices]

            xyz = (c.unsqueeze(1)
                   + u_flat.unsqueeze(0).unsqueeze(-1) * (sx.unsqueeze(1) * r0.unsqueeze(1))
                   + v_flat.unsqueeze(0).unsqueeze(-1) * (sy.unsqueeze(1) * r1.unsqueeze(1)))
            xyz_flat = xyz.reshape(-1, 3)

            with torch.no_grad():
                hash_feat = ingp._encode_3D(xyz_flat)
                # MLP and input are FP16 to match training's in-kernel
                # __half2 math (training uploads FP16 weights via
                # float2half_kernel; any FP32 Python path diverges numerically).
                mlp_input = torch.zeros(xyz_flat.shape[0], mlp_input_padded,
                                        device='cuda', dtype=torch.float16)
                mlp_input[:, :hash_dim] = hash_feat[:, :hash_dim].to(torch.float16)
                # 3D_SH_res: bias_col is None (pure zero-padded input, matching training).
                # 3D_SH_cat: bias_col = hash_dim+3, holds implicit bias = 1.
                if bias_col is not None:
                    mlp_input[:, bias_col] = 1.0

                if is_cat_mode:
                    dc_batch = dc_sh[batch_indices]
                    dc_expanded = dc_batch.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, 3)
                    mlp_input[:, dc_start:dc_start+3] = dc_expanded.to(torch.float16)

                mlp_out = mlp(mlp_input)  # FP16 in, FP16 out
                rgb_residual = mlp_out[:, :3].to(torch.float32)  # Raw residual; render kernel applies ReLU

            residual = rgb_residual.reshape(n_batch, bake_res_x, bake_res_y, 3)

            if ss > 1:
                residual = residual.view(n_batch, res_x, ss, res_y, ss, 3).mean(dim=(2, 4))

            # Write to atlas. UV lattice is (res_x, res_y); atlas stores as
            # [row=v, col=u], so transpose the (u, v) axes when copying.
            rects = atlas_rects[batch_indices]
            for b in range(n_batch):
                u0 = int(rects[b, 0].item())
                v0 = int(rects[b, 1].item())
                atlas[v0:v0+res_y, u0:u0+res_x, :] = residual[b].permute(1, 0, 2)

            del xyz, xyz_flat, hash_feat, mlp_input, mlp_out, rgb_residual, residual
            torch.cuda.empty_cache()

            if n_group > chunk_size and ci_start % (chunk_size * 5) == 0 and ci_start > 0:
                print(f"  {ci_start}/{n_group}")

    # =========================================================================
    # Step 4: Save outputs
    # =========================================================================
    output_dir = bake_args.output_dir or os.path.join(bake_args.model_path, "baked_atlas")
    os.makedirs(output_dir, exist_ok=True)

    # PLY (SH unchanged)
    ply_path = os.path.join(output_dir, "baked.ply")
    gaussians.save_ply(ply_path)
    print(f"\n[BAKE] Saved baked.ply → {ply_path}")

    # Atlas texture. Default path is uint8 RGBA (±sigma·std quantization, matches the
    # CUDA runtime g_atlas_use_uint8 path verbatim). Heavy-tail clipping actually
    # improves quality vs lossless FP16 on our scenes (acts as a regularizer).
    # Legacy FP16 [H,W,3] path is available via --atlas_format fp16.
    atlas_path = os.path.join(output_dir, "atlas_texture.pt")
    if bake_args.atlas_format == "uint8":
        atlas_fp32 = atlas.detach().to(torch.float32)
        a_mean = float(atlas_fp32.mean().item())
        a_std  = float(atlas_fp32.std().item())
        k = float(bake_args.atlas_sigma)
        atlas_offset = a_mean - k * a_std
        atlas_scale  = max(2.0 * k * a_std, 1e-6)
        # Quantize: u8 = clamp((val - offset) / scale * 255, 0, 255). Pad to RGBA so
        # Halloumi/wgpu can use unpack4x8unorm (1 u32 per texel, hw bilinear ready).
        atlas_u8_rgb = torch.clamp(
            (atlas_fp32 - atlas_offset) / atlas_scale * 255.0, 0.0, 255.0
        ).round().to(torch.uint8)                                # [H, W, 3]
        alpha = torch.zeros(atlas_u8_rgb.shape[0], atlas_u8_rgb.shape[1], 1,
                            dtype=torch.uint8, device=atlas_u8_rgb.device)
        atlas_out = torch.cat([atlas_u8_rgb, alpha], dim=-1).contiguous().cpu()
        torch.save(atlas_out, atlas_path)
        atlas_mb_out = atlas_out.numel() / 1024 / 1024
        print(f"[BAKE] Saved atlas_texture.pt → {atlas_path} "
              f"({atlas_mb_out:.1f} MB, uint8 RGBA, scale={atlas_scale:.6f}, "
              f"offset={atlas_offset:.6f})")
    else:
        torch.save(atlas.cpu().half(), atlas_path)
        atlas_offset = 0.0
        atlas_scale  = 1.0
        print(f"[BAKE] Saved atlas_texture.pt → {atlas_path} ({atlas_mb:.1f} MB, fp16 RGB)")

    # Atlas rects [N, 4] float32
    rects_path = os.path.join(output_dir, "atlas_rects.pt")
    torch.save(atlas_rects.cpu(), rects_path)
    print(f"[BAKE] Saved atlas_rects.pt → {rects_path}")

    # Spherical-Beta params (if --feature beta was used during training).
    sb_number = 0
    sb_path = None
    if bake_feature_mode == "beta" and hasattr(gaussians, '_sb_params') and gaussians._sb_params.numel() > 0:
        sb_tensor = gaussians._sb_params.detach().cpu().float()
        sb_number = int(sb_tensor.shape[1])
        sb_path = os.path.join(output_dir, "sb_params.pt")
        torch.save(sb_tensor, sb_path)
        print(f"[BAKE] Saved sb_params.pt → {sb_path}  shape={list(sb_tensor.shape)}  "
              f"(K={sb_number} lobes per Gaussian)")

    # Residual stats
    print(f"\n[BAKE] Atlas residual stats: mean={atlas.mean():.6f}, std={atlas.std():.6f}, "
          f"min={atlas.min():.6f}, max={atlas.max():.6f}")

    # Metadata
    res_dist = {}
    for (rx_v, ry_v) in unique_pairs:
        cnt = int(((resolutions[:, 0] == rx_v) & (resolutions[:, 1] == ry_v)).sum().item())
        res_dist[f"{rx_v}x{ry_v}"] = cnt

    meta = {
        "texture_mode": "atlas",
        "residual_dim": 3,
        "uv_extent": uv_extent,
        "supersample": ss,
        "num_gaussians": N,
        "iteration": iteration,
        "method": args.method,
        "kernel": getattr(args, 'kernel', 'gaussian'),
        "kernel_type_int": bake_kernel_type,
        "sh_degree": 3,
        "atlas_width": atlas_width,
        "atlas_height": atlas_height,
        "max_res": bake_args.max_res,
        "min_res": bake_args.min_res,
        "cell_size": cell_size,
        "finest_resolution": finest_resolution,
        "resolution_distribution": res_dist,
        # Training-time activation + Compact Box knobs so the renderer can
        # reproduce the training kernel exactly.
        "sh_bias": bake_sh_bias,
        "res_bias": bake_res_bias,
        "compact_mult": bake_fastgs_mult,
        "feature_mode": bake_feature_mode,
        # Atlas texel format + dequant parameters — NAT2 exporter + runtime
        # readers use these to interpret atlas_texture.pt correctly.
        "atlas_format": "uint8_rgba" if bake_args.atlas_format == "uint8" else "fp16_rgb",
        "atlas_scale": atlas_scale,
        "atlas_offset": atlas_offset,
        "sb_number": sb_number,
        "sb_params_file": "sb_params.pt" if sb_number > 0 else None,
    }
    meta_path = os.path.join(output_dir, "bake_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[BAKE] Saved bake_meta.json → {meta_path}")

    print(f"\n[BAKE] Done! Output: {output_dir}")


if __name__ == "__main__":
    main()

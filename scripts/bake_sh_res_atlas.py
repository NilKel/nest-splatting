#!/usr/bin/env python3
"""
Bake 3D_SH_res model with adaptive per-Gaussian UV resolution into a packed atlas.

Resolution per Gaussian = next power of 2 above (hash cells across Gaussian).
This ensures every texel is finer than the finest hashgrid cell.

Output:
  - baked.ply: Gaussians with original SH coefficients (unchanged)
  - atlas_texture.pt: [H, W, 3] FP16 packed atlas of RGB residuals
  - atlas_rects.pt: [N, 4] float32 (u0_px, v0_px, u_span, v_span)
  - bake_meta.json: metadata

Usage:
    python scripts/bake_sh_res_atlas.py --model_path outputs/nerf_synthetic/chair/3D_SH_res/betscaled
    python scripts/bake_sh_res_atlas.py --model_path ... --max_res 64 --ss 2
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
    """
    Per-Gaussian resolution = next power of 2 >= 2 * n_cells (Nyquist criterion).

    The Gaussian's UV texture covers [-uv_extent*sx, +uv_extent*sx] in world space
    along the u axis (and similarly for v). The number of hash cells across is:
        n_cells = 2 * uv_extent * max(sx, sy) / cell_size
    Nyquist requires 2 samples per cell, so we need 2 * n_cells texels.
    We round up to the next power of 2.
    """
    max_scale = scales.max(dim=1).values  # [N]
    n_cells = 2.0 * uv_extent * max_scale / cell_size
    # Nyquist: 2 texels per hash cell
    nyquist_samples = 2.0 * n_cells
    log2_res = torch.ceil(torch.log2(nyquist_samples.clamp(min=1.0)))
    resolutions = (2.0 ** log2_res).int()
    return resolutions.clamp(min=min_res, max=max_res)


# ---------------------------------------------------------------------------
# Atlas packing (shelf-first-fit-decreasing)
# ---------------------------------------------------------------------------
def shelf_pack_atlas(resolutions, atlas_width=4096):
    """Pack square patches into atlas. Returns (rects [N,4], height, used_rows, utilization%)."""
    N = len(resolutions)
    res_cpu = resolutions.cpu().numpy()

    # Compute needed height
    height = 0
    for sz in sorted(set(int(x) for x in res_cpu), reverse=True):
        count = int((res_cpu == sz).sum())
        per_row = atlas_width // sz
        rows = (count + per_row - 1) // per_row
        height += rows * sz
    atlas_height = max(((height + 63) // 64) * 64, 64)

    order = np.argsort(-res_cpu)  # descending by resolution
    shelves = []  # [y_start, height, x_cursor]
    rects = np.zeros((N, 4), dtype=np.float32)

    for idx in order:
        sz = int(res_cpu[idx])
        placed = False
        for shelf in shelves:
            if shelf[1] >= sz and shelf[2] + sz <= atlas_width:
                rects[idx] = [shelf[2], shelf[0], sz, sz]
                shelf[2] += sz
                placed = True
                break
        if not placed:
            y_start = max((s[0] + s[1] for s in shelves), default=0)
            if y_start + sz > atlas_height:
                rects[idx] = [0, 0, 2, 2]  # fallback
                continue
            shelves.append([y_start, sz, sz])
            rects[idx] = [0, y_start, sz, sz]

    used_rows = max((s[0] + s[1] for s in shelves), default=0)
    total_area = float(np.sum(res_cpu.astype(np.int64) ** 2))
    utilization = total_area / (atlas_width * atlas_height) * 100

    return torch.from_numpy(rects).float(), atlas_height, used_rows, utilization


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="Bake 3D_SH_res → adaptive atlas (3D RGB residual)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--uv_extent", type=float, default=4.0)
    parser.add_argument("--max_res", type=int, default=128, help="Max per-Gaussian resolution")
    parser.add_argument("--min_res", type=int, default=4, help="Min per-Gaussian resolution")
    parser.add_argument("--atlas_width", type=int, default=4096)
    parser.add_argument("--ss", type=int, default=1, help="Supersample factor")
    parser.add_argument("--output_dir", type=str, default=None)
    bake_args = parser.parse_args()

    # Load training config
    print(f"\n[BAKE] Loading config from: {bake_args.model_path}")
    with open(os.path.join(bake_args.model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = bake_args.model_path
    args.eval = True

    config_yaml_path = os.path.join(bake_args.model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    assert args.method == "3D_SH_res", f"Expected 3D_SH_res, got {args.method}"

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
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                     '_scaling', '_rotation', '_appearance_level']:
            tensor = getattr(gaussians, attr)
            setattr(gaussians, attr, tensor[valid_mask])
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask.to(gaussians._shape.device)]

    N = len(gaussians.get_xyz)
    print(f"\n[BAKE] Gaussians after pruning: {N:,}")

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

    print(f"\n[BAKE] Adaptive resolution distribution:")
    unique_res = resolutions.unique().sort().values
    for res_val in unique_res:
        count = (resolutions == res_val.item()).sum().item()
        print(f"  {res_val.item():>4}×{res_val.item():<4}: {count:>7,} Gaussians")

    # =========================================================================
    # Step 2: Pack atlas
    # =========================================================================
    atlas_rects, atlas_height, used_rows, utilization = shelf_pack_atlas(
        resolutions, atlas_width=bake_args.atlas_width)
    atlas_rects = atlas_rects.cuda()
    atlas_width = bake_args.atlas_width

    atlas_mb = atlas_height * atlas_width * 3 * 2 / 1024 / 1024
    print(f"\n[ATLAS] Packed: {atlas_width}×{atlas_height}, "
          f"used {used_rows}/{atlas_height} rows, {utilization:.1f}% util, {atlas_mb:.1f} MB")

    # Allocate atlas (FP32 for accumulation, convert to FP16 at end)
    atlas = torch.zeros(atlas_height, atlas_width, 3, device='cuda', dtype=torch.float32)

    # =========================================================================
    # Step 3: Bake hash+MLP per resolution group
    # =========================================================================
    centers = gaussians.get_xyz        # [N, 3]
    quats = gaussians.get_rotation     # [N, 4]
    scales = gaussians.get_scaling     # [N, 2]
    R0, R1 = quat_to_rotcols(quats)   # [N, 3] each

    mlp = ingp.mlp_fused
    mlp.eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_input_padded = mlp[0].weight.shape[1]
    bias_col = hash_dim

    ss = bake_args.ss

    for res_val in unique_res:
        res = res_val.item()
        bake_res = res * ss  # supersample resolution
        mask = (resolutions == res)
        indices = mask.nonzero(as_tuple=True)[0]
        n_group = len(indices)

        print(f"\n[BAKE] Resolution {res}×{res} (ss={ss}× → {bake_res}×{bake_res}): "
              f"{n_group:,} Gaussians")

        # Build UV grid for this resolution (texel-center convention)
        step = 2.0 * uv_extent / bake_res
        coords = torch.arange(bake_res, dtype=torch.float32, device='cuda')
        uv_1d = (coords + 0.5) * step - uv_extent

        # meshgrid with indexing='ij':
        #   uu[i, j] = uv_1d[i]  (i = u index, varies along dim 0)
        #   vv[i, j] = uv_1d[j]  (j = v index, varies along dim 1)
        uu, vv = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
        u_flat = uu.reshape(-1)  # [bake_res^2], flat index k = i*bake_res + j
        v_flat = vv.reshape(-1)
        n_pts = bake_res * bake_res

        # Process in chunks to limit VRAM
        max_batch = max(1, 2 * (1024**3) // (3 * 4 * n_pts))
        chunk_size = min(n_group, max_batch)

        for ci_start in range(0, n_group, chunk_size):
            ci_end = min(ci_start + chunk_size, n_group)
            batch_indices = indices[ci_start:ci_end]
            n_batch = len(batch_indices)

            c = centers[batch_indices]             # [B, 3]
            sx = scales[batch_indices, 0:1]        # [B, 1]
            sy = scales[batch_indices, 1:2]        # [B, 1]
            r0 = R0[batch_indices]                 # [B, 3]
            r1 = R1[batch_indices]                 # [B, 3]

            # xyz = center + u * sx * R0 + v * sy * R1   [B, n_pts, 3]
            xyz = (c.unsqueeze(1)
                   + u_flat.unsqueeze(0).unsqueeze(-1) * (sx.unsqueeze(1) * r0.unsqueeze(1))
                   + v_flat.unsqueeze(0).unsqueeze(-1) * (sy.unsqueeze(1) * r1.unsqueeze(1)))
            xyz_flat = xyz.reshape(-1, 3)  # [B*n_pts, 3]

            with torch.no_grad():
                hash_feat = ingp._encode_3D(xyz_flat)
                mlp_input = torch.zeros(xyz_flat.shape[0], mlp_input_padded, device='cuda')
                mlp_input[:, :hash_dim] = hash_feat[:, :hash_dim]
                mlp_input[:, bias_col] = 1.0
                mlp_out = mlp(mlp_input)
                rgb_residual = mlp_out[:, :3]  # [B*n_pts, 3]

            # Reshape to [B, bake_res, bake_res, 3] where dim1=u(i), dim2=v(j)
            residual = rgb_residual.reshape(n_batch, bake_res, bake_res, 3)

            # Box-filter downsample if ss > 1
            if ss > 1:
                residual = residual.view(n_batch, res, ss, res, ss, 3).mean(dim=(2, 4))
                # Now [B, res, res, 3] where dim1=u(i), dim2=v(j)

            # Write to atlas
            # Atlas convention: atlas[row, col, ch] where row=v, col=u
            # Bake convention: residual[b, i(u), j(v), ch]
            # Render kernel: au = u0 + (s.x+E)/(2E)*u_span - 0.5  → col from s.x
            #                av = v0 + (s.y+E)/(2E)*v_span - 0.5  → row from s.y
            # At bake texel (i, j): s.x = uv_1d[i], s.y = uv_1d[j]
            #   → atlas col = u0 + i, atlas row = v0 + j
            # So: atlas[v0+j, u0+i] = residual[b, i, j]
            #   = atlas[v0:v0+res, u0:u0+res] = residual[b].permute(1, 0, 2)  (swap u↔v)
            rects = atlas_rects[batch_indices]  # [B, 4]
            for b in range(n_batch):
                u0 = int(rects[b, 0].item())
                v0 = int(rects[b, 1].item())
                # residual[b] is [res(u), res(v), 3] → transpose to [res(v), res(u), 3] for atlas
                atlas[v0:v0+res, u0:u0+res, :] = residual[b].permute(1, 0, 2)

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

    # Atlas texture [H, W, 3] FP16
    atlas_path = os.path.join(output_dir, "atlas_texture.pt")
    torch.save(atlas.half().cpu(), atlas_path)
    print(f"[BAKE] Saved atlas_texture.pt → {atlas_path} ({atlas_mb:.1f} MB)")

    # Atlas rects [N, 4] float32
    rects_path = os.path.join(output_dir, "atlas_rects.pt")
    torch.save(atlas_rects.cpu(), rects_path)
    print(f"[BAKE] Saved atlas_rects.pt → {rects_path}")

    # Residual stats
    print(f"\n[BAKE] Atlas residual stats: mean={atlas.mean():.6f}, std={atlas.std():.6f}, "
          f"min={atlas.min():.6f}, max={atlas.max():.6f}")

    # Metadata
    res_dist = {}
    for res_val in unique_res:
        count = (resolutions == res_val.item()).sum().item()
        res_dist[str(res_val.item())] = count

    meta = {
        "texture_mode": "atlas",
        "residual_dim": 3,
        "uv_extent": uv_extent,
        "supersample": ss,
        "num_gaussians": N,
        "iteration": iteration,
        "method": args.method,
        "kernel": getattr(args, 'kernel', 'gaussian'),
        "sh_degree": 3,
        "atlas_width": atlas_width,
        "atlas_height": atlas_height,
        "max_res": bake_args.max_res,
        "min_res": bake_args.min_res,
        "cell_size": cell_size,
        "finest_resolution": finest_resolution,
        "resolution_distribution": res_dist,
    }
    meta_path = os.path.join(output_dir, "bake_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[BAKE] Saved bake_meta.json → {meta_path}")

    print(f"\n[BAKE] Done! Output: {output_dir}")


if __name__ == "__main__":
    main()

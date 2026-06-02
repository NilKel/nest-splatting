#!/usr/bin/env python3
"""
Unified bake + benchmark script for 3D_SH_res models.

Reports:
  1. Neural renderer metrics (from training_log.txt and test_metrics.txt)
  2. Baked renderer metrics (atlas bake → render test views → PSNR/SSIM/FPS)

The atlas is allocated on CPU to avoid OOM — only MLP evaluation chunks use GPU.

Usage:
    python scripts/benchmark_baked.py --model_path outputs/mip_360/bonsai/3D_SH_res/simpleres
    python scripts/benchmark_baked.py --model_path ... --max_res 32 --atlas_width 4096
"""

import os, sys, json, pickle, glob, math, re, time
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams
from utils.render_utils import save_img_u8
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_training_log(model_path):
    """Extract metrics from training_log.txt."""
    log_path = os.path.join(model_path, "training_log.txt")
    info = {}
    if not os.path.exists(log_path):
        return info
    with open(log_path) as f:
        text = f.read()
    m = re.search(r"Render FPS:\s*([\d.]+)", text)
    if m:
        info["train_fps"] = float(m.group(1))
    m = re.search(r"Time per frame:\s*([\d.]+)", text)
    if m:
        info["train_ms"] = float(m.group(1))
    m = re.search(r"Number of Gaussians:\s*([\d,]+)", text)
    if m:
        info["num_gaussians"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Resolution:\s*(\d+)x(\d+)", text)
    if m:
        info["resolution"] = f"{m.group(1)}x{m.group(2)}"
    return info


def parse_test_metrics(model_path):
    """Extract metrics from test_metrics.txt."""
    path = os.path.join(model_path, "test_metrics.txt")
    info = {}
    if not os.path.exists(path):
        return info
    with open(path) as f:
        text = f.read()
    m = re.search(r"Average PSNR:\s*([\d.]+)", text)
    if m:
        info["neural_psnr"] = float(m.group(1))
    m = re.search(r"Average SSIM:\s*([\d.]+)", text)
    if m:
        info["neural_ssim"] = float(m.group(1))
    m = re.search(r"Average LPIPS:\s*([\d.]+)", text)
    if m:
        info["neural_lpips"] = float(m.group(1))
    return info


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def quat_to_rotcols(quats):
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm
    r00 = 1 - 2*(y*y + z*z); r10 = 2*(x*y + w*z); r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z); r11 = 1 - 2*(x*x + z*z); r21 = 2*(y*z + w*x)
    return torch.stack([r00, r10, r20], dim=-1), torch.stack([r01, r11, r21], dim=-1)


# ---------------------------------------------------------------------------
# Adaptive resolution
# ---------------------------------------------------------------------------
def compute_adaptive_resolution(scales, cell_size, uv_extent=4.0, max_res=64, min_res=4,
                                 large_gauss_cap=None):
    """Per-axis Nyquist resolution: each surfel gets (res_x, res_y) where each
    axis is sized by its own scale (not the larger axis). Saves atlas texels on
    anisotropic surfels (edges, hair) without losing fidelity.
    Returns [N, 2] int tensor of (res_x, res_y) powers of two in [min_res, max_res].

    NOTE: This is the *encoding-Nyquist* path — each surfel sized to capture
    the finest hashgrid spatial frequency. See `compute_view_aware_resolution`
    for the *viewing-Nyquist* alternative, which is typically much smaller.

    `large_gauss_cap` (int, optional): scale-aware lite cap. Surfels whose
    natural per-axis res WOULD have been clamped at `max_res` (the visually-
    large blobs whose footprint covers many screen pixels) get clamped at
    `large_gauss_cap` instead. Each affected axis shrinks by `(cap/max_res)`
    linearly → `(cap/max_res)²` area, on the rects that dominate the atlas.
    Leaves the small / detail-carrying surfels at their natural res. Use for
    a real "lite" tier that targets the biggest bytes, not a blunt global
    `--max_res` halving.
    """
    # scales is [N, 2] for 2DGS surfels — column 0 = sx, column 1 = sy.
    n_cells = 2.0 * uv_extent * scales / cell_size           # [N, 2]
    nyquist_samples = 2.0 * n_cells                          # [N, 2]
    log2_res = torch.ceil(torch.log2(nyquist_samples.clamp(min=1.0)))
    resolutions = (2.0 ** log2_res).int()
    if large_gauss_cap is not None and large_gauss_cap < max_res:
        # Per-axis: an axis is "saturated" (naturally exceeds max_res) iff its
        # uncapped resolution > max_res. Saturated axes get the tighter cap;
        # unsaturated axes are unchanged. Keeps small surfels at full fidelity
        # while squeezing the 64×64 blob bucket that dominates the atlas.
        saturated = resolutions > max_res                       # [N, 2] bool
        cap_t = torch.full_like(resolutions, int(large_gauss_cap))
        resolutions = torch.where(saturated, cap_t, resolutions)
    return resolutions.clamp(min=min_res, max=max_res)       # [N, 2]


@torch.no_grad()
def compute_view_max_footprint(xyz, scaling, rotation, train_cameras, k_sigma=4.0,
                               batch_size=200_000):
    """Per-surfel max projected (w_px, h_px) across all train cameras.

    For each surfel and each camera, projects the 4 disk corners
    (center ± k*sx*R[:,0] ± k*sy*R[:,1]) through the projmatrix; takes the
    pixel-space bbox; keeps the elementwise max across cameras. Surfels never
    visible (all corners behind the camera in every view) get (0, 0).

    Returns [N, 2] float tensor of pixel counts.
    """
    from utils.general_utils import build_rotation
    N = xyz.shape[0]
    device = xyz.device
    max_w = torch.zeros(N, device=device, dtype=torch.float32)
    max_h = torch.zeros(N, device=device, dtype=torch.float32)

    R = build_rotation(rotation)                                       # [N, 3, 3]
    Tu = (k_sigma * scaling[:, 0:1]) * R[:, :, 0]                      # [N, 3]
    Tv = (k_sigma * scaling[:, 1:2]) * R[:, :, 1]                      # [N, 3]
    corners = torch.stack([
        xyz + Tu + Tv, xyz + Tu - Tv,
        xyz - Tu + Tv, xyz - Tu - Tv,
    ], dim=1)                                                          # [N, 4, 3]
    ones4 = torch.ones(N, 4, 1, device=device, dtype=corners.dtype)
    homog = torch.cat([corners, ones4], dim=-1)                        # [N, 4, 4]

    for cam in train_cameras:
        W, H = int(cam.image_width), int(cam.image_height)
        # full_proj_transform is row-major in our codebase (matches the C++
        # GLM constructor's transposed feed). Right-multiply.
        P = cam.full_proj_transform.to(device).contiguous()            # [4, 4]
        # Stream in batches to bound peak memory (corners @ P allocates [N, 4, 4]).
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            clip = homog[s:e] @ P                                      # [B, 4, 4]
            valid = clip[..., 3] > 1e-6                                # [B, 4]
            all_valid = valid.all(dim=1)                               # [B]
            w_clip = clip[..., 3].clamp(min=1e-6)
            ndc_x = clip[..., 0] / w_clip
            ndc_y = clip[..., 1] / w_clip
            sx = (ndc_x * 0.5 + 0.5) * W
            sy = (ndc_y * 0.5 + 0.5) * H
            # Clip the bbox to the image rect — surfel pieces beyond the frame
            # never need atlas detail.
            sx = sx.clamp(min=0.0, max=float(W))
            sy = sy.clamp(min=0.0, max=float(H))
            w_px = (sx.amax(dim=1) - sx.amin(dim=1))
            h_px = (sy.amax(dim=1) - sy.amin(dim=1))
            w_px = torch.where(all_valid, w_px, torch.zeros_like(w_px))
            h_px = torch.where(all_valid, h_px, torch.zeros_like(h_px))
            max_w[s:e] = torch.maximum(max_w[s:e], w_px)
            max_h[s:e] = torch.maximum(max_h[s:e], h_px)

    return torch.stack([max_w, max_h], dim=-1)                         # [N, 2]


def compute_view_aware_resolution(max_footprint_px, max_res=64, min_res=4,
                                   nyquist_factor=2.0):
    """Convert max-footprint pixel counts to atlas resolution.

    `nyquist_factor=2.0` keeps two atlas texels per pixel at the closest train
    view (Nyquist criterion). Snaps to the next power of two in [min_res, max_res].
    Surfels with zero footprint (never visible) clamp to min_res.
    """
    nyquist = nyquist_factor * max_footprint_px                         # [N, 2]
    log2_res = torch.ceil(torch.log2(nyquist.clamp(min=1.0)))
    resolutions = (2.0 ** log2_res).int()
    return resolutions.clamp(min=min_res, max=max_res)                  # [N, 2]


@torch.no_grad()
def compute_per_gaussian_contribution(gaussians, ingp, train_cameras, pipe, background,
                                      cfg_model, beta, iteration):
    """Walk all training views with the neural renderer (record_transmittance=True);
    return per-Gaussian importance = sum of alpha*T over visible pixels in all views.

    Reuses the same accumulator GSpa Phase-1 uses (`transmittance_avg` /
    `cover_pixels` from the rasterizer) — see optimizing_spa.py.
    """
    from gaussian_renderer import render
    N = gaussians.get_xyz.shape[0]
    imp = torch.zeros(N, device='cuda')
    cover = torch.zeros(N, device='cuda')
    for view in train_cameras:
        pkg = render(view, gaussians, pipe, background, beta=beta,
                     iteration=iteration, cfg=cfg_model, ingp=ingp,
                     record_transmittance=True, is_training=False)
        imp += pkg['transmittance_avg'].squeeze().to(imp.device)
        cover += pkg['cover_pixels'].squeeze().to(cover.device)
    imp[cover == 0] = 0.0
    return imp


# ---------------------------------------------------------------------------
# Atlas packing (shelf-first-fit-decreasing)
# ---------------------------------------------------------------------------
def shelf_pack_atlas(resolutions, atlas_width=4096):
    """Shelf-first-fit-decreasing on arbitrary rectangles.
    `resolutions` is [N, 2] = (res_x, res_y). Shelves are indexed by shelf_height
    (largest res_y dominates shelf height). Items ordered by res_y descending.
    Returns rects as [N, 4] = (u0, v0, w, h) in atlas pixels.
    """
    res_cpu = resolutions.cpu().numpy().astype(np.int64)   # [N, 2]
    N = res_cpu.shape[0]
    rx = res_cpu[:, 0]
    ry = res_cpu[:, 1]

    # Estimate atlas height by shelves of distinct ry values.
    height = 0
    for sz in sorted(set(int(y) for y in ry), reverse=True):
        sel = (ry == sz)
        # Widths of items that will go on shelves of this height.
        rxs = rx[sel]
        # Greedy linear tally: sum widths, divide by atlas_width, round up.
        total_w = int(rxs.sum())
        rows_for_sz = (total_w + atlas_width - 1) // atlas_width
        height += rows_for_sz * sz
    atlas_height = max(((height + 63) // 64) * 64, 64)

    # Order items by res_y desc (taller first), break ties by res_x desc.
    order = np.lexsort((-rx, -ry))

    # Each shelf: [y_start, shelf_height, next_x]
    shelves = []
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
                rects[idx] = [0, 0, 2, 2]  # fallback: tiny placeholder
                continue
            shelves.append([y_start, iy, ix])
            rects[idx] = [0, y_start, ix, iy]

    used_rows = max((s[0] + s[1] for s in shelves), default=0)
    total_area = float((rx * ry).sum())
    utilization = total_area / (atlas_width * atlas_height) * 100

    return torch.from_numpy(rects).float(), atlas_height, used_rows, utilization


# ---------------------------------------------------------------------------
# Bake: atlas on CPU, MLP chunks on GPU
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_atlas_quant_range(ingp, gaussians, uv_extent=4.0,
                                  n_gaussians=20000, n_uvs_per_gaussian=64,
                                  margin=0.02, k_sigma=None):
    """Estimate (offset, scale) for uint8/BC7 quantization of the MLP residual,
    reversible via `x ≈ q/255 * scale + offset`.

    Uses the EMPIRICAL [min, max] over a large Gaussian × UV sample (all
    Gaussians up to `n_gaussians`, `n_uvs_per_gaussian` random UVs each in the
    same [-uv_extent, uv_extent] domain the bake samples) plus a small relative
    `margin`. This guarantees no residual peak is hard-clamped at bake time —
    the old mean±6σ over ~8K texels clipped the heavy tails, and clamping the
    bright residual peaks uniformly is exactly a dull-intensity bias (not a
    structural one). Pass `k_sigma` (e.g. 6.0) to fall back to the legacy
    mean±k·σ behaviour.
    """
    from utils.general_utils import build_rotation
    N = gaussians.get_xyz.shape[0]
    K = min(n_gaussians, N)
    sample_idx = torch.randperm(N, device='cuda')[:K]
    centers = gaussians.get_xyz[sample_idx]
    scales = gaussians.get_scaling[sample_idx]
    R = build_rotation(gaussians.get_rotation[sample_idx])
    R0, R1 = R[:, :, 0], R[:, :, 1]

    uv = (torch.rand(K, n_uvs_per_gaussian, 2, device='cuda') * 2 - 1) * uv_extent
    xyz = (centers.unsqueeze(1)
           + uv[..., 0:1] * (scales[:, 0:1].unsqueeze(1) * R0.unsqueeze(1))
           + uv[..., 1:2] * (scales[:, 1:2].unsqueeze(1) * R1.unsqueeze(1)))
    xyz_flat = xyz.reshape(-1, 3)

    mlp = ingp.mlp_fused.half().eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_input_padded = mlp[0].weight.shape[1]

    # Chunked forward — K·n_uvs can be ~1M+ rows at full sampling.
    CH = 262144
    lo = torch.tensor(float('inf'), device='cuda')
    hi = torch.tensor(float('-inf'), device='cuda')
    s1 = torch.zeros((), device='cuda'); s2 = torch.zeros((), device='cuda')
    cnt = 0
    for i in range(0, xyz_flat.shape[0], CH):
        xb = xyz_flat[i:i + CH]
        hf = ingp._encode_3D(xb)
        mi = torch.zeros(xb.shape[0], mlp_input_padded, device='cuda', dtype=torch.float16)
        mi[:, :hash_dim] = hf[:, :hash_dim].to(torch.float16)
        rb = mlp(mi)[:, :3].to(torch.float32)
        lo = torch.minimum(lo, rb.min()); hi = torch.maximum(hi, rb.max())
        s1 += rb.sum(); s2 += (rb * rb).sum(); cnt += rb.numel()

    lo = lo.item(); hi = hi.item()
    mean = (s1 / cnt).item()
    std = max((s2 / cnt - mean * mean), 0.0) ** 0.5
    n_tex = K * n_uvs_per_gaussian

    if k_sigma is not None:
        offset = mean - k_sigma * std
        scale = max(2.0 * k_sigma * std, 1e-6)
        print(f"[BAKE] (legacy k_sigma={k_sigma}) range over {n_tex:,} texels: "
              f"mean={mean:.4f} std={std:.4f} → offset={offset:.4f} scale={scale:.4f}")
        return offset, scale

    span = max(hi - lo, 1e-6)
    offset = lo - margin * span
    scale = span * (1.0 + 2.0 * margin)
    # How much the OLD mean±6σ range would have clipped, for visibility.
    old_lo, old_hi = mean - 6.0 * std, mean + 6.0 * std
    clip_frac = ((lo < old_lo) or (hi > old_hi))
    print(f"[BAKE] Empirical residual range over {n_tex:,} texels: "
          f"min={lo:.4f} max={hi:.4f} (mean={mean:.4f} std={std:.4f})")
    print(f"[BAKE]   → offset={offset:.4f} scale={scale:.4f} "
          f"(±{margin*100:.0f}% margin; spans full [min,max], no clamp)")
    if clip_frac:
        print(f"[BAKE]   NOTE old mean±6σ=[{old_lo:.4f},{old_hi:.4f}] would have "
              f"clipped the true [{lo:.4f},{hi:.4f}] tails → dull-intensity bias.")
    return offset, scale


def bake_atlas(ingp, gaussians, uv_extent, max_res, min_res, atlas_width, ss,
               atlas_budget_mb=2048,
               train_cameras=None, view_aware=False,
               prune_low_contrib=0.0, skip_texture_low_contrib=0.0,
               prune_thresh=-1.0, skip_texture_thresh=-1.0,
               importance_render_args=None,
               budget_mode="uniform",
               importance_for_budget=None,
               bake_dtype="fp16",
               large_gauss_cap=None):
    """Bake hash MLP residual into a CPU-resident atlas. Returns (atlas_cpu, atlas_rects, meta).

    Resolution selection:
      - `view_aware=False` (default): hashgrid-Nyquist (encoding frequency).
      - `view_aware=True`: per-surfel max projected footprint across all train cams
        (viewing frequency). Requires `train_cameras`.

    Pruning / texture-skip (require `train_cameras` + `importance_render_args`):
      - `prune_low_contrib > 0`: drop the bottom fraction of Gaussians by accumulated
        alpha*T importance across train views. Modifies `gaussians` in place.
      - `skip_texture_low_contrib > 0`: among survivors, mark the bottom fraction
        as zero-rect → renderer falls through to SH-only (no atlas lookup, no
        per-Gaussian texels). Stacks on top of `prune_low_contrib`.
    """

    hash_encoding = ingp.hash_encoding
    embeddings, offsets, num_levels, per_level_scale, base_resolution, align_corners, interp_id = hash_encoding.get_params()
    voxel_min = ingp.voxel_range[0]
    voxel_max = ingp.voxel_range[1]
    finest_resolution = base_resolution * (per_level_scale ** (num_levels - 1))
    cell_size = (voxel_max - voxel_min) / finest_resolution

    print(f"[BAKE] Hash grid: {num_levels} levels, finest_res={finest_resolution:.0f}, "
          f"cell_size={cell_size:.6f}")

    # Optional importance-based pruning + skip-texture (BEFORE resolution sizing,
    # so we don't pay to compute footprints for soon-to-be-deleted Gaussians).
    # Two scoring modes coexist:
    #   - fraction: prune_low_contrib / skip_texture_low_contrib (0..1)
    #   - threshold: prune_thresh / skip_texture_thresh (absolute alpha*T)
    # If both are set for the same stage, the threshold takes precedence.
    def _bake_time_prune(gaussians, keep_mask):
        # Mirror the dead-Gaussian prune in main(): edit attributes in place
        # because the optimizer doesn't exist at bake time, so prune_points
        # would AttributeError.
        keep_mask_dev = keep_mask
        attrs = ['_xyz', '_features_dc', '_features_rest', '_opacity',
                 '_scaling', '_rotation', '_appearance_level',
                 '_gaussian_features', '_shape', '_flex_beta',
                 '_sb_params', '_sg_directions', '_sg_sharpness_sg', '_sg_rgb',
                 '_sv_sites', '_sv_colors', '_sv_tau', '_sv_dc',
                 '_gamma', '_adaptive_features',
                 '_adaptive_cat_weight', '_adaptive_zero_weight', '_gate_logits',
                 # `--method mixed[_3d]`: keep the per-Gauss textured flag AND
                 # the EWA 3rd-axis scale aligned through bake-time pruning
                 # (else they desync from _xyz / each other).
                 '_is_textured', '_scaling_z']
        n_total = keep_mask.shape[0]
        for attr in attrs:
            tensor = getattr(gaussians, attr, None)
            if tensor is not None and tensor.numel() > 0 and tensor.shape[0] == n_total:
                setattr(gaussians, attr, tensor[keep_mask_dev.to(tensor.device)])

    skip_texture_mask = None
    need_imp = (prune_low_contrib > 0.0 or skip_texture_low_contrib > 0.0
                or prune_thresh >= 0.0 or skip_texture_thresh >= 0.0)
    if need_imp:
        if train_cameras is None or importance_render_args is None:
            raise RuntimeError("Importance-based pruning/skipping requires "
                               "train_cameras + importance_render_args.")
        n_before = gaussians.get_xyz.shape[0]
        print(f"[BAKE] Scoring importance over {len(train_cameras)} train views...")
        imp = compute_per_gaussian_contribution(
            gaussians, ingp, train_cameras, **importance_render_args)
        imp_max = imp.max().item()
        imp_med = imp.median().item()
        imp_zero = int((imp == 0).sum())
        print(f"[BAKE] Importance: max={imp_max:.4g}  median={imp_med:.4g}  "
              f"never-visible={imp_zero:,} / {n_before:,}")
        # 1) Prune.
        if prune_thresh >= 0.0:
            keep_mask = imp > prune_thresh
            print(f"[BAKE] prune_thresh={prune_thresh:.4g}: "
                  f"{n_before:,} -> {int(keep_mask.sum()):,} "
                  f"(dropped {n_before - int(keep_mask.sum()):,})")
            _bake_time_prune(gaussians, keep_mask)
            imp = imp[keep_mask]
        elif prune_low_contrib > 0.0:
            n_keep = int(round(n_before * (1.0 - prune_low_contrib)))
            n_keep = max(n_keep, 1)
            thresh = torch.kthvalue(imp, n_before - n_keep).values.item() \
                if n_before > n_keep else -1.0
            keep_mask = imp > thresh
            if keep_mask.sum().item() > n_keep:
                ties = (imp == thresh).nonzero(as_tuple=True)[0]
                drop_n = int(keep_mask.sum().item() - n_keep)
                keep_mask[ties[:drop_n]] = False
            print(f"[BAKE] prune_low_contrib={prune_low_contrib:.3f}: "
                  f"{n_before:,} -> {int(keep_mask.sum()):,} "
                  f"(dropped {n_before - int(keep_mask.sum()):,})")
            _bake_time_prune(gaussians, keep_mask)
            imp = imp[keep_mask]
        # 2) Skip-texture (after pruning so the threshold is applied to survivors).
        if skip_texture_thresh >= 0.0:
            skip_texture_mask = imp <= skip_texture_thresh
            print(f"[BAKE] skip_texture_thresh={skip_texture_thresh:.4g}: "
                  f"{int(skip_texture_mask.sum()):,} survivors get SH-only (zero rect)")
        elif skip_texture_low_contrib > 0.0:
            n_now = imp.shape[0]
            n_skip = int(round(n_now * skip_texture_low_contrib))
            if n_skip > 0:
                _, skip_idx = torch.topk(imp, n_skip, largest=False)
                skip_texture_mask = torch.zeros(n_now, dtype=torch.bool, device=imp.device)
                skip_texture_mask[skip_idx] = True
                print(f"[BAKE] skip_texture_low_contrib={skip_texture_low_contrib:.3f}: "
                      f"{int(skip_texture_mask.sum()):,} survivors get SH-only (zero rect)")

    # Compute ideal per-axis resolutions, then shrink to fit atlas budget.
    # resolutions is [N, 2] = (res_x, res_y).
    res_hash = compute_adaptive_resolution(
        gaussians.get_scaling, cell_size, uv_extent=uv_extent,
        max_res=max_res, min_res=min_res,
        large_gauss_cap=large_gauss_cap)
    if view_aware:
        if train_cameras is None:
            raise RuntimeError("--view_aware_res requires train_cameras to be passed.")
        print(f"[BAKE] Walking {len(train_cameras)} train views for max projected footprints...")
        max_footprint = compute_view_max_footprint(
            gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation,
            train_cameras, k_sigma=4.0)
        res_view = compute_view_aware_resolution(
            max_footprint, max_res=max_res, min_res=min_res, nyquist_factor=2.0)
        # When the closest pixel footprint > hashgrid cell size (i.e.
        # res_view < res_hash), the viewer can't resolve sub-cell detail —
        # drop atlas resolution to res_view to save memory. When pixels are
        # smaller than cells (res_view >= res_hash), we keep res_hash because
        # the bandlimited hashgrid signal has nothing finer to give. Net:
        # take the elementwise min. Trade-off: when res_view < res_hash, the
        # bake point-samples a richer-than-Nyquist signal → mild aliasing
        # in the atlas. Visible as noise/moiré on distant surfels but usually
        # imperceptible because viewer's pixels are bigger than the aliased
        # texels anyway.
        resolutions = torch.minimum(res_view, res_hash)
        n_unseen = int((max_footprint.amax(dim=1) <= 0.0).sum())
        if n_unseen > 0:
            print(f"[BAKE] {n_unseen:,} Gaussians never visible from any train cam "
                  f"(footprint=0) → assigned min_res={min_res} as a safety floor.")
        med_w = max_footprint[:, 0].median().item()
        med_h = max_footprint[:, 1].median().item()
        view_lim = (res_view < res_hash).any(dim=1).sum().item()
        hash_lim = (res_hash <= res_view).all(dim=1).sum().item()
        print(f"[BAKE] Footprint median (px): w={med_w:.1f} h={med_h:.1f} | "
              f"view-limited (smaller atlas): {view_lim:,}  "
              f"hashgrid-limited: {hash_lim:,}")
    else:
        resolutions = res_hash

    # Budget allocation. Two strategies:
    #   uniform   : iteratively halve the global max_res until atlas fits.
    #               Penalizes important and unimportant Gaussians equally.
    #   importance: greedy fill in descending importance order; tail goes
    #               to zero-rect (SH-only). Requires per-Gaussian importance
    #               (auto-computed if not provided).
    if budget_mode == "importance" and atlas_budget_mb > 0:
        if importance_for_budget is None:
            if train_cameras is None or importance_render_args is None:
                raise RuntimeError(
                    "--bake_budget_mode importance requires train_cameras + "
                    "importance_render_args (or pre-computed importance).")
            print(f"[BAKE] (budget-mode=importance) scoring "
                  f"{gaussians.get_xyz.shape[0]:,} Gaussians over {len(train_cameras)} train views...")
            importance_for_budget = compute_per_gaussian_contribution(
                gaussians, ingp, train_cameras, **importance_render_args)
        budget_texels = atlas_budget_mb * 1024 * 1024 // 6  # FP16 RGB → 6 B/texel
        texels = resolutions[:, 0].long() * resolutions[:, 1].long()
        # Graceful degradation: walk in importance order, give each Gaussian
        # the largest power-of-2 resolution that still fits the remaining
        # budget. Tail (won't fit even at min_res) → zero-rect (SH-only).
        order = torch.argsort(importance_for_budget - 1e-9 * texels.float(),
                              descending=True).cpu().numpy()
        res_np = resolutions.cpu().numpy().astype(np.int64).copy()
        budget_left = int(budget_texels)
        n_full, n_reduced, n_skip = 0, 0, 0
        for idx in range(order.shape[0]):
            i = int(order[idx])
            rx, ry = int(res_np[i, 0]), int(res_np[i, 1])
            req_rx, req_ry = rx, ry
            placed = False
            while rx >= min_res and ry >= min_res:
                cost = rx * ry
                if cost <= budget_left:
                    res_np[i, 0] = rx
                    res_np[i, 1] = ry
                    budget_left -= cost
                    placed = True
                    break
                if rx == min_res and ry == min_res:
                    break
                rx = max(rx // 2, min_res)
                ry = max(ry // 2, min_res)
            if placed:
                if rx == req_rx and ry == req_ry:
                    n_full += 1
                else:
                    n_reduced += 1
            else:
                res_np[i, 0] = 0
                res_np[i, 1] = 0
                n_skip += 1
        resolutions = torch.from_numpy(res_np).to(resolutions.device)
        used_texels = int((resolutions[:, 0].long() * resolutions[:, 1].long()).sum())
        used_mb = used_texels * 6 / (1024**2)
        print(f"[BAKE] (budget-mode=importance) atlas budget {atlas_budget_mb} MB → "
              f"used {used_mb:.1f} MB | full-res: {n_full:,}  reduced: {n_reduced:,}  "
              f"skip-texture: {n_skip:,}")
    else:
        effective_max = max_res
        while atlas_budget_mb > 0 and effective_max > min_res:
            clamped = resolutions.clamp(max=effective_max)
            total_texels = (clamped[:, 0].long() * clamped[:, 1].long()).sum().item()
            atlas_size_mb = total_texels * 3 * 2 / (1024 * 1024)  # FP16, 3 channels
            if atlas_size_mb <= atlas_budget_mb:
                break
            effective_max //= 2

        if effective_max != max_res:
            print(f"[BAKE] (budget-mode=uniform) clamped max_res {max_res} -> {effective_max} "
                  f"to fit {atlas_budget_mb} MB")
            resolutions = resolutions.clamp(min=min_res, max=effective_max)

    # `--method mixed`: untextured surfels have NO MLP residual to bake — they
    # render SV-only (simple 2DGS). That is exactly the skip-texture (zero-rect)
    # behavior, so fold ~_is_textured into the skip-texture mask. The bake-render
    # kernel already treats a zero rect as "no atlas lookup → SV/SH baseline
    # only", which IS the untextured render path. No kernel is_textured plumbing
    # needed on the render side; the rect geometry carries the signal.
    _it = getattr(gaussians, '_is_textured', None)
    _n_now = gaussians.get_xyz.shape[0]
    if _it is not None and _it.numel() == _n_now and not bool(_it.all()):
        _untex = ~_it.to(resolutions.device)
        if skip_texture_mask is None:
            skip_texture_mask = _untex.clone()
        else:
            skip_texture_mask = skip_texture_mask | _untex
        print(f"[BAKE] --method mixed: {int(_untex.sum()):,} untextured surfels "
              f"→ zero-rect (SV-only, no residual baked)")

    # Apply skip-texture mask: zero out resolutions for low-contributors so
    # the shelf-packer assigns them no atlas pixels and the bake loop emits
    # nothing. The renderer treats atlas_rect.w*h == 0 as "SH only".
    if skip_texture_mask is not None:
        resolutions = resolutions.clone()
        resolutions[skip_texture_mask] = 0

    # Print distribution grouped by (res_x, res_y) pair.
    print(f"[BAKE] Adaptive resolution distribution (res_x × res_y):")
    # Unique pair keys = res_x * 10000 + res_y (both fit in 4 digits since <= 128).
    res_cpu = resolutions.cpu()
    pair_keys = (res_cpu[:, 0].long() * 10000 + res_cpu[:, 1].long())
    unique_keys, counts = torch.unique(pair_keys, return_counts=True)
    # Sort by resolution area desc.
    areas = (unique_keys // 10000) * (unique_keys % 10000)
    sort_idx = torch.argsort(-areas)
    for k_idx in sort_idx:
        key = unique_keys[k_idx].item()
        cnt = counts[k_idx].item()
        rx_v, ry_v = key // 10000, key % 10000
        print(f"  {rx_v:>4}x{ry_v:<4}: {cnt:>7,} Gaussians")

    # Auto-grow atlas_width if a default-width shelf-pack would push atlas_height
    # past CUDA's cudaArray 2D max (65536 in either dim). Target a 60k margin and
    # snap to mults of 64 for alignment. Required for BC7 cudaArray on dense
    # scenes (bicycle/treehill/garden at full Nyquist on 0w0g configs).
    SAFE_HEIGHT = 60000
    total_texels = int((resolutions[:, 0].long() * resolutions[:, 1].long()).sum().item())
    needed_width = (int(total_texels * 1.1) + SAFE_HEIGHT - 1) // SAFE_HEIGHT
    needed_width = ((needed_width + 63) // 64) * 64
    needed_width = min(max(needed_width, atlas_width), 65536)
    if needed_width != atlas_width:
        print(f"[BAKE] Auto-grew atlas_width {atlas_width} → {needed_width} "
              f"so atlas_height stays ≤ {SAFE_HEIGHT} (cudaArray 2D dim cap).")
        atlas_width = needed_width
    atlas_rects, atlas_height, used_rows, utilization = shelf_pack_atlas(
        resolutions, atlas_width=atlas_width)
    atlas_mb = atlas_height * atlas_width * 3 * 2 / 1024 / 1024
    print(f"[ATLAS] Packed: {atlas_width}x{atlas_height}, "
          f"used {used_rows}/{atlas_height} rows, {utilization:.1f}% util, {atlas_mb:.1f} MB (FP16)")

    # Atlas dtype + quantization range (used by uint8 / BC7 paths).
    atlas_offset, atlas_scale = 0.0, 1.0
    if bake_dtype in ("uint8", "bc7"):
        atlas_offset, atlas_scale = precompute_atlas_quant_range(
            ingp, gaussians, uv_extent=uv_extent)
        atlas_cpu = torch.zeros(atlas_height, atlas_width, 3, dtype=torch.uint8)
        print(f"[BAKE] Atlas dtype: uint8  ({atlas_height * atlas_width * 3 / (1024**2):.0f} MB peak CPU RAM)")
    else:
        # Default FP16 atlas — residuals are small (mean ~0.01) so FP16 is fine.
        atlas_cpu = torch.zeros(atlas_height, atlas_width, 3, dtype=torch.float16)
        print(f"[BAKE] Atlas dtype: fp16  ({atlas_height * atlas_width * 6 / (1024**2):.0f} MB peak CPU RAM)")

    centers = gaussians.get_xyz
    quats = gaussians.get_rotation
    scales = gaussians.get_scaling
    R0, R1 = quat_to_rotcols(quats)

    # Cast the training MLP to FP16 to match the in-kernel __half2 math the
    # training-time forward uses (float2half_kernel uploads + FP16 GEMM).
    # Without this, Python evaluates in FP32 and the residual numerically drifts
    # from what the training render kernel produced.
    mlp = ingp.mlp_fused.half().eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_input_padded = mlp[0].weight.shape[1]
    bias_col = hash_dim

    # Iterate per unique (res_x, res_y) pair; Gaussians within the same pair
    # share the UV lattice shape, so the MLP eval can be batched together.
    unique_pairs = set(
        (int(resolutions[i, 0].item()), int(resolutions[i, 1].item()))
        for i in range(resolutions.shape[0])
    )
    for (res_x, res_y) in sorted(unique_pairs, key=lambda p: -(p[0] * p[1])):
        # Skip-texture Gaussians get (0, 0) resolution → no atlas data needed.
        if res_x == 0 or res_y == 0:
            continue
        bake_res_x = res_x * ss
        bake_res_y = res_y * ss
        mask = (resolutions[:, 0] == res_x) & (resolutions[:, 1] == res_y)
        indices = mask.nonzero(as_tuple=True)[0]
        n_group = len(indices)

        print(f"[BAKE] {res_x}x{res_y} (ss={ss}x -> {bake_res_x}x{bake_res_y}): {n_group:,} Gaussians")

        step_x = 2.0 * uv_extent / bake_res_x
        step_y = 2.0 * uv_extent / bake_res_y
        u_coords = (torch.arange(bake_res_x, dtype=torch.float32, device='cuda') + 0.5) * step_x - uv_extent
        v_coords = (torch.arange(bake_res_y, dtype=torch.float32, device='cuda') + 0.5) * step_y - uv_extent
        uu, vv = torch.meshgrid(u_coords, v_coords, indexing='ij')
        u_flat = uu.reshape(-1)
        v_flat = vv.reshape(-1)
        n_pts = bake_res_x * bake_res_y

        # Chunk to limit GPU memory (~4 GB budget)
        # Per Gaussian: xyz(12) + hash(16) + mlp_in(64) + mlp_out(64) + residual(12) ≈ 168 bytes per texel
        bytes_per_gaussian = 168 * n_pts
        gpu_budget = 4 * (1024**3)
        max_batch = max(1, gpu_budget // max(bytes_per_gaussian, 1))
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
                # 3D_SH_res has NO bias column in its input — training builds the
                # MLP with `bias=False` on every layer and pads the unused tail
                # of the input with zeros (see hash_encoder/modules.py:336 "no bias
                # in input"). Writing 1.0 anywhere here would diverge from the
                # training-time forward pass. (Earlier code wrote 1.0 at hash_dim
                # which silently corrupted bakes for hash_dim < 16.)
                mlp_out = mlp(mlp_input)  # FP16 in, FP16 out
                rgb_residual = mlp_out[:, :3].to(torch.float32)

            residual = rgb_residual.reshape(n_batch, bake_res_x, bake_res_y, 3)
            if ss > 1:
                residual = residual.view(n_batch, res_x, ss, res_y, ss, 3).mean(dim=(2, 4))

            # Quantize per-chunk if bake_dtype is uint8 — keeps peak CPU RAM at
            # `atlas_h * atlas_w * 3 B` (3× smaller than FP16) instead of holding
            # the full FP16 atlas alive until the end of the bake loop.
            if bake_dtype in ("uint8", "bc7"):
                q = ((residual - atlas_offset) / atlas_scale * 255.0).clamp(0, 255)
                residual_cpu = q.to(torch.uint8).cpu()
            else:
                residual_cpu = residual.half().cpu()
            rects = atlas_rects[batch_indices.cpu()]
            for b in range(n_batch):
                u0 = int(rects[b, 0].item())
                v0 = int(rects[b, 1].item())
                # UV lattice is (res_x, res_y); atlas stores as [row=v, col=u],
                # so transpose the (u, v) axes when copying.
                atlas_cpu[v0:v0+res_y, u0:u0+res_x, :] = residual_cpu[b].permute(1, 0, 2)

            del xyz, xyz_flat, hash_feat, mlp_input, mlp_out, rgb_residual, residual
            torch.cuda.empty_cache()

    if atlas_cpu.dtype == torch.uint8:
        # Dequantize back to float to report stats in residual units (matches FP16 path).
        a32 = atlas_cpu.float() / 255.0 * atlas_scale + atlas_offset
        print(f"[BAKE] Atlas residual stats (uint8 → float): "
              f"mean={a32.mean():.6f}, std={a32.std():.6f}, "
              f"min={a32.min():.6f}, max={a32.max():.6f}")
    else:
        print(f"[BAKE] Atlas residual stats: mean={atlas_cpu.mean():.6f}, "
              f"std={atlas_cpu.std():.6f}, min={atlas_cpu.min():.6f}, max={atlas_cpu.max():.6f}")

    res_dist = {}
    for (rx_v, ry_v) in unique_pairs:
        cnt = int(((resolutions[:, 0] == rx_v) & (resolutions[:, 1] == ry_v)).sum().item())
        res_dist[f"{rx_v}x{ry_v}"] = cnt
    meta = {
        "texture_mode": "atlas",
        "residual_dim": 3,
        "uv_extent": uv_extent,
        "supersample": ss,
        "num_gaussians": len(gaussians.get_xyz),
        "method": "3D_SH_res",
        "atlas_width": atlas_width,
        "atlas_height": atlas_height,
        "max_res": max_res,
        "min_res": min_res,
        "cell_size": cell_size,
        "finest_resolution": finest_resolution,
        "resolution_distribution": res_dist,
        "atlas_dtype": bake_dtype,
        "atlas_offset": float(atlas_offset),
        "atlas_scale": float(atlas_scale),
    }

    return atlas_cpu, atlas_rects, meta


# ---------------------------------------------------------------------------
# Render baked model
# ---------------------------------------------------------------------------
def _make_sv_state(gaussians):
    """Snapshot pre-activated SV tensors for the fused-CUDA voronoi path.

    Returns a dict `{sites, tau, colors, K}` ready to pass to the rasterizer
    via `voronoi_sites/voronoi_tau/voronoi_colors/voronoi_K`. The CUDA kernel
    (`computeColorFromVoronoi`) expects normalized sites + post-exp tau +
    raw colors (ReLU applied inside the kernel via the existing sh_bias).

    Eval-time site mask: bit-equivalent to the torch path's
    `where(mask, dir, vec3(1e8))` trick — masked sites get distance ≈ 1e8
    so their softmax weight ≈ 0.
    """
    sv_sites = gaussians._sv_sites                            # [N, K, 3]
    if sv_sites.numel() == 0:
        return None
    K = sv_sites.shape[1]

    sites_n = torch.nn.functional.normalize(sv_sites, dim=-1)
    sv_tau_raw = getattr(gaussians, '_sv_tau', None)
    if sv_tau_raw is not None and sv_tau_raw.numel() > 0:
        tau = torch.exp(sv_tau_raw)                           # [N, K]
    else:
        tau = torch.norm(sv_sites, dim=-1)                    # legacy: ||sites||

    sv_mask = getattr(gaussians, '_sv_mask', None)
    apply_mask = (
        sv_mask is not None
        and (not getattr(gaussians, '_sv_training_flag', True))
        and sv_mask.shape[0] == sv_sites.shape[0]
    )
    if apply_mask:
        # Bake the mask into sites: dead sites → vec3(1e8) so their
        # softmax-distance logit ≈ -inf. Same trick as eval_voronoi_sv_feat.
        far = torch.full_like(sites_n, 1e8)
        sites_n = torch.where(sv_mask.unsqueeze(-1), sites_n, far)

    sv_dc_param = getattr(gaussians, '_sv_dc', None)
    if sv_dc_param is not None and sv_dc_param.numel() > 0:
        # --sv_dc legacy mode: bake the DC into all K colors so
        # softmax(W) · (colors+dc) = (softmax(W) · colors) + dc (since W sums to 1).
        colors = (gaussians._sv_colors + sv_dc_param.unsqueeze(1)).contiguous()
    else:
        colors = gaussians._sv_colors.contiguous()

    return {
        'sites': sites_n.contiguous(),                        # [N, K, 3]
        'tau':   tau.contiguous(),                            # [N, K]
        'colors': colors,                                     # [N, K, 3]
        'K':     int(K),
    }


def _make_sv_eval(gaussians):
    """Legacy Option-A torch path (per-frame fake-SH-DC). Kept for parity-
    debug only — production benches go through `_make_sv_state` + the fused
    CUDA voronoi kernel.
    """
    from gaussian_renderer import eval_voronoi_sv_feat

    SH_C0 = 0.28209479177387814

    # Snapshot the SV tensors once — they're constant for the whole render.
    sv_sites = gaussians._sv_sites
    sv_colors = gaussians._sv_colors
    sv_tau_raw = getattr(gaussians, '_sv_tau', None)
    sv_tau = (torch.exp(sv_tau_raw)
              if sv_tau_raw is not None and sv_tau_raw.numel() > 0
              else None)
    sv_dc_param = getattr(gaussians, '_sv_dc', None)
    sv_dc = (sv_dc_param
             if sv_dc_param is not None and sv_dc_param.numel() > 0
             else None)

    # Eval-time site mask (set by gaussians.update_sites_mask()) — applied
    # only at inference, mirroring nest's `_nn_sites[~mask] = 1e8` trick.
    sv_mask = getattr(gaussians, '_sv_mask', None)
    apply_mask = (
        sv_mask is not None
        and (not getattr(gaussians, '_sv_training_flag', True))
        and sv_mask.shape[0] == sv_sites.shape[0]
    )
    sites_mask_tensor = sv_mask if apply_mask else None

    means3D = gaussians.get_xyz                      # [N, 3]
    N = means3D.shape[0]
    M = (gaussians.active_sh_degree + 1) ** 2

    def _eval(viewpoint_camera):
        cam_center = viewpoint_camera.camera_center.to(means3D.device)
        view_dirs = means3D - cam_center.unsqueeze(0)
        view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        feat = eval_voronoi_sv_feat(
            sv_sites, sv_colors, view_dirs,
            sv_tau=sv_tau, sites_mask=sites_mask_tensor)
        if sv_dc is not None:
            feat = feat + sv_dc

        # SH-DC slot trick: rasterizer evaluates SH_C0 * fake_dc + sh_bias =
        # feat + sh_bias, then clamps. Higher orders stay zero.
        fake = torch.zeros((N, M, 3), dtype=feat.dtype, device=feat.device)
        fake[:, 0, :] = feat / SH_C0
        return fake.contiguous()

    return _eval


def render_baked(viewpoint_camera, gaussian_pkg, background,
                 beta=0.0, sh_degree=3, aabb_mode=3, sort_mode=0,
                 atlas_texture=None, atlas_rects=None, atlas_width=0,
                 sb_params=None, sb_number=0,
                 sv_state=None, final_relu=False):
    """Render one view. `gaussian_pkg` is the dict from
    `prepare_gaussian_inputs(gaussians, ...)` — pre-activated tensors that are
    constant for the whole scene.

    `sort_mode`: 0 = legacy 64-bit single sort, 1 = FastGS two-stage sort.

    `sv_state`: optional dict `{sites, tau, colors, K}` from `_make_sv_state`.
    When set, the rasterizer's `voronoi_*` inputs activate the fused CUDA
    SV path — `preprocessCUDA` calls `computeColorFromVoronoi` directly,
    skipping the SH branch.  `gaussian_pkg['shs']` is still passed (and
    ignored in CUDA when voronoi_K > 0) so the wrapper's "exactly one of
    shs/colors_precomp" guard stays happy.
    """
    from diff_surfel_bake_render import get_rasterizer

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    rasterizer = get_rasterizer(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=background,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        sh_degree=sh_degree, beta=beta, aabb_mode=aabb_mode, sort_mode=sort_mode,
    )

    if sv_state is not None:
        v_sites, v_tau, v_colors, v_K = (
            sv_state['sites'], sv_state['tau'], sv_state['colors'], sv_state['K'])
    else:
        v_sites = v_tau = v_colors = None
        v_K = 0

    color, _ = rasterizer(
        means3D=gaussian_pkg['means3D'],
        opacities=gaussian_pkg['opacities'],
        shs=gaussian_pkg['shs'],
        scales=gaussian_pkg['scales'],
        rotations=gaussian_pkg['rotations'],
        shapes=gaussian_pkg['shapes'],
        kernel_type=gaussian_pkg['kernel_type'],
        atlas_texture=atlas_texture,
        atlas_rects=atlas_rects,
        atlas_width=atlas_width,
        sb_params=sb_params,
        sb_number=sb_number,
        voronoi_sites=v_sites,
        voronoi_tau=v_tau,
        voronoi_colors=v_colors,
        voronoi_K=v_K,
        # `--method mixed_3d`: untextured surfels render as EWA 3D ellipsoids
        # (FastGS conic). None ⇒ pure 2DGS bake (every other method unchanged).
        is_textured=gaussian_pkg.get('is_textured'),
        scaling_z=gaussian_pkg.get('scaling_z'),
    )
    # `--method mixed` (residual_mode==2): the kernel emits a signed per-Gauss
    # color (no per-Gauss outer ReLU); the ReLU is on the FINAL blended pixel,
    # matching diff_surfel_mixed training (torch.relu on the rendered image).
    if final_relu:
        color = torch.relu(color)
    return color


def evaluate_baked(test_cameras, gaussians, bg_color, beta, kernel_type,
                   atlas_texture=None, atlas_rects=None,
                   atlas_width=0, num_warmup=10, num_benchmark=100, save_dir=None,
                   aabb_mode=3, sort_mode=0,
                   sb_params=None, sb_number=0,
                   sv_state=None, final_relu=False):
    """Render all test views, compute metrics, benchmark FPS.

    `sv_state`: optional dict from `_make_sv_state` — activates the fused
    CUDA voronoi color path. None ⇒ standard SH path.
    """
    from diff_surfel_bake_render import prepare_gaussian_inputs

    # Snapshot post-activation tensors once for the whole scene.
    gaussian_pkg = prepare_gaussian_inputs(
        gaussians, sh_degree=gaussians.active_sh_degree, kernel_type=kernel_type)

    psnrs, l1s, ssims_list, lpips_list = [], [], [], []
    kwargs = dict(atlas_texture=atlas_texture, atlas_rects=atlas_rects,
                  atlas_width=atlas_width, aabb_mode=aabb_mode, sort_mode=sort_mode,
                  sb_params=sb_params, sb_number=sb_number,
                  sh_degree=gaussians.active_sh_degree,
                  sv_state=sv_state, final_relu=final_relu)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for cam in test_cameras:
            rendered = render_baked(cam, gaussian_pkg, bg_color,
                                    beta=beta, **kwargs)
            gt = cam.original_image[:3].cuda()
            psnrs.append(psnr(rendered, gt).mean().item())
            l1s.append(l1_loss(rendered, gt).item())
            ssims_list.append(ssim(rendered, gt).item())
            # LPIPS expects [B, 3, H, W] in [0, 1]; rendered/gt are already [3, H, W].
            lpips_list.append(lpips(rendered.clamp(0, 1).unsqueeze(0),
                                    gt.clamp(0, 1).unsqueeze(0),
                                    net_type='vgg').item())

            if save_dir is not None:
                img_np = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                save_img_u8(img_np, os.path.join(save_dir, f"{cam.image_name}.png"))

        # FPS warmup
        for i in range(num_warmup):
            cam = test_cameras[i % len(test_cameras)]
            _ = render_baked(cam, gaussian_pkg, bg_color,
                             beta=beta, **kwargs)
        torch.cuda.synchronize()

        # FPS benchmark using CUDA events. We time each frame with its own
        # event pair so the measurement excludes Python-side wall-clock noise.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmark)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmark)]
        for i in range(num_benchmark):
            cam = test_cameras[i % len(test_cameras)]
            starts[i].record()
            _ = render_baked(cam, gaussian_pkg, bg_color,
                             beta=beta, **kwargs)
            ends[i].record()
        torch.cuda.synchronize()
        # elapsed_time returns ms; convert to seconds for parity with old API.
        times_ms = [starts[i].elapsed_time(ends[i]) for i in range(num_benchmark)]
        mean_ms = float(np.mean(times_ms))

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims_list)),
        "lpips": float(np.mean(lpips_list)),
        "l1": float(np.mean(l1s)),
        "fps": float(1000.0 / mean_ms),
        "ms_per_frame": mean_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="Bake + benchmark 3D_SH_res model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--uv_extent", type=float, default=4.0)
    parser.add_argument("--max_res", type=int, default=128)
    parser.add_argument("--min_res", type=int, default=4)
    parser.add_argument("--large_gauss_cap", type=int, default=-1,
                        help="Scale-aware lite cap: any per-axis rect that "
                             "would have been clamped at --max_res gets "
                             "clamped at this value instead. Leaves the "
                             "small/detail-carrying surfels at full res; "
                             "only squeezes the visually-large blobs that "
                             "dominate the atlas. -1 ⇒ disabled.")
    parser.add_argument("--atlas_width", type=int, default=4096)
    parser.add_argument("--atlas_budget_mb", type=int, default=2048,
                        help="Max atlas size in MB (FP16). Resolutions auto-shrink to fit.")
    parser.add_argument("--ss", type=int, default=1, help="Supersample factor")
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_benchmark", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_bake", action="store_true", help="Skip baking, use existing atlas")
    parser.add_argument("--rvq", action="store_true",
                        help="Use the RVQ atlas decoder. Loads "
                             "vq/{codebooks,indices,block_meta}.pt from the "
                             "bake_dir and installs via set_atlas_rvq() — the "
                             "render kernel does per-fragment codebook decode "
                             "instead of tex2D / BC7.")
    parser.add_argument("--rvq_nearest", action="store_true",
                        help="With --rvq, sample with nearest neighbour "
                             "(~4× fewer codebook reads, ~−2 dB atlas PSNR).")
    parser.add_argument("--rvq_shared_cb", action="store_true",
                        help="With --rvq, load codebook into __shared__ at "
                             "kernel start (faster lookup, lower occupancy).")
    parser.add_argument("--rvq_tex_cb", action="store_true",
                        help="With --rvq, read codebook via cudaTextureObject.")
    parser.add_argument("--rvq_tex_idx", action="store_true",
                        help="With --rvq, read indices via cudaTextureObject.")
    parser.add_argument("--bply", type=str, default=None,
                        help="Optional: path to a .bply (compressed PLY). "
                             "Decompressed to a temp .ply before rendering. "
                             "Use to bench the 'RVQply' combo (8-bit PLY + RVQ atlas).")
    parser.add_argument("--aabb_mode", type=int, default=3,
                        help="AABB mode: 0=square, 1=square+AdR, 2=rect, 3=rect+AdR (default: 3)")
    parser.add_argument("--sort_mode", type=int, default=0,
                        help="Sort scheme: 0=legacy 64-bit single sort, 1=FastGS two-stage "
                             "(32-bit depth on n_visible + 32-bit tile on n_instances). default 0")
    parser.add_argument("--view_aware_res", action="store_true",
                        help="Size each surfel's atlas resolution by its max projected "
                             "footprint across all training views (viewing-Nyquist). "
                             "Typically yields 5-10x atlas shrink vs hashgrid-Nyquist on "
                             "outdoor scenes, with no quality loss on training-distribution views.")
    parser.add_argument("--bake_prune_low_contrib", type=float, default=0.0,
                        help="Drop the bottom fraction of Gaussians by accumulated "
                             "alpha*T importance over all train views. e.g. 0.10 prunes "
                             "the bottom 10%%. Default 0 (no pruning).")
    parser.add_argument("--bake_skip_texture_low_contrib", type=float, default=0.0,
                        help="After --bake_prune_low_contrib, mark this fraction of the "
                             "remaining lowest-importance Gaussians as zero-rect — the "
                             "renderer falls through to SH-only color (no atlas lookup, "
                             "no per-Gaussian texels). Default 0.")
    parser.add_argument("--bake_prune_thresh", type=float, default=-1.0,
                        help="Drop Gaussians whose accumulated alpha*T over all train "
                             "views is <= this absolute value. -1 disables. Common: 0.0001 "
                             "to drop never-contributing Gaussians.")
    parser.add_argument("--bake_skip_texture_thresh", type=float, default=-1.0,
                        help="Mark Gaussians whose accumulated alpha*T <= this value as "
                             "zero-rect (SH only). -1 disables. Common: 0.001 to skip "
                             "low-contributors that don't warrant a residual texture.")
    parser.add_argument("--bake_budget_mode", type=str, default="uniform",
                        choices=["uniform", "importance"],
                        help="Atlas-budget allocator: 'uniform' (default) iteratively "
                             "halves the global max_res to fit; 'importance' keeps the "
                             "highest-importance Gaussians at their requested resolution "
                             "and zero-rects the tail. importance mode requires train "
                             "views (auto-walks them).")
    parser.add_argument("--bake_dtype", type=str, default="fp16",
                        choices=["fp16", "uint8", "bc7"],
                        help="In-memory atlas dtype during bake. fp16 (default, 6 B/texel) "
                             "matches existing artifacts. uint8 (3 B/texel) halves CPU peak "
                             "RAM by quantizing per-chunk with a pre-sampled ±6σ range; "
                             "saves an `atlas_offset/atlas_scale` pair in bake_meta.json. "
                             "bc7 = uint8 + BC7 block compression (8 bpp, 3× shrink over uint8) "
                             "saved as `atlas_texture.bc7`; runtime samples it via the CUDA "
                             "BC7 hardware texture path.")
    parser.add_argument("--atlas_quant", type=str, default="uint8",
                        choices=["uint8", "half4", "software"],
                        help="Atlas texture encoding. uint8 (default) = quantized ±6σ, ¼ memory, "
                             "hw bilinear. half4 = lossless vs training FP16 storage (slower, larger). "
                             "software = legacy pre-texture path (raw FP16 global reads, manual "
                             "bilinear in kernel — slowest, but no hw-texture dependency).")
    bargs = parser.parse_args()

    model_path = bargs.model_path

    # =====================================================================
    # 1. Report neural renderer metrics from logs
    # =====================================================================
    print("=" * 70)
    print("  BENCHMARK: 3D_SH_res Baked Rendering")
    print("=" * 70)

    train_info = parse_training_log(model_path)
    test_info = parse_test_metrics(model_path)

    print(f"\n[NEURAL RENDERER] (from training logs)")
    if train_info:
        print(f"  Gaussians:  {train_info.get('num_gaussians', '?'):,}")
        print(f"  Resolution: {train_info.get('resolution', '?')}")
        print(f"  Render FPS: {train_info.get('train_fps', '?')}")
        print(f"  ms/frame:   {train_info.get('train_ms', '?')}")
    if test_info:
        print(f"  Test PSNR:  {test_info.get('neural_psnr', '?')} dB")
        print(f"  Test SSIM:  {test_info.get('neural_ssim', '?')}")
        print(f"  Test LPIPS: {test_info.get('neural_lpips', '?')}")

    # =====================================================================
    # 2. Load model and bake atlas
    # =====================================================================
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True

    config_yaml_path = os.path.join(model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    iteration = bargs.iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)
    print(f"\n[CONFIG] Iteration: {iteration}, method: {args.method}")

    output_dir = bargs.output_dir or os.path.join(model_path, "baked_atlas")

    if not bargs.skip_bake:
        # Load INGP
        ingp = INGP(cfg, args=args).to('cuda')
        ingp.load_model(model_path, iteration)
        ingp.set_active_levels(iteration)

        # Load Gaussians
        temp_parser = ArgumentParser()
        model_params = ModelParams(temp_parser, sentinel=True)
        dataset = model_params.extract(args)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
        gaussians.base_opacity = cfg.surfel.tg_base_alpha
        if hasattr(args, 'kernel'):
            gaussians.kernel_type = args.kernel
        gaussians.kernel_type2 = getattr(args, 'kernel2', None)

        # Prune dead Gaussians — must prune EVERY per-Gaussian tensor so save_ply
        # doesn't see a size mismatch. Optional banks (SB/SG/SV/flex_beta/etc.)
        # are only pruned when actually populated.
        dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
        n_dead = dead_mask.sum().item()
        if n_dead > 0:
            valid_mask = ~dead_mask
            for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                         '_scaling', '_rotation', '_appearance_level',
                         '_gaussian_features',
                         '_shape', '_flex_beta',
                         '_sb_params',
                         '_sg_directions', '_sg_sharpness_sg', '_sg_rgb',
                         '_sv_sites', '_sv_colors', '_sv_tau', '_sv_dc',
                         '_gamma', '_adaptive_features',
                         '_adaptive_cat_weight', '_adaptive_zero_weight',
                         '_gate_logits',
                         # `--method mixed[_3d]`: keep the per-Gauss textured
                         # flag AND the EWA 3rd-axis scale aligned through the
                         # dead-Gaussian prune, else save_ply's is_textured /
                         # scale_z columns desync from _xyz.
                         '_is_textured', '_scaling_z']:
                tensor = getattr(gaussians, attr, None)
                if tensor is not None and tensor.numel() > 0 and tensor.shape[0] == valid_mask.shape[0]:
                    setattr(gaussians, attr, tensor[valid_mask.to(tensor.device)])

        N = len(gaussians.get_xyz)
        print(f"[BAKE] {N:,} Gaussians after pruning ({n_dead:,} pruned)")

        # Train cameras + render plumbing for view-aware Nyquist,
        # importance-based pruning / skip-texture, and importance-priority
        # budget allocation.
        need_imp = (bargs.bake_prune_low_contrib > 0
                    or bargs.bake_skip_texture_low_contrib > 0
                    or bargs.bake_prune_thresh >= 0
                    or bargs.bake_skip_texture_thresh >= 0
                    or bargs.bake_budget_mode == "importance")
        train_cameras = scene.getTrainCameras() if (bargs.view_aware_res or need_imp) else None
        importance_render_args = None
        if need_imp:
            from arguments import PipelineParams
            pipe = PipelineParams(ArgumentParser()).extract(args)
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
            importance_render_args = dict(
                pipe=pipe, background=background, cfg_model=cfg,
                beta=getattr(args, 'tg_beta', 0.0), iteration=iteration)

        # Bake
        atlas_cpu, atlas_rects, bake_meta = bake_atlas(
            ingp, gaussians, bargs.uv_extent, bargs.max_res, bargs.min_res,
            bargs.atlas_width, bargs.ss, atlas_budget_mb=bargs.atlas_budget_mb,
            train_cameras=train_cameras, view_aware=bargs.view_aware_res,
            prune_low_contrib=bargs.bake_prune_low_contrib,
            skip_texture_low_contrib=bargs.bake_skip_texture_low_contrib,
            prune_thresh=bargs.bake_prune_thresh,
            skip_texture_thresh=bargs.bake_skip_texture_thresh,
            importance_render_args=importance_render_args,
            budget_mode=bargs.bake_budget_mode,
            bake_dtype=bargs.bake_dtype,
            large_gauss_cap=(bargs.large_gauss_cap if bargs.large_gauss_cap > 0 else None))
        bake_meta["iteration"] = iteration
        bake_meta["kernel"] = getattr(args, 'kernel', 'gaussian')
        # `--method mixed_3d --kernel2`: untextured-EWA kernel override (None ⇒
        # untextured used --kernel). The bake-render kernel reads this via
        # set_untex_kernel() so baked geometry matches training.
        bake_meta["kernel2"] = getattr(args, 'kernel2', None)
        bake_meta["sh_degree"] = 3

        # --- Training-time config snapshot: activation biases + Compact Box ---
        _ab = getattr(args, 'activation_bias', [0.5, 0.0])
        _sh_bias_train = float(_ab[0]) if isinstance(_ab, (list, tuple)) else 0.5
        _res_bias_train = float(_ab[1]) if isinstance(_ab, (list, tuple)) else 0.0
        _fastgs_mult_train = float(getattr(args, 'fastgs_mult', 1.0)) if getattr(args, 'fastgs', False) else 1.0
        _feature_mode_train = getattr(args, 'feature', 'sh')
        bake_meta["sh_bias"] = _sh_bias_train
        bake_meta["res_bias"] = _res_bias_train
        bake_meta["compact_mult"] = _fastgs_mult_train
        bake_meta["feature_mode"] = _feature_mode_train
        bake_meta["sb_number"] = 0
        bake_meta["sb_params_file"] = None
        # Override the legacy hardcoded `"method": "3D_SH_res"` in the bake-atlas
        # default meta with the actual training method, so mixed / mixed_3d
        # round-trip correctly in bake_meta + downstream consumers.
        bake_meta["method"] = getattr(args, 'method', bake_meta.get("method", "3D_SH_res"))
        # Derive from args.method, mirroring train.py exactly:
        #   3D_SH_add                       → 1  (separate ReLUs)
        #   mixed_sep / mixed_3d_sep        → 2  (signed per-Gauss residual;
        #                                         per-pixel ReLU re-applied
        #                                         in Python at render time —
        #                                         see render_baked /
        #                                         evaluate_baked final_relu)
        #   everything else, incl. plain
        #     3D_SH_res / mixed / mixed_3d → 0  (per-Gauss outer ReLU in the
        #                                         kernel; NO per-pixel ReLU)
        _method_train = getattr(args, 'method', '')
        if _method_train == "3D_SH_add":
            _rm = 1
        elif _method_train in ("mixed_sep", "mixed_3d_sep"):
            _rm = 2
        else:
            _rm = 0
        bake_meta["residual_mode"] = _rm
        # `--method mixed`: record the textured/untextured split so the bake is
        # self-describing. Untextured surfels were folded into the skip-texture
        # (zero-rect) set in bake_atlas → they render SV-only at no atlas cost.
        _it_bm = getattr(gaussians, '_is_textured', None)
        if _it_bm is not None and _it_bm.numel() == gaussians.get_xyz.shape[0]:
            _tex_bm = int(_it_bm.sum().item())
            bake_meta["mixed_textured"] = _tex_bm
            bake_meta["mixed_untextured"] = int(_it_bm.numel()) - _tex_bm

        # Save
        os.makedirs(output_dir, exist_ok=True)

        ply_path = os.path.join(output_dir, "baked.ply")
        gaussians.save_ply(ply_path)

        atlas_path = os.path.join(output_dir, "atlas_texture.pt")
        torch.save(atlas_cpu, atlas_path)

        # Optional BC7 compression of the uint8 atlas. Saved as raw BC7 byte
        # stream + bake_meta gets `atlas_bc7_file` + dimensions. The runtime
        # path picks up BC7 when present and uses cudaMallocArray with the
        # BC7 channel descriptor (hardware-decompressed sampling).
        if bargs.bake_dtype == "bc7":
            if atlas_cpu.dtype != torch.uint8:
                raise RuntimeError("--bake_dtype bc7 requires uint8 atlas first.")
            try:
                import bc7encoder
            except ImportError:
                raise RuntimeError("bc7encoder not installed — build it from "
                                   "submodules/bc7enc_lib first.")
            import time as _t
            H, W, _ = atlas_cpu.shape
            # Pad to mults of 4 for BC7 4×4 blocks.
            Hp = ((H + 3) // 4) * 4
            Wp = ((W + 3) // 4) * 4
            rgba = np.zeros((Hp, Wp, 4), dtype=np.uint8)
            rgba[:H, :W, :3] = atlas_cpu.numpy()
            rgba[..., 3] = 255  # full alpha (BC7 expects RGBA input)
            t0 = _t.time()
            bc7_bytes = bc7encoder.encode_image_rgba(rgba, uber_level=1, perceptual=False)
            dt = _t.time() - t0
            n_blocks = (Hp // 4) * (Wp // 4)
            bc7_mb = len(bc7_bytes) / (1024 ** 2)
            print(f"[BAKE] BC7 encoded {n_blocks:,} blocks in {dt:.1f}s → {bc7_mb:.1f} MB "
                  f"({len(bc7_bytes)/(H*W):.2f} B/texel; {atlas_cpu.nelement()/len(bc7_bytes):.1f}× shrink vs uint8)")
            bc7_path = os.path.join(output_dir, "atlas_texture.bc7")
            with open(bc7_path, "wb") as f:
                f.write(bc7_bytes)
            bake_meta["atlas_bc7_file"] = "atlas_texture.bc7"
            bake_meta["atlas_bc7_padded_h"] = Hp
            bake_meta["atlas_bc7_padded_w"] = Wp
            bake_meta["atlas_bc7_bytes"] = len(bc7_bytes)

        rects_path = os.path.join(output_dir, "atlas_rects.pt")
        torch.save(atlas_rects.cpu(), rects_path)

        # If trained with --feature beta, snapshot SB params for the baked renderer.
        if _feature_mode_train == "beta" and hasattr(gaussians, '_sb_params') and gaussians._sb_params.numel() > 0:
            sb_tensor = gaussians._sb_params.detach().cpu().float()
            bake_meta["sb_number"] = int(sb_tensor.shape[1])
            bake_meta["sb_params_file"] = "sb_params.pt"
            torch.save(sb_tensor, os.path.join(output_dir, "sb_params.pt"))
            print(f"[BAKE] Saved sb_params.pt  shape={list(sb_tensor.shape)}  "
                  f"(K={bake_meta['sb_number']} lobes)")

        # --feature SV: SV state is carried inside the baked PLY (sv_site_*,
        # sv_col_*, sv_tau_*, optional sv_dc_*) via GaussianModel.save_ply.
        # bake_meta records K + dc-flag so the render side can flip into the
        # SV color path on load (Option A: per-frame torch _build_fake_shs_from_SV
        # — no CUDA kernel changes).
        if _feature_mode_train == "SV" and hasattr(gaussians, '_sv_sites') and gaussians._sv_sites.numel() > 0:
            bake_meta["sv_number"] = int(gaussians._sv_sites.shape[1])
            bake_meta["sv_has_dc"] = bool(
                hasattr(gaussians, '_sv_dc') and gaussians._sv_dc.numel() > 0)
            print(f"[BAKE] --feature SV: K={bake_meta['sv_number']} sites/Gaussian, "
                  f"sv_dc={'on' if bake_meta['sv_has_dc'] else 'off'} (in baked.ply)")

        meta_path = os.path.join(output_dir, "bake_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(bake_meta, f, indent=2)

        atlas_mb = atlas_cpu.nelement() * 2 / 1024 / 1024
        print(f"[BAKE] Saved to {output_dir} ({atlas_mb:.1f} MB atlas)")

        # Free INGP from GPU
        del ingp
        torch.cuda.empty_cache()
    else:
        print(f"[BAKE] Skipping bake, loading from {output_dir}")

    # =====================================================================
    # 3. Render baked model on test views
    # =====================================================================
    print(f"\n[RENDER] Loading baked model...")

    # Re-setup for rendering (need test cameras from Scene)
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    baked_ply = os.path.join(output_dir, "baked.ply")
    # --bply: decompress the 8-bit .bply into a temp .ply, use that instead.
    if bargs.bply is not None:
        rt_ply = os.path.join(output_dir, "baked_bply_roundtrip.ply")
        if not os.path.exists(rt_ply) or os.path.getmtime(bargs.bply) > os.path.getmtime(rt_ply):
            import subprocess
            print(f"[BPLY] decompressing {bargs.bply} → {rt_ply}")
            subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "decompress_baked_bply.py"),
                "--input", bargs.bply, "--output", rt_ply,
            ], check=True)
        baked_ply = rt_ply
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel
    gaussians.kernel_type2 = getattr(args, 'kernel2', None)

    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(getattr(args, 'kernel', 'gaussian'), 0)

    # Atlas encoding: uint8 quantized (default), lossless half4, or the
    # pre-texture software path (raw FP16 global reads + manual bilinear).
    from diff_surfel_bake_render import (
        set_atlas_use_uint8, set_use_atlas_tex_object, clear_atlas_cache,
        set_atlas_bc7, clear_atlas_bc7)
    if bargs.atlas_quant == "software":
        set_use_atlas_tex_object(False)  # kernel falls through to software bilinear
    else:
        set_use_atlas_tex_object(True)
        set_atlas_use_uint8(bargs.atlas_quant == "uint8")
    clear_atlas_cache()
    clear_atlas_bc7()  # default: not using BC7
    print(f"[RENDER] atlas_quant={bargs.atlas_quant}")

    # Pull bake_meta early — uint8 dequant + BC7 path need it.
    meta_path = os.path.join(output_dir, "bake_meta.json")
    bake_meta_render = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            bake_meta_render = json.load(f)

    # RVQ path: install codebooks + indices, skip BC7 entirely. Takes
    # precedence when --rvq is set.
    if bargs.rvq:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bench_rvq_render_lib import install_rvq_atlas
        from diff_surfel_bake_render import (set_atlas_rvq_bilinear,
                                              set_atlas_rvq_use_shared_cb,
                                              set_atlas_rvq_use_tex_cb,
                                              set_atlas_rvq_use_tex_idx)
        install_rvq_atlas(output_dir, device='cuda')
        set_atlas_rvq_bilinear(not bargs.rvq_nearest)
        set_atlas_rvq_use_shared_cb(bargs.rvq_shared_cb)
        set_atlas_rvq_use_tex_cb(bargs.rvq_tex_cb)
        set_atlas_rvq_use_tex_idx(bargs.rvq_tex_idx)
        print(f"[RENDER] RVQ sampling: "
              f"{'nearest' if bargs.rvq_nearest else 'bilinear'}, "
              f"shared_cb={'on' if bargs.rvq_shared_cb else 'off'}, "
              f"tex_cb={'on' if bargs.rvq_tex_cb else 'off'}, "
              f"tex_idx={'on' if bargs.rvq_tex_idx else 'off'}")
        # Placeholder atlas tensor (kernel ignores it when RVQ is installed).
        atlas_tex = torch.zeros(1, 1, 3, dtype=torch.float16, device='cuda')
        bc7_file = None
    # BC7 fast path: when bake produced a `.bc7` file, install it on the
    # device and skip the FP16/uint8 atlas upload entirely.
    elif (bc7_file := bake_meta_render.get("atlas_bc7_file")) is not None:
        bc7_path = os.path.join(output_dir, bc7_file)
        with open(bc7_path, "rb") as f:
            bc7_bytes = f.read()
        bc7_tensor = torch.frombuffer(bytearray(bc7_bytes), dtype=torch.uint8).cuda()
        bc7_W = int(bake_meta_render.get("atlas_bc7_padded_w"))
        bc7_H = int(bake_meta_render.get("atlas_bc7_padded_h"))
        atlas_offset_meta = float(bake_meta_render.get("atlas_offset", 0.0))
        atlas_scale_meta  = float(bake_meta_render.get("atlas_scale",  1.0))
        set_atlas_bc7(bc7_tensor, bc7_W, bc7_H, atlas_offset_meta, atlas_scale_meta)
        print(f"[RENDER] BC7 atlas installed: {bc7_W}x{bc7_H}, "
              f"{len(bc7_bytes)/(1024**2):.1f} MB")
        # The C++ side still expects atlas_texture as `at::Half` (the dtype is
        # used to derive atlas_texture_ptr), but the kernel doesn't read it
        # when the BC7 fast path's `atlas_tex_obj` is set. Pass a 1×1×3 FP16
        # placeholder — must match dtype to avoid PyTorch's data_ptr type check.
        atlas_tex = torch.zeros(1, 1, 3, dtype=torch.float16, device='cuda')
    else:
        # Load atlas onto GPU for rendering.
        atlas_tex = torch.load(os.path.join(output_dir, "atlas_texture.pt")).cuda()

    # uint8 atlas on disk (new default) → dequantize to FP16 RGB for the CUDA
    # kernel (which expects at::Half). The CUDA runtime may then re-quantize to
    # uint8 internally for the hw-texture path (round-trip is numerically stable
    # with matched scale/offset).
    # The uint8-on-disk path needs a dequant to FP16 for the legacy renderer
    # cudaArray builder. Skip when BC7 is active (renderer's BC7 fast path
    # ignores atlas_texture entirely).
    if atlas_tex.dtype == torch.uint8 and bc7_file is None:
        atlas_scale_meta  = float(bake_meta_render.get("atlas_scale",  1.0))
        atlas_offset_meta = float(bake_meta_render.get("atlas_offset", 0.0))
        atlas_rgb = atlas_tex[..., :3] if atlas_tex.shape[-1] == 4 else atlas_tex
        atlas_tex = (atlas_rgb.to(torch.float32) / 255.0 * atlas_scale_meta
                     + atlas_offset_meta).to(torch.float16).contiguous()
        print(f"[RENDER] Dequantized uint8 atlas → FP16 "
              f"(scale={atlas_scale_meta:.4f}, offset={atlas_offset_meta:.4f})")

    atlas_width = atlas_tex.shape[1]
    atlas_texture = atlas_tex.reshape(-1).contiguous()
    atlas_rects_gpu = torch.load(os.path.join(output_dir, "atlas_rects.pt")).cuda().contiguous()
    N = len(gaussians.get_xyz)
    _sh_bias = float(bake_meta_render.get("sh_bias", getattr(args, 'activation_bias', [0.5, 0.0])[0]))
    _res_bias = float(bake_meta_render.get("res_bias", getattr(args, 'activation_bias', [0.5, 0.0])[1]))
    _compact_mult = float(bake_meta_render.get("compact_mult", 1.0))
    from diff_surfel_bake_render import (set_activation_bias, set_compact_mult,
                                         set_residual_mode, set_untex_kernel)
    set_activation_bias(_sh_bias, _res_bias)
    set_compact_mult(_compact_mult)
    # 0 = 3D_SH_res outer ReLU; 1 = 3D_SH_add separate ReLUs. Default 0 if absent.
    _residual_mode = int(bake_meta_render.get("residual_mode", 0))
    set_residual_mode(_residual_mode)
    # `--method mixed_3d --kernel2`: kernel for the UNTEXTURED EWA half. Recorded
    # in bake_meta at bake time; -1 (unset) → untextured use the run's --kernel.
    _kmap2 = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4, 'nexel': 5}
    _k2_str = bake_meta_render.get("kernel2", None)
    _untex_kt = _kmap2.get(_k2_str, -1) if _k2_str else -1
    set_untex_kernel(_untex_kt)
    print(f"[RENDER] set_activation_bias(sh={_sh_bias}, res={_res_bias})  "
          f"set_compact_mult({_compact_mult})  set_residual_mode({_residual_mode})  "
          f"set_untex_kernel({_untex_kt}{' = '+_k2_str if _k2_str else ''})")

    # SB params (only present if training used --feature beta).
    sb_params = None
    sb_number = int(bake_meta_render.get("sb_number", 0))
    sb_file = bake_meta_render.get("sb_params_file")
    if sb_number > 0 and sb_file is not None:
        sb_path = os.path.join(output_dir, sb_file)
        if os.path.exists(sb_path):
            sb_t = torch.load(sb_path).float().cuda().contiguous()
            sb_params = sb_t.reshape(-1).contiguous()  # [N*K*6]
            print(f"[RENDER] Loaded SB params: shape={list(sb_t.shape)} (K={sb_number} lobes)")
        else:
            print(f"[RENDER] bake_meta referenced {sb_file} but file missing; SB disabled.")
            sb_number = 0

    atlas_mb = atlas_tex.nelement() * 2 / 1024 / 1024
    print(f"[RENDER] {N:,} Gaussians, atlas {atlas_tex.shape[0]}x{atlas_tex.shape[1]} ({atlas_mb:.1f} MB)")

    # Load test cameras (Scene overwrites PLY, so reload after)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cameras = scene.getTestCameras()
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    print(f"[RENDER] {len(test_cameras)} test cameras")

    # Activate the SV color path (Option A) when the trained feature_mode was
    # 'SV'. load_ply just round-tripped _sv_sites/_sv_colors/_sv_tau/_sv_dc;
    # setting feature_mode lets render_baked decide to inject per-frame fake
    # shs via _build_fake_shs_from_SV. No CUDA changes — the rasterizer sees
    # the SV-evaluated color through the existing SH-DC slot.
    _feat_mode_render = bake_meta_render.get("feature_mode", "sh")
    gaussians.feature_mode = _feat_mode_render
    sv_state = None
    if _feat_mode_render == "SV":
        sv_K = int(bake_meta_render.get("sv_number", 0))
        sv_dc_on = bool(bake_meta_render.get("sv_has_dc", False))
        if (not hasattr(gaussians, '_sv_sites')) or gaussians._sv_sites.numel() == 0:
            raise RuntimeError(
                f"bake_meta says feature_mode='SV' (K={sv_K}) but the baked PLY "
                f"didn't restore _sv_sites — was it saved with the SV-aware save_ply?")
        print(f"[RENDER] feature_mode=SV active: K={sv_K}, sv_dc={'on' if sv_dc_on else 'off'} "
              f"(fused CUDA voronoi path)")
        sv_state = _make_sv_state(gaussians)

    beta = cfg.surfel.tg_beta
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    render_dir = os.path.join(output_dir, "renders")

    # --- SH only ---
    sh_save_dir = os.path.join(render_dir, "sh_only")
    print(f"\n[RENDER] Evaluating SH only -> {sh_save_dir}")
    sh_metrics = evaluate_baked(
        test_cameras, gaussians, bg_color, beta, kernel_type,
        num_warmup=bargs.num_warmup, num_benchmark=bargs.num_benchmark,
        save_dir=sh_save_dir, aabb_mode=bargs.aabb_mode, sort_mode=bargs.sort_mode,
        sb_params=sb_params, sb_number=sb_number,
        sv_state=sv_state, final_relu=(_residual_mode == 2))

    # --- SH + Atlas residual ---
    atlas_save_dir = os.path.join(render_dir, "sh_atlas")
    print(f"[RENDER] Evaluating SH + Atlas residual -> {atlas_save_dir}")
    baked_metrics = evaluate_baked(
        test_cameras, gaussians, bg_color, beta, kernel_type,
        atlas_texture=atlas_texture, atlas_rects=atlas_rects_gpu,
        atlas_width=atlas_width,
        num_warmup=bargs.num_warmup, num_benchmark=bargs.num_benchmark,
        save_dir=atlas_save_dir, aabb_mode=bargs.aabb_mode, sort_mode=bargs.sort_mode,
        sb_params=sb_params, sb_number=sb_number,
        sv_state=sv_state, final_relu=(_residual_mode == 2))

    # =====================================================================
    # 4. Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    n_gauss = int(gaussians.get_xyz.shape[0])
    print(f"\n  Gaussians: {n_gauss:,}")
    print(f"\n  {'Mode':<25} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FPS':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    if test_info.get("neural_psnr"):
        neural_fps = train_info.get("train_fps", "?")
        print(f"  {'Neural renderer':<25} {test_info['neural_psnr']:>7.2f}  "
              f"{test_info.get('neural_ssim', 0):>7.4f}  "
              f"{test_info.get('neural_lpips', 0):>7.4f}  {neural_fps:>7}")

    print(f"  {'Baked (SH only)':<25} {sh_metrics['psnr']:>7.2f}  "
          f"{sh_metrics['ssim']:>7.4f}  {sh_metrics['lpips']:>7.4f}  {sh_metrics['fps']:>7.1f}")
    print(f"  {'Baked (SH + atlas)':<25} {baked_metrics['psnr']:>7.2f}  "
          f"{baked_metrics['ssim']:>7.4f}  {baked_metrics['lpips']:>7.4f}  {baked_metrics['fps']:>7.1f}")

    if test_info.get("neural_psnr"):
        delta = baked_metrics['psnr'] - test_info['neural_psnr']
        print(f"\n  Bake quality loss: {delta:+.2f} dB vs neural renderer")
        speedup = baked_metrics['fps'] / train_info['train_fps'] if train_info.get('train_fps') else 0
        if speedup:
            print(f"  Bake speedup:     {speedup:.1f}x")

    # Save all metrics
    all_metrics = {
        "neural": {**train_info, **test_info},
        "baked_sh_only": sh_metrics,
        "baked_sh_atlas": baked_metrics,
    }
    metrics_path = os.path.join(output_dir, "benchmark_results.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved to: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

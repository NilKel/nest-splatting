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
def compute_adaptive_resolution(scales, cell_size, uv_extent=4.0, max_res=64, min_res=4):
    """Per-axis Nyquist resolution: each surfel gets (res_x, res_y) where each
    axis is sized by its own scale (not the larger axis). Saves atlas texels on
    anisotropic surfels (edges, hair) without losing fidelity.
    Returns [N, 2] int tensor of (res_x, res_y) powers of two in [min_res, max_res].
    """
    # scales is [N, 2] for 2DGS surfels — column 0 = sx, column 1 = sy.
    n_cells = 2.0 * uv_extent * scales / cell_size           # [N, 2]
    nyquist_samples = 2.0 * n_cells                          # [N, 2]
    log2_res = torch.ceil(torch.log2(nyquist_samples.clamp(min=1.0)))
    resolutions = (2.0 ** log2_res).int()
    return resolutions.clamp(min=min_res, max=max_res)       # [N, 2]


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
def bake_atlas(ingp, gaussians, uv_extent, max_res, min_res, atlas_width, ss,
               atlas_budget_mb=2048):
    """Bake hash MLP residual into a CPU-resident atlas. Returns (atlas_cpu, atlas_rects, meta)."""

    hash_encoding = ingp.hash_encoding
    embeddings, offsets, num_levels, per_level_scale, base_resolution, align_corners, interp_id = hash_encoding.get_params()
    voxel_min = ingp.voxel_range[0]
    voxel_max = ingp.voxel_range[1]
    finest_resolution = base_resolution * (per_level_scale ** (num_levels - 1))
    cell_size = (voxel_max - voxel_min) / finest_resolution

    print(f"[BAKE] Hash grid: {num_levels} levels, finest_res={finest_resolution:.0f}, "
          f"cell_size={cell_size:.6f}")

    # Compute ideal per-axis resolutions, then shrink to fit atlas budget.
    # resolutions is [N, 2] = (res_x, res_y).
    resolutions = compute_adaptive_resolution(
        gaussians.get_scaling, cell_size, uv_extent=uv_extent,
        max_res=max_res, min_res=min_res)

    # Budget-constrain: iteratively halve max_res until atlas fits (0 = unlimited).
    # Texel cost per Gaussian is res_x * res_y, so anisotropic surfels stay cheap.
    effective_max = max_res
    while atlas_budget_mb > 0 and effective_max > min_res:
        clamped = resolutions.clamp(max=effective_max)
        total_texels = (clamped[:, 0].long() * clamped[:, 1].long()).sum().item()
        atlas_size_mb = total_texels * 3 * 2 / (1024 * 1024)  # FP16, 3 channels
        if atlas_size_mb <= atlas_budget_mb:
            break
        effective_max //= 2

    if effective_max != max_res:
        print(f"[BAKE] Budget {atlas_budget_mb} MB: clamped max_res {max_res} -> {effective_max}")
        resolutions = resolutions.clamp(min=min_res, max=effective_max)

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

    atlas_rects, atlas_height, used_rows, utilization = shelf_pack_atlas(
        resolutions, atlas_width=atlas_width)
    atlas_mb = atlas_height * atlas_width * 3 * 2 / 1024 / 1024
    print(f"[ATLAS] Packed: {atlas_width}x{atlas_height}, "
          f"used {used_rows}/{atlas_height} rows, {utilization:.1f}% util, {atlas_mb:.1f} MB (FP16)")

    # Allocate atlas on CPU in FP16 to fit in system RAM
    # (residuals are small — mean ~0.01, FP16 precision is sufficient)
    atlas_cpu = torch.zeros(atlas_height, atlas_width, 3, dtype=torch.float16)

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

            # Write to CPU atlas (convert to FP16 to match atlas dtype)
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
    }

    return atlas_cpu, atlas_rects, meta


# ---------------------------------------------------------------------------
# Render baked model
# ---------------------------------------------------------------------------
def render_baked(viewpoint_camera, gaussian_pkg, background,
                 beta=0.0, sh_degree=3, aabb_mode=3,
                 atlas_texture=None, atlas_rects=None, atlas_width=0,
                 sb_params=None, sb_number=0):
    """Render one view. `gaussian_pkg` is the dict from
    `prepare_gaussian_inputs(gaussians, ...)` — pre-activated tensors that are
    constant for the whole scene."""
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
        sh_degree=sh_degree, beta=beta, aabb_mode=aabb_mode,
    )

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
    )
    return color


def evaluate_baked(test_cameras, gaussians, bg_color, beta, kernel_type,
                   atlas_texture=None, atlas_rects=None,
                   atlas_width=0, num_warmup=10, num_benchmark=100, save_dir=None,
                   aabb_mode=3,
                   sb_params=None, sb_number=0):
    """Render all test views, compute metrics, benchmark FPS."""
    from diff_surfel_bake_render import prepare_gaussian_inputs

    # Snapshot post-activation tensors once for the whole scene.
    gaussian_pkg = prepare_gaussian_inputs(
        gaussians, sh_degree=gaussians.active_sh_degree, kernel_type=kernel_type)

    psnrs, l1s, ssims_list, lpips_list = [], [], [], []
    kwargs = dict(atlas_texture=atlas_texture, atlas_rects=atlas_rects,
                  atlas_width=atlas_width, aabb_mode=aabb_mode,
                  sb_params=sb_params, sb_number=sb_number,
                  sh_degree=gaussians.active_sh_degree)

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
    parser.add_argument("--atlas_width", type=int, default=4096)
    parser.add_argument("--atlas_budget_mb", type=int, default=2048,
                        help="Max atlas size in MB (FP16). Resolutions auto-shrink to fit.")
    parser.add_argument("--ss", type=int, default=1, help="Supersample factor")
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_benchmark", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_bake", action="store_true", help="Skip baking, use existing atlas")
    parser.add_argument("--aabb_mode", type=int, default=3,
                        help="AABB mode: 0=square, 1=square+AdR, 2=rect, 3=rect+AdR (default: 3)")
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
                         '_sv_sites', '_sv_colors',
                         '_gamma', '_adaptive_features',
                         '_adaptive_cat_weight', '_adaptive_zero_weight',
                         '_gate_logits']:
                tensor = getattr(gaussians, attr, None)
                if tensor is not None and tensor.numel() > 0 and tensor.shape[0] == valid_mask.shape[0]:
                    setattr(gaussians, attr, tensor[valid_mask.to(tensor.device)])

        N = len(gaussians.get_xyz)
        print(f"[BAKE] {N:,} Gaussians after pruning ({n_dead:,} pruned)")

        # Bake
        atlas_cpu, atlas_rects, bake_meta = bake_atlas(
            ingp, gaussians, bargs.uv_extent, bargs.max_res, bargs.min_res,
            bargs.atlas_width, bargs.ss, atlas_budget_mb=bargs.atlas_budget_mb)
        bake_meta["iteration"] = iteration
        bake_meta["kernel"] = getattr(args, 'kernel', 'gaussian')
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

        # Save
        os.makedirs(output_dir, exist_ok=True)

        ply_path = os.path.join(output_dir, "baked.ply")
        gaussians.save_ply(ply_path)

        atlas_path = os.path.join(output_dir, "atlas_texture.pt")
        torch.save(atlas_cpu, atlas_path)

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
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(getattr(args, 'kernel', 'gaussian'), 0)

    # Atlas encoding: uint8 quantized (default), lossless half4, or the
    # pre-texture software path (raw FP16 global reads + manual bilinear).
    from diff_surfel_bake_render import (
        set_atlas_use_uint8, set_use_atlas_tex_object, clear_atlas_cache)
    if bargs.atlas_quant == "software":
        set_use_atlas_tex_object(False)  # kernel falls through to software bilinear
    else:
        set_use_atlas_tex_object(True)
        set_atlas_use_uint8(bargs.atlas_quant == "uint8")
    clear_atlas_cache()
    print(f"[RENDER] atlas_quant={bargs.atlas_quant}")

    # Load atlas onto GPU for rendering.
    atlas_tex = torch.load(os.path.join(output_dir, "atlas_texture.pt")).cuda()

    # Pull bake_meta early — uint8 dequant below needs atlas_scale/offset.
    meta_path = os.path.join(output_dir, "bake_meta.json")
    bake_meta_render = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            bake_meta_render = json.load(f)

    # uint8 atlas on disk (new default) → dequantize to FP16 RGB for the CUDA
    # kernel (which expects at::Half). The CUDA runtime may then re-quantize to
    # uint8 internally for the hw-texture path (round-trip is numerically stable
    # with matched scale/offset).
    if atlas_tex.dtype == torch.uint8:
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
    from diff_surfel_bake_render import set_activation_bias, set_compact_mult
    set_activation_bias(_sh_bias, _res_bias)
    set_compact_mult(_compact_mult)
    print(f"[RENDER] set_activation_bias(sh={_sh_bias}, res={_res_bias})  "
          f"set_compact_mult({_compact_mult})")

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

    beta = cfg.surfel.tg_beta
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    render_dir = os.path.join(output_dir, "renders")

    # --- SH only ---
    sh_save_dir = os.path.join(render_dir, "sh_only")
    print(f"\n[RENDER] Evaluating SH only -> {sh_save_dir}")
    sh_metrics = evaluate_baked(
        test_cameras, gaussians, bg_color, beta, kernel_type,
        num_warmup=bargs.num_warmup, num_benchmark=bargs.num_benchmark,
        save_dir=sh_save_dir, aabb_mode=bargs.aabb_mode,
        sb_params=sb_params, sb_number=sb_number)

    # --- SH + Atlas residual ---
    atlas_save_dir = os.path.join(render_dir, "sh_atlas")
    print(f"[RENDER] Evaluating SH + Atlas residual -> {atlas_save_dir}")
    baked_metrics = evaluate_baked(
        test_cameras, gaussians, bg_color, beta, kernel_type,
        atlas_texture=atlas_texture, atlas_rects=atlas_rects_gpu,
        atlas_width=atlas_width,
        num_warmup=bargs.num_warmup, num_benchmark=bargs.num_benchmark,
        save_dir=atlas_save_dir, aabb_mode=bargs.aabb_mode,
        sb_params=sb_params, sb_number=sb_number)

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

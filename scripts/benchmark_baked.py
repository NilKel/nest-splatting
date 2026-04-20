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
    max_scale = scales.max(dim=1).values
    n_cells = 2.0 * uv_extent * max_scale / cell_size
    nyquist_samples = 2.0 * n_cells
    log2_res = torch.ceil(torch.log2(nyquist_samples.clamp(min=1.0)))
    resolutions = (2.0 ** log2_res).int()
    return resolutions.clamp(min=min_res, max=max_res)


# ---------------------------------------------------------------------------
# Atlas packing (shelf-first-fit-decreasing)
# ---------------------------------------------------------------------------
def shelf_pack_atlas(resolutions, atlas_width=4096):
    N = len(resolutions)
    res_cpu = resolutions.cpu().numpy()

    height = 0
    for sz in sorted(set(int(x) for x in res_cpu), reverse=True):
        count = int((res_cpu == sz).sum())
        per_row = atlas_width // sz
        rows = (count + per_row - 1) // per_row
        height += rows * sz
    atlas_height = max(((height + 63) // 64) * 64, 64)

    order = np.argsort(-res_cpu)
    shelves = []
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
                rects[idx] = [0, 0, 2, 2]
                continue
            shelves.append([y_start, sz, sz])
            rects[idx] = [0, y_start, sz, sz]

    used_rows = max((s[0] + s[1] for s in shelves), default=0)
    total_area = float(np.sum(res_cpu.astype(np.int64) ** 2))
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

    # Compute ideal resolutions, then shrink to fit atlas budget
    resolutions = compute_adaptive_resolution(
        gaussians.get_scaling, cell_size, uv_extent=uv_extent,
        max_res=max_res, min_res=min_res)

    # Budget-constrain: iteratively halve max_res until atlas fits (0 = unlimited)
    effective_max = max_res
    while atlas_budget_mb > 0 and effective_max > min_res:
        clamped = resolutions.clamp(max=effective_max)
        total_texels = (clamped.long() ** 2).sum().item()
        atlas_size_mb = total_texels * 3 * 2 / (1024 * 1024)  # FP16, 3 channels
        if atlas_size_mb <= atlas_budget_mb:
            break
        effective_max //= 2

    if effective_max != max_res:
        print(f"[BAKE] Budget {atlas_budget_mb} MB: clamped max_res {max_res} -> {effective_max}")
        resolutions = resolutions.clamp(min=min_res, max=effective_max)

    print(f"[BAKE] Adaptive resolution distribution:")
    unique_res = resolutions.unique().sort().values
    for res_val in unique_res:
        count = (resolutions == res_val.item()).sum().item()
        print(f"  {res_val.item():>4}x{res_val.item():<4}: {count:>7,} Gaussians")

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

    mlp = ingp.mlp_fused
    mlp.eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_input_padded = mlp[0].weight.shape[1]
    bias_col = hash_dim

    for res_val in unique_res:
        res = res_val.item()
        bake_res = res * ss
        mask = (resolutions == res)
        indices = mask.nonzero(as_tuple=True)[0]
        n_group = len(indices)

        print(f"[BAKE] {res}x{res} (ss={ss}x -> {bake_res}x{bake_res}): {n_group:,} Gaussians")

        step = 2.0 * uv_extent / bake_res
        coords = torch.arange(bake_res, dtype=torch.float32, device='cuda')
        uv_1d = (coords + 0.5) * step - uv_extent
        uu, vv = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
        u_flat = uu.reshape(-1)
        v_flat = vv.reshape(-1)
        n_pts = bake_res * bake_res

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
                mlp_input = torch.zeros(xyz_flat.shape[0], mlp_input_padded, device='cuda')
                mlp_input[:, :hash_dim] = hash_feat[:, :hash_dim]
                mlp_input[:, bias_col] = 1.0
                mlp_out = mlp(mlp_input)
                rgb_residual = mlp_out[:, :3]

            residual = rgb_residual.reshape(n_batch, bake_res, bake_res, 3)
            if ss > 1:
                residual = residual.view(n_batch, res, ss, res, ss, 3).mean(dim=(2, 4))

            # Write to CPU atlas (convert to FP16 to match atlas dtype)
            residual_cpu = residual.half().cpu()
            rects = atlas_rects[batch_indices.cpu()]
            for b in range(n_batch):
                u0 = int(rects[b, 0].item())
                v0 = int(rects[b, 1].item())
                atlas_cpu[v0:v0+res, u0:u0+res, :] = residual_cpu[b].permute(1, 0, 2)

            del xyz, xyz_flat, hash_feat, mlp_input, mlp_out, rgb_residual, residual
            torch.cuda.empty_cache()

    print(f"[BAKE] Atlas residual stats: mean={atlas_cpu.mean():.6f}, "
          f"std={atlas_cpu.std():.6f}, min={atlas_cpu.min():.6f}, max={atlas_cpu.max():.6f}")

    res_dist = {str(r.item()): int((resolutions == r).sum().item()) for r in unique_res}
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
def render_baked(viewpoint_camera, gaussians, background,
                 residual_textures=None, beta=0.0, kernel_type=0,
                 atlas_texture=None, atlas_rects=None, atlas_width=0,
                 aabb_mode=3):
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=background, scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=False, beta=beta,
        aabb_mode=aabb_mode,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    shapes = None
    if kernel_type > 0 and hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape

    color, radii = rasterizer(
        means3D=gaussians.get_xyz,
        means2D=torch.zeros_like(gaussians.get_xyz[:, :2], requires_grad=False),
        opacities=gaussians.get_opacity,
        shs=gaussians.get_features,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        shapes=shapes, kernel_type=kernel_type,
        residual_textures=residual_textures,
        atlas_texture=atlas_texture,
        atlas_rects=atlas_rects,
        atlas_width=atlas_width,
    )
    return color


def evaluate_baked(test_cameras, gaussians, bg_color, beta, kernel_type,
                   residual_textures=None, atlas_texture=None, atlas_rects=None,
                   atlas_width=0, num_warmup=10, num_benchmark=100, save_dir=None,
                   aabb_mode=3):
    """Render all test views, compute metrics, benchmark FPS. Optionally save images."""
    psnrs, l1s, ssims_list = [], [], []
    kwargs = dict(residual_textures=residual_textures,
                  atlas_texture=atlas_texture, atlas_rects=atlas_rects,
                  atlas_width=atlas_width, aabb_mode=aabb_mode)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for cam in test_cameras:
            rendered = render_baked(cam, gaussians, bg_color,
                                    beta=beta, kernel_type=kernel_type, **kwargs)
            gt = cam.original_image[:3].cuda()
            psnrs.append(psnr(rendered, gt).mean().item())
            l1s.append(l1_loss(rendered, gt).item())
            ssims_list.append(ssim(rendered, gt).item())

            if save_dir is not None:
                img_np = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                save_img_u8(img_np, os.path.join(save_dir, f"{cam.image_name}.png"))

        # FPS warmup
        for i in range(num_warmup):
            cam = test_cameras[i % len(test_cameras)]
            _ = render_baked(cam, gaussians, bg_color,
                             beta=beta, kernel_type=kernel_type, **kwargs)
        torch.cuda.synchronize()

        # FPS benchmark
        times = []
        for i in range(num_benchmark):
            cam = test_cameras[i % len(test_cameras)]
            torch.cuda.synchronize()
            t0 = time.time()
            _ = render_baked(cam, gaussians, bg_color,
                             beta=beta, kernel_type=kernel_type, **kwargs)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims_list)),
        "l1": float(np.mean(l1s)),
        "fps": float(1.0 / np.mean(times)),
        "ms_per_frame": float(np.mean(times) * 1000),
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
        print(f"[BAKE] {N:,} Gaussians after pruning ({n_dead:,} pruned)")

        # Bake
        atlas_cpu, atlas_rects, bake_meta = bake_atlas(
            ingp, gaussians, bargs.uv_extent, bargs.max_res, bargs.min_res,
            bargs.atlas_width, bargs.ss, atlas_budget_mb=bargs.atlas_budget_mb)
        bake_meta["iteration"] = iteration
        bake_meta["kernel"] = getattr(args, 'kernel', 'gaussian')
        bake_meta["sh_degree"] = 3

        # Save
        os.makedirs(output_dir, exist_ok=True)

        ply_path = os.path.join(output_dir, "baked.ply")
        gaussians.save_ply(ply_path)

        atlas_path = os.path.join(output_dir, "atlas_texture.pt")
        torch.save(atlas_cpu, atlas_path)

        rects_path = os.path.join(output_dir, "atlas_rects.pt")
        torch.save(atlas_rects.cpu(), rects_path)

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

    # Load atlas onto GPU for rendering
    atlas_tex = torch.load(os.path.join(output_dir, "atlas_texture.pt")).cuda()
    atlas_width = atlas_tex.shape[1]
    atlas_texture = atlas_tex.reshape(-1).contiguous()
    atlas_rects_gpu = torch.load(os.path.join(output_dir, "atlas_rects.pt")).cuda().contiguous()
    N = len(gaussians.get_xyz)

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
        save_dir=sh_save_dir, aabb_mode=bargs.aabb_mode)

    # --- SH + Atlas residual ---
    atlas_save_dir = os.path.join(render_dir, "sh_atlas")
    print(f"[RENDER] Evaluating SH + Atlas residual -> {atlas_save_dir}")
    baked_metrics = evaluate_baked(
        test_cameras, gaussians, bg_color, beta, kernel_type,
        atlas_texture=atlas_texture, atlas_rects=atlas_rects_gpu,
        atlas_width=atlas_width,
        num_warmup=bargs.num_warmup, num_benchmark=bargs.num_benchmark,
        save_dir=atlas_save_dir, aabb_mode=bargs.aabb_mode)

    # =====================================================================
    # 4. Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Mode':<25} {'PSNR':>8} {'SSIM':>8} {'FPS':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

    if test_info.get("neural_psnr"):
        neural_fps = train_info.get("train_fps", "?")
        print(f"  {'Neural renderer':<25} {test_info['neural_psnr']:>7.2f}  "
              f"{test_info.get('neural_ssim', 0):>7.4f}  {neural_fps:>7}")

    print(f"  {'Baked (SH only)':<25} {sh_metrics['psnr']:>7.2f}  "
          f"{sh_metrics['ssim']:>7.4f}  {sh_metrics['fps']:>7.1f}")
    print(f"  {'Baked (SH + atlas)':<25} {baked_metrics['psnr']:>7.2f}  "
          f"{baked_metrics['ssim']:>7.4f}  {baked_metrics['fps']:>7.1f}")

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

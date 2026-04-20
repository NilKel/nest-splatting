"""Standalone test harness for comparing depth-reinit scale algorithms.

Loads a saved pre-reinit PLY (the ones --mini1 dumps to `reinit_plys/`),
runs a depth reinit with a chosen scale-init algorithm, then renders the
first training camera and saves:
  - `<out>/{tag}_maxcontrib_id.png`  — per-pixel max-contributor id map
  - `<out>/{tag}_rgb.png`            — rendered RGB (uses an untrained hash/MLP,
                                       so colors will be rough — the goal is
                                       to compare silhouettes, not photometry)
  - `<out>/{tag}_depth_maxcontrib.png` — depth of the max-weight Gaussian

Usage:
    python scripts/test_reinit_algorithm.py \\
        --model_path outputs/.../bREetamini2OPACITYFBIGSOFTaggnegfixtest \\
        --ply        outputs/.../reinit_plys/point_cloud_iter5000.ply \\
        --algo       current        # or 'silhouette' or 'none' (no reinit)
        --out_dir    outputs/.../reinit_tests

The script reproduces the same Scene / GaussianModel / INGP setup as
training would, but skips the iter loop — it only executes one render pass
and writes images. Run it multiple times with different --algo values to
produce comparison images.
"""
import os
import sys
import argparse
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from utils.render_utils import save_img_u8, convert_gray_to_cmap
from hash_encoder.config import Config
from hash_encoder.modules import INGP
from gaussian_renderer import render


def _colorize_max_contrib_idx(max_idx_map):
    """Hash-based RGB colormap for per-pixel Gaussian ids.
    Matches the helper defined in train.py."""
    arr = max_idx_map.squeeze().detach().cpu().numpy().astype(np.int64)
    H, W = arr.shape
    invalid = arr < 0
    hashed = np.clip(arr, 0, None)
    r = ((hashed * 2654435761) & 0xFFFFFF) / 0xFFFFFF
    g = ((hashed * 40503 + 31) & 0xFFFFFF) / 0xFFFFFF
    b = ((hashed * 1442695040888963407 + 11) & 0xFFFFFF) / 0xFFFFFF
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    if invalid.any():
        rgb[invalid] = 0.0
    return rgb


# ============================================================================
# Silhouette-aware scale algorithm (new, proposed).
# ============================================================================
#
# For each depth-reinit pixel, compute a "safe radius" from the nearest
# silhouette/crease/background edge, converted to world units via the
# per-pixel depth × (1 / focal). Take the min of nn_dist and safe_radius.
#
# Boundary mask = depth edge OR normal crease OR alpha edge. Distance
# transform → world-space safe radius.
#
def compute_safe_radius_world(depth, alpha, normal, fx, fy,
                              rel_depth_thresh=0.05,
                              normal_dot_thresh=0.70,
                              alpha_thresh=0.01,
                              use_normal_edge=False):
    """Per-pixel world-space safe radius (distance to the nearest edge * footprint).

    Args:
        depth:  [1, H, W] or [H, W]  (depth_max_contributor from rasterizer)
        alpha:  [1, H, W] or [H, W]  (rend_alpha)
        normal: [3, H, W]            (rend_normal, world-space unit)
        fx, fy: camera focal in pixels
    Returns:
        [H, W] world-units safe radius (float)
    """
    depth = depth.squeeze()
    alpha = alpha.squeeze()
    H, W = depth.shape
    device = depth.device

    # Boundary mask components.
    # (a) relative depth jump > threshold
    dz_x = torch.abs(depth[:, 1:] - depth[:, :-1])
    dz_y = torch.abs(depth[1:, :] - depth[:-1, :])
    dep_x = depth[:, :-1]
    dep_y = depth[:-1, :]
    rel_x = dz_x > rel_depth_thresh * dep_x.clamp_min(1e-6)
    rel_y = dz_y > rel_depth_thresh * dep_y.clamp_min(1e-6)
    depth_edge = torch.zeros_like(depth, dtype=torch.bool)
    depth_edge[:, :-1] |= rel_x
    depth_edge[:, 1:]  |= rel_x
    depth_edge[:-1, :] |= rel_y
    depth_edge[1:, :]  |= rel_y

    # (b) normal crease: neighbor dot product below threshold.
    # DISABLED by default — curved surfaces (chair armrests, legs) produce
    # continuous normal variation, and a strict dot threshold like 0.94 treats
    # half the object as "edges", blowing up the boundary mask and making all
    # safe radii tiny. Only enable for scenes with genuine sharp creases.
    if use_normal_edge:
        n_dot_x = (normal[:, :, 1:] * normal[:, :, :-1]).sum(dim=0)
        n_dot_y = (normal[:, 1:, :] * normal[:, :-1, :]).sum(dim=0)
        n_edge_x = n_dot_x < normal_dot_thresh
        n_edge_y = n_dot_y < normal_dot_thresh
        normal_edge = torch.zeros_like(depth, dtype=torch.bool)
        normal_edge[:, :-1] |= n_edge_x
        normal_edge[:, 1:]  |= n_edge_x
        normal_edge[:-1, :] |= n_edge_y
        normal_edge[1:, :]  |= n_edge_y
    else:
        normal_edge = torch.zeros_like(depth, dtype=torch.bool)

    # (c) alpha / background boundary
    alpha_edge = alpha < alpha_thresh

    boundary = depth_edge | normal_edge | alpha_edge

    # Fast distance transform: scipy is CPU but quick for a single 800x800 image.
    try:
        from scipy.ndimage import distance_transform_edt
        dist_px = distance_transform_edt((~boundary).cpu().numpy())
        dist_px = torch.from_numpy(dist_px).to(device=device, dtype=torch.float32)
    except ImportError:
        # Fallback: iterative erosion (bounded at ~20 steps).
        dist_px = torch.zeros_like(depth, dtype=torch.float32)
        frontier = boundary.clone()
        for step in range(1, 21):
            import torch.nn.functional as F
            grown = F.max_pool2d(frontier.float().unsqueeze(0).unsqueeze(0),
                                  kernel_size=3, stride=1, padding=1)[0, 0] > 0
            newly_hit = grown & ~frontier
            dist_px[newly_hit] = step
            frontier = grown
            if frontier.all():
                break
        dist_px[~boundary] = dist_px[~boundary].clamp_min(1.0)

    # Convert to world units via per-pixel footprint: depth * (1 / min_focal)
    inv_focal = max(1.0 / float(fx), 1.0 / float(fy))
    safe_radius_world = dist_px * depth * inv_focal
    return safe_radius_world


def silhouette_aware_depth_reinit(gaussians, scene, render_fn, pipe, background,
                                  beta, iteration, cfg, ingp=None,
                                  footprint_scale_cap=0.0,
                                  rel_depth_thresh=0.05,
                                  use_normal_edge=False):
    """Like `mini_depth_reinit + reinitial_from_depth` but with silhouette-aware
    scale clamping. Each reinit point's scale = min(nn_dist,
    footprint_cap * pixel_footprint, safe_radius_world).
    """
    import math
    views = scene.getTrainCameras()
    all_pts, all_colors, all_normals, all_safe_r, all_footprint = [], [], [], [], []

    # Snapshot old SH for transfer via max_contrib_idx.
    old_features_dc = gaussians._features_dc.detach().clone()
    old_features_rest = gaussians._features_rest.detach().clone()
    N_total = len(gaussians._xyz)

    for v in views:
        with torch.no_grad():
            rpkg = render_fn(v, gaussians, pipe, background, beta=beta,
                             iteration=iteration, cfg=cfg, ingp=ingp,
                             record_transmittance=False, is_training=False)
            gt_img = v.original_image.cuda()
            depth = rpkg.get('depth_max_contributor', None)
            if depth is None or depth.numel() == 0:
                depth = rpkg['depth_median']
            alpha = rpkg['rend_alpha']
            normal = rpkg['rend_normal']
            max_idx_map = rpkg.get('max_contrib_idx', None)

            H, W = v.image_height, v.image_width
            if hasattr(v, 'focal_x'):
                fx, fy = v.focal_x, v.focal_y
            else:
                fx = W / (2.0 * np.tan(v.FoVx / 2.0))
                fy = H / (2.0 * np.tan(v.FoVy / 2.0))
            cx, cy = W / 2.0, H / 2.0

            # Silhouette-aware safe radius (world units per pixel).
            safe_r_world = compute_safe_radius_world(
                depth, alpha, normal, fx, fy,
                rel_depth_thresh=rel_depth_thresh,
                use_normal_edge=use_normal_edge,
            )

            # Valid pixel mask (same as mini_depth_reinit)
            alpha_flat = alpha.squeeze().reshape(-1)
            depth_flat = depth.squeeze().reshape(-1)
            valid_mask = (alpha_flat > 0.01) & (depth_flat > 0.01)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            if valid_indices.shape[0] == 0:
                continue

            num_sample = max(1, int(N_total / len(views)))
            num_sample = min(num_sample, valid_indices.shape[0])
            # Importance-weight by (1 - alpha) for hole-filling
            prob = (1.0 - alpha_flat[valid_indices]).clamp(min=0)
            if prob.sum() > 0:
                prob = prob / prob.sum()
                sampled = torch.multinomial(prob, num_sample, replacement=False)
            else:
                sampled = torch.randperm(valid_indices.shape[0], device="cuda")[:num_sample]
            indices = valid_indices[sampled]

            iy = (indices // W).float()
            ix = (indices % W).float()
            d = depth.squeeze().reshape(-1)[indices]

            x_cam = (ix - cx) / fx * d
            y_cam = (iy - cy) / fy * d
            z_cam = d
            pts_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)
            w2c = v.world_view_transform.T
            c2w = torch.inverse(w2c)
            pts_world = (c2w @ pts_cam.T).T[:, :3]

            all_pts.append(pts_world)
            all_colors.append(gt_img.reshape(3, -1)[:, indices].T)
            nmap = normal.reshape(3, -1)[:, indices].T
            nmap = torch.nn.functional.normalize(nmap, dim=-1)
            all_normals.append(nmap)
            all_safe_r.append(safe_r_world.reshape(-1)[indices])
            inv_focal = max(1.0 / float(fx), 1.0 / float(fy))
            all_footprint.append(d * inv_focal)
            del rpkg
            torch.cuda.empty_cache()

    if not all_pts:
        return
    new_xyz = torch.cat(all_pts, dim=0)
    new_colors = torch.cat(all_colors, dim=0)
    new_normals = torch.cat(all_normals, dim=0)
    new_safe_r = torch.cat(all_safe_r, dim=0)
    new_footprint = torch.cat(all_footprint, dim=0)

    # Now replace the Gaussians in-place using the usual reinitial_from_depth path,
    # but override the scale-init with a silhouette-aware cap.
    from simple_knn._C import distCUDA2
    from utils.sh_utils import RGB2SH

    M = new_xyz.shape[0]
    features = torch.zeros((M, 3, (gaussians.max_sh_degree + 1) ** 2), device="cuda")
    features[:, :, 0] = RGB2SH(new_colors)

    dist2 = torch.clamp_min(distCUDA2(new_xyz), 1e-7)
    nn_dist = torch.sqrt(dist2)
    # Silhouette-aware cap (loose variant):
    #   scale = min(nn_dist, safe_radius_world)
    # The footprint cap is applied only if footprint_scale_cap > 0.
    # Interior points where safe_radius >> nn_dist keep their natural NN size.
    # Points near silhouettes get clamped to their distance-to-edge in world units.
    nn_dist = torch.minimum(nn_dist, new_safe_r.clamp_min(1e-7))
    if footprint_scale_cap > 0.0:
        cap_foot = footprint_scale_cap * new_footprint.clamp_min(1e-7)
        nn_dist = torch.minimum(nn_dist, cap_foot)
    log_scales = torch.log(nn_dist)[..., None].repeat(1, 2)

    # Normal-based quaternions
    rots = gaussians._normal_to_quaternion(new_normals)
    opacities = torch.logit(torch.ones(M, 1, device="cuda") * 0.8)
    ap_levels = torch.ones(M, 1, device="cuda") * 24

    import torch.nn as nn
    gaussians._xyz = nn.Parameter(new_xyz.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._scaling = nn.Parameter(log_scales.requires_grad_(True))
    gaussians._rotation = nn.Parameter(rots.requires_grad_(True))
    gaussians._opacity = nn.Parameter(opacities.requires_grad_(True))
    gaussians._appearance_level = nn.Parameter(ap_levels.requires_grad_(False))
    if hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
        shape_init = 1.386 * torch.ones((M, 1), dtype=torch.float, device="cuda")
        gaussians._shape = nn.Parameter(shape_init.requires_grad_(True))
    gaussians.xyz_gradient_accum = torch.zeros(M, 1, device="cuda")
    gaussians.feat_gradient_accum = torch.zeros(M, 1, device="cuda")
    gaussians.denom = torch.zeros(M, 1, device="cuda")
    gaussians.max_radii2D = torch.zeros(M, device="cuda")
    print(f"[SILHOUETTE REINIT] {N_total} -> {M} Gaussians")


# ============================================================================


def reconstruct_args(model_path):
    """Rebuild the training Namespace from cfg_args + command_line.txt."""
    # cfg_args is a Namespace(...) literal — eval it into a dict
    with open(os.path.join(model_path, "cfg_args"), "r") as f:
        text = f.read().strip()
    ns = eval(text, {"Namespace": Namespace})
    base = vars(ns).copy()

    # Parse command_line.txt for the additional training args the cfg_args
    # doesn't record (method, kernel, hybrid_levels, etc.)
    with open(os.path.join(model_path, "command_line.txt"), "r") as f:
        cmd = f.read().strip().split()
    i = 0
    extra = {}
    while i < len(cmd):
        if cmd[i].startswith("--"):
            key = cmd[i][2:]
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
                val = cmd[i + 1]
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                extra[key] = val
                i += 2
            else:
                extra[key] = True
                i += 1
        else:
            i += 1
    base.update(extra)
    return Namespace(**base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the training output directory (contains cfg_args, cameras.json, etc.)")
    parser.add_argument("--ply", type=str, required=True,
                        help="Path to the saved pre-reinit PLY")
    parser.add_argument("--algo", type=str, default="none",
                        choices=["none", "current", "silhouette"],
                        help="'none' = render as-is, 'current' = existing mini_depth_reinit, 'silhouette' = new silhouette-aware scale")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir for comparison PNGs (defaults to <model_path>/reinit_tests)")
    parser.add_argument("--cam_idx", type=int, default=0,
                        help="Train camera index to render from (default 0)")
    parser.add_argument("--footprint_scale_cap", type=float, default=0.0,
                        help="Silhouette algo: hard cap on scale = this × pixel_footprint. 0 = disabled. Default 0.")
    parser.add_argument("--rel_depth_thresh", type=float, default=0.05,
                        help="Silhouette algo: depth-edge relative threshold (neighbor Δdepth > this × depth). Default 0.05 (5%%).")
    parser.add_argument("--use_normal_edge", action="store_true",
                        help="Silhouette algo: enable normal-crease boundary detection (not recommended for curved objects).")
    parser.add_argument("--tag_suffix", type=str, default="",
                        help="Optional suffix appended to the output filenames to distinguish tunings.")
    args_cli = parser.parse_args()

    safe_state(True)
    torch.cuda.empty_cache()

    # 1. Reconstruct training args
    args = reconstruct_args(args_cli.model_path)
    args.model_path = args_cli.model_path
    print(f"Reconstructed args: source={args.source_path}, method={getattr(args, 'method', 'baseline')}, "
          f"kernel={getattr(args, 'kernel', 'gaussian')}")

    # 2. Load cfg_model yaml
    yaml_path = getattr(args, 'yaml', None)
    if yaml_path is None:
        raise RuntimeError("Missing --yaml in command_line.txt; can't reconstruct cfg_model")
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), yaml_path)
    cfg_model = Config(yaml_path)
    if getattr(args, 'cold', False):
        cfg_model.ingp_stage.initialize = 0
        cfg_model.ingp_stage.switch_iter = 0

    # 3. Build Scene + GaussianModel
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.feature_mode = getattr(args, 'feature', 'sh')

    # Minimal parser to extract ModelParams/PipelineParams/OptimizationParams defaults
    tmp_parser = argparse.ArgumentParser()
    lp = ModelParams(tmp_parser)
    op = OptimizationParams(tmp_parser)
    pp = PipelineParams(tmp_parser)
    dummy_ns = tmp_parser.parse_args([])
    # Override from the training args
    for k, v in vars(args).items():
        if hasattr(dummy_ns, k):
            setattr(dummy_ns, k, v)
        if hasattr(dummy_ns, "_" + k):
            setattr(dummy_ns, "_" + k, v)
    dataset = lp.extract(dummy_ns)
    pipe = pp.extract(dummy_ns)
    opt = op.extract(dummy_ns)
    # Cold opts
    dataset.source_path = os.path.abspath(args.source_path)

    print(f"Building Scene from {dataset.source_path}...")
    scene = Scene(dataset, gaussians, resolution_scales=[1.0], full_args=args)

    # 4. Load the saved PLY into the GaussianModel (replaces create_from_pcd's init)
    print(f"Loading PLY: {args_cli.ply}")
    gaussians.load_ply(args_cli.ply, args=args)
    print(f"  → {len(gaussians._xyz)} Gaussians loaded")
    # training_setup is needed for the optimizer — use the existing opt params
    gaussians.training_setup(opt)

    # 5. Build INGP (matches training config)
    print("Building INGP...")
    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.training_setup(cfg_model.optim)
    ingp.set_active_levels(current_iter=5000)  # sets `ingp.active_levels` used in render()

    # 6. Background
    bg_color = [1, 1, 1] if getattr(args, 'white_background', False) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    beta = 0.0
    iteration = 5000  # arbitrary, just needs a value for the render dispatch

    # 7. Apply the requested reinit algorithm
    if args_cli.algo == "current":
        print("Running CURRENT mini_depth_reinit algorithm...")
        views = scene.getTrainCameras()
        all_reinit_data = []
        old_dc = gaussians._features_dc.detach().clone()
        old_rest = gaussians._features_rest.detach().clone()
        for v in views:
            with torch.no_grad():
                rpkg = render(v, gaussians, pipe, background, beta=beta,
                              iteration=iteration, cfg=cfg_model, ingp=ingp,
                              record_transmittance=False, is_training=False)
                gt_img = v.original_image.cuda()
                depth = rpkg.get('depth_max_contributor', None)
                if depth is None or depth.numel() == 0:
                    depth = rpkg['depth_median']
                max_idx = rpkg.get('max_contrib_idx', None)
                data = gaussians.mini_depth_reinit(
                    [depth.detach()], [rpkg['rend_alpha'].detach()], [v],
                    gt_images=[gt_img], normal_maps=[rpkg['rend_normal'].detach()],
                    num_total_views=len(views),
                    max_idx_maps=[max_idx.detach()] if max_idx is not None else None,
                    src_features_dc=old_dc, src_features_rest=old_rest)
                if data is not None:
                    all_reinit_data.append({k: t.cpu() for k, t in data.items()})
                del rpkg
            torch.cuda.empty_cache()
        if all_reinit_data:
            merged = {
                'xyz': torch.cat([d['xyz'] for d in all_reinit_data], dim=0).cuda(),
                'colors': torch.cat([d['colors'] for d in all_reinit_data], dim=0).cuda() if 'colors' in all_reinit_data[0] else None,
                'normals': torch.cat([d['normals'] for d in all_reinit_data], dim=0).cuda() if 'normals' in all_reinit_data[0] else None,
                'sh_dc': torch.cat([d['sh_dc'] for d in all_reinit_data], dim=0).cuda() if 'sh_dc' in all_reinit_data[0] else None,
                'sh_rest': torch.cat([d['sh_rest'] for d in all_reinit_data], dim=0).cuda() if 'sh_rest' in all_reinit_data[0] else None,
            }
            gaussians.reinitial_from_depth(merged)
            gaussians.training_setup(opt)
    elif args_cli.algo == "silhouette":
        print(f"Running SILHOUETTE-aware depth reinit "
              f"(footprint_cap={args_cli.footprint_scale_cap}, "
              f"rel_depth_thresh={args_cli.rel_depth_thresh}, "
              f"use_normal_edge={args_cli.use_normal_edge})...")
        silhouette_aware_depth_reinit(gaussians, scene, render, pipe, background,
                                       beta=beta, iteration=iteration, cfg=cfg_model,
                                       ingp=ingp,
                                       footprint_scale_cap=args_cli.footprint_scale_cap,
                                       rel_depth_thresh=args_cli.rel_depth_thresh,
                                       use_normal_edge=args_cli.use_normal_edge)
        gaussians.training_setup(opt)
    else:
        print("Skipping reinit (algo=none).")

    # 8. Render the chosen train camera and save images
    out_dir = args_cli.out_dir or os.path.join(args_cli.model_path, "reinit_tests")
    os.makedirs(out_dir, exist_ok=True)
    view = scene.getTrainCameras()[args_cli.cam_idx]
    print(f"Rendering train camera [{args_cli.cam_idx}] → {out_dir}")
    with torch.no_grad():
        rpkg = render(view, gaussians, pipe, background, beta=beta,
                      iteration=iteration, cfg=cfg_model, ingp=ingp,
                      record_transmittance=False, is_training=False)
        tag = f"iter5000_algo_{args_cli.algo}"
        if args_cli.tag_suffix:
            tag = f"{tag}_{args_cli.tag_suffix}"
        rgb = torch.clamp(rpkg['render'], 0, 1)
        save_img_u8(rgb.permute(1, 2, 0).cpu().numpy(),
                    os.path.join(out_dir, f"{tag}_rgb.png"))
        mci = rpkg.get('max_contrib_idx', None)
        if mci is not None:
            save_img_u8(_colorize_max_contrib_idx(mci),
                        os.path.join(out_dir, f"{tag}_maxcontrib_id.png"))
        d_max = rpkg.get('depth_max_contributor', None)
        if d_max is not None:
            save_img_u8(convert_gray_to_cmap(d_max.squeeze().cpu().numpy(),
                                             map_mode='turbo', revert=False),
                        os.path.join(out_dir, f"{tag}_depth_maxcontrib.png"))
        # Normal map — stored as rend_normal [3, H, W] world-space unit vectors.
        # Visualize as (n * 0.5 + 0.5) mapped into [0, 1] RGB.
        rnormal = rpkg.get('rend_normal', None)
        if rnormal is not None and rnormal.numel() > 0:
            n_vis = (rnormal * 0.5 + 0.5).clamp(0.0, 1.0)
            save_img_u8(n_vis.permute(1, 2, 0).cpu().numpy(),
                        os.path.join(out_dir, f"{tag}_normal.png"))
    print(f"Saved images with tag {tag} in {out_dir}")


if __name__ == "__main__":
    main()

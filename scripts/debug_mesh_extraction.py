#!/usr/bin/env python3
"""
debug_mesh_extraction.py
========================

Sweep TSDF mesh extraction params on the textured-surfel half to figure out
WHY the resulting mesh is missing parts of the scene (walls, bicycle, etc.)
that are clearly visible in the textured-only render.

Saves to <bake_subdir>/mesh_debug/:
  - view0_depth.png / view0_alpha.png / view0_rgb.png
        — what TSDF actually sees for camera 0 (input to integration).
  - mesh_raw.ply
        — RAW TSDF output with NO post-processing. Reveals whether
          post_process_mesh's cluster filter is what's dropping the walls.
  - mesh_kept_<N>.ply
        — post_process_mesh with several cluster_to_keep values, so you
          can see how many clusters survive at each.
  - mesh_depthtrunc_<X>.ply
        — sweep depth_trunc to see if far elements are being clipped.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser

import open3d as o3d
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.mesh_utils import GaussianExtractor, post_process_mesh, to_cam_open3d
from utils.system_utils import searchForMaxIteration
from utils.render_utils import save_img_u8, convert_gray_to_cmap
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args

from bake_mesh_texture import install_setter_mirror, fold_train_args, slice_to_textured


def integrate_tsdf(extractor, voxel_size, sdf_trunc, depth_trunc,
                   depth_floor=0.0, alpha_min=0.0):
    """Run TSDF integration over `extractor.viewpoint_stack` with the cached
    depth/rgb/alpha maps. Optionally gate by `alpha < alpha_min` (zero those
    pixels) and `depth < depth_floor` (also zeroed). Returns the mesh."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    cams_o3d = to_cam_open3d(extractor.viewpoint_stack)
    for i, cam_o3d in enumerate(cams_o3d):
        rgb = extractor.rgbmaps[i]
        depth = extractor.depthmaps[i].clone()
        alpha = extractor.alphamaps[i] if extractor.alphamaps else None
        if alpha is not None and alpha_min > 0:
            depth[alpha < alpha_min] = 0
        if depth_floor > 0:
            depth[depth < depth_floor] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
                               .astype(np.float32).copy() * 255).astype(np.uint8) if False
            else o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1, 2, 0).cpu().numpy(),
                                                       0.0, 1.0) * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")),
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False, depth_scale=1.0)
        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
    return volume.extract_triangle_mesh()


def main():
    parser = ArgumentParser(description="Diagnose TSDF mesh extraction on textured-half")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--bake_subdir", default="mesh_bake", type=str)
    parser.add_argument("--mesh_res", default=1024, type=int)
    parser.add_argument("--depth_trunc_sweep", default="auto,5,10,20", type=str,
                        help="Comma-separated depth_trunc values to try")
    parser.add_argument("--cluster_keep_sweep", default="200,1000,5000,50000", type=str)
    parser.add_argument("--limit_views", default=-1, type=int)
    args = get_combined_args(parser)
    args = fold_train_args(args, args.model_path)

    exp_path = args.model_path
    iteration = args.iteration
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(exp_path, "point_cloud"))

    install_setter_mirror(getattr(args, "method", None))

    yaml_file = getattr(args, "yaml", None) or "tiny"
    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    print(f"[DBG] Model: {exp_path}")
    print(f"[DBG] Iter:  {iteration}")

    # Setup
    ingp = INGP(cfg_model, args=args).to("cuda")
    ingp.load_model(exp_path, iteration)
    ingp.set_active_levels(iteration)
    if args.method in ("mixed", "mixed_3d"):
        from diff_surfel_3D_sh_res import (
            set_residual_mode, set_activation_bias, set_compact_mult)
        sh_b, res_b = getattr(args, "activation_bias", [0.5, 0.0])
        set_activation_bias(sh_bias=float(sh_b), res_bias=float(res_b))
        set_residual_mode(2)
        set_compact_mult(float(getattr(args, "fastgs_mult", 0.5)))

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.kernel_type = getattr(args, "kernel", "gaussian")
    gaussians.kernel_type2 = getattr(args, "kernel2", None)
    gaussians.feature_mode = getattr(args, "feature", "sh")
    gaussians._sv_training_flag = False

    n_tex = int(gaussians._is_textured.sum().item())
    print(f"[DBG] {gaussians.get_xyz.shape[0]:,} surfels total, textured={n_tex:,}")
    slice_to_textured(gaussians)
    print(f"[DBG] After slice: {gaussians.get_xyz.shape[0]:,} textured-only surfels")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    beta_cfg = cfg_model.surfel.tg_beta

    out_dir = os.path.join(exp_path, args.bake_subdir, "mesh_debug")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[DBG] Output: {out_dir}")

    train_cams = scene.getTrainCameras().copy()
    if args.limit_views > 0:
        train_cams = train_cams[: args.limit_views]
    print(f"[DBG] Reconstructing {len(train_cams)} views (TSDF source) …")

    extractor = GaussianExtractor(
        render, gaussians, pipe, bg, ingp=ingp,
        beta=beta_cfg, iteration=iteration, cfg=cfg_model)
    extractor.reconstruction(train_cams)
    auto_depth_trunc = extractor.radius * 2.0
    voxel_size = (auto_depth_trunc / args.mesh_res)
    sdf_trunc = 5.0 * voxel_size
    print(f"[DBG] camera_radius = {extractor.radius:.3f}  auto_depth_trunc = {auto_depth_trunc:.3f}")
    print(f"[DBG] voxel_size = {voxel_size:.5f}  sdf_trunc = {sdf_trunc:.5f}")

    # --- Save the depth/alpha/rgb that TSDF will see for view 0 -----------
    rgb0 = extractor.rgbmaps[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    depth0 = extractor.depthmaps[0].squeeze().cpu().numpy()
    alpha0 = (extractor.alphamaps[0].squeeze().cpu().numpy()
              if extractor.alphamaps else None)
    save_img_u8(rgb0, os.path.join(out_dir, "view0_rgb.png"))
    d_norm = np.clip(depth0 / max(1e-6, auto_depth_trunc), 0, 1)
    save_img_u8(convert_gray_to_cmap(d_norm, map_mode='turbo', revert=False),
                os.path.join(out_dir, "view0_depth.png"))
    if alpha0 is not None:
        save_img_u8(np.stack([alpha0]*3, axis=-1), os.path.join(out_dir, "view0_alpha.png"))
    # Plain bool mask of where depth is finite & nonzero
    valid = (depth0 > 0).astype(np.float32)
    save_img_u8(np.stack([valid]*3, axis=-1), os.path.join(out_dir, "view0_depth_valid_mask.png"))
    print(f"[DBG] view0 depth: min={depth0.min():.3f} max={depth0.max():.3f} "
          f"nonzero_frac={float((depth0>0).mean()):.3f}")
    if alpha0 is not None:
        print(f"[DBG] view0 alpha: mean={alpha0.mean():.3f} max={alpha0.max():.3f}")

    # --- 1. Raw TSDF, no post-processing, default depth_trunc -------------
    print(f"[DBG] Extracting RAW mesh (no post-process)…")
    raw_mesh = extractor.extract_mesh_bounded(
        voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=auto_depth_trunc)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh_raw.ply"), raw_mesh)
    print(f"[DBG]   raw V={len(raw_mesh.vertices):,}  F={len(raw_mesh.triangles):,}")

    # Count clusters
    print(f"[DBG] Clustering raw mesh…")
    triangle_clusters, cluster_n_triangles, cluster_area = (raw_mesh.cluster_connected_triangles())
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    print(f"[DBG]   total clusters = {len(cluster_n_triangles)}")
    print(f"[DBG]   largest cluster sizes (top 10): "
          f"{np.sort(cluster_n_triangles)[-10:][::-1].tolist()}")
    print(f"[DBG]   cluster size median = {int(np.median(cluster_n_triangles))}")

    # --- 2. Sweep cluster_keep -------------------------------------------
    for keep_s in args.cluster_keep_sweep.split(","):
        keep = int(keep_s.strip())
        if keep <= 0:
            continue
        try:
            kept = post_process_mesh(raw_mesh, cluster_to_keep=keep)
            o3d.io.write_triangle_mesh(
                os.path.join(out_dir, f"mesh_kept_{keep}.ply"), kept)
            print(f"[DBG]   keep={keep}: V={len(kept.vertices):,} "
                  f"F={len(kept.triangles):,}")
        except Exception as e:
            print(f"[DBG]   keep={keep}: FAILED — {e}")

    # --- 3. Sweep depth_trunc --------------------------------------------
    print(f"[DBG] Sweeping depth_trunc …")
    for trunc_s in args.depth_trunc_sweep.split(","):
        trunc_s = trunc_s.strip()
        if trunc_s.lower() == "auto":
            tr = auto_depth_trunc
        else:
            tr = float(trunc_s)
        vs = tr / args.mesh_res
        st = 5.0 * vs
        try:
            m = extractor.extract_mesh_bounded(
                voxel_size=vs, sdf_trunc=st, depth_trunc=tr)
            o3d.io.write_triangle_mesh(
                os.path.join(out_dir, f"mesh_depthtrunc_{trunc_s}.ply"), m)
            print(f"[DBG]   depth_trunc={tr:.2f}: V={len(m.vertices):,} "
                  f"F={len(m.triangles):,}")
        except Exception as e:
            print(f"[DBG]   depth_trunc={trunc_s}: FAILED — {e}")

    print(f"[DBG] Done. Inspect: {out_dir}")
    print(f"[DBG]   - mesh_raw.ply       : pre-post-process baseline")
    print(f"[DBG]   - mesh_kept_*.ply    : effect of cluster_to_keep")
    print(f"[DBG]   - mesh_depthtrunc_*.ply : effect of depth_trunc")
    print(f"[DBG]   - view0_depth.png    : what TSDF sees per pixel")


if __name__ == "__main__":
    main()

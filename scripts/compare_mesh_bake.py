#!/usr/bin/env python3
"""
compare_mesh_bake.py
====================

Side-by-side comparison of a baked mesh RGBA layer against the original
textured-surfel render, on N test views.

Renders, for each test view:
  - target.png   : original textured surfels, full pipeline (SV + MLP residual + ReLU)
  - sv_only.png  : the same textured surfels with MLP zeroed
                   (= the "untextured surfels" the bake replaces them with)
  - mesh.png     : the mesh-Gaussian RGBA layer alone (signed RGB; clipped for save)
  - pred.png     : composite ReLU(α_mesh·rgb_mesh + (1-α_mesh)·sv_only)
                   — this is what the bake reproduces in lieu of `target`
  - diff.png     : |target - pred| · 5 (×5 boost for visibility)

Terminology (from user clarification):
  - "textured surfels"   = 2DGS beta_scaled kernels with MLP residual.
  - "untextured Gaussians" = the EWA 3D ellipsoids (the other half of mixed_3d).
    Not touched by this bake.
  - "untextured surfels" = textured surfels with MLP discarded (SV-only on 2DGS).
    Generated at render time via decompose_mode='sh_only'. The bake targets
    these as the background; the mesh RGBA layer compensates for the lost
    residual.
"""

import os
import sys
import math
import pickle
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
from utils.system_utils import searchForMaxIteration
from utils.render_utils import save_img_u8
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args

# Reuse helpers from the bake script.
from bake_mesh_texture import (
    install_setter_mirror, fold_train_args, slice_to_textured,
    build_mesh_gaussian_model,
)


def load_rgba_from_ply(ply_path: str):
    """Read per-vertex (x,y,z) + (R,G,B,A) FP32 + triangle indices from the PLY
    saved by bake_mesh_texture.save_mesh_rgba."""
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    verts = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], axis=1)
    rgb = np.stack([np.asarray(v['red']), np.asarray(v['green']), np.asarray(v['blue'])], axis=1).astype(np.float32)
    alpha = np.asarray(v['alpha']).astype(np.float32)
    faces = np.stack(plydata['face']['vertex_indices'], axis=0).astype(np.int32)
    return verts, rgb, alpha, faces


def composite(view, mesh_gm, mesh_rgb_param, surf_rgb, pipe, bg_zero, cfg_model):
    """Render the mesh-Gaussian layer with signed override_color and alpha-over
    onto surf_rgb. Returns (mesh_rgb_rendered, mesh_alpha, pred)."""
    mesh_pkg = render(
        view, mesh_gm, pipe, bg_zero,
        ingp=None, beta=0.0, iteration=0, cfg=cfg_model,
        override_color=mesh_rgb_param,
        skybox=None, background_mode="none", bg_hashgrid=None,
        is_training=False,
    )
    mesh_rgb_rendered = mesh_pkg["render"]                  # signed
    mesh_alpha = mesh_pkg.get("rend_alpha", None)
    if mesh_alpha is None:
        mesh_alpha = (mesh_rgb_rendered.abs().sum(0, keepdim=True) > 0).float()
    pred = mesh_alpha * mesh_rgb_rendered + (1.0 - mesh_alpha) * surf_rgb
    pred = torch.relu(pred).clamp(0.0, 1.0)
    return mesh_rgb_rendered, mesh_alpha, pred


def main():
    parser = ArgumentParser(description="Compare baked mesh RGBA vs original textured render")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--bake_subdir", default="mesh_bake", type=str,
                        help="Subdir under model_path containing mesh_rgba_final.ply")
    parser.add_argument("--bake_ply", default="mesh_rgba_final.ply", type=str,
                        help="Filename of the baked PLY (default: final)")
    parser.add_argument("--num_views", default=5, type=int,
                        help="Number of test views to render (default 5, -1 = all)")
    parser.add_argument("--out_subdir", default=None, type=str,
                        help="Output subdir (default: <bake_subdir>/compare)")
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

    print(f"[CMP] Model:   {exp_path}")
    print(f"[CMP] Iter:    {iteration}")
    print(f"[CMP] Method:  {getattr(args, 'method', '?')}")

    # INGP + CUDA globals
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

    # Scene
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.kernel_type = getattr(args, "kernel", "gaussian")
    gaussians.kernel_type2 = getattr(args, "kernel2", None)
    gaussians.feature_mode = getattr(args, "feature", "sh")
    gaussians._sv_training_flag = False

    print(f"[CMP] Loaded model: {gaussians.get_xyz.shape[0]:,} surfels "
          f"(textured {int(gaussians._is_textured.sum().item()):,})")
    slice_to_textured(gaussians)
    print(f"[CMP] Sliced to textured-only: {gaussians.get_xyz.shape[0]:,}")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    bg_zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta_cfg = cfg_model.surfel.tg_beta

    # Load baked PLY
    bake_dir = os.path.join(exp_path, args.bake_subdir)
    ply_path = os.path.join(bake_dir, args.bake_ply)
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"No baked PLY at {ply_path}")
    print(f"[CMP] Loading baked PLY: {ply_path}")
    verts, rgb, alpha, faces = load_rgba_from_ply(ply_path)
    print(f"[CMP]   V={verts.shape[0]:,}  F={faces.shape[0]:,}  "
          f"RGB range=[{rgb.min():+.2f}, {rgb.max():+.2f}]  "
          f"alpha mean={alpha.mean():.3f}")

    # Rebuild a triangle mesh (Open3D) so build_mesh_gaussian_model can fetch
    # vertex normals via mesh.compute_vertex_normals(). Geometry, not bake.
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    # disk_scale: same heuristic as bake script (so the mesh-Gaussian layout
    # matches what optimization saw).
    if faces.shape[0] > 0:
        e0 = np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=1)
        e1 = np.linalg.norm(verts[faces[:, 1]] - verts[faces[:, 2]], axis=1)
        e2 = np.linalg.norm(verts[faces[:, 2]] - verts[faces[:, 0]], axis=1)
        mean_edge = float(np.mean(np.concatenate([e0, e1, e2])))
    else:
        mean_edge = 1.0
    disk_scale = 0.7 * mean_edge

    mesh_gm, _, _ = build_mesh_gaussian_model(o3d_mesh, sh_degree=0, disk_scale=disk_scale)

    # Inject the BAKED RGBA into the mesh-Gaussian sheet:
    #   mesh_rgb_param = signed FP32 RGB (verbatim from PLY)
    #   _opacity = logit(alpha)  so that get_opacity(via sigmoid) returns `alpha`.
    mesh_rgb_param = torch.tensor(rgb, dtype=torch.float32, device="cuda")
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device="cuda").clamp(1e-6, 1 - 1e-6)
    logit_alpha = torch.log(alpha_t / (1 - alpha_t)).unsqueeze(-1)
    mesh_gm._opacity = nn.Parameter(logit_alpha, requires_grad=False)

    # Select views
    test_cams = scene.getTestCameras().copy()
    if args.num_views > 0:
        test_cams = test_cams[: args.num_views]
    print(f"[CMP] Rendering {len(test_cams)} test views")

    # `get_combined_args` selectively folds saved cfg into the namespace and
    # can drop newly-added CLI flags; use getattr with default to be safe.
    _out_sub = getattr(args, "out_subdir", None)
    if _out_sub:
        out_dir = _out_sub if os.path.isabs(_out_sub) else os.path.join(exp_path, _out_sub)
    else:
        out_dir = os.path.join(bake_dir, "compare")  # bake_dir is already absolute
    for sub in ("target", "sv_only", "mesh", "pred", "diff"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    print(f"[CMP] Output:  {out_dir}")

    psnrs = []
    with torch.no_grad():
        for idx, cam in enumerate(tqdm(test_cams, desc="compare")):
            # Target: textured surfels, full pipeline
            tgt_pkg = render(cam, gaussians, pipe, bg, ingp=ingp,
                             beta=beta_cfg, iteration=iteration, cfg=cfg_model,
                             skybox=None, background_mode="none", bg_hashgrid=None)
            target = torch.clamp(tgt_pkg["render"], 0.0, 1.0)
            # SV-only: textured surfels with MLP zeroed ("untextured surfels")
            sv_pkg = render(cam, gaussians, pipe, bg, ingp=ingp,
                            beta=beta_cfg, iteration=iteration, cfg=cfg_model,
                            skybox=None, background_mode="none", bg_hashgrid=None,
                            decompose_mode='sh_only')
            surf_rgb = sv_pkg["render"]  # unclamped
            # Mesh + composite
            mesh_rgb_rendered, mesh_alpha, pred = composite(
                cam, mesh_gm, mesh_rgb_param, surf_rgb, pipe, bg_zero, cfg_model)

            # PSNR (pred vs target, both clamped)
            mse = ((pred - target) ** 2).mean().item()
            psnrs.append(-10.0 * math.log10(mse + 1e-12))

            # Save PNGs
            name = f"{idx:03d}"
            save_img_u8(target.permute(1, 2, 0).cpu().numpy(),
                        os.path.join(out_dir, "target",  f"{name}.png"))
            save_img_u8(torch.clamp(surf_rgb, 0.0, 1.0).permute(1, 2, 0).cpu().numpy(),
                        os.path.join(out_dir, "sv_only", f"{name}.png"))
            # mesh layer alone — clamp signed to [0,1] just for visualization
            save_img_u8(torch.clamp(mesh_rgb_rendered, 0.0, 1.0)
                            .permute(1, 2, 0).cpu().numpy(),
                        os.path.join(out_dir, "mesh",    f"{name}.png"))
            save_img_u8(pred.permute(1, 2, 0).cpu().numpy(),
                        os.path.join(out_dir, "pred",    f"{name}.png"))
            diff = (target - pred).abs().clamp(0.0, 1.0) * 5.0  # 5× boost
            save_img_u8(diff.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(),
                        os.path.join(out_dir, "diff",    f"{name}.png"))

    print(f"[CMP] Test PSNR (pred vs target): "
          f"mean {np.mean(psnrs):.2f} dB, median {np.median(psnrs):.2f} dB, "
          f"min {np.min(psnrs):.2f}, max {np.max(psnrs):.2f}")
    print(f"[CMP] Wrote target/sv_only/mesh/pred/diff PNGs to {out_dir}")


if __name__ == "__main__":
    main()

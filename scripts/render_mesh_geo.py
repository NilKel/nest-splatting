#!/usr/bin/env python3
"""
render_mesh_geo.py
==================

Render the BARE MESH (no learned color, no SV background) from N test views,
to diagnose whether the TSDF mesh geometry itself is good or whether bake
artifacts are coming from mesh holes / wrong topology.

Two outputs per view (under <bake_subdir>/mesh_geo/):
  - mesh_disks/<NNN>.png : mesh-Gaussian sheet rendered with white color +
                           full opacity (raw _opacity=+5 → sigmoid≈1).
                           Bright = disk coverage; dark = gap or missing
                           geometry. Reveals disk-overlap artifacts directly.
  - mesh_alpha/<NNN>.png : same, but visualized as the rendered alpha map
                           alone — pure coverage mask.

Also tries to render true mesh triangles via Open3D's OffscreenRenderer
(falls back silently if headless GL isn't available).
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
from utils.system_utils import searchForMaxIteration
from utils.render_utils import save_img_u8
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args

from bake_mesh_texture import (
    install_setter_mirror, fold_train_args, slice_to_textured,
    build_mesh_gaussian_model,
)
from compare_mesh_bake import load_rgba_from_ply


def try_render_o3d_triangles(mesh_path, cameras, out_dir):
    """Best-effort: render the mesh's TRIANGLES (not Gaussian disks) with
    Open3D's offscreen renderer. Returns True on success.
    Requires a working GL/EGL context — silently skipped on headless boxes."""
    try:
        import open3d.visualization.rendering as rendering
    except (ImportError, AttributeError) as e:
        print(f"[GEO] Open3D rendering module unavailable: {e}")
        return False
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        # Strip any vertex colors so material color shows through.
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.zeros((len(mesh.vertices), 3)) + 0.85)

        for idx, cam in enumerate(cameras):
            W, H = cam.image_width, cam.image_height
            renderer = rendering.OffscreenRenderer(W, H)
            renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
            mat = rendering.MaterialRecord()
            mat.base_color = [0.85, 0.85, 0.85, 1.0]
            mat.shader = "defaultLit"
            renderer.scene.add_geometry("mesh", mesh, mat)
            # Intrinsics / extrinsics from the nest-splatting camera
            fx = W / (2.0 * math.tan(cam.FoVx / 2.0))
            fy = H / (2.0 * math.tan(cam.FoVy / 2.0))
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            extrinsic = np.asarray(cam.world_view_transform.T.cpu().numpy())
            renderer.setup_camera(intrinsic, extrinsic)
            img = renderer.render_to_image()
            os.makedirs(out_dir, exist_ok=True)
            o3d.io.write_image(os.path.join(out_dir, f"{idx:03d}.png"), img)
            del renderer  # release GL context
        return True
    except Exception as e:
        print(f"[GEO] Open3D offscreen render failed (headless?): {e}")
        return False


def main():
    parser = ArgumentParser(description="Render bare mesh geometry from test views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--bake_subdir", default="mesh_bake", type=str)
    parser.add_argument("--bake_ply", default="mesh_rgba_final.ply", type=str)
    parser.add_argument("--num_views", default=5, type=int)
    parser.add_argument("--mesh_disk_scale", default=0.7, type=float,
                        help="Same as bake script — to match the same disk size used in optimization")
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

    print(f"[GEO] Model: {exp_path}")
    print(f"[GEO] Iter:  {iteration}")

    # INGP + globals (not used for the bare-mesh render, but kept so
    # Scene/render() pipelines are happy and the camera intrinsics match.)
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

    bg_zero = torch.zeros(3, dtype=torch.float32, device="cuda")

    # Load mesh (use the SAME ply the bake produced — same vertex set / faces).
    bake_dir = os.path.join(exp_path, args.bake_subdir)
    ply_path = os.path.join(bake_dir, args.bake_ply)
    if not os.path.exists(ply_path):
        # Fall back to the raw TSDF mesh if the textured PLY isn't there.
        ply_path = os.path.join(bake_dir, "mesh_textured.ply")
    print(f"[GEO] Mesh: {ply_path}")
    verts, _, _, faces = load_rgba_from_ply(ply_path) if 'rgba' in os.path.basename(ply_path) \
                        else (np.asarray(o3d.io.read_triangle_mesh(ply_path).vertices),
                              None, None,
                              np.asarray(o3d.io.read_triangle_mesh(ply_path).triangles).astype(np.int32))

    # Reconstruct an Open3D mesh from verts/faces for build_mesh_gaussian_model
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    print(f"[GEO] V={len(o3d_mesh.vertices):,}  F={len(o3d_mesh.triangles):,}")

    # disk_scale from mean face edge — identical to bake script
    if faces.shape[0] > 0:
        e0 = np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=1)
        e1 = np.linalg.norm(verts[faces[:, 1]] - verts[faces[:, 2]], axis=1)
        e2 = np.linalg.norm(verts[faces[:, 2]] - verts[faces[:, 0]], axis=1)
        mean_edge = float(np.mean(np.concatenate([e0, e1, e2])))
    else:
        mean_edge = 1.0
    disk_scale = args.mesh_disk_scale * mean_edge
    print(f"[GEO] mean_edge={mean_edge:.5f}, disk_scale={disk_scale:.5f}")

    mesh_gm, _, _ = build_mesh_gaussian_model(o3d_mesh, sh_degree=0, disk_scale=disk_scale)
    M = mesh_gm.get_xyz.shape[0]

    # Force full opacity (sigmoid(+5) ≈ 0.993) — bypass the optimized alpha so
    # we see PURE GEOMETRY coverage, not where the optimizer decided to keep
    # alpha high.
    mesh_gm._opacity = nn.Parameter(torch.full((M, 1), 5.0, device="cuda"),
                                    requires_grad=False)
    # White color via override_color in render() (signed FP32 path)
    mesh_rgb_white = torch.ones((M, 3), device="cuda")

    # Cameras
    test_cams = scene.getTestCameras().copy()
    if args.num_views > 0:
        test_cams = test_cams[: args.num_views]
    print(f"[GEO] Rendering {len(test_cams)} test views")

    out_dir = os.path.join(bake_dir, "mesh_geo")
    disk_dir = os.path.join(out_dir, "mesh_disks")
    alpha_dir = os.path.join(out_dir, "mesh_alpha")
    tri_dir = os.path.join(out_dir, "mesh_triangles_o3d")
    os.makedirs(disk_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)
    print(f"[GEO] Output: {out_dir}")

    # ---- 1. Gaussian-disk render (always works) ----
    with torch.no_grad():
        cover_means = []
        for idx, cam in enumerate(tqdm(test_cams, desc="disks")):
            pkg = render(cam, mesh_gm, pipe, bg_zero,
                         ingp=None, beta=0.0, iteration=0, cfg=cfg_model,
                         override_color=mesh_rgb_white,
                         skybox=None, background_mode="none", bg_hashgrid=None,
                         is_training=False)
            rgb = torch.clamp(pkg["render"], 0.0, 1.0)
            alpha = pkg.get("rend_alpha", None)
            save_img_u8(rgb.permute(1, 2, 0).cpu().numpy(),
                        os.path.join(disk_dir, f"{idx:03d}.png"))
            if alpha is not None:
                a = alpha.clamp(0.0, 1.0).expand(3, -1, -1).permute(1, 2, 0).cpu().numpy()
                save_img_u8(a, os.path.join(alpha_dir, f"{idx:03d}.png"))
                cover_means.append(float(alpha.mean().item()))
    if cover_means:
        print(f"[GEO] Mesh coverage (mean alpha across views): "
              f"{np.mean(cover_means):.3f} (min {np.min(cover_means):.3f}, "
              f"max {np.max(cover_means):.3f})")

    # ---- 2. True triangle render via Open3D (best-effort) ----
    print(f"[GEO] Trying Open3D triangle rasterization → {tri_dir}")
    ok = try_render_o3d_triangles(ply_path, test_cams, tri_dir)
    if ok:
        print(f"[GEO] Triangle renders saved.")
    else:
        print(f"[GEO] Skipped triangle pass (no working offscreen GL).")

    print(f"[GEO] Done.")


if __name__ == "__main__":
    main()

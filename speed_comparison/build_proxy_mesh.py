#!/usr/bin/env python3
"""Build a per-scene proxy occlusion mesh via TSDF fusion over the neural
renderer's median-depth maps at each training/test view. Result is a
triangle mesh that sits at (or very near) the 'opaque frontier' for
each view — the depth at which cumulative T drops through 0.5.

Use case: on WebGPU we have no per-pixel early termination. If we can
depth-test surfels against this mesh in a pre-pass, we reject the
ones behind the opaque frontier before they run the fragment shader.
The mesh needs to be right AT or slightly BEHIND the last visible
depth for every training-view pixel so we never accidentally cull
something visible.

Depth used: median depth from `out_others[5:6]` (T=0.5 crossing). Set
`--depth_ratio 0` to blend in the expected depth (unbounded-scene mode
from the 2DGS paper). `--push_back` adds a multiplicative fudge to
push the mesh slightly BEHIND the frontier so an erosion step later
is easier.

Uses `utils/mesh_utils.py`'s existing `GaussianExtractor` +
`ScalableTSDFVolume` pipeline (Open3D CPU-side TSDF integration).

Usage:
    conda run -n nest_splatting python speed_comparison/build_proxy_mesh.py \
        --model_path outputs/mip_360/bonsai/3D_SH_res/... \
        --out_dir speed_comparison/proxy_meshes/bonsai \
        [--depth_ratio 1.0] [--push_back 1.0] [--voxel 0.004]
"""
import argparse, glob, json, os, pickle, sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.mesh_utils import GaussianExtractor, post_process_mesh


def load_train_args(model_path: Path):
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    return Namespace(**json.load(open(model_path / "args.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--out_dir",    type=Path, required=True)
    ap.add_argument("--use_train_views", action="store_true",
                    help="Fuse over train + test cameras instead of just test.")
    ap.add_argument("--depth_ratio", type=float, default=1.0,
                    help="1.0 = pure median depth (T=0.5 crossing, deeper), "
                         "0.0 = pure expected depth (weighted avg, tends closer). "
                         "Bounded/indoor scenes: use 1.0. Under LAST_DEPTH_MODE build "
                         "the median slot carries the deepest-contributor depth.")
    ap.add_argument("--push_back", type=float, default=1.0,
                    help="Multiply all depth values by this before fusion. >1 pushes "
                         "the mesh farther from the camera. BUT: multi-view fusion "
                         "gets confused because the shift is view-dependent (near "
                         "objects shift less absolutely than far ones), producing "
                         "fragmented meshes. Prefer --depth_bias for a uniform "
                         "additive shift.")
    ap.add_argument("--depth_bias", type=float, default=0.0,
                    help="Add this METRES to every depth value before fusion. Uniform "
                         "additive shift: all cameras agree on the same world-space "
                         "surface position (just biased outward by depth_bias along "
                         "each ray). Positive = mesh sits farther from cameras.")
    ap.add_argument("--min_alpha", type=float, default=0.99,
                    help="Fuse only pixels where render_alpha >= this. render_alpha "
                         "= 1 - final_T, so 0.99 means 'T saturated'. Background pixels "
                         "(where T never reached zero) are skipped → mesh has HOLES "
                         "there instead of a phantom back-of-scene surface. That's "
                         "correct for Z-cull: WebGPU can't skip surfels for background "
                         "pixels; the mesh should only exist where the CUDA renderer "
                         "actually saturated.")
    ap.add_argument("--voxel", type=float, default=0.004,
                    help="TSDF voxel size (world units). Smaller = finer mesh, slower.")
    ap.add_argument("--sdf_trunc", type=float, default=0.02,
                    help="TSDF truncation band. Rule of thumb: 5-10× voxel.")
    ap.add_argument("--depth_trunc", type=float, default=6.0,
                    help="Skip depth values beyond this. Prevents skybox contamination.")
    ap.add_argument("--mask_background", action="store_true",
                    help="Mask depths where alpha < 0.5 as background (skip).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    train_args = load_train_args(args.model_path)
    train_args.model_path = str(args.model_path)
    train_args.eval = True
    cfg_yaml = args.model_path / "config.yaml"
    cfg_model = Config(str(cfg_yaml)) if cfg_yaml.exists() else Config(train_args.yaml)

    ngp_files = glob.glob(str(args.model_path / "ngp_*.pth"))
    iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                    for f in ngp_files)

    tp = argparse.ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(train_args)
    pipe    = PipelineParams(tp).extract(train_args)
    # Force the depth_ratio surf_depth blend used inside the renderer.
    pipe.depth_ratio = float(args.depth_ratio)

    ingp = INGP(cfg_model, args=train_args).to("cuda")
    ingp.load_model(str(args.model_path), iteration)
    ingp.set_active_levels(iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(train_args, "kernel"):
        gaussians.kernel_type = train_args.kernel

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta = float(cfg_model.surfel.tg_beta) if hasattr(cfg_model.surfel, "tg_beta") else 0.0

    # ---- Cameras to fuse over ----
    cams = scene.getTestCameras()
    if args.use_train_views:
        cams = scene.getTrainCameras() + cams
    print(f"[proxy_mesh] fusing over {len(cams)} views, {gaussians.get_xyz.shape[0]:,} Gauss")
    print(f"[proxy_mesh] depth_ratio={args.depth_ratio} push_back={args.push_back} "
          f"voxel={args.voxel} sdf_trunc={args.sdf_trunc} depth_trunc={args.depth_trunc}")

    # ---- Render depth per view via the existing GaussianExtractor ----
    # It binds pipe/bg/ingp/beta/iteration/cfg into a partial() itself.
    extractor = GaussianExtractor(render, gaussians, pipe, background=bg,
                                   ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model)
    extractor.reconstruction(cams, render_rays=False)

    # ---- Alpha-mask: skip pixels where the rasterizer never saturated. ----
    # Background pixels (T stayed high, alpha < 1) should NOT contribute to the
    # mesh — we want holes there so the WebGPU Z-cull test naturally passes
    # every surfel through for those pixels. We flip them to a sentinel depth
    # larger than depth_trunc so Open3D's TSDF integrator drops them.
    n_masked = 0; n_total = 0
    for i in range(len(extractor.depthmaps)):
        d     = extractor.depthmaps[i]
        alpha = extractor.alphamaps[i]
        bg    = (alpha < args.min_alpha)
        n_masked += int(bg.sum().item())
        n_total  += int(bg.numel())
        # depth_trunc + big => TSDF skips this pixel
        extractor.depthmaps[i] = torch.where(bg,
                                              torch.full_like(d, args.depth_trunc + 100.0),
                                              d)
    print(f"[proxy_mesh] alpha-mask: {n_masked:,}/{n_total:,} pixels dropped "
          f"({100*n_masked/n_total:.1f}%) — below alpha {args.min_alpha}")

    # ---- Apply push_back fudge in-place before TSDF ----
    if args.push_back != 1.0:
        for i in range(len(extractor.depthmaps)):
            extractor.depthmaps[i] = extractor.depthmaps[i] * args.push_back
        print(f"[proxy_mesh] pushed all depth maps by {args.push_back}×")

    # ---- Apply additive depth bias (uniform metric shift, view-consistent) ----
    if args.depth_bias != 0.0:
        for i in range(len(extractor.depthmaps)):
            extractor.depthmaps[i] = extractor.depthmaps[i] + args.depth_bias
        print(f"[proxy_mesh] added {args.depth_bias:+.4f} m depth bias to all depths")

    # ---- Standard TSDF fusion (Open3D CPU) ----
    mesh = extractor.extract_mesh_bounded(
        voxel_size=args.voxel, sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc, mask_backgrond=args.mask_background,
    )
    print(f"[proxy_mesh] raw mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    # ---- Post-process: keep the largest connected component and any others
    #       with >1000 triangles. This drops dust flakes but keeps the scene body.
    # Clamp cluster_to_keep to the actual cluster count — post_process_mesh
    # indexes `sorted_sizes[-cluster_to_keep]` and hard-crashes when the mesh
    # has FEWER clusters than requested (first hit: BFC-trained model's
    # coarse fusion produced only 168 clusters — a good problem to have).
    import numpy as _np
    _tc, _cnt, _ = mesh.cluster_connected_triangles()
    _n_clusters = len(_cnt)
    cleaned = post_process_mesh(mesh, cluster_to_keep=min(1000, max(1, _n_clusters)))
    print(f"[proxy_mesh] cleaned mesh: {len(cleaned.vertices):,} verts, "
          f"{len(cleaned.triangles):,} tris")

    # ---- Save PLY (raw AND cleaned) ----
    import open3d as o3d
    raw_path     = args.out_dir / "proxy_mesh_raw.ply"
    cleaned_path = args.out_dir / "proxy_mesh_cleaned.ply"
    o3d.io.write_triangle_mesh(str(raw_path),     mesh)
    o3d.io.write_triangle_mesh(str(cleaned_path), cleaned)

    # ---- Also save a mid-view depth PNG for visual inspection ----
    if len(extractor.depthmaps) > 0:
        d = extractor.depthmaps[len(extractor.depthmaps) // 2].squeeze().cpu().numpy()
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        try:
            import imageio
            imageio.imwrite(args.out_dir / "sample_depth.png",
                            (d_norm * 255).astype(np.uint8))
        except Exception:
            pass

    print(f"[proxy_mesh] wrote:")
    print(f"  {raw_path}")
    print(f"  {cleaned_path}")
    if (args.out_dir / "sample_depth.png").exists():
        print(f"  {args.out_dir / 'sample_depth.png'}   (mid-view depth for eyeballing)")


if __name__ == "__main__":
    main()

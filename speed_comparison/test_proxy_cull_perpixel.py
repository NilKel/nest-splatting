#!/usr/bin/env python3
"""Per-pixel Z-cull quality test using the newly-added set_occluder_depth
device-global on the baked rasterizer. Semantics:

  For each pixel, fragments with surfel `depth > occluder_depth[pixel]`
  are dropped BY THE CUDA KERNEL. Non-finite entries (no mesh hit) skip
  the cull. Where the mesh IS hit and only behind-mesh surfels would have
  contributed, background color fills the remaining transmittance —
  behaves like an opaque proxy mesh silhouette.

  This is stricter than the per-Gauss opacity zeroing in
  test_proxy_cull_quality_baked.py: a Gauss straddling the mesh boundary
  contributes on its in-front pixels but is dropped on its behind pixels.

Usage:
    conda run -n nest_splatting python speed_comparison/test_proxy_cull_perpixel.py \
        --model_path outputs/mip_360/bonsai/... \
        --bake_dir   .../baked_atlas \
        --mesh_paths <mesh> --mesh_names name \
        --out_dir speed_comparison/... --n_views 3 --margin 0.03
"""
import argparse, glob, json, math, os, pickle, sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import torch
import numpy as np
import diff_surfel_bake_render as prod
from diff_surfel_bake_render._C import set_occluder_depth, clear_occluder_depth

from scene import Scene, GaussianModel
from arguments import ModelParams
from utils.render_utils import save_img_u8
from benchmark_baked import _make_sv_state
import open3d as o3d


def load_train_args(model_path: Path):
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    return Namespace(**json.load(open(model_path / "args.json")))


def load_bc7(bake_dir: Path):
    meta = json.load(open(bake_dir / "bake_meta.json"))
    W = int(meta["atlas_width"]); H = int(meta["atlas_height"])
    atlas_scale = float(meta["atlas_scale"]); atlas_offset = float(meta["atlas_offset"])
    bc7_bytes = (bake_dir / "atlas_texture.bc7").read_bytes()
    bc7 = torch.from_numpy(np.frombuffer(bc7_bytes, dtype=np.uint8).copy()).cuda()
    return bc7, W, H, atlas_offset, atlas_scale, meta


def rasterize_mesh_depth_gpu(mesh_path: str, cam, margin: float):
    """Raycast a mesh from Camera → per-pixel cam-z depth as a CUDA fp32
    [H, W] tensor. Non-hit pixels are set to +inf so the CUDA cull will
    skip them (isfinite check). Adds `margin` per pixel to the raw hit
    depth (breathing room)."""
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))

    H, W = cam.image_height, cam.image_width
    fovx, fovy = float(cam.FoVx), float(cam.FoVy)
    fx = W / (2.0 * np.tan(fovx / 2.0))
    fy = H / (2.0 * np.tan(fovy / 2.0))
    cx = W / 2.0; cy = H / 2.0
    W2C = cam.world_view_transform.detach().cpu().numpy().T
    C2W = np.linalg.inv(W2C)

    js, is_ = np.meshgrid(np.arange(W), np.arange(H))
    xs = (js - cx) / fx
    ys = (is_ - cy) / fy
    dirs = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)
    R = C2W[:3, :3].astype(np.float32); t = C2W[:3, 3].astype(np.float32)
    dirs_w = dirs @ R.T
    dirs_w = dirs_w / np.linalg.norm(dirs_w, axis=-1, keepdims=True)
    origins = np.broadcast_to(t, dirs_w.shape).copy()

    rays = o3d.core.Tensor(
        np.concatenate([origins.reshape(-1, 3), dirs_w.reshape(-1, 3)], axis=1),
        dtype=o3d.core.Dtype.Float32)
    t_hit = scene.cast_rays(rays)['t_hit'].numpy().reshape(H, W)
    unnorm = np.sqrt(xs*xs + ys*ys + 1.0)
    depth_camz = (t_hit / unnorm).astype(np.float32)
    # Non-hit pixels come back as +inf from Open3D — leave as inf so CUDA skips.
    # Add margin only to finite (hit) pixels.
    hit = np.isfinite(depth_camz)
    depth_camz[hit] += float(margin)
    return torch.from_numpy(depth_camz).contiguous().cuda()


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else -10 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--bake_dir",   type=Path, required=True)
    ap.add_argument("--mesh_paths", type=str, nargs="+", required=True)
    ap.add_argument("--mesh_names", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir",    type=Path, required=True)
    ap.add_argument("--n_views",    type=int, default=5)
    ap.add_argument("--margin",     type=float, default=0.03)
    args = ap.parse_args()
    assert len(args.mesh_paths) == len(args.mesh_names)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_args = load_train_args(args.model_path)
    train_args.model_path = str(args.model_path)
    train_args.eval = True
    ngp_files = glob.glob(str(args.model_path / "ngp_*.pth"))
    iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                    for f in ngp_files)

    gaussians = GaussianModel(train_args.sh_degree)
    gaussians.load_ply(str(args.bake_dir / "baked.ply"), args=train_args)
    if hasattr(train_args, "kernel"):
        gaussians.kernel_type = train_args.kernel
    if hasattr(gaussians, "update_sites_mask"):
        try:
            gaussians._sv_training_flag = False
            gaussians.update_sites_mask()
        except Exception:
            pass

    tp = argparse.ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(train_args)
    scene = Scene(dataset, GaussianModel(train_args.sh_degree),
                  load_iteration=iteration, shuffle=False)
    test_cams = scene.getTestCameras()

    atlas_rects = torch.load(args.bake_dir / "atlas_rects.pt").cuda()
    bc7_tensor, bc7_W, bc7_H, atlas_offset, atlas_scale, bake_meta = load_bc7(args.bake_dir)
    atlas_texture_fp16 = torch.zeros(3, dtype=torch.float16, device="cuda")
    _compact_mult_train = float(bake_meta.get("compact_mult", 1.0))
    _residual_mode = int(bake_meta.get("residual_mode", 0))
    kernel_type = {"gaussian": 0, "beta": 1, "beta_scaled": 4}.get(train_args.kernel, 0)
    aabb_mode = 5; sort_mode = 0
    beta_val = float(getattr(train_args, "beta_scaled_beta", 2.0))

    pkg_prod = prod.prepare_gaussian_inputs(
        gaussians, sh_degree=train_args.sh_degree, kernel_type=kernel_type)
    prod.set_activation_bias(0.5, 0.0)
    prod.set_compact_mult(_compact_mult_train)
    prod.set_beta_mult(_compact_mult_train)
    prod.set_residual_mode(_residual_mode)
    if hasattr(prod, "set_untex_kernel"): prod.set_untex_kernel(-1)
    prod.set_atlas_bc7(bc7_tensor, bc7_W, bc7_H, atlas_offset, atlas_scale)

    sv = _make_sv_state(gaussians) or {'sites': None, 'tau': None, 'colors': None, 'K': 0}
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    def rasterize(cam):
        r = prod.get_rasterizer(
            image_height=int(cam.image_height), image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform, campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb_mode, sort_mode=sort_mode)
        color, _ = r(
            means3D=pkg_prod['means3D'], opacities=pkg_prod['opacities'],
            shs=pkg_prod['shs'], scales=pkg_prod['scales'],
            rotations=pkg_prod['rotations'], shapes=pkg_prod.get("shapes"),
            kernel_type=pkg_prod['kernel_type'],
            atlas_texture=atlas_texture_fp16, atlas_rects=atlas_rects,
            atlas_width=bc7_W,
            voronoi_sites=sv['sites'], voronoi_tau=sv['tau'],
            voronoi_colors=sv['colors'], voronoi_K=sv['K'],
            is_textured=pkg_prod.get('is_textured'),
            scaling_z=pkg_prod.get('scaling_z'))
        return color.clamp(0, 1)

    N_test = len(test_cams)
    view_ids = np.linspace(0, N_test - 1, args.n_views, dtype=int).tolist()
    print(f"[perpixel_cull] {N_test} test cams; sampling {view_ids}")
    print(f"[perpixel_cull] margin={args.margin*100:.1f} cm")
    results = {name: [] for name in args.mesh_names}

    for vid in view_ids:
        cam = test_cams[vid]
        print(f"\n[perpixel_cull] view {vid}  ({cam.image_width}×{cam.image_height})")

        with torch.no_grad():
            clear_occluder_depth()  # baseline: no cull
            img_base = rasterize(cam)
        gt = cam.original_image[:3].cuda().clamp(0, 1)
        psnr_base_gt = psnr(img_base, gt)
        save_img_u8(img_base.permute(1,2,0).cpu().numpy(),
                    str(args.out_dir / f"view{vid:03d}_baseline.png"))
        print(f"  baseline vs GT: {psnr_base_gt:.2f} dB")

        for name, mp in zip(args.mesh_names, args.mesh_paths):
            depth_gpu = rasterize_mesh_depth_gpu(mp, cam, args.margin)
            n_hit = int(torch.isfinite(depth_gpu).sum().item())
            hit_pct = 100 * n_hit / depth_gpu.numel()

            with torch.no_grad():
                set_occluder_depth(depth_gpu)  # install per-pixel Z-cull
                img_cull = rasterize(cam)
                clear_occluder_depth()
            psnr_c_b  = psnr(img_cull, img_base)
            psnr_c_gt = psnr(img_cull, gt)
            l1 = float((img_cull - img_base).abs().mean().item())
            print(f"  [{name}] hit {hit_pct:.1f}%   "
                  f"PSNR c/b = {psnr_c_b:.2f} dB   (vs GT: {psnr_c_gt:.2f})")

            save_img_u8(img_cull.permute(1,2,0).cpu().numpy(),
                        str(args.out_dir / f"view{vid:03d}_{name}_culled.png"))
            diff = (img_cull - img_base).abs().clamp(0, 1)
            save_img_u8((diff * 5.0).clamp(0, 1).permute(1,2,0).cpu().numpy(),
                        str(args.out_dir / f"view{vid:03d}_{name}_diff_x5.png"))
            results[name].append({
                "view": vid, "psnr_cull_vs_base": psnr_c_b,
                "psnr_cull_vs_gt": psnr_c_gt, "psnr_base_vs_gt": psnr_base_gt,
                "L1": l1, "mesh_hit_pct": hit_pct,
            })

    print(f"\n[perpixel_cull] Summary:")
    for name in args.mesh_names:
        hs = [r['mesh_hit_pct']       for r in results[name]]
        ps = [r['psnr_cull_vs_base']  for r in results[name]]
        print(f"  {name:>16}: mean hit = {np.mean(hs):.1f}%, "
              f"mean PSNR c/b = {np.mean(ps):.2f} dB")
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[perpixel_cull] wrote → {args.out_dir}")


if __name__ == "__main__":
    main()

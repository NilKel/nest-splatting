#!/usr/bin/env python3
"""Test the quality impact of per-Gauss proxy-mesh occlusion culling.

For each of a few test views:
  1. Baseline render (all Gauss)
  2. Rasterize the proxy mesh from that camera → per-pixel proxy_depth
  3. Project each Gauss center to screen, sample proxy_depth there
  4. Mask Gauss whose center depth > proxy_depth (they'd be occluded)
  5. Re-render with masked Gauss opacity → 0
  6. Compute PSNR/SSIM culled-vs-baseline and culled-vs-GT

Approximation: uses per-Gauss CENTER depth, not per-pixel per-Gauss disc
depth. A big Gauss straddling the mesh boundary is either fully kept or
fully dropped by its center. This is what a compute-shader cull pass on
the WebGPU side would do too, so it's the right approximation for the
proposal.

Usage:
    conda run -n nest_splatting python speed_comparison/test_proxy_cull_quality.py \
        --model_path outputs/mip_360/bonsai/... \
        --mesh_paths speed_comparison/proxy_meshes/bonsai/proxy_mesh_cleaned.ply \
                     speed_comparison/proxy_meshes/bonsai_saturation/proxy_mesh_cleaned.ply \
        --mesh_names median saturation \
        --out_dir speed_comparison/proxy_cull_quality/bonsai \
        --n_views 5
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
from utils.render_utils import save_img_u8

import open3d as o3d


def load_train_args(model_path: Path):
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    return Namespace(**json.load(open(model_path / "args.json")))


def rasterize_mesh_depth(mesh_path: str, cam):
    """Raycast a mesh from a Camera to produce per-pixel first-hit depth.
    Returns an H×W numpy array of depths (inf for pixels that miss the mesh)."""
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))

    H, W = cam.image_height, cam.image_width
    fovx, fovy = float(cam.FoVx), float(cam.FoVy)

    # Camera intrinsics via FOV.
    fx = W / (2.0 * np.tan(fovx / 2.0))
    fy = H / (2.0 * np.tan(fovy / 2.0))
    cx = W / 2.0; cy = H / 2.0

    # Camera → world 4x4 (Camera class stores world_view_transform, transpose to world_c2w).
    W2C = cam.world_view_transform.detach().cpu().numpy().T   # column-major → row-major
    C2W = np.linalg.inv(W2C)

    # Build ray_origins + ray_directions.
    js, is_ = np.meshgrid(np.arange(W), np.arange(H))
    xs = (js - cx) / fx
    ys = (is_ - cy) / fy
    zs = np.ones_like(xs)
    dirs = np.stack([xs, ys, zs], axis=-1).astype(np.float32)   # H,W,3 (camera space)
    # rotate to world
    R = C2W[:3, :3].astype(np.float32)
    t = C2W[:3, 3].astype(np.float32)
    dirs_w = dirs @ R.T
    dirs_w = dirs_w / np.linalg.norm(dirs_w, axis=-1, keepdims=True)
    origins = np.broadcast_to(t, dirs_w.shape).copy()

    rays = o3d.core.Tensor(
        np.concatenate([origins.reshape(-1, 3), dirs_w.reshape(-1, 3)], axis=1),
        dtype=o3d.core.Dtype.Float32,
    )
    hit = scene.cast_rays(rays)
    t_hit = hit['t_hit'].numpy().reshape(H, W)   # distance along ray

    # Convert distance along ray → camera-space z (positive-z forward). Since
    # the rays go through the pixel-plane at z=1 in camera space, and we
    # normalized, we recover cam-z by projecting the world-space hit back.
    # Simpler: t_hit * cos(pixel angle) = cam-space z if origin is camera.
    # But cos comes from the un-normalized dirs.length before normalizing:
    #     cam_z = t_hit / |dirs_w_pre_norm|   where the pre-norm dir has z=1.
    unnorm_len = np.sqrt(xs*xs + ys*ys + 1.0)
    depth_camz = t_hit / unnorm_len
    # Miss pixels: t_hit == inf; leave inf here so caller can mask.
    return depth_camz.astype(np.float32)


def compute_occlusion_mask(gaussians, cam, proxy_depth_camz: np.ndarray, margin: float):
    """For each Gauss, project its 3D center into camera space, look up the
    proxy depth at that screen position, return a bool tensor: True = the
    Gauss is BEHIND the mesh at its projected position (occluded)."""
    with torch.no_grad():
        means = gaussians.get_xyz               # [N, 3] world
        H, W = cam.image_height, cam.image_width
        fovx, fovy = float(cam.FoVx), float(cam.FoVy)
        fx = W / (2.0 * np.tan(fovx / 2.0))
        fy = H / (2.0 * np.tan(fovy / 2.0))
        cx = W / 2.0; cy = H / 2.0

        # world → camera
        W2C = cam.world_view_transform.T.to(means.device).float()  # camera-major stored transposed
        ones = torch.ones(means.shape[0], 1, device=means.device)
        pts_h = torch.cat([means, ones], dim=1)
        cam_pts = (pts_h @ W2C.T)[:, :3]   # world_view_transform is already row-major C2W^-1
        cam_x, cam_y, cam_z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]

        # project
        u = fx * (cam_x / cam_z.clamp_min(1e-6)) + cx
        v = fy * (cam_y / cam_z.clamp_min(1e-6)) + cy
        in_view = (cam_z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # sample proxy depth at (u, v)
        proxy_t = torch.from_numpy(proxy_depth_camz).to(means.device)  # H, W
        ui = u.long().clamp(0, W-1); vi = v.long().clamp(0, H-1)
        proxy_at_gauss = proxy_t[vi, ui]                              # [N]

        # occluded = in_view AND proxy has hit (< inf) AND gauss center is behind
        occluded = in_view & torch.isfinite(proxy_at_gauss) & (cam_z > proxy_at_gauss + margin)
    return occluded


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else -10 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--mesh_paths", type=str, nargs="+", required=True)
    ap.add_argument("--mesh_names", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_views", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.02,
                    help="Gauss center depth must exceed proxy depth BY THIS AMOUNT "
                         "to be occluded. Small margin protects Gauss straddling the mesh.")
    args = ap.parse_args()
    assert len(args.mesh_paths) == len(args.mesh_names)
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

    ingp = INGP(cfg_model, args=train_args).to("cuda")
    ingp.load_model(str(args.model_path), iteration)
    ingp.set_active_levels(iteration)

    # Prime the CUDA device-global activation bias / residual mode so the
    # residual composite math matches what the trained model expects. train.py
    # normally does this once at startup; our standalone test needs the same.
    # Without these, render() runs the fragment composite with default bias
    # values and the hash+MLP residual either gets zeroed or misapplied →
    # only the SH+SV base color shows through.
    try:
        from diff_surfel_3D_sh_res import set_activation_bias as _set_ab, set_residual_mode as _set_rm
        _sh_bias, _res_bias = getattr(train_args, "activation_bias", [0.5, 0.0])
        _set_ab(sh_bias=float(_sh_bias), res_bias=float(_res_bias))
        _rm = int(getattr(train_args, "_residual_mode", 0))
        if _rm in (1, 2):
            _set_rm(_rm)
        # Also prime the Python-side default so downstream ReLU passes use the same.
        try:
            from gaussian_renderer import set_default_activation_bias as _sdab
            _sdab(float(_sh_bias), float(_res_bias))
        except Exception:
            pass
        print(f"[cull_test] primed CUDA activation bias sh={_sh_bias}, res={_res_bias}, residual_mode={_rm}")
    except Exception as e:
        print(f"[cull_test] WARN: activation-bias priming failed: {e}")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(train_args, "kernel"):
        gaussians.kernel_type = train_args.kernel

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta = float(cfg_model.surfel.tg_beta) if hasattr(cfg_model.surfel, "tg_beta") else 0.0

    test_cams = scene.getTestCameras()
    N_test = len(test_cams)
    view_ids = np.linspace(0, N_test - 1, args.n_views, dtype=int).tolist()
    print(f"[cull_test] {N_test} test views; sampling {view_ids}")
    print(f"[cull_test] {gaussians.get_xyz.shape[0]:,} Gauss, {len(args.mesh_paths)} meshes")

    # ---- Run each mesh × each view ----
    results = {name: [] for name in args.mesh_names}
    for i, vid in enumerate(view_ids):
        cam = test_cams[vid]
        print(f"\n[cull_test] view {vid}  ({cam.image_width}×{cam.image_height})")

        # Baseline (no cull)
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg, beta=beta, iteration=iteration,
                          cfg=cfg_model, ingp=ingp)
            img_base = pkg['render'].clamp(0, 1)
        gt = cam.original_image[:3].cuda().clamp(0, 1)
        psnr_base_gt = psnr(img_base, gt)
        save_img_u8(img_base.permute(1, 2, 0).cpu().numpy(),
                    str(args.out_dir / f"view{vid:03d}_baseline.png"))

        for name, mp in zip(args.mesh_names, args.mesh_paths):
            print(f"  [{name}] raycasting mesh {Path(mp).name}...")
            proxy = rasterize_mesh_depth(mp, cam)
            n_hit = int(np.isfinite(proxy).sum())
            n_total = proxy.size
            print(f"           mesh hits {n_hit:,}/{n_total:,} pixels "
                  f"({100*n_hit/n_total:.1f}%)")

            occ = compute_occlusion_mask(gaussians, cam, proxy, margin=args.margin)
            n_gauss = int(gaussians.get_xyz.shape[0])
            n_occ = int(occ.sum().item())
            print(f"           masking {n_occ:,}/{n_gauss:,} Gauss ({100*n_occ/n_gauss:.1f}%)")

            # Cull by opacity → very-negative raw so activated = 0
            orig_op = gaussians._opacity.data.clone()
            with torch.no_grad():
                gaussians._opacity.data[occ] = -30.0     # sigmoid(-30) ≈ 1e-13
            with torch.no_grad():
                pkg = render(cam, gaussians, pipe, bg, beta=beta, iteration=iteration,
                              cfg=cfg_model, ingp=ingp)
                img_cull = pkg['render'].clamp(0, 1)
            gaussians._opacity.data.copy_(orig_op)   # restore

            psnr_cull_gt   = psnr(img_cull, gt)
            psnr_cull_base = psnr(img_cull, img_base)
            l1 = float((img_cull - img_base).abs().mean().item())
            print(f"           PSNR culled vs baseline = {psnr_cull_base:.2f} dB   "
                  f"(vs GT: {psnr_cull_gt:.2f}, base vs GT: {psnr_base_gt:.2f}); "
                  f"L1 = {l1:.4f}")

            save_img_u8(img_cull.permute(1, 2, 0).cpu().numpy(),
                        str(args.out_dir / f"view{vid:03d}_{name}_culled.png"))
            diff = (img_cull - img_base).abs().clamp(0, 1)
            save_img_u8((diff * 5.0).clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
                        str(args.out_dir / f"view{vid:03d}_{name}_diff_x5.png"))

            results[name].append({
                "view": vid, "psnr_cull_vs_base": psnr_cull_base,
                "psnr_cull_vs_gt": psnr_cull_gt, "psnr_base_vs_gt": psnr_base_gt,
                "L1": l1, "n_occluded": n_occ, "n_total_gauss": n_gauss,
                "mesh_hit_pct": 100*n_hit/n_total,
            })

    # ---- Summary ----
    print(f"\n[cull_test] Summary — proxy mesh Z-cull quality:\n")
    print(f"{'mesh':>14} {'view':>4} {'mesh_hit%':>10} {'occluded%':>10} "
          f"{'psnr_c_v_b':>11} {'psnr_c_v_gt':>12} {'L1':>7}")
    for name in args.mesh_names:
        for r in results[name]:
            print(f"{name:>14} {r['view']:>4} {r['mesh_hit_pct']:>9.1f}% "
                  f"{100*r['n_occluded']/r['n_total_gauss']:>9.1f}% "
                  f"{r['psnr_cull_vs_base']:>10.2f}  {r['psnr_cull_vs_gt']:>11.2f}  "
                  f"{r['L1']:>7.4f}")
    print()
    for name in args.mesh_names:
        mean_psnr_c_v_b = np.mean([r['psnr_cull_vs_base'] for r in results[name]])
        mean_occ_pct = np.mean([100*r['n_occluded']/r['n_total_gauss'] for r in results[name]])
        print(f"  {name:>14}: mean PSNR culled vs baseline = {mean_psnr_c_v_b:.2f} dB   "
              f"occluded {mean_occ_pct:.1f}% of Gauss")

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[cull_test] wrote images + summary.json → {args.out_dir}")


if __name__ == "__main__":
    main()

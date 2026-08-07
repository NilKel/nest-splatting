"""
Render per-view intersection heatmaps (Gauss-per-pixel count) with the mesh
occluder installed, matching what `final_test_intersection/` shows but AFTER
the per-pixel Z-cull. Uses the training-time neural renderer
(`diff_surfel_3D_sh_res`) — its render_pkg emits `gaussian_num` which is what
`create_intersection_heatmap` visualizes. The lean CONIC baked renderer
doesn't emit this counter, so we use the neural path for this diagnostic.

Also saves the color render for the same view (with cull applied) so you can
sanity-check the count map against the pixels it counts.

Output layout:
  {out_dir}/{idx:03d}_{image_name}_color.png            — culled RGB render
  {out_dir}/{idx:03d}_{image_name}_intersection.png     — heatmap + legend
  {out_dir}/{idx:03d}_{image_name}_histogram.png        — distribution + stats
"""
from __future__ import annotations
import os, sys, argparse, pickle, math
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import imageio.v2 as imageio
import open3d as o3d

from scene import Scene, GaussianModel
from gaussian_renderer import render, set_default_activation_bias
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.render_utils import (create_intersection_heatmap,
                                 create_intersection_histogram, save_img_u8)


def load_model(model_path: str, iteration: int):
    with open(os.path.join(model_path, "args.pkl"), "rb") as f:
        train_args = pickle.load(f)
    train_args.model_path = model_path
    train_args.eval = True
    cfg = Config(os.path.join(model_path, "config.yaml"))

    from diff_surfel_3D_sh_res import set_activation_bias, set_residual_mode
    ab = getattr(train_args, "activation_bias", [0.5, 0.0])
    set_activation_bias(sh_bias=float(ab[0]), res_bias=float(ab[1]))
    set_residual_mode(int(getattr(train_args, "_residual_mode", 0)))
    set_default_activation_bias(float(ab[0]), float(ab[1]))

    ingp = INGP(cfg, args=train_args).to("cuda")
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)

    tp = argparse.ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(train_args)
    pipe = PipelineParams(tp).extract(train_args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration,
                  shuffle=False, full_args=train_args)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(train_args, "kernel"):
        gaussians.kernel_type = train_args.kernel
    gaussians.feature_mode = getattr(train_args, "feature", "sh")
    gaussians._sv_training_flag = False
    if hasattr(gaussians, "update_sites_mask"):
        gaussians.update_sites_mask()
    beta_kern = float(cfg.surfel.tg_beta) if hasattr(cfg.surfel, "tg_beta") else 0.0
    return train_args, cfg, ingp, gaussians, scene, pipe, beta_kern


class MeshDepthBaker:
    def __init__(self, mesh_path: str):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def cam_depth(self, cam, margin: float) -> torch.Tensor:
        H, W = cam.image_height, cam.image_width
        fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
        fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
        cx, cy = W / 2.0, H / 2.0
        W2C = cam.world_view_transform.detach().cpu().numpy().T
        C2W = np.linalg.inv(W2C)
        js, is_ = np.meshgrid(np.arange(W), np.arange(H))
        xs = (js - cx) / fx
        ys = (is_ - cy) / fy
        dirs_cam = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)
        R = C2W[:3, :3].astype(np.float32); t = C2W[:3, 3].astype(np.float32)
        dirs_w = dirs_cam @ R.T
        dirs_w /= np.linalg.norm(dirs_w, axis=-1, keepdims=True)
        origins = np.broadcast_to(t, dirs_w.shape).copy()
        rays = o3d.core.Tensor(
            np.concatenate([origins.reshape(-1, 3), dirs_w.reshape(-1, 3)], axis=1),
            dtype=o3d.core.Dtype.Float32)
        t_hit = self.scene.cast_rays(rays)['t_hit'].numpy().reshape(H, W)
        unnorm = np.sqrt(xs * xs + ys * ys + 1.0).astype(np.float32)
        depth = (t_hit / unnorm).astype(np.float32)
        hit = np.isfinite(depth)
        depth[hit] += float(margin)
        return torch.from_numpy(depth).contiguous().cuda()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--mesh_margin", type=float, default=0.03)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_display", type=int, default=200,
                   help="Heatmap upper bound (surfels/pixel).")
    p.add_argument("--also_no_cull", action="store_true",
                   help="Also emit the intersection map WITHOUT the mesh cull "
                        "into a sibling `_nocull` sub-dir for A/B.")
    args = p.parse_args()

    iteration = args.iteration
    if iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                        for f in ngps)
    print(f"[intersection_cull] loading iter={iteration}")

    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, iteration)
    test_cams = scene.getTestCameras()
    print(f"[intersection_cull] {len(test_cams)} test cams")

    baker = MeshDepthBaker(args.mesh_ply)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    from diff_surfel_3D_sh_res import set_occluder_depth, clear_occluder_depth

    out_cull = Path(args.out_dir)
    out_cull.mkdir(parents=True, exist_ok=True)
    out_nocull = Path(args.out_dir + "_nocull") if args.also_no_cull else None
    if out_nocull:
        out_nocull.mkdir(parents=True, exist_ok=True)

    # Reproduce the training preview's per-Gauss cull semantics: project each
    # Gauss centre, look up mesh depth at that pixel, drop the whole surfel if
    # its centre is behind. This matches what the finetune script did during
    # training, so gaussian_num reflects the surfel set the model was trained
    # to render. (The CUDA per-fragment cull via set_occluder_depth would give
    # a subtler reduction — only the behind-mesh fragments of straddling
    # surfels get dropped, not the whole surfel.)
    @torch.no_grad()
    def _gauss_keep(centers_w, cam, mesh_depth):
        H, W = cam.image_height, cam.image_width
        fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
        fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
        cx, cy = W / 2.0, H / 2.0
        W2C = cam.world_view_transform.to(centers_w.device).T
        R = W2C[:3, :3]; t = W2C[:3, 3]
        cam_xyz = centers_w @ R.T + t
        z = cam_xyz[:, 2]
        keep = torch.ones(centers_w.shape[0], dtype=torch.bool, device=centers_w.device)
        infront = z > 1e-4
        px = (cam_xyz[:, 0] / z.clamp(min=1e-4)) * fx + cx
        py = (cam_xyz[:, 1] / z.clamp(min=1e-4)) * fy + cy
        inbounds = infront & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if inbounds.any():
            px_i = px[inbounds].long().clamp(0, W - 1)
            py_i = py[inbounds].long().clamp(0, H - 1)
            mesh_z = mesh_depth[py_i, px_i]
            cull = torch.isfinite(mesh_z) & (z[inbounds] > mesh_z)
            idx_v = torch.nonzero(inbounds, as_tuple=False).squeeze(-1)
            keep[idx_v[cull]] = False
        return keep

    def _emit(cam, idx, install_occluder: bool, out_root: Path):
        override_op = None
        clear_occluder_depth()
        if install_occluder:
            mesh_z = baker.cam_depth(cam, args.mesh_margin)
            keep = _gauss_keep(gaussians.get_xyz, cam, mesh_z)
            mask = keep.to(gaussians.get_xyz.dtype).view(-1, 1)
            override_op = gaussians.get_opacity * mask
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg, beta=beta_kern,
                         iteration=iteration, cfg=cfg, ingp=ingp,
                         is_training=False, lowpass=True,
                         override_opacity=override_op)
        img = pkg["render"].clamp(0, 1)
        img_u8 = (img.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                  ).clip(0, 255).astype(np.uint8)
        gnum = pkg.get("gaussian_num")
        heat, min_c, max_c = create_intersection_heatmap(gnum, max_display=args.max_display)
        hist, stats = create_intersection_histogram(gnum, max_display=args.max_display)

        name = cam.image_name
        stem = f"{idx:03d}_{name}"
        imageio.imwrite(str(out_root / f"{stem}_color.png"), img_u8)
        save_img_u8(heat, str(out_root / f"{stem}_intersection.png"))
        save_img_u8(hist, str(out_root / f"{stem}_histogram.png"))
        gn_np = gnum.detach().cpu().numpy() if hasattr(gnum, "cpu") else np.asarray(gnum)
        return {"idx": idx, "name": name,
                "min": int(gn_np.min()), "max": int(gn_np.max()),
                "mean": float(gn_np.mean()), "median": float(np.median(gn_np))}

    print(f"[intersection_cull] rendering all {len(test_cams)} test cams "
          f"WITH mesh cull → {out_cull}")
    for idx, cam in enumerate(test_cams):
        stats = _emit(cam, idx, install_occluder=True, out_root=out_cull)
        print(f"  {idx:03d} {stats['name']:<40s}  "
              f"mean/median/max gs/pix = {stats['mean']:.1f}/{stats['median']:.1f}/{stats['max']}")

    if out_nocull:
        print(f"\n[intersection_cull] rendering same views WITHOUT cull → {out_nocull}")
        for idx, cam in enumerate(test_cams):
            stats = _emit(cam, idx, install_occluder=False, out_root=out_nocull)
            print(f"  {idx:03d} {stats['name']:<40s}  "
                  f"mean/median/max gs/pix = {stats['mean']:.1f}/{stats['median']:.1f}/{stats['max']}")

    clear_occluder_depth()
    print(f"\n[intersection_cull] done.")


if __name__ == "__main__":
    main()

"""
Render SV-only (no MLP residual, no atlas) for the finetuned model, with the
mesh cull applied. Compare against the full (SV + residual) render to see
whether the floating artifacts come from the SV base or the residual.

Strategy: disable the residual via set_activation_bias(sh=0.5, res=-999.0).
That drives (residual + res_bias) very negative → ReLU clips to 0 → color =
ReLU(SH+sh_bias) + 0 = SV base only. Standard 3D_SH_res code path, no other
changes required.

Outputs (per test view):
  view{i}_A_svOnly_noCull.png    — SV base, no mesh
  view{i}_B_svOnly_pgCull.png    — SV base + per-Gauss mesh cull (matches training preview's cull)
  view{i}_C_full_pgCull.png      — full render (SV + residual) + per-Gauss cull
  view{i}_GT.png                 — GT reference

If B still has floaters, the SV base is the culprit. If B is clean but C has
floaters, the residual/atlas is where the artifact lives.
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
        R = C2W[:3, :3].astype(np.float32)
        t = C2W[:3, 3].astype(np.float32)
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


@torch.no_grad()
def gauss_occ_mask(centers_w, cam, mesh_depth):
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
        idx = torch.nonzero(inbounds, as_tuple=False).squeeze(-1)
        keep[idx[cull]] = False
    return keep


def to_u8(img_chw):
    return (img_chw.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--mesh_margin", type=float, default=0.03)
    p.add_argument("--n_views", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    iteration = args.iteration
    if iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                        for f in ngps)
    print(f"[sv_only_cull] loading iter={iteration}")

    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, iteration)

    test_cams = scene.getTestCameras()
    rng = np.random.default_rng(args.seed)
    picks = list(rng.choice(len(test_cams), size=min(args.n_views, len(test_cams)),
                            replace=False))
    print(f"[sv_only_cull] rendering test views: {picks}")

    baker = MeshDepthBaker(args.mesh_ply)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    from diff_surfel_3D_sh_res import set_activation_bias
    ab = getattr(train_args, "activation_bias", [0.5, 0.0])
    sh_bias, res_bias = float(ab[0]), float(ab[1])

    with torch.no_grad():
        for slot, cam_idx in enumerate(picks):
            cam = test_cams[cam_idx]
            print(f"\n=== view {slot} (cam_idx={cam_idx}) ===")

            mesh_z = baker.cam_depth(cam, args.mesh_margin)
            keep = gauss_occ_mask(gaussians.get_xyz, cam, mesh_z)
            cull_pct = 100.0 * (1 - keep.float().mean().item())
            mask = keep.to(gaussians.get_xyz.dtype).view(-1, 1)
            override_pg = gaussians.get_opacity * mask
            print(f"    per-Gauss cull: {cull_pct:.1f}%")

            # ---- SV-only: disable residual via res_bias = -999 ----
            # ReLU(residual + (-999)) = 0 for realistic residual values.
            set_activation_bias(sh_bias=sh_bias, res_bias=-999.0)

            #  A) SV-only, no cull
            pkg_a = render(cam, gaussians, pipe, bg, beta=beta_kern,
                           iteration=iteration, cfg=cfg, ingp=ingp,
                           is_training=False, lowpass=True)
            img_a = pkg_a["render"].clamp(0, 1)

            #  B) SV-only, per-Gauss cull
            pkg_b = render(cam, gaussians, pipe, bg, beta=beta_kern,
                           iteration=iteration, cfg=cfg, ingp=ingp,
                           is_training=False, lowpass=True,
                           override_opacity=override_pg)
            img_b = pkg_b["render"].clamp(0, 1)

            # ---- Restore residual for the "full" comparison ----
            set_activation_bias(sh_bias=sh_bias, res_bias=res_bias)

            #  C) full render (SV + residual), per-Gauss cull  — matches finetune preview
            pkg_c = render(cam, gaussians, pipe, bg, beta=beta_kern,
                           iteration=iteration, cfg=cfg, ingp=ingp,
                           is_training=False, lowpass=True,
                           override_opacity=override_pg)
            img_c = pkg_c["render"].clamp(0, 1)

            base = f"view{slot:02d}_cam{cam_idx:03d}"
            imageio.imwrite(os.path.join(args.out_dir, f"{base}_A_svOnly_noCull.png"),
                            to_u8(img_a))
            imageio.imwrite(os.path.join(args.out_dir, f"{base}_B_svOnly_pgCull.png"),
                            to_u8(img_b))
            imageio.imwrite(os.path.join(args.out_dir, f"{base}_C_full_pgCull.png"),
                            to_u8(img_c))
            imageio.imwrite(os.path.join(args.out_dir, f"{base}_GT.png"),
                            to_u8(cam.original_image[:3].cuda()))
            print(f"    wrote A/B/C + GT to {args.out_dir}")

    print(f"\n[sv_only_cull] done. output at {args.out_dir}")


if __name__ == "__main__":
    main()

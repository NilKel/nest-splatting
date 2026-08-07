"""
Render three occlusion variants for a checkpoint on random test views
with a randomized background color per view:

  A. no_cull:      baseline neural render (no mesh) — bg_color used as bg.
  B. mesh_blocks:  per-pixel Z-cull (fragments behind mesh dropped). Where the
                   mesh is hit AND nothing in front covers → bg_color shows.
                   Where the mesh is hit AND some surfel is in front → that
                   surfel blends over bg_color as usual.
  C. mesh_bg:      variant B, then paint the mesh silhouette solid bg_color
                   (mesh acts as a fully opaque surface — surfels IN FRONT of
                   the mesh don't get to show through either).

Uses the finetune-script loader (feature_mode=SV + activation biases +
setter mirror), so it works on either the base checkpoint or a fine-tuned
one whose out_dir lacks args.pkl (pass --base_args_dir for that case).
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


def load_model(base_args_dir: str, ckpt_dir: str, iteration: int):
    with open(os.path.join(base_args_dir, "args.pkl"), "rb") as f:
        train_args = pickle.load(f)
    train_args.model_path = ckpt_dir
    train_args.eval = True
    cfg = Config(os.path.join(base_args_dir, "config.yaml"))

    from diff_surfel_3D_sh_res import set_activation_bias, set_residual_mode
    ab = getattr(train_args, "activation_bias", [0.5, 0.0])
    set_activation_bias(sh_bias=float(ab[0]), res_bias=float(ab[1]))
    set_residual_mode(int(getattr(train_args, "_residual_mode", 0)))
    set_default_activation_bias(float(ab[0]), float(ab[1]))

    ingp = INGP(cfg, args=train_args).to("cuda")
    ingp.load_model(ckpt_dir, iteration)
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
    if hasattr(train_args, "kernel2"):
        gaussians.kernel_type2 = getattr(train_args, "kernel2", None)
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
            np.concatenate([origins.reshape(-1, 3),
                            dirs_w.reshape(-1, 3)], axis=1),
            dtype=o3d.core.Dtype.Float32)
        t_hit = self.scene.cast_rays(rays)['t_hit'].numpy().reshape(H, W)
        unnorm = np.sqrt(xs * xs + ys * ys + 1.0).astype(np.float32)
        depth = (t_hit / unnorm).astype(np.float32)
        hit = np.isfinite(depth)
        depth[hit] += float(margin)
        return torch.from_numpy(depth).contiguous().cuda()


def to_u8(img_chw: torch.Tensor) -> np.ndarray:
    """[3,H,W] float in [0,1] → [H,W,3] uint8."""
    return (img_chw.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Checkpoint dir (has args.pkl + config.yaml + ngp_*.pth).")
    p.add_argument("--base_args_dir", default=None,
                   help="If args.pkl / config.yaml live elsewhere (e.g. finetune output), "
                        "pass the base dir here.")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--mesh_margin", type=float, default=0.03)
    p.add_argument("--n_views", type=int, default=6,
                   help="Number of random test views to render.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    base_args_dir = args.base_args_dir or args.model_path
    iteration = args.iteration
    if iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                        for f in ngps)
    print(f"[occlusion_variants] loading model iter={iteration}")
    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(base_args_dir, args.model_path, iteration)

    test_cams = scene.getTestCameras()
    rng = np.random.default_rng(args.seed)
    picks = list(rng.choice(len(test_cams), size=min(args.n_views, len(test_cams)),
                            replace=False))
    print(f"[occlusion_variants] rendering test views: {picks}")

    print("[occlusion_variants] loading mesh + baking depths ...")
    baker = MeshDepthBaker(args.mesh_ply)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    from diff_surfel_3D_sh_res import set_occluder_depth, clear_occluder_depth

    with torch.no_grad():
        for slot, cam_idx in enumerate(picks):
            cam = test_cams[cam_idx]
            # Random bg color per view (fixed seed downstream so reproducible).
            bg_np = rng.random(3).astype(np.float32)
            bg = torch.from_numpy(bg_np).cuda()
            bg_label = f"rgb({bg_np[0]:.2f},{bg_np[1]:.2f},{bg_np[2]:.2f})"

            print(f"\n=== view {slot} (cam_idx={cam_idx})  bg={bg_label} ===")

            mesh_z = baker.cam_depth(cam, args.mesh_margin)         # [H, W] fp32, +inf where miss
            mesh_hit = torch.isfinite(mesh_z)                       # [H, W] bool
            hit_pct = 100.0 * float(mesh_hit.float().mean().item())
            print(f"    mesh hit: {hit_pct:.1f}% of pixels")

            # ---- Variant A: no cull ----
            clear_occluder_depth()
            pkg_a = render(cam, gaussians, pipe, bg, beta=beta_kern,
                           iteration=iteration, cfg=cfg, ingp=ingp,
                           is_training=False, lowpass=True)
            img_a = pkg_a["render"]

            # ---- Variant B: per-pixel Z-cull ----
            set_occluder_depth(mesh_z)
            pkg_b = render(cam, gaussians, pipe, bg, beta=beta_kern,
                           iteration=iteration, cfg=cfg, ingp=ingp,
                           is_training=False, lowpass=True)
            img_b = pkg_b["render"]
            clear_occluder_depth()

            # ---- Variant C: mesh silhouette painted opaque bg ----
            # Start from variant B (cull kept), then overwrite the mesh-hit
            # pixels with bg_color so the mesh acts as an opaque surface.
            img_c = img_b.clone()
            bg3 = bg.view(3, 1, 1).expand_as(img_c)
            img_c = torch.where(mesh_hit.unsqueeze(0).expand_as(img_c), bg3, img_c)

            # ---- Save PNGs ----
            base = f"view{slot:02d}_cam{cam_idx:03d}"
            imageio.imwrite(os.path.join(args.out_dir, f"{base}_A_nocull.png"), to_u8(img_a))
            imageio.imwrite(os.path.join(args.out_dir, f"{base}_B_meshblocks.png"), to_u8(img_b))
            imageio.imwrite(os.path.join(args.out_dir, f"{base}_C_meshbg.png"), to_u8(img_c))
            # Optional: also save GT for reference.
            imageio.imwrite(os.path.join(args.out_dir, f"{base}_GT.png"),
                            to_u8(cam.original_image[:3].cuda()))
            print(f"    wrote {base}_A / _B / _C + _GT to {args.out_dir}")

    print(f"\n[occlusion_variants] done. output at {args.out_dir}")


if __name__ == "__main__":
    main()

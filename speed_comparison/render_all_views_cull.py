"""
Render every train + test view with the per-Gauss mesh cull applied AND
save the mesh depth map for each view — so you can eyeball whether the
mesh actually lines up with the model's geometry from every angle.

Outputs per view:
  {out_dir}/{split}/{idx:03d}_{name}_color.png       — culled render
  {out_dir}/{split}/{idx:03d}_{name}_meshdepth.png   — mesh cam-Z (viridis)
  {out_dir}/{split}/{idx:03d}_{name}_gt.png          — GT reference

`split` = train | test. Use these to check mesh alignment view-by-view;
a misaligned mesh will show up as a depth silhouette that DOESN'T match
the object outline in GT.
"""
from __future__ import annotations
import os, sys, argparse, math
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import imageio.v2 as imageio

from scene import Scene, GaussianModel
from gaussian_renderer import render, set_default_activation_bias
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
import open3d as o3d


def load_model(model_path, iteration):
    import pickle
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
    """Per-view mesh-depth for the cull.

    Two margin modes:
      * `inflate_margin_normal` (default when > 0): geometrically inflate the
        mesh by pushing each vertex outward along its area-weighted vertex
        normal, ONCE at construction. The raycast scene sees the inflated
        mesh; downstream `cam_depth(margin=0)` gets a hit for the extra
        breathing room and the silhouette grows correspondingly so surfels
        near the boundary land INSIDE the inflated region and get finite
        mesh_z. Fixes the "boundary surfels wrongly killed" symptom of the
        old per-pixel margin.
      * per-call `margin` (fallback, for backward compat): pushes only the
        hit pixels' depth farther by `margin` metres. Pixels JUST outside the
        original silhouette stay at inf (no ray hit) — no breathing room
        near the boundary, exactly what the normal-mode fixes.
    Both can stack (they compose additively), but for the normal fix pass
    `margin=0.0` to `cam_depth()`."""
    def __init__(self, mesh_path, inflate_margin_normal: float = 0.0):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if inflate_margin_normal != 0.0:
            # Angle-weighted vertex normals — Open3D's default; already
            # normalized. Fallback to face-normal averaging if the mesh
            # has no shared-vertex normals.
            mesh.compute_vertex_normals(normalized=True)
            V = np.asarray(mesh.vertices)             # [nV, 3]
            N = np.asarray(mesh.vertex_normals)       # [nV, 3]
            mesh.vertices = o3d.utility.Vector3dVector(
                V + float(inflate_margin_normal) * N)
            print(f"[MeshDepthBaker] inflated {V.shape[0]} vertices along "
                  f"vertex normals by {inflate_margin_normal:+.4f} m")
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def cam_depth(self, cam, margin):
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
        if margin != 0.0:
            depth[hit] += float(margin)
        return torch.from_numpy(depth).contiguous().cuda()


@torch.no_grad()
def gauss_keep(centers, cam, mesh_depth):
    H, W = cam.image_height, cam.image_width
    fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
    fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
    cx, cy = W / 2.0, H / 2.0
    W2C = cam.world_view_transform.to(centers.device).T
    R = W2C[:3, :3]; t = W2C[:3, 3]
    cam_xyz = centers @ R.T + t
    z = cam_xyz[:, 2]
    keep = torch.ones(centers.shape[0], dtype=torch.bool, device=centers.device)
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


def to_u8(img):
    return (img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)


def depth_to_viridis(depth_hw: torch.Tensor):
    """Simple built-in viridis-ish colorization; +inf -> black (no hit)."""
    d = depth_hw.detach().cpu().numpy()
    hit = np.isfinite(d)
    out = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    if hit.any():
        dv = d[hit]
        lo, hi = float(dv.min()), float(dv.max())
        norm = np.zeros_like(d)
        if hi > lo:
            norm[hit] = (d[hit] - lo) / (hi - lo)
        # Small inline viridis LUT
        try:
            import matplotlib.cm as cm
            rgba = (cm.viridis(norm) * 255).astype(np.uint8)
            out = rgba[..., :3]
            out[~hit] = 0
        except Exception:
            g = (norm * 255).astype(np.uint8)
            out = np.stack([g, g, g], axis=-1)
            out[~hit] = 0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--mesh_margin", type=float, default=0.03,
                   help="Legacy per-pixel depth bump — see MeshDepthBaker docstring.")
    p.add_argument("--mesh_normal_margin", type=float, default=0.0,
                   help="Geometric normal-inflation (metres) done once at mesh load. "
                        "Grows the silhouette so boundary surfels get a finite mesh_z. "
                        "Prefer this over --mesh_margin.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--only", choices=["train", "test", "both"], default="both")
    p.add_argument("--view", type=int, default=-1,
                   help="Render just this cam index (within the selected split). -1 = all.")
    args = p.parse_args()

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)
    print(f"[all_views] iter={args.iteration}")

    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, args.iteration)
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    print(f"[all_views] {len(train_cams)} train + {len(test_cams)} test cams")

    baker = MeshDepthBaker(args.mesh_ply,
                           inflate_margin_normal=float(args.mesh_normal_margin))
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    def _emit(cam, idx, out_root):
        mesh_z = baker.cam_depth(cam, args.mesh_margin)
        keep = gauss_keep(gaussians.get_xyz, cam, mesh_z)
        mask = keep.to(gaussians.get_xyz.dtype).view(-1, 1)
        override_op = gaussians.get_opacity * mask
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg, beta=beta_kern,
                         iteration=args.iteration, cfg=cfg, ingp=ingp,
                         is_training=False, lowpass=True,
                         override_opacity=override_op)
        img = pkg["render"].clamp(0, 1)
        stem = f"{idx:03d}_{cam.image_name}"
        imageio.imwrite(str(out_root / f"{stem}_color.png"), to_u8(img))
        imageio.imwrite(str(out_root / f"{stem}_meshdepth.png"),
                        depth_to_viridis(mesh_z))
        imageio.imwrite(str(out_root / f"{stem}_gt.png"),
                        to_u8(cam.original_image[:3].cuda()))
        return {
            "hit_pct": 100.0 * float(torch.isfinite(mesh_z).float().mean().item()),
            "cull_pct": 100.0 * (1.0 - float(keep.float().mean().item())),
        }

    splits = [("train", train_cams), ("test", test_cams)]
    if args.only != "both":
        splits = [(s, c) for s, c in splits if s == args.only]
    for split, cams in splits:
        out = Path(args.out_dir) / split
        out.mkdir(parents=True, exist_ok=True)
        iterset = (
            [(args.view, cams[args.view])] if 0 <= args.view < len(cams)
            else list(enumerate(cams))
        )
        print(f"\n[all_views] {split}: rendering {len(iterset)}/{len(cams)} cams → {out}")
        for idx, cam in iterset:
            s = _emit(cam, idx, out)
            print(f"  {idx:03d} {cam.image_name:<38s}  mesh_hit={s['hit_pct']:5.1f}%  "
                  f"cull={s['cull_pct']:5.1f}%")
    print("\n[all_views] done.")


if __name__ == "__main__":
    main()

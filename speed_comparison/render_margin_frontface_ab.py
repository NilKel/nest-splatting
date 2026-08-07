"""
A/B renders for two mesh-cull margin strategies on all test views:

  A) depth   — legacy per-pixel depth push: mesh_z += 0.03 at hit pixels.
  B) frontface — vertices deflated 0.03 m INWARD along vertex normals,
       and BACKFACE hits discarded during the raycast (treated as no-hit).
       Rationale: deflation moves back-side triangles TOWARD the camera;
       in thin regions they can cross in front of the true front surface
       and poison mesh_z with too-shallow values → false culls. Keeping
       only front-facing hits removes that failure mode while the front
       surface itself sits 3 cm behind where it was (forgiving cull).

Outputs per view into {out_dir}/{config}/:
  {idx:03d}_{name}_color.png       culled render
  {idx:03d}_{name}_meshdepth.png   the depth map used (viridis; inf=black)
Plus a per-view cull% log line for each config.
"""
from __future__ import annotations
import os, sys, math, argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import imageio.v2 as imageio
import open3d as o3d

from speed_comparison.render_all_views_cull import (
    load_model, gauss_keep, to_u8, depth_to_viridis,
)
from gaussian_renderer import render


class FrontfaceDepthBaker:
    """Raycast depth with optional inward normal deflation + backface-hit
    rejection. Open3D's RaycastingScene returns the FIRST hit regardless of
    facing; we test dot(ray_dir, hit_normal) and discard hits on triangles
    facing away from the camera (dot > 0) by setting depth = inf there."""

    def __init__(self, mesh_path: str, normal_offset: float = 0.0,
                 drop_backfaces: bool = False):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if normal_offset != 0.0:
            mesh.compute_vertex_normals(normalized=True)
            V = np.asarray(mesh.vertices)
            N = np.asarray(mesh.vertex_normals)
            mesh.vertices = o3d.utility.Vector3dVector(V + float(normal_offset) * N)
            print(f"[FrontfaceBaker] offset {V.shape[0]:,} verts by "
                  f"{normal_offset:+.4f} m along vertex normals")
        self.drop_backfaces = bool(drop_backfaces)
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
        ans = self.scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy().reshape(H, W)

        if self.drop_backfaces:
            # primitive_normals = geometric normal of the hit triangle (from
            # winding). Backface hit ⇔ ray direction and normal agree
            # (dot > 0): the triangle faces AWAY from the camera. Discard.
            n_hit = ans['primitive_normals'].numpy().reshape(H, W, 3)
            dots = np.einsum('hwc,hwc->hw', dirs_w.reshape(H, W, 3), n_hit)
            back = np.isfinite(t_hit) & (dots > 0.0)
            t_hit = t_hit.copy()
            t_hit[back] = np.inf
            print(f"    backface hits discarded: {back.sum():,} px "
                  f"({100.0 * back.sum() / (H * W):.1f}%)")

        unnorm = np.sqrt(xs * xs + ys * ys + 1.0).astype(np.float32)
        depth = (t_hit / unnorm).astype(np.float32)
        hit = np.isfinite(depth)
        if margin != 0.0:
            depth[hit] += float(margin)
        return torch.from_numpy(depth).contiguous().cuda()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--offset", type=float, default=0.03,
                   help="Magnitude used for both configs (depth +M / normal −M).")
    args = p.parse_args()

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)
    print(f"[ffab] iter={args.iteration}")

    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, args.iteration)
    test_cams = list(scene.getTestCameras())
    print(f"[ffab] {len(test_cams)} test cams")
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    M = float(args.offset)
    configs = [
        # (label, baker, per-call margin)
        ("depth_p03",
         FrontfaceDepthBaker(args.mesh_ply, normal_offset=0.0, drop_backfaces=False),
         +M),
        ("normal_m03_frontface",
         FrontfaceDepthBaker(args.mesh_ply, normal_offset=-M, drop_backfaces=True),
         0.0),
    ]

    for label, baker, margin in configs:
        out = Path(args.out_dir) / label
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[ffab] === {label} ===")
        for idx, cam in enumerate(test_cams):
            mesh_z = baker.cam_depth(cam, margin)
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
            imageio.imwrite(str(out / f"{stem}_color.png"), to_u8(img))
            imageio.imwrite(str(out / f"{stem}_meshdepth.png"),
                            depth_to_viridis(mesh_z))
            cull_pct = 100.0 * (1.0 - float(keep.float().mean().item()))
            hit_pct = 100.0 * float(torch.isfinite(mesh_z).float().mean().item())
            print(f"  {stem:<44s} mesh_hit={hit_pct:5.1f}%  cull={cull_pct:5.1f}%")

    print(f"\n[ffab] done → {args.out_dir}/")


if __name__ == "__main__":
    main()

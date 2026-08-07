"""
Test-view color renders for a --backface_cull-trained checkpoint, applying
the SAME per-view backface cull the training used (disc normal = 3rd rotation
axis, sign-oriented outward via cloud centroid, cull when
dot(view_dir, outward_normal) > cos threshold), optionally combined with the
per-Gauss proxy-mesh cull.

Outputs per view into {out_dir}/:
  {idx:03d}_{name}_bfc.png           BFC only (training-render semantics)
  {idx:03d}_{name}_bfc_meshcull.png  BFC + mesh cull (deployment semantics)
  {idx:03d}_{name}_gt.png            GT
Log line per view: BFC-culled %, mesh-culled %, combined-kept %.
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch
import imageio.v2 as imageio

from speed_comparison.render_all_views_cull import (
    load_model, MeshDepthBaker, gauss_keep, to_u8,
)
from gaussian_renderer import render
from utils.general_utils import build_rotation


@torch.no_grad()
def bfc_keep_mask(gaussians, cam, cos_thr: float) -> torch.Tensor:
    """Same math as train.py --backface_cull: outward-oriented disc normals
    via cloud centroid; cull when dot(view_dir, outward) > cos_thr."""
    xyz = gaussians.get_xyz.detach()
    nrm = build_rotation(gaussians.get_rotation.detach())[:, :, 2]
    ctr = xyz.mean(dim=0, keepdim=True)
    sgn = torch.sign((nrm * (xyz - ctr)).sum(dim=1, keepdim=True))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    out_n = nrm * sgn
    vdir = xyz - cam.camera_center.cuda().view(1, 3)
    vdir = vdir / (vdir.norm(dim=1, keepdim=True) + 1e-12)
    return (vdir * out_n).sum(dim=1) <= cos_thr        # True = keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", default=None,
                   help="Optional proxy mesh for the combined BFC+meshcull render.")
    p.add_argument("--mesh_margin", type=float, default=0.0)
    p.add_argument("--mesh_normal_margin", type=float, default=0.0,
                   help="Vertex offset along outward normals applied once at mesh "
                        "load. NEGATIVE = deflate inward (mesh sits deeper → more "
                        "forgiving cull near the surface).")
    p.add_argument("--bfc_cos", type=float, default=0.2,
                   help="Same threshold the training used (--backface_cull_cos).")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)
    print(f"[bfc_views] iter={args.iteration}  bfc_cos={args.bfc_cos}")

    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, args.iteration)
    test_cams = list(scene.getTestCameras())
    baker = (MeshDepthBaker(args.mesh_ply,
                            inflate_margin_normal=float(args.mesh_normal_margin))
             if args.mesh_ply else None)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for idx, cam in enumerate(test_cams):
        keep_bfc = bfc_keep_mask(gaussians, cam, args.bfc_cos)          # [N] bool

        def _render(keep):
            ov = gaussians.get_opacity * keep.to(gaussians.get_opacity.dtype).view(-1, 1)
            with torch.no_grad():
                pkg = render(cam, gaussians, pipe, bg, beta=beta_kern,
                             iteration=args.iteration, cfg=cfg, ingp=ingp,
                             is_training=False, lowpass=True,
                             override_opacity=ov)
            return pkg["render"].clamp(0, 1)

        stem = f"{idx:03d}_{cam.image_name}"
        imageio.imwrite(str(out / f"{stem}_bfc.png"), to_u8(_render(keep_bfc)))

        msg = (f"  {stem:<44s} bfc_culled={100.0 * (1 - keep_bfc.float().mean()).item():5.1f}%")
        if baker is not None:
            mesh_z = baker.cam_depth(cam, args.mesh_margin)
            keep_mesh = gauss_keep(gaussians.get_xyz, cam, mesh_z)
            keep_both = keep_bfc & keep_mesh
            imageio.imwrite(str(out / f"{stem}_bfc_meshcull.png"), to_u8(_render(keep_both)))
            msg += (f"  mesh_culled={100.0 * (1 - keep_mesh.float().mean()).item():5.1f}%"
                    f"  combined_kept={100.0 * keep_both.float().mean().item():5.1f}%")
        imageio.imwrite(str(out / f"{stem}_gt.png"),
                        to_u8(cam.original_image[:3].cuda().clamp(0, 1)))
        print(msg)

    print(f"[bfc_views] done → {out}/")


if __name__ == "__main__":
    main()

"""
Intersection heatmaps (Gauss-per-pixel contributor count from the NEURAL
renderer's `gaussian_num` output) for the BFC-trained checkpoint, in up to
three per-Gauss cull variants:

  plain      — no culls
  bfc        — centroid-oriented backface cull (--bfc_cos)
  bfc_mesh   — BFC + per-Gauss mesh cull (offset mesh)

Select with --variants (comma list). Run under a NO_EARLY_EXIT=1 rasterizer
build with `--variants plain --tag noearly` to get the no-early-exit
contributor counts (n_contrib = every alpha-surviving fragment, matching
what a renderer without per-pixel T-saturation exit pays — the delta vs the
default build's plain map = fragments skipped by the early exit).

Outputs per view into {out_dir}/{variant}{tag}/:
  {idx:03d}_{name}_color.png / _intersection.png / _histogram.png
plus a per-view mean/median/max gs/pix log line.
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import imageio.v2 as imageio

from speed_comparison.render_all_views_cull import (
    load_model, MeshDepthBaker, gauss_keep, to_u8,
)
from speed_comparison.render_bfc_cull_views import bfc_keep_mask
from gaussian_renderer import render
from utils.render_utils import (create_intersection_heatmap,
                                create_intersection_histogram, save_img_u8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", default=None)
    p.add_argument("--mesh_margin", type=float, default=0.0)
    p.add_argument("--mesh_normal_margin", type=float, default=-0.012)
    p.add_argument("--bfc_cos", type=float, default=0.2)
    p.add_argument("--variants", default="plain,bfc,bfc_mesh")
    p.add_argument("--tag", default="",
                   help="Suffix appended to each variant's out dir (e.g. '_noearly').")
    p.add_argument("--max_display", type=int, default=100)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"[ix_bfc] iter={args.iteration}  variants={variants}  tag='{args.tag}'")

    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, args.iteration)
    test_cams = list(scene.getTestCameras())
    baker = (MeshDepthBaker(args.mesh_ply,
                            inflate_margin_normal=float(args.mesh_normal_margin))
             if args.mesh_ply else None)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    for variant in variants:
        out = Path(args.out_dir) / f"{variant}{args.tag}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[ix_bfc] === {variant}{args.tag} ===")
        stats = []
        for idx, cam in enumerate(test_cams):
            keep = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            if variant in ("bfc", "bfc_mesh"):
                keep &= bfc_keep_mask(gaussians, cam, args.bfc_cos)
            if variant == "bfc_mesh":
                assert baker is not None, "--mesh_ply required for bfc_mesh"
                mesh_z = baker.cam_depth(cam, args.mesh_margin)
                keep &= gauss_keep(gaussians.get_xyz, cam, mesh_z)
            override = None
            if variant != "plain":
                override = gaussians.get_opacity * keep.to(
                    gaussians.get_opacity.dtype).view(-1, 1)
            with torch.no_grad():
                pkg = render(cam, gaussians, pipe, bg, beta=beta_kern,
                             iteration=args.iteration, cfg=cfg, ingp=ingp,
                             is_training=False, lowpass=True,
                             override_opacity=override)
            img = pkg["render"].clamp(0, 1)
            gnum = pkg.get("gaussian_num")
            stem = f"{idx:03d}_{cam.image_name}"
            imageio.imwrite(str(out / f"{stem}_color.png"), to_u8(img))
            if gnum is not None:
                g = gnum.squeeze().detach()
                heat, _min_c, _max_c = create_intersection_heatmap(
                    g.cpu().numpy(), max_display=args.max_display)
                save_img_u8(heat, str(out / f"{stem}_intersection.png"))
                hist, _stats = create_intersection_histogram(
                    g.cpu().numpy(), max_display=args.max_display)
                save_img_u8(hist, str(out / f"{stem}_histogram.png"))
                gm = float(g.float().mean().item())
                gmed = float(g.float().median().item())
                gmax = int(g.max().item())
                stats.append((gm, gmed, gmax))
                print(f"  {stem:<44s} mean/median/max gs/pix = "
                      f"{gm:.1f}/{gmed:.0f}/{gmax}")
        if stats:
            print(f"  [{variant}{args.tag}] AGG mean gs/pix = "
                  f"{np.mean([s[0] for s in stats]):.2f}   "
                  f"max = {max(s[2] for s in stats)}")

    print(f"\n[ix_bfc] done → {args.out_dir}/")


if __name__ == "__main__":
    main()

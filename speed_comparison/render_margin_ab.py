"""
Render the SAME test view under several mesh-cull margin configurations
so we can visually pick the right direction for --mesh_normal_margin.

Outputs a single stitched PNG (variants stacked horizontally) plus per-
variant PNGs alongside, into <model_path>/margin_ab/.

Variants (fixed set — tweak `_variants()` if you want a different sweep):
  0. no cull                     (mesh cull disabled)
  1. mesh_margin  +M             (legacy per-pixel push; forgiving on front)
  2. normal_margin +M            (outward inflation; expands silhouette)
  3. normal_margin −M            (inward deflation; shrinks silhouette)
  4. mesh_margin +M AND normal +M  (stacked; per-pixel on top of inflation)

Each render uses the SAME per-Gauss mesh cull as the working CUDA path
(override_opacity from a keep-mask against the per-view mesh depth). The
GT image + a mesh-silhouette overlay are also emitted for reference.
"""
from __future__ import annotations
import os, sys, math, argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import imageio.v2 as imageio

from speed_comparison.render_all_views_cull import (
    load_model, MeshDepthBaker, gauss_keep, to_u8, depth_to_viridis,
)
from gaussian_renderer import render


def _variants(M: float):
    """Return list of (label, mesh_margin, normal_margin, cull_on) tuples."""
    return [
        ("no_cull",                  0.0,  0.0,  False),
        (f"mesh_+{M:g}",             M,    0.0,  True),
        (f"normal_+{M:g}",           0.0,  M,    True),
        (f"normal_-{M:g}",           0.0, -M,    True),
        (f"mesh_+{M:g}_normal_+{M:g}", M,  M,    True),
    ]


def _label_strip(img_u8: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    """Prepend a black strip with white ASCII text (no font deps — 5x7 bitmap
    at 3x scale would need a font atlas). Skip actual rendering — instead
    just leave a solid band the caller draws OS-side. We keep it dead simple:
    return the image with a colored top strip so the user knows a label goes
    there; the per-variant filename is the source of truth."""
    H, W, _ = img_u8.shape
    strip = np.full((height, W, 3), 24, dtype=np.uint8)   # near-black
    return np.vstack([strip, img_u8])


def _stitch_h(imgs: list[np.ndarray]) -> np.ndarray:
    return np.hstack(imgs)


@torch.no_grad()
def render_one(cam, gaussians, cfg, pipe, ingp, beta_kern, iteration,
                baker: MeshDepthBaker, mesh_margin: float, cull_on: bool):
    """Render `cam` with the given cull config. Returns (img_u8, mesh_z_torch)."""
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    mesh_z = baker.cam_depth(cam, mesh_margin)
    if cull_on:
        keep = gauss_keep(gaussians.get_xyz, cam, mesh_z)
        mask = keep.to(gaussians.get_xyz.dtype).view(-1, 1)
        override_op = gaussians.get_opacity * mask
    else:
        override_op = None
    pkg = render(cam, gaussians, pipe, bg, beta=beta_kern,
                 iteration=iteration, cfg=cfg, ingp=ingp,
                 is_training=False, lowpass=True,
                 override_opacity=override_op)
    img = pkg["render"].clamp(0, 1)
    return to_u8(img), mesh_z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--margin", "-M", type=float, default=0.03,
                   help="Magnitude used for all sweep variants (see script docstring).")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--view_index", type=int, default=0,
                   help="Cam index within the chosen split (default = first).")
    p.add_argument("--out_dir", default=None,
                   help="Default: <model_path>/margin_ab/")
    args = p.parse_args()

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)
    print(f"[ab] iter={args.iteration}")

    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, args.iteration)
    cams = list(scene.getTrainCameras()) if args.split == "train" else list(scene.getTestCameras())
    if not (0 <= args.view_index < len(cams)):
        raise SystemExit(f"view_index {args.view_index} out of range [0, {len(cams)})")
    cam = cams[args.view_index]
    print(f"[ab] rendering {args.split}[{args.view_index}] = {cam.image_name}")

    out_root = Path(args.out_dir) if args.out_dir else Path(args.model_path) / "margin_ab"
    out_root.mkdir(parents=True, exist_ok=True)

    # A separate baker per unique normal_margin — inflation is baked in at
    # construction time. Cache by normal_margin so we don't rebuild for stacks.
    variants = _variants(args.margin)
    normal_margins = sorted({nm for _, _, nm, _ in variants})
    bakers: dict[float, MeshDepthBaker] = {}
    for nm in normal_margins:
        print(f"[ab] building baker with normal_margin={nm:+.4f}")
        bakers[nm] = MeshDepthBaker(args.mesh_ply, inflate_margin_normal=float(nm))

    # Render each variant + save individual PNGs.
    tiles: list[np.ndarray] = []
    labels: list[str] = []
    for label, mm, nm, cull_on in variants:
        img_u8, mesh_z = render_one(cam, gaussians, cfg, pipe, ingp, beta_kern,
                                    args.iteration, bakers[nm], mm, cull_on)
        out_png = out_root / f"{args.split}_{args.view_index:03d}_{label}.png"
        imageio.imwrite(str(out_png), img_u8)
        # Also dump the mesh_z map used for this variant — helps confirm
        # inflation actually shifted the silhouette.
        imageio.imwrite(
            str(out_root / f"{args.split}_{args.view_index:03d}_{label}_meshdepth.png"),
            depth_to_viridis(mesh_z))
        pct_hit = 100.0 * float(torch.isfinite(mesh_z).float().mean().item())
        print(f"  {label:<28s}  mesh_hit={pct_hit:5.1f}%   {out_png.name}")
        tiles.append(_label_strip(img_u8, label))
        labels.append(label)

    # Also emit the GT (unchanged) for eyeballing.
    gt = to_u8(cam.original_image[:3].cuda().clamp(0, 1))
    imageio.imwrite(str(out_root / f"{args.split}_{args.view_index:03d}_gt.png"), gt)
    tiles.append(_label_strip(gt, "GT"))
    labels.append("GT")

    # Side-by-side comparison.
    strip = _stitch_h(tiles)
    combo_path = out_root / f"{args.split}_{args.view_index:03d}_ab_strip.png"
    imageio.imwrite(str(combo_path), strip)
    print(f"\n[ab] strip: {combo_path}")
    print(f"[ab] variants (left→right): {' | '.join(labels)}")
    print(f"[ab] all outputs in: {out_root}/")


if __name__ == "__main__":
    main()

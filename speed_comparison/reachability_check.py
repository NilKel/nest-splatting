"""
Count how many surfels are never visible in ANY view (train + test). Two
counts reported:
  - frustum-unreached : centre never in-frustum from any camera
  - opacity-invisible : centre in-frustum somewhere BUT opacity < threshold

Runs on the raw finetune/base checkpoint (no mesh cull); tells us how much
of the model is pure dead weight independent of the mesh.
"""
from __future__ import annotations
import os, sys, argparse, math, pickle
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from scene import Scene, GaussianModel
from gaussian_renderer import set_default_activation_bias
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams


def load(model_path, iteration):
    with open(os.path.join(model_path, "args.pkl"), "rb") as f:
        train_args = pickle.load(f)
    train_args.model_path = model_path
    train_args.eval = True
    cfg = Config(os.path.join(model_path, "config.yaml"))
    from diff_surfel_3D_sh_res import set_activation_bias, set_residual_mode
    ab = getattr(train_args, "activation_bias", [0.5, 0.0])
    set_activation_bias(float(ab[0]), float(ab[1]))
    set_residual_mode(int(getattr(train_args, "_residual_mode", 0)))
    set_default_activation_bias(float(ab[0]), float(ab[1]))
    ingp = INGP(cfg, args=train_args).to("cuda")
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)
    tp = argparse.ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(train_args)
    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g, load_iteration=iteration,
                  shuffle=False, full_args=train_args)
    g.base_opacity = cfg.surfel.tg_base_alpha
    if hasattr(train_args, "kernel"):
        g.kernel_type = train_args.kernel
    g.feature_mode = getattr(train_args, "feature", "sh")
    return train_args, g, scene


@torch.no_grad()
def in_frustum(centers, cam, near=1e-4):
    """Return bool [N] : centre is inside this cam's view frustum."""
    H, W = cam.image_height, cam.image_width
    fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
    fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
    cx, cy = W / 2.0, H / 2.0
    W2C = cam.world_view_transform.to(centers.device).T
    R = W2C[:3, :3]; t = W2C[:3, 3]
    xyz_cam = centers @ R.T + t
    z = xyz_cam[:, 2]
    infront = z > near
    zc = z.clamp(min=near)
    px = (xyz_cam[:, 0] / zc) * fx + cx
    py = (xyz_cam[:, 1] / zc) * fy + cy
    return infront & (px >= 0) & (px < W) & (py >= 0) & (py < H)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--opacity_thresh", type=float, default=0.005,
                   help="Surfels with get_opacity < this are counted separately.")
    args = p.parse_args()

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)
    print(f"[reach] iter={args.iteration}")

    train_args, g, scene = load(args.model_path, args.iteration)
    N = int(g.get_xyz.shape[0])
    print(f"[reach] N={N:,} surfels")

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    all_cams = list(train_cams) + list(test_cams)
    print(f"[reach] scanning {len(train_cams)} train + {len(test_cams)} test = "
          f"{len(all_cams)} cams total")

    centers = g.get_xyz.detach()
    reached = torch.zeros(N, dtype=torch.bool, device=centers.device)
    for cam in all_cams:
        reached |= in_frustum(centers, cam)
        if reached.all():
            break

    n_unreached = int((~reached).sum().item())
    n_reached = N - n_unreached
    print()
    print(f"  reached (in some view frustum): {n_reached:>8,} / {N:,}  "
          f"({100.0 * n_reached / N:.2f}%)")
    print(f"  unreached (dead — no frustum):  {n_unreached:>8,} / {N:,}  "
          f"({100.0 * n_unreached / N:.2f}%)")

    # Also count opacity-invisible.
    opacity = g.get_opacity.squeeze(-1).detach()
    low_opac = opacity < args.opacity_thresh
    n_low_opac = int(low_opac.sum().item())
    n_low_opac_reached = int((low_opac & reached).sum().item())
    print()
    print(f"  opacity < {args.opacity_thresh}: {n_low_opac:>8,}  "
          f"(of which {n_low_opac_reached:,} are also in-frustum somewhere)")

    # Combined: dead weight = unreached OR (reached but too transparent)
    dead = (~reached) | low_opac
    n_dead = int(dead.sum().item())
    print(f"  combined dead-weight (unreached OR opacity<{args.opacity_thresh}): "
          f"{n_dead:,} / {N:,}  ({100.0 * n_dead / N:.2f}%)")


if __name__ == "__main__":
    main()

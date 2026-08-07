"""
Render test view + intersection map for a mesh-culled fine-tuned checkpoint
AND its base checkpoint into the SAME output dir for easy comparison.

Uses the same load-and-configure path as finetune_mesh_cull.py (feature_mode
setter, activation_bias etc.), which is required for SV mode to produce a
sane render — regular render scripts need cfg_args/args.pkl in the target
dir, which fine-tune outputs don't contain.
"""
from __future__ import annotations
import os, sys, argparse, pickle
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
from utils.render_utils import create_intersection_heatmap


def load_model(base_args_dir: str, ckpt_dir: str, iteration: int):
    """base_args_dir supplies args.pkl / config.yaml; ckpt_dir supplies the
    PLY + ngp_<iter>.pth. They can be the same directory or different (the
    finetune output only has the ckpts)."""
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


def render_view(name, ckpt_dir, iteration, base_args_dir, out_dir, frame_idx):
    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(base_args_dir, ckpt_dir, iteration)
    test_cams = scene.getTestCameras()
    assert 0 <= frame_idx < len(test_cams), \
        f"frame {frame_idx} out of range 0..{len(test_cams)-1}"
    cam = test_cams[frame_idx]
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg, beta=beta_kern,
                     iteration=iteration, cfg=cfg, ingp=ingp,
                     is_training=False, lowpass=True)
    img = pkg["render"].clamp(0, 1)
    gt = cam.original_image[:3].cuda().clamp(0, 1)
    mse = ((img - gt) ** 2).mean().item()
    psnr = -10 * np.log10(mse) if mse > 0 else float("inf")

    gnum = pkg.get("gaussian_num")
    if gnum is None:
        print(f"[{name}] gaussian_num missing from render pkg — cannot make "
              f"intersection map.")
        heatmap = None
    else:
        heatmap, _, _ = create_intersection_heatmap(gnum, max_display=200)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_u8 = (img.detach().cpu().permute(1, 2, 0).numpy() * 255.0
              ).clip(0, 255).astype(np.uint8)
    img_path = os.path.join(out_dir, f"{name}_iter{iteration}_render.png")
    imageio.imwrite(img_path, img_u8)
    print(f"[{name}] iter={iteration}  PSNR={psnr:.2f} dB  → {img_path}")

    if heatmap is not None:
        heat_path = os.path.join(out_dir, f"{name}_iter{iteration}_intersection.png")
        # create_intersection_heatmap returns HWC uint8-ish; save with imageio.
        heatmap_np = heatmap
        if isinstance(heatmap_np, torch.Tensor):
            heatmap_np = heatmap_np.detach().cpu().numpy()
        if heatmap_np.dtype != np.uint8:
            heatmap_np = np.clip(heatmap_np * (255.0 if heatmap_np.max() <= 1.5 else 1.0),
                                 0, 255).astype(np.uint8)
        imageio.imwrite(heat_path, heatmap_np)
        print(f"[{name}] intersection map → {heat_path}")

    # Free before next model loads
    del ingp, gaussians, scene
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True,
                   help="Original checkpoint dir (has args.pkl / config.yaml / ngp_*.pth).")
    p.add_argument("--ft_model", required=True,
                   help="Fine-tuned checkpoint dir (only PLY + ngp_*.pth).")
    p.add_argument("--base_iter", type=int, default=35000)
    p.add_argument("--ft_iter", type=int, default=40000)
    p.add_argument("--frame", type=int, default=0,
                   help="Test-camera index to render.")
    p.add_argument("--out_dir", default=None,
                   help="Where to write PNGs (default: <ft_model>/comparison).")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.ft_model, "comparison")
    render_view("base", args.base_model, args.base_iter,
                base_args_dir=args.base_model, out_dir=out_dir, frame_idx=args.frame)
    render_view("ft", args.ft_model, args.ft_iter,
                base_args_dir=args.base_model, out_dir=out_dir, frame_idx=args.frame)
    print(f"\nAll renders in: {out_dir}")


if __name__ == "__main__":
    main()

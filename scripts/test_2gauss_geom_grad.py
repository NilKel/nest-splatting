#!/usr/bin/env python3
"""
2-Gaussian geometry gradient comparison: 3D_direct vs 3D_direct_lean.

Loads from checkpoint, zeros opacity of all but 2 Gaussians, compares every gradient.
"""

import os
import sys
import glob
import pickle
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams

MODEL_PATH = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"
NUM_KEEP = 2


def load_training_config(model_path):
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        return pickle.load(f)


def mask_gaussians(gaussians, n):
    """Zero opacity of all but the top-n Gaussians. Returns kept indices."""
    opacities = gaussians.get_opacity.squeeze()
    _, top_idx = torch.topk(opacities, min(n, len(opacities)))
    top_idx = top_idx.sort().values

    print(f"Keeping {n} Gaussians (indices: {top_idx.tolist()}) out of {len(opacities)}")
    for i in top_idx:
        print(f"  [{i.item()}] xyz={gaussians._xyz.data[i].tolist()}, "
              f"opacity_logit={gaussians._opacity.data[i].item():.4f}, "
              f"opacity={opacities[i].item():.6f}")

    # Set all other opacities to very negative (sigmoid → ~0)
    mask = torch.ones(len(opacities), dtype=torch.bool, device="cuda")
    mask[top_idx] = False
    with torch.no_grad():
        gaussians._opacity.data[mask] = -20.0  # sigmoid(-20) ≈ 2e-9

    return top_idx


def run_mode(gaussians, camera, method, ingp, beta, iteration, cfg, args_base):
    """Render + backward, return (image, grads_dict)."""
    args = Namespace(**vars(args_base))
    args.method = method

    # Enable gradients
    for p in [gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity]:
        p.requires_grad_(True)
        p.grad = None
    if gaussians._gaussian_features is not None:
        gaussians._gaussian_features.requires_grad_(True)
        gaussians._gaussian_features.grad = None
    for param in ingp.parameters():
        param.requires_grad_(True)
        param.grad = None

    result = render(
        camera, gaussians, args, torch.zeros(3, device="cuda"),
        ingp=ingp, beta=beta, iteration=iteration, cfg=cfg
    )
    rendered = result["render"]
    loss = rendered.sum()
    loss.backward()

    grads = {}
    for name, param in [('xyz', gaussians._xyz), ('scales', gaussians._scaling),
                         ('rotations', gaussians._rotation), ('opacity', gaussians._opacity),
                         ('gaussian_features', gaussians._gaussian_features)]:
        grads[name] = param.grad.clone() if param.grad is not None else None

    if hasattr(ingp, 'hash_encoding') and ingp.hash_encoding.embeddings.grad is not None:
        grads['hash_embeddings'] = ingp.hash_encoding.embeddings.grad.clone()

    # MLP grads
    if method == "3D_direct" and hasattr(ingp, 'mlp_3D_direct') and ingp.mlp_3D_direct is not None:
        mlp = ingp.mlp_3D_direct
        for lname, layer_idx in [('W1', 0), ('b1', 0), ('W2', 2), ('b2', 2), ('W3', 4), ('b3', 4)]:
            layer = mlp[layer_idx]
            if 'W' in lname:
                grads[lname] = layer.weight.grad.clone() if layer.weight.grad is not None else None
            else:
                grads[lname] = layer.bias.grad.clone() if layer.bias.grad is not None else None
    elif method == "3D_direct_lean":
        from diff_surfel_3D import get_mlp_grads
        mlp_grads = get_mlp_grads()
        if mlp_grads is not None:
            for i, lname in enumerate(['W1', 'b1', 'W2', 'b2', 'W3', 'b3']):
                grads[lname] = mlp_grads[i].clone() if mlp_grads[i] is not None else None

    return rendered, grads


def main():
    args = load_training_config(MODEL_PATH)
    args.model_path = MODEL_PATH
    args.eval = True
    cfg = Config(os.path.join(MODEL_PATH, "config.yaml"))
    beta = cfg.surfel.tg_beta

    # Find iteration
    ngp_files = glob.glob(os.path.join(MODEL_PATH, "ngp_*.pth"))
    iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
    iteration = max(iterations)

    # Load full model
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    camera = scene.getTestCameras()[0]

    print("=" * 80)
    print(f"{NUM_KEEP}-GAUSSIAN GEOMETRY GRADIENT COMPARISON")
    print("=" * 80)

    # Save original opacity to restore between runs
    orig_opacity = gaussians._opacity.data.clone()

    # Mask to top-N
    kept_idx = mask_gaussians(gaussians, NUM_KEEP)

    # ========== 3D_direct ==========
    print(f"\n{'='*40} 3D_direct {'='*40}")
    args_d = Namespace(**vars(args))
    args_d.method = "3D_direct"
    ingp_d = INGP(cfg, args=args_d).to('cuda')
    ingp_d.load_model(MODEL_PATH, iteration)
    ingp_d.set_active_levels(iteration)

    render_d, grads_d = run_mode(gaussians, camera, "3D_direct", ingp_d, beta, iteration, cfg, args)
    print(f"  Render: mean={render_d.mean().item():.8f}, max={render_d.max().item():.8f}, nonzero_pix={(render_d > 1e-6).sum().item()}")

    # Restore opacity for lean run
    with torch.no_grad():
        gaussians._opacity.data.copy_(orig_opacity)
    mask_gaussians(gaussians, NUM_KEEP)

    # ========== 3D_direct_lean ==========
    print(f"\n{'='*40} 3D_direct_lean {'='*40}")
    args_l = Namespace(**vars(args))
    args_l.method = "3D_direct_lean"
    ingp_l = INGP(cfg, args=args_l).to('cuda')
    ingp_l.load_model(MODEL_PATH, iteration)
    ingp_l.set_active_levels(iteration)

    render_l, grads_l = run_mode(gaussians, camera, "3D_direct_lean", ingp_l, beta, iteration, cfg, args)
    print(f"  Render: mean={render_l.mean().item():.8f}, max={render_l.max().item():.8f}, nonzero_pix={(render_l > 1e-6).sum().item()}")

    # ========== COMPARISON ==========
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    fwd_diff = (render_d - render_l).abs()
    print(f"\nForward diff: mean={fwd_diff.mean().item():.8f}, max={fwd_diff.max().item():.8f}")

    # Print gradients only for the kept Gaussians
    for name in ['xyz', 'scales', 'rotations', 'opacity', 'gaussian_features']:
        g_d = grads_d.get(name)
        g_l = grads_l.get(name)
        print(f"\n--- {name} ---")
        if g_d is None and g_l is None:
            print(f"  Both None")
            continue
        if g_d is None or g_l is None:
            print(f"  3D_direct: {'None' if g_d is None else 'exists'}")
            print(f"  lean:      {'None' if g_l is None else 'exists'}")
            continue

        for idx in kept_idx:
            i = idx.item()
            vals_d = g_d[i].flatten().cpu().tolist()
            vals_l = g_l[i].flatten().cpu().tolist()
            print(f"  Gaussian [{i}]:")
            for j, (vd, vl) in enumerate(zip(vals_d, vals_l)):
                marker = " " if abs(vd - vl) < max(abs(vd), abs(vl), 1e-6) * 0.01 else "***"
                print(f"    [{j:2d}] direct={vd:14.6f}  lean={vl:14.6f}  diff={abs(vd-vl):14.6f} {marker}")

        # Also check if any non-kept Gaussians got gradients
        nonzero_d = (g_d.abs() > 1e-10).any(dim=-1) if g_d.dim() > 1 else g_d.abs() > 1e-10
        nonzero_l = (g_l.abs() > 1e-10).any(dim=-1) if g_l.dim() > 1 else g_l.abs() > 1e-10
        mask_kept = torch.zeros(g_d.shape[0], dtype=torch.bool, device="cuda")
        mask_kept[kept_idx] = True
        extra_d = (nonzero_d & ~mask_kept).sum().item()
        extra_l = (nonzero_l & ~mask_kept).sum().item()
        print(f"  Non-kept Gaussians with nonzero grad: direct={extra_d}, lean={extra_l}")

    # MLP grads summary
    print(f"\n--- MLP gradients ---")
    for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        g_d = grads_d.get(name)
        g_l = grads_l.get(name)
        if g_d is not None and g_l is not None:
            cos = torch.nn.functional.cosine_similarity(
                g_d.flatten().unsqueeze(0).float(),
                g_l.flatten().unsqueeze(0).float()
            ).item() if g_d.abs().sum() > 1e-10 and g_l.abs().sum() > 1e-10 else 0.0
            print(f"  {name}: cos_sim={cos:.6f}, mag_d={g_d.abs().mean():.6f}, mag_l={g_l.abs().mean():.6f}")


if __name__ == "__main__":
    main()

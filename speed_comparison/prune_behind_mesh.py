"""
Permanent prune: drop any surfel whose centre projects BEHIND the proxy mesh
across EVERY train view (with a small breathing margin).

Rationale — CUDA runtime cull (`set_occluder_depth`) and the WebGPU
`mesh_cull` toggle both work per-view: on any given frame, ~40-60% of
surfels drop out. But a large fraction of THOSE are surfels that would be
culled from any viewpoint (they live inside the mesh's occluded volume and
were needed at training only to fit the random-bg composited image, not to
show a real object). Prune them once, permanently, and downstream:
  - The .bitymi ships smaller.
  - The atlas can be re-baked with fewer rects → smaller texture.
  - Any renderer (WebGPU, mobile, native) benefits without any runtime mesh.
  - The runtime mesh cull is STILL beneficial on top for the per-view slice.

Loads the finetune checkpoint via `finetune_mesh_cull.load_checkpoint`
(preserves feature_mode/SV wiring), iterates the full train camera set,
raycasts the mesh from each view, and marks a Gauss as SURVIVOR if it is
in front of the mesh at its projected pixel in at least ONE view (with the
same `--mesh_margin` used during finetune training). Non-survivors are
dropped from every per-Gauss tensor. The pruned model is saved as a new
`point_cloud/iteration_<N>/point_cloud.ply` in `--out_dir`.
"""
from __future__ import annotations
import os, sys, argparse, math, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

# Reuse the well-tested loader from the finetune script.
from finetune_mesh_cull import (load_checkpoint, MeshDepthBaker, gauss_occ_mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Finetune output dir (needs args.pkl + config.yaml + "
                        "ngp_<iter>.pth + point_cloud/iteration_<iter>/).")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--mesh_margin", type=float, default=0.03,
                   help="Same margin used at training; a Gauss is kept if it "
                        "is EVER in front of (mesh_depth + margin) in any view.")
    p.add_argument("--mesh_normal_margin", type=float, default=0.0,
                   help="Geometric normal-inflation of the proxy mesh (metres, done "
                        "once at load). Use the same value the finetune used so "
                        "prune matches training cull. Prefer this over --mesh_margin.")
    p.add_argument("--out_dir", required=True,
                   help="Fresh dir for the pruned model (args.pkl/config.yaml "
                        "get copied over so downstream bake works).")
    p.add_argument("--use_train_only", action="store_true",
                   help="Prune based on train cams only (default). Off = also "
                        "count test cams as survival evidence — but that leaks "
                        "test-view information into what surfels survive.")
    args = p.parse_args()

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)
    print(f"[prune] loading iter={args.iteration}")

    train_args, cfg, ingp, gaussians, scene, pipe, dataset, beta_kern = \
        load_checkpoint(args.model_path, args.iteration)
    N = int(gaussians.get_xyz.shape[0])
    print(f"[prune] loaded N={N:,} surfels")

    cams = scene.getTrainCameras()
    print(f"[prune] evaluating survival across {len(cams)} train cameras "
          f"(margin={args.mesh_margin} m)")

    baker = MeshDepthBaker(args.mesh_ply,
                           inflate_margin_normal=float(getattr(args, 'mesh_normal_margin', 0.0)))
    centers = gaussians.get_xyz.detach()   # [N, 3] on cuda
    survivor = torch.zeros(N, dtype=torch.bool, device=centers.device)

    for cam in tqdm(cams, desc="cams"):
        mesh_z = baker.cam_depth(cam, args.mesh_margin)
        keep = gauss_occ_mask(centers, cam, mesh_z)   # [N] bool, True = in-front
        survivor |= keep
        if survivor.all().item():
            print(f"\n[prune] all Gauss survive at least one view; nothing to prune.")
            break

    n_survive = int(survivor.sum().item())
    n_drop = N - n_survive
    print(f"[prune] survivors: {n_survive:,} / {N:,}  "
          f"({100.0 * n_survive / N:.1f}%)  drop {n_drop:,}")

    if n_drop == 0:
        print("[prune] nothing to drop — copying model as-is.")

    # --- Apply the drop to every per-Gauss tensor GaussianModel owns. ---
    dead = ~survivor
    dead_mask_cpu_bool = dead.cpu()
    ATTRS = [
        "_xyz", "_features_dc", "_features_rest",
        "_opacity", "_scaling", "_rotation",
        "_appearance_level",
        "_shape", "_flex_beta",
        "_sv_sites", "_sv_colors", "_sv_dc", "_sv_tau",
        "_gaussian_features",
        "_is_textured", "_scaling_z",
        "_film_params",
        "_sv_mask",
    ]
    keep_mask = survivor.to(centers.device)
    for name in ATTRS:
        t = getattr(gaussians, name, None)
        if t is None or not torch.is_tensor(t) or t.numel() == 0:
            continue
        if t.shape[0] != N:
            continue
        setattr(gaussians, name, t[keep_mask])
        print(f"  pruned {name:<22s}  {tuple(t.shape)} -> "
              f"{tuple(getattr(gaussians, name).shape)}")

    # --- Save the pruned model. ---
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pc_dir = out / "point_cloud" / f"iteration_{args.iteration}"
    pc_dir.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(pc_dir / "point_cloud.ply"))
    print(f"[prune] wrote {pc_dir / 'point_cloud.ply'}")

    # --- Copy the config artifacts so downstream bake works out of the box. ---
    import shutil
    src_dir = Path(args.model_path)
    for name in ("args.pkl", "args.json", "config.yaml", "cfg_args",
                 "cameras.json", "command_line.txt", "input.ply"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out / name)
    # ngp checkpoint (hash+MLP): permanently pruning surfels doesn't touch the
    # hash grid or the MLP, so we can copy the ngp_*.pth verbatim.
    ngp_src = src_dir / f"ngp_{args.iteration}.pth"
    if ngp_src.exists():
        shutil.copy2(ngp_src, out / f"ngp_{args.iteration}.pth")
    print(f"[prune] config artifacts copied to {out}")
    print(f"[prune] done — pruned model ready at {out}")


if __name__ == "__main__":
    main()

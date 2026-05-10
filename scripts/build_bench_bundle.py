"""Build a self-contained "bench bundle" for one trained-and-baked scene.

Outputs everything bench_minimal.py needs to run FPS/PSNR/SSIM/LPIPS on
another machine, with NO nest-splatting source dependency on the receiver:

  bundle/<scene_name>/
    gaussian_state.pt   pre-activated tensors snapshot
    atlas_texture.bc7   raw BC7 byte stream
    atlas_rects.pt      [N, 4] float32
    sb_params.pt        [N, K, 6] float32       (if --feature beta)
    bake_meta.json      runtime activation/dequant/feature config
    cameras.pt          list[ dict(...) ] for the test split
    images/<name>.png   ground-truth test images at eval resolution

The bundle is keyed only on the model_path; this script runs on the
training/bake host where Scene + GaussianModel work, so we can
"freeze" the eval-side state once and replay it elsewhere.
"""
import argparse, json, os, pickle, shutil, sys
from argparse import Namespace
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from hash_encoder.config import Config


def _load_args(model_path):
    p = os.path.join(model_path, "args.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    p = os.path.join(model_path, "args.json")
    if os.path.exists(p):
        with open(p) as f:
            return Namespace(**json.load(f))
    raise FileNotFoundError(f"No args.pkl/args.json in {model_path}")


def _make_sv_state(g):
    if g._sv_sites.numel() == 0:
        return None
    K = g._sv_sites.shape[1]
    sites_n = torch.nn.functional.normalize(g._sv_sites, dim=-1)
    if g._sv_tau is not None and g._sv_tau.numel() > 0:
        tau = torch.exp(g._sv_tau)
    else:
        tau = torch.norm(g._sv_sites, dim=-1)
    mask = getattr(g, "_sv_mask", None)
    apply_mask = (
        mask is not None
        and (not getattr(g, "_sv_training_flag", True))
        and mask.shape[0] == g._sv_sites.shape[0]
    )
    if apply_mask:
        far = torch.full_like(sites_n, 1e8)
        sites_n = torch.where(mask.unsqueeze(-1), sites_n, far)
    sv_dc = getattr(g, "_sv_dc", None)
    if sv_dc is not None and sv_dc.numel() > 0:
        colors = (g._sv_colors + sv_dc.unsqueeze(1)).contiguous()
    else:
        colors = g._sv_colors.contiguous()
    return {"sites": sites_n.contiguous().cpu(),
            "tau":   tau.contiguous().cpu(),
            "colors": colors.cpu(),
            "K": int(K)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="Path containing baked_atlas/atlas_texture.bc7")
    ap.add_argument("--out_dir", required=True,
                    help="Bundle output directory")
    ap.add_argument("--iteration", type=int, default=-1)
    args = ap.parse_args()

    bake_dir = os.path.join(args.model_path, "baked_atlas")
    bc7 = os.path.join(bake_dir, "atlas_texture.bc7")
    if not os.path.exists(bc7):
        raise FileNotFoundError(f"No baked atlas at {bc7}")

    train_args = _load_args(args.model_path)
    train_args.model_path = args.model_path
    train_args.eval = True

    cfg_yaml = os.path.join(args.model_path, "config.yaml")
    if not os.path.exists(cfg_yaml):
        cfg_yaml = train_args.yaml
    cfg = Config(cfg_yaml)

    # Auto-detect iteration
    iteration = args.iteration
    if iteration == -1:
        import glob
        plys = glob.glob(os.path.join(args.model_path, "point_cloud", "iteration_*"))
        iters = [int(os.path.basename(p).split("_")[1]) for p in plys]
        iteration = max(iters)
    print(f"[BUNDLE] Using iteration {iteration}")

    # Pull in the scene to extract test cameras (training PLY is loaded then
    # immediately replaced with baked.ply for the activated-tensor snapshot).
    gaussians = GaussianModel(train_args.sh_degree)
    gaussians.kernel_type = getattr(train_args, "kernel", "gaussian")
    temp = argparse.ArgumentParser()
    mp = ModelParams(temp, sentinel=True)
    dataset = mp.extract(train_args)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cams = scene.getTestCameras()
    print(f"[BUNDLE] {len(test_cams)} test cameras")

    # Reload baked PLY (Scene() ctor overwrote it with the training PLY).
    baked_ply = os.path.join(bake_dir, "baked.ply")
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha

    # Resolve kernel_type int.
    kernel_map = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}
    kernel_type = kernel_map.get(getattr(train_args, "kernel", "gaussian"), 0)

    # Bake meta — read first so we can branch on feature_mode below.
    with open(os.path.join(bake_dir, "bake_meta.json")) as f:
        bake_meta = json.load(f)

    # Snapshot pre-activated Gaussian state.
    # `--feature beta` uses only the SH DC slot (SB lobes carry directional
    # color); _features_rest is empty along axis 1, so get_features returns
    # [N, 1, 3]. Mirror that with active_sh_degree=0 so the rasterizer reads
    # only the DC coefficient (M=1 instead of M=16).
    _is_beta = (bake_meta.get("feature_mode", "sh") == "beta")
    state = {
        "means3D":   gaussians.get_xyz.detach().cpu().contiguous(),
        "opacities": gaussians.get_opacity.detach().cpu().contiguous(),
        "scales":    gaussians.get_scaling.detach().cpu().contiguous(),
        "rotations": gaussians.get_rotation.detach().cpu().contiguous(),
        "shs":       gaussians.get_features.detach().cpu().contiguous(),
        "kernel_type": int(kernel_type),
        "active_sh_degree": (0 if _is_beta else int(gaussians.active_sh_degree)),
    }
    if kernel_type > 0 and hasattr(gaussians, "_shape") \
            and gaussians._shape is not None and gaussians._shape.numel() > 0:
        state["shapes"] = gaussians.get_shape.detach().cpu().contiguous()
    else:
        state["shapes"] = None

    # SV fused-CUDA state (None when not --feature SV).
    state["sv_state"] = _make_sv_state(gaussians) if hasattr(gaussians, "_sv_sites") else None

    # Output skeleton
    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save Gaussian state + atlas + meta.
    torch.save(state, os.path.join(args.out_dir, "gaussian_state.pt"))
    shutil.copy2(bc7, os.path.join(args.out_dir, "atlas_texture.bc7"))
    shutil.copy2(os.path.join(bake_dir, "atlas_rects.pt"),
                 os.path.join(args.out_dir, "atlas_rects.pt"))
    if bake_meta.get("sb_number", 0) > 0 and bake_meta.get("sb_params_file"):
        shutil.copy2(os.path.join(bake_dir, bake_meta["sb_params_file"]),
                     os.path.join(args.out_dir, "sb_params.pt"))
    with open(os.path.join(args.out_dir, "bake_meta.json"), "w") as f:
        json.dump(bake_meta, f, indent=2)

    # Snapshot test cameras + ground-truth images.
    cams = []
    for cam in test_cams:
        cams.append({
            "image_name": cam.image_name,
            "image_width":  int(cam.image_width),
            "image_height": int(cam.image_height),
            "FoVx": float(cam.FoVx),
            "FoVy": float(cam.FoVy),
            # Snapshot the already-built transforms — bench-time we don't
            # need to know R, T, znear, zfar; we just feed these to the rasterizer.
            "world_view_transform": cam.world_view_transform.detach().cpu(),
            "full_proj_transform":  cam.full_proj_transform.detach().cpu(),
            "camera_center":        cam.camera_center.detach().cpu(),
        })
        # Save GT image as lossless PNG (uint8). Original_image is in [0,1]
        # already clamped; convert to [0, 255] uint8.
        gt = (cam.original_image[:3].clamp(0, 1).cpu()
              .permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        from PIL import Image
        Image.fromarray(gt).save(
            os.path.join(images_dir, f"{cam.image_name}.png"), optimize=False)
    torch.save(cams, os.path.join(args.out_dir, "cameras.pt"))

    # Manifest
    n_g = state["means3D"].shape[0]
    h, w = test_cams[0].image_height, test_cams[0].image_width
    manifest = {
        "n_gaussians": int(n_g),
        "n_test_cameras": len(test_cams),
        "image_resolution": f"{w}x{h}",
        "kernel_type": int(kernel_type),
        "feature_mode": bake_meta.get("feature_mode", "sh"),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[BUNDLE] OK  N={n_g:,}  cams={len(test_cams)}  res={w}x{h}  → {args.out_dir}")


if __name__ == "__main__":
    main()

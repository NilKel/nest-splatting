#!/usr/bin/env python3
"""
FiLM decomposition renders for a trained `--method film` model.

For a stride of test views, saves three variants per view (+ GT):
  1) full              : f = gamma * H(x) + beta            (normal FiLM render)
  2) gamma0_beta_only  : f = beta            (gamma set to 0 -> hash masked out)
  3) beta0_gamma_hash  : f = gamma * H(x)    (beta set to 0  -> just the modulated hash)

Usage:
  python scripts/render_film_decomp.py [MODEL_PATH] [--stride 50]
"""
import os, sys, glob, pickle, argparse
import numpy as np
import torch
from PIL import Image
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams
from gaussian_renderer import render

p = argparse.ArgumentParser()
p.add_argument("model_path", nargs="?",
               default="outputs/nerf_synthetic/chair/film/REDO2ldeb")
p.add_argument("--stride", type=int, default=50)
cli = p.parse_args()

model_path = cli.model_path
out_dir = os.path.join(model_path, "decomp")
os.makedirs(out_dir, exist_ok=True)

# --- args + config ---
with open(os.path.join(model_path, "args.pkl"), "rb") as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True
assert getattr(args, "method", None) == "film", f"not a film run: method={getattr(args,'method',None)}"
cfg_model = Config(os.path.join(model_path, "config.yaml"))
iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                for f in glob.glob(os.path.join(model_path, "ngp_*.pth")))
print(f"[decomp] model={model_path}  iteration={iteration}")

# --- Gaussians + Scene (loads the film PLY at this iteration; film_* -> _film_params) ---
dataset = ModelParams(argparse.ArgumentParser(), sentinel=True).extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
gaussians.kernel_type = getattr(args, "kernel", "gaussian")   # beta_scaled -> kernel_type 4 in renderer
test_cameras = scene.getTestCameras()

# --- INGP (hash + 64-wide MLP, whatever this run used) ---
ingp = INGP(cfg_model, args=args).to("cuda")
ingp.load_model(model_path, iteration)
ingp.set_active_levels(iteration)

pipe = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.0)
bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")  # white_background: false
beta = cfg_model.surfel.tg_beta

assert hasattr(gaussians, "_film_params") and gaussians._film_params.numel() > 0, \
    "no _film_params loaded from PLY!"
orig = gaussians._film_params.data.clone()
print(f"[decomp] _film_params {tuple(orig.shape)} | gamma mean={orig[:,0].mean():.4f} "
      f"beta absmax={orig[:,1:].abs().max():.4f}")

def _set(full=False, gamma0=False, beta0=False):
    t = orig.clone()
    if gamma0: t[:, 0] = 0.0     # f = beta
    if beta0:  t[:, 1:] = 0.0    # f = gamma * H
    gaussians._film_params.data.copy_(t)

variants = [
    ("full",             dict(full=True)),
    ("gamma0_beta_only", dict(gamma0=True)),
    ("beta0_gamma_hash", dict(beta0=True)),
]

def save(t, path):
    img = t.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)

idxs = list(range(0, len(test_cameras), cli.stride))
print(f"[decomp] {len(test_cameras)} test views; stride={cli.stride}; rendering idx {idxs}")

with torch.no_grad():
    for name, kw in variants:
        _set(**kw)
        for idx in idxs:
            cam = test_cameras[idx]
            cam_name = getattr(cam, "image_name", f"view_{idx:03d}")
            pkg = render(cam, gaussians, pipe, bg, ingp=ingp, beta=beta,
                         iteration=iteration, cfg=cfg_model)
            save(pkg["render"], os.path.join(out_dir, f"{idx:03d}_{cam_name}_{name}.png"))
            if name == "full":
                save(cam.original_image[:3].cuda(),
                     os.path.join(out_dir, f"{idx:03d}_{cam_name}_gt.png"))
        print(f"[decomp]   rendered '{name}' for {len(idxs)} views")

gaussians._film_params.data.copy_(orig)  # restore
print(f"[decomp] done -> {out_dir}")

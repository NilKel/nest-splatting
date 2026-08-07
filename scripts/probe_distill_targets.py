"""Precompute per-train-view TEACHER residual images for proberes distillation.

For each training view this renders the teacher's SIGNED blended residual

    R_teacher(pixel) = sum_i  T_i * alpha_i * residual_i          (no ReLU)

using `decompose_mode='tex_only_raw'` (sh_bias=-999 kills the SV base; the
mode-2 per-pixel clamp is bypassed). Under residual_mode 2 — the post-flip
res_switch state — the per-Gauss outer ReLU is gone, so this sum is LINEAR in
the per-fragment residuals, and therefore linear in the atlas texels a probe
model fetches. That makes "match the teacher's residual" a well-conditioned
target with no dead gradient zones, unlike mode 0 where ReLU(SH+res)<=0 zeroes
the texture gradient.

Output: <out>/targets.pt  = {image_name: fp16 [3,H,W]} + meta.

  conda run -n nest_splatting python scripts/probe_distill_targets.py \
      -m outputs/nerf_synthetic/chair/res_switch/erefnr --iteration 30000
"""
import os, sys, pickle, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from hash_encoder.config import Config
from hash_encoder.modules import INGP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_path', required=True)
    ap.add_argument('--iteration', type=int, default=-1)
    ap.add_argument('--out_tag', type=str, default='distill_targets')
    a = ap.parse_args()

    with open(os.path.join(a.model_path, 'args.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.model_path = a.model_path
    it = a.iteration
    if it == -1:
        import glob
        it = max(int(os.path.basename(f)[4:-4])
                 for f in glob.glob(os.path.join(a.model_path, 'ngp_*.pth')))

    cfg_yaml = os.path.join(a.model_path, 'config.yaml')
    cfg = Config(cfg_yaml if os.path.exists(cfg_yaml) else args.yaml)

    tp = ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.feature_mode = 'SV' if getattr(args, 'feature', 'sh') == 'SV' else 'sh'
    gaussians.kernel_type = getattr(args, 'kernel', 'gaussian')
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False, full_args=args)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(a.model_path, it)
    ingp.set_active_levels(10 ** 6)          # all hash levels on (post-c2f teacher)
    ingp.hashgrid_disabled = False

    # res_switch POST-FLIP state: signed residual, no per-Gauss outer ReLU.
    from diff_surfel_3D_sh_res import set_residual_mode
    set_residual_mode(2)
    ingp.is_mixed_deferred_relu_mode = True
    ingp.lru_slope = float(getattr(args, 'lru', 0.0) or 0.0)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device='cuda')
    cams = scene.getTrainCameras()
    print(f'[DISTILL] {len(cams)} train views, residual_mode=2, teacher iter={it}')

    out = {}
    with torch.no_grad():
        for i, cam in enumerate(cams):
            pkg = render(cam, gaussians, pipe, bg, ingp=ingp, cfg=cfg,
                         iteration=it, decompose_mode='tex_only_raw')
            out[cam.image_name] = pkg['render'].detach().half().cpu()
            if i % 25 == 0:
                r = pkg['render']
                print(f'  [{i:4d}/{len(cams)}] {cam.image_name}  '
                      f'mean={r.mean():+.4f} std={r.std():.4f} '
                      f'min={r.min():+.3f} max={r.max():+.3f}')

    d = os.path.join(a.model_path, a.out_tag)
    os.makedirs(d, exist_ok=True)
    torch.save({'targets': out, 'iteration': it, 'residual_mode': 2,
                'method': getattr(args, 'method', None)},
               os.path.join(d, 'targets.pt'))
    mb = sum(v.numel() * 2 for v in out.values()) / 1048576
    print(f'[DISTILL] saved {len(out)} targets ({mb:.0f} MB) -> {d}/targets.pt')


if __name__ == '__main__':
    main()

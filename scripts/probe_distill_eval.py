"""Score a TRAINED proberes run against the cached teacher residual targets.

Read-only. Renders each train view's SIGNED blended residual through the trained
atlas/probes ('tex_only_raw', residual_mode 2) and reports L1 vs the teacher's,
i.e. exactly the quantity --probe_distill_only minimizes.

The point: if a GT-trained run scores WELL on images but POORLY here, then
faithfulness to the teacher's residual and image quality are different objectives
at this atlas capacity — and blend-matching is the wrong distillation target
regardless of optimizer settings.

  conda run -n nest_splatting python scripts/probe_distill_eval.py \
      -m outputs/nerf_synthetic/chair/proberes/rs_frz_gt \
      --targets outputs/nerf_synthetic/chair/res_switch/erefnr/distill_targets
"""
import os, sys, glob, pickle, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from hash_encoder.config import Config
from hash_encoder.modules import INGP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_path', required=True, help='a TRAINED proberes run dir')
    ap.add_argument('--targets', required=True, help='dir holding targets.pt')
    ap.add_argument('--iteration', type=int, default=-1)
    ap.add_argument('--override_bake', type=str, default=None,
                    help='replace the trained atlas/probes with a bake dir (tex_init.pt + '
                         'probes.pt) -> scores the UNTRAINED bake using the same harness')
    a = ap.parse_args()

    with open(os.path.join(a.model_path, 'args.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.model_path = a.model_path
    it = a.iteration
    if it == -1:
        it = max(int(os.path.basename(f)[4:-4])
                 for f in glob.glob(os.path.join(a.model_path, 'ngp_*.pth')))

    cfg_yaml = os.path.join(a.model_path, 'config.yaml')
    cfg = Config(cfg_yaml if os.path.exists(cfg_yaml) else args.yaml)
    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.feature_mode = 'SV' if getattr(args, 'feature', 'sh') == 'SV' else 'sh'
    gaussians.kernel_type = getattr(args, 'kernel', 'gaussian')
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False, full_args=args)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(a.model_path, it)
    ingp.set_active_levels(10 ** 6)
    ingp.hashgrid_disabled = False

    # Same state the distillation loss saw: mode 2, signed residual.
    from gaussian_renderer import _sh_res_setter_mod
    _sh_res_setter_mod(ingp).set_residual_mode(2)
    ingp.is_mixed_deferred_relu_mode = True
    ingp.lru_slope = float(getattr(args, 'lru', 0.0) or 0.0)

    if a.override_bake:
        T0 = torch.load(os.path.join(a.override_bake, 'tex_init.pt'), map_location='cuda')
        pr = torch.load(os.path.join(a.override_bake, 'probes.pt'), map_location='cuda')
        pr = pr['probes'] if isinstance(pr, dict) else pr
        with torch.no_grad():
            ingp.probe_field.pixels.copy_(T0.to(ingp.probe_field.pixels.dtype))
            ingp.probe_head.fixed_probes.data = pr.to(
                ingp.probe_head.fixed_probes.dtype if hasattr(ingp.probe_head.fixed_probes,'dtype') else torch.float32)
        print(f'[EVAL] overrode atlas+probes with the BAKE at {a.override_bake}')

    tgts = torch.load(os.path.join(a.targets, 'targets.pt'), map_location='cpu')['targets']
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device='cuda')

    # Also score the FULL composite on the held-out test views, so each model has
    # both coordinates of the (blend-match, image-quality) tradeoff curve.
    from utils.image_utils import psnr as _psnr
    psnrs = []
    with torch.no_grad():
        for cam in scene.getTestCameras():
            img = render(cam, gaussians, pipe, bg, ingp=ingp, cfg=cfg, iteration=it,
                         aabb_mode=getattr(args, 'aabb', 'rect'),
                         lowpass=getattr(args, 'lowpass', False))['render'].clamp(0, 1)
            psnrs.append(_psnr(img, cam.original_image.cuda().clamp(0, 1)).mean().item())

    l1s, ratios = [], []
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            t = tgts.get(cam.image_name)
            if t is None:
                continue
            R = render(cam, gaussians, pipe, bg, ingp=ingp, cfg=cfg, iteration=it,
                       aabb_mode=getattr(args, 'aabb', 'rect'),
                       lowpass=getattr(args, 'lowpass', False),
                       decompose_mode='tex_only_raw')['render']
            t = t.to('cuda', torch.float32)
            l1s.append((R - t).abs().mean().item())
            ratios.append((R.std() / t.std().clamp_min(1e-9)).item())

    l1 = torch.tensor(l1s)
    print(f'\n=== {os.path.basename(a.model_path)}  ({len(l1s)} train views) ===')
    print(f'  L_distill (L1 vs teacher blend): mean={l1.mean():.5f}  median={l1.median():.5f}  '
          f'min={l1.min():.5f}  max={l1.max():.5f}')
    print(f'  std(R_model)/std(R_teacher)    : mean={torch.tensor(ratios).mean():.4f}')
    if psnrs:
        print(f'  TEST PSNR (full composite)     : {torch.tensor(psnrs).mean():.3f} dB  '
              f'({len(psnrs)} views)')


if __name__ == '__main__':
    main()

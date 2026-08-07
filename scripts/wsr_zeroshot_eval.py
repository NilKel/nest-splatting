"""Zero-shot WSR evaluation on a trained proberes run (no finetuning).

Loads the checkpoint, distill-initializes the per-surfel occlusion from a sorted
record_transmittance dump (occ = Σ(α·T)/Σα over train views), then scores the
held-out test views three ways:

  1. sorted        — through the wsr clone with set_wsr(0); must reproduce the
                     run's original test PSNR (sanity),
  2. wsr occ=1     — naive sort-free weighted mean (raw leakage floor),
  3. wsr distilled — the zero-shot number: how far view-averaged occlusion
                     alone closes the gap. This is the finetune's starting point.

  conda run -n nest_splatting python scripts/wsr_zeroshot_eval.py \
      -m /mnt/nilkel_hdd/outputs/mip_360/room/proberes/room8k_p32
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
    ap.add_argument('--iteration', type=int, default=-1)
    ap.add_argument('--save_occ', type=str, default=None,
                    help='optionally save the distilled occ tensor (for reuse)')
    ap.add_argument('--gate_taus', type=float, nargs='*', default=[],
                    help='sweep the 2-pass transmittance gate at these taus '
                         '(uses the PLY occ as-is; skips the distill dump)')
    ap.add_argument('--dgate_margins', type=float, nargs='*', default=[],
                    help='sweep the mean-depth gate (?wsr=3) at these margins')
    ap.add_argument('--skip_distill', action='store_true',
                    help='use the occ already in the PLY (trained checkpoints)')
    ap.add_argument('--active_iter', type=int, default=10 ** 6,
                    help='iteration fed to set_active_levels (probe c2f gating). '
                         'Finetunes renumbered to 1..5000 trained with '
                         '1+iter//probe_c2f_interval field levels — eval must match.')
    a = ap.parse_args()

    with open(os.path.join(a.model_path, 'args.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.model_path = a.model_path
    args.wsr = True  # load_ply creates _wsr_occ; INGP sets is_wsr_mode
    it = a.iteration
    if it == -1:
        it = max(int(os.path.basename(f)[4:-4])
                 for f in glob.glob(os.path.join(a.model_path, 'ngp_*.pth')))

    cfg_yaml = os.path.join(a.model_path, 'config.yaml')
    cfg = Config(cfg_yaml if os.path.exists(cfg_yaml) else args.yaml)
    # Mirror train.py's --cold override (train.py:~9192). Without this, a
    # finetune whose iterations were renumbered below the yaml's
    # ingp_stage.switch_iter (10k for 360_indoor) silently renders MODE 0 —
    # no hash/probe residual, SV-only, ~9 dB low. Bitten hard 2026-08-05.
    if getattr(args, 'cold', False):
        cfg.ingp_stage.switch_iter = 0
    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.feature_mode = 'SV' if getattr(args, 'feature', 'sh') == 'SV' else 'sh'
    gaussians.kernel_type = getattr(args, 'kernel', 'gaussian')
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False, full_args=args)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    N = gaussians.get_xyz.shape[0]
    if gaussians._wsr_occ.numel() == 0:
        # load_ply didn't create it (e.g. args pickled without wsr routing) — do it here.
        gaussians._wsr_occ = torch.nn.Parameter(
            torch.full((N, 1), 4.595, device='cuda').requires_grad_(True))
    print(f'[WSR-EVAL] {N} surfels, iteration {it}')

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(a.model_path, it)
    ingp.set_active_levels(a.active_iter)
    ingp.hashgrid_disabled = False
    assert ingp.is_wsr_mode, 'INGP did not pick up args.wsr'

    # Configure the wsr module's device globals like train.py startup would.
    from gaussian_renderer import _sh_res_setter_mod
    mod = _sh_res_setter_mod(ingp)
    assert 'wsr' in mod.__name__, f'setter mod is {mod.__name__}, expected the wsr clone'
    ab = getattr(args, 'activation_bias', [0.5, 0.0])
    mod.set_residual_mode(int(getattr(args, '_residual_mode', 0) or 0))
    mod.set_activation_bias(float(ab[0]), float(ab[1]))
    mod.set_lru_slope(float(getattr(args, 'lru', 0.0) or 0.0))
    mod.set_contrib_thresh(0.0); mod.set_count_thresh(0)
    mod.set_opacity_thresh(0.0); mod.set_dropout(0.0, 0)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device='cuda')
    rkw = dict(ingp=ingp, cfg=cfg, iteration=it,
               aabb_mode=getattr(args, 'aabb', 'rect'),
               lowpass=getattr(args, 'lowpass', False),
               pixel_center=getattr(args, 'pixel_center', False))

    from utils.image_utils import psnr as _psnr

    def eval_test(tag):
        vals = []
        with torch.no_grad():
            for cam in scene.getTestCameras():
                img = render(cam, gaussians, pipe, bg, **rkw)['render'].clamp(0, 1)
                vals.append(_psnr(img, cam.original_image.cuda().clamp(0, 1)).mean().item())
        t = torch.tensor(vals)
        print(f'  {tag:16s}: {t.mean():.3f} dB  (min {t.min():.2f}, max {t.max():.2f}, '
              f'{len(vals)} views)')
        return t.mean().item()

    # --- 1. sorted sanity ---
    ingp.wsr_sorted = True
    eval_test('sorted (sanity)')

    if a.skip_distill or a.gate_taus or a.dgate_margins:
        # Trained checkpoint path: occ comes from the PLY; evaluate the wsr
        # operator as-is, then sweep the gates.
        ingp.wsr_sorted = False
        ingp.wsr_gate_tau = 0.0
        ingp.wsr_dgate_margin = 0.0
        eval_test('wsr (ply occ)')
        for tau in a.gate_taus:
            ingp.wsr_gate_tau = float(tau)
            eval_test(f'wsr gate tau={tau:g}')
        ingp.wsr_gate_tau = 0.0
        for m in a.dgate_margins:
            ingp.wsr_dgate_margin = float(m)
            eval_test(f'wsr dgate m={m:g}')
        ingp.wsr_dgate_margin = 0.0
        return

    # --- distill dump over train views (still sorted) ---
    num = torch.zeros(N, device='cuda')
    den = torch.zeros(N, device='cuda')
    with torch.no_grad():
        cams = scene.getTrainCameras()
        for i, cam in enumerate(cams):
            pkg = render(cam, gaussians, pipe, bg, record_transmittance=True, **rkw)
            num += pkg['transmittance_avg'].view(-1)
            den += pkg['cover_pixels'].view(-1)
            if (i + 1) % 50 == 0:
                print(f'  [dump] {i + 1}/{len(cams)} views')
    occ0 = (num / den.clamp_min(1e-6)).clamp(1e-3, 1.0 - 1e-3)
    occ0[den < 1e-6] = 0.5
    print(f'[WSR-EVAL] distilled occ: mean {occ0.mean():.4f}  p10 {occ0.quantile(0.1):.4f}  '
          f'p50 {occ0.quantile(0.5):.4f}  p90 {occ0.quantile(0.9):.4f}  '
          f'unseen {(den < 1e-6).sum().item()}')
    if a.save_occ:
        torch.save({'occ': occ0.cpu(), 'num': num.cpu(), 'den': den.cpu()}, a.save_occ)
        print(f'[WSR-EVAL] saved occ -> {a.save_occ}')

    # --- 2. WSR, occ = 1 (naive weighted mean) ---
    ingp.wsr_sorted = False
    gaussians._wsr_occ.data.fill_(12.0)  # sigmoid ≈ 1
    eval_test('wsr occ=1')

    # --- 3. WSR, distilled occ ---
    gaussians._wsr_occ.data = torch.log(occ0 / (1 - occ0)).view(-1, 1)
    eval_test('wsr distilled')


if __name__ == '__main__':
    main()

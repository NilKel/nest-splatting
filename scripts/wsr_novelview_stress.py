"""Novel-view stress test for the WSR/composite operators (CUDA, headless).

Renders a trained --wsr / --wsr_composite checkpoint from CAMERA POSES OFF THE
TRAINING HULL (dolly toward the scene + lateral shifts of a test camera) under
sorted / wsr / composite operators, and saves the images side by side.

Purpose: decide whether interactive-viewer transparency reports are a viewer
bug or an operator limitation (baked occ + K=1 core cannot express stacked
semi-transparent occluders at novel rays).

  conda run -n nest_splatting python scripts/wsr_novelview_stress.py \
      -m outputs/mip_360/room/proberes/room_htc5k --cam 0
"""
import os, sys, glob, pickle, argparse, copy
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from hash_encoder.config import Config
from hash_encoder.modules import INGP


def shifted_camera(cam, forward=0.0, right=0.0, up=0.0):
    """Clone `cam` with its center translated in CAMERA axes (metres)."""
    c = copy.copy(cam)
    wvt = cam.world_view_transform  # [4,4], stored TRANSPOSED (row-vector convention)
    # world_view_transform[:3,:3] = R_w2v^T → its COLUMNS index view axes, so
    # the camera axes expressed in world coords are wvt[:3, i] for i=x,y,z.
    cam_right = wvt[:3, 0]; cam_up = wvt[:3, 1]; cam_fwd = wvt[:3, 2]
    delta_w = forward * cam_fwd + right * cam_right + up * cam_up
    c2w = torch.inverse(wvt.transpose(0, 1))
    c2w[:3, 3] += delta_w
    w2c = torch.inverse(c2w)
    new_wvt = w2c.transpose(0, 1).contiguous()
    c.world_view_transform = new_wvt
    c.full_proj_transform = (new_wvt.unsqueeze(0).bmm(
        cam.projection_matrix.unsqueeze(0))).squeeze(0)
    c.camera_center = new_wvt.inverse()[3, :3]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_path', required=True)
    ap.add_argument('--cam', type=int, default=0, help='test camera index to perturb')
    ap.add_argument('--out', default=None)
    ap.add_argument('--gate_tau', type=float, default=0.0,
                    help='>0: arm the 2-pass transmittance gate on the wsr/composite renders')
    ap.add_argument('--dgate_margin', type=float, default=0.0,
                    help='>0: arm the mean-depth gate (?wsr=3) instead')
    a = ap.parse_args()
    out_dir = a.out or os.path.join(a.model_path, 'novelview_stress')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(a.model_path, 'args.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.model_path = a.model_path
    it = max(int(os.path.basename(f)[4:-4])
             for f in glob.glob(os.path.join(a.model_path, 'ngp_*.pth')))
    cfg = Config(os.path.join(a.model_path, 'config.yaml'))
    # Mirror train.py's --cold override — without it, finetunes renumbered
    # below switch_iter (10k) render MODE 0 (no residual). See wsr_zeroshot_eval.
    if getattr(args, 'cold', False):
        cfg.ingp_stage.switch_iter = 0
    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.feature_mode = 'SV' if getattr(args, 'feature', 'sh') == 'SV' else 'sh'
    gaussians.kernel_type = getattr(args, 'kernel', 'gaussian')
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False, full_args=args)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    assert gaussians._wsr_occ.numel() > 0, 'no wsr_occ in PLY'

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(a.model_path, it)
    ingp.set_active_levels(10 ** 6)
    ingp.hashgrid_disabled = False

    from gaussian_renderer import _sh_res_setter_mod
    mod = _sh_res_setter_mod(ingp)
    ab = getattr(args, 'activation_bias', [0.5, 0.0])
    mod.set_residual_mode(0); mod.set_activation_bias(float(ab[0]), float(ab[1]))
    mod.set_lru_slope(0.0); mod.set_contrib_thresh(0.0); mod.set_count_thresh(0)
    mod.set_opacity_thresh(0.0); mod.set_dropout(0.0, 0)

    bg = torch.zeros(3, device='cuda')
    rkw = dict(ingp=ingp, cfg=cfg, iteration=it,
               aabb_mode=getattr(args, 'aabb', 'rect'),
               lowpass=getattr(args, 'lowpass', False))

    from torchvision.utils import save_image
    base = scene.getTestCameras()[a.cam]
    poses = [('orig', base)]
    for f in (0.4, 0.8, 1.2):
        poses.append((f'dolly{f:g}', shifted_camera(base, forward=f)))
    poses.append(('right0.5', shifted_camera(base, right=0.5)))
    poses.append(('down0.3_fwd0.6', shifted_camera(base, forward=0.6, up=-0.3)))

    modes = [('sorted', True, None), ('composite', False, True), ('wsr', False, False)]
    ingp.wsr_gate_tau = a.gate_tau
    ingp.wsr_dgate_margin = a.dgate_margin
    gate_sfx = f'_g{a.gate_tau:g}' if a.gate_tau > 0 else (
        f'_d{a.dgate_margin:g}' if a.dgate_margin > 0 else '')
    with torch.no_grad():
        for pname, cam in poses:
            for mname, srt, comp in modes:
                ingp.wsr_sorted = srt
                if comp is not None:
                    ingp.is_wsr_composite = comp
                sfx = gate_sfx if not srt else ''
                img = render(cam, gaussians, pipe, bg, **rkw)['render'].clamp(0, 1)
                save_image(img, os.path.join(out_dir, f'{pname}_{mname}{sfx}.png'))
            print(f'  {pname}: rendered sorted/composite/wsr (gate_tau={a.gate_tau:g})')
    ingp.wsr_sorted = False
    print(f'[stress] wrote {len(poses) * len(modes)} images -> {out_dir}')


if __name__ == '__main__':
    main()

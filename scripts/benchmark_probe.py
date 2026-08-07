"""Benchmark a trained proberes run through the dedicated CUDA renderer.

`diff_surfel_probe_render` is a clone of `diff_surfel_bake_render_lean`
(LEAN_FLAGS=CONIC) with the per-surfel atlas RECT replaced by a per-surfel
general AFFINE (the probe), sampling ONE shared texture:

    baked : au = clamp(base + scale*s.x, rect)      2 FMA + 4 clamps
    probe : au = A00*s.x + A01*s.y + t0             4 FMA, no clamp

`atlas_rects` is reinterpreted as the [N,6] probe array, so no signatures
changed. Reuses benchmark_baked's render/eval path via a sys.modules alias.

  conda run -n nest_splatting python scripts/benchmark_probe.py \
      -m outputs/mip_360/room/proberes/oct8k_p32_frzsv
"""
import os, sys, glob, math, pickle, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Route every `from diff_surfel_bake_render import X` inside benchmark_baked to
# the probe build. Must happen BEFORE importing benchmark_baked.
import diff_surfel_probe_render as _probe_pkg
sys.modules['diff_surfel_bake_render'] = _probe_pkg

from argparse import ArgumentParser
from arguments import ModelParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from scripts.benchmark_baked import evaluate_baked, _make_sv_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_path', required=True, help='trained proberes run dir')
    ap.add_argument('--iteration', type=int, default=-1)
    ap.add_argument('--num_warmup', type=int, default=10)
    ap.add_argument('--num_benchmark', type=int, default=100)
    ap.add_argument('--save_dir', type=str, default=None)
    ap.add_argument('--atlas_dtype', choices=['fp16', 'uint8', 'bc7'], default='fp16',
                    help='fp16 = raw texture (verification); uint8 = quantized hw texture; '
                         'bc7 = u8-quantized + BC7 block compression (1 B/texel, hw decode)')
    ap.add_argument('--sh_only', action='store_true',
                    help='bypass the atlas entirely (FPS attribution: geometry lane only)')
    ap.add_argument('--clamp_pct', type=float, default=99.9,
                    help='quantization range = [P(100-x), P(x)] percentiles of the atlas. '
                         '100 = full min/max (documented dull-render failure at ~4 levels/sigma)')
    a = ap.parse_args()

    with open(os.path.join(a.model_path, 'args.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.model_path = a.model_path
    it = a.iteration
    if it == -1:
        it = max(int(os.path.basename(f)[4:-4])
                 for f in glob.glob(os.path.join(a.model_path, 'ngp_*.pth')))

    # ---- probes + atlas straight out of the trained checkpoint ----------------
    sd = torch.load(os.path.join(a.model_path, f'ngp_{it}.pth'),
                    map_location='cpu')['model_state_dict']
    probes = sd['probe_head.fixed_probes'].float().contiguous().cuda()      # [N,6] texels
    atlas = sd['probe_field.pixels'].float()                                # [R,R,3]
    R = atlas.shape[0]
    print(f'[PROBE] probes {tuple(probes.shape)}  atlas {tuple(atlas.shape)}  iter={it}')
    print(f'[PROBE] atlas range [{atlas.min():+.4f},{atlas.max():+.4f}] '
          f'mean={atlas.mean():+.4f} std={atlas.std():.4f}')

    # ---- scene / gaussians ---------------------------------------------------
    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.feature_mode = 'SV' if getattr(args, 'feature', 'sh') == 'SV' else 'sh'
    gaussians.kernel_type = getattr(args, 'kernel', 'gaussian')
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False, full_args=args)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    N = gaussians.get_xyz.shape[0]
    assert probes.shape[0] == N, f'probe/gauss mismatch: {probes.shape[0]} vs {N}'

    from hash_encoder.config import Config
    cfg_yaml = os.path.join(a.model_path, 'config.yaml')
    cfg = Config(cfg_yaml if os.path.exists(cfg_yaml) else args.yaml)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha

    _probe_pkg.set_activation_bias(*getattr(args, 'activation_bias', [0.5, 0.0]))
    _probe_pkg.set_residual_mode(int(getattr(args, '_residual_mode', 0) or 0))
    _probe_pkg.clear_atlas_cache()
    _probe_pkg.clear_atlas_bc7()
    _probe_pkg.set_use_atlas_tex_object(True)
    _probe_pkg.set_atlas_use_uint8(a.atlas_dtype == 'uint8')

    if a.atlas_dtype == 'bc7':
        import numpy as np, time as _t
        import bc7encoder
        # u8 quantize with percentile clamp: val = q*scale + offset  (q in [0,1] from BC7)
        if a.clamp_pct >= 100.0:
            lo, hi = float(atlas.min()), float(atlas.max())
        else:
            _a = atlas.numpy().ravel()
            lo = float(np.percentile(_a, 100 - a.clamp_pct))
            hi = float(np.percentile(_a, a.clamp_pct))
            del _a
        scale, offset = hi - lo, lo
        q = ((atlas - lo) / max(scale, 1e-8)).clamp_(0, 1)
        u8 = (q * 255.0 + 0.5).to(torch.uint8).numpy()
        clipped = ((atlas < lo) | (atlas > hi)).float().mean().item()
        rgba = np.zeros((R, R, 4), dtype=np.uint8)
        rgba[..., :3] = u8; rgba[..., 3] = 255
        t0 = _t.time()
        bc7_bytes = bc7encoder.encode_image_rgba(rgba, uber_level=1, perceptual=False)
        print(f'[PROBE] BC7: clamp P{a.clamp_pct} -> range [{lo:+.4f},{hi:+.4f}] '
              f'(step/sigma = {scale/255/max(atlas.std().item(),1e-8):.3f}, {clipped*100:.2f}% clipped) '
              f'encoded {len(bc7_bytes)/1048576:.0f} MB in {_t.time()-t0:.1f}s')
        bc7_tensor = torch.frombuffer(bytearray(bc7_bytes), dtype=torch.uint8).cuda()
        _probe_pkg.set_atlas_bc7(bc7_tensor, R, R, offset, scale)
        # kernel ignores atlas_texture when the BC7 tex object is set; dtype must be Half
        atlas_tex = torch.zeros(1, 1, 3, dtype=torch.float16, device='cuda')
        print(f'[PROBE] atlas as BC7 texture: {R}x{R}, {len(bc7_bytes)/1048576:.0f} MB resident')
    else:
        atlas_tex = atlas.half().cuda()
        mb = atlas_tex.numel() * 2 / 1048576
        print(f'[PROBE] atlas as {a.atlas_dtype} texture: {R}x{R}, {mb:.0f} MB resident')

    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(getattr(args, 'kernel', 'gaussian'), 0)
    sv_state = _make_sv_state(gaussians) if gaussians.feature_mode == 'SV' else None
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device='cuda')

    test_cams = scene.getTestCameras()
    print(f'[PROBE] {N:,} surfels, {len(test_cams)} test cameras, '
          f'{test_cams[0].image_width}x{test_cams[0].image_height}')

    if a.sh_only:
        atlas_tex, probes, R = None, None, 0
        print('[PROBE] SH-ONLY lane (atlas bypassed)')

    # Cold-start: the atlas cudaTextureObject is created lazily on first use, and
    # evaluate_baked runs its metric loop BEFORE its FPS warmup — so without this
    # the first view renders with no atlas bound (measured: 10 dB vs 31 on room,
    # dragging the 39-view mean down ~0.5 dB).
    from scripts.benchmark_baked import render_baked
    from diff_surfel_bake_render import prepare_gaussian_inputs as _prep
    with torch.no_grad():
        _pkg = _prep(gaussians, sh_degree=gaussians.active_sh_degree, kernel_type=kernel_type)
        for _ in range(3):
            _ = render_baked(test_cams[0], _pkg, bg, beta=cfg.surfel.tg_beta,
                             atlas_texture=atlas_tex, atlas_rects=probes, atlas_width=R,
                             aabb_mode=3, sort_mode=0, sh_degree=gaussians.active_sh_degree,
                             sv_state=sv_state, final_relu=False)
        torch.cuda.synchronize()
    print('[PROBE] atlas texture warm')

    m = evaluate_baked(test_cams, gaussians, bg, cfg.surfel.tg_beta, kernel_type,
                       atlas_texture=atlas_tex, atlas_rects=probes, atlas_width=R,
                       num_warmup=a.num_warmup, num_benchmark=a.num_benchmark,
                       save_dir=a.save_dir,
                       aabb_mode=3, sort_mode=0, sv_state=sv_state,
                       final_relu=(int(getattr(args, '_residual_mode', 0) or 0) == 2))
    print('\n' + '=' * 62)
    print(f'  PROBE RENDERER — {os.path.basename(a.model_path)}')
    print('=' * 62)
    for k in ('psnr', 'ssim', 'lpips', 'fps', 'ms_per_frame'):
        if k in m:
            print(f'  {k:14s} {m[k]:.4f}' if k != 'fps' else f'  {k:14s} {m[k]:.1f}')
    print('=' * 62)


if __name__ == '__main__':
    main()

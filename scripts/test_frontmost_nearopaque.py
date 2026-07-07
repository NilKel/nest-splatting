"""Near-opaque stress test for the GES frontmost-first promotion (harden rasterizer).

Covers the doc's stated test gap (FRONTMOST_2PASS_DEBUG.md): alphas at the 0.99
clamp, beta_scaled flat-top (low beta, the dG-pole regime), F deep in tile order,
plus an isolated single-surfel pixel (pass-2-blends-nothing, n_contrib == 0).
Checks: forward image finite, EVERY gradient finite, collab == scalar, and grad
magnitudes stay sane (the old bug produced x100-per-contributor blowups -> inf).

  python stress_frontmost_nearopaque.py run {on|off} {collab|scalar}
  python stress_frontmost_nearopaque.py compare
"""
import sys, os, math
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')
# run from repo root

from scripts.test_frontmost_collab import build_camera, quat, YAML  # reuse scene machinery

DEV = 'cuda'
OUT_DIR = '/tmp/claude-1000/fm_nearopaque_check'


def build_args():
    return Namespace(
        method='res_switch', is_gestex=True, hybrid_levels=2, disable_c2f=True,
        freeze_mlp=False, hash_lr_scale=1.0, res_lr_scale=1.0, ste=False, lru=0.01,
        adaptive_cat_inference=False, adaptive_gate_inference=False,
        adaptive_zero_inference=False, params=None, freeze_mlp_from=None,
        feature='sh', kernel='beta_scaled', kernel2=None, activation_bias=[0.5, 0.0],
        sh_degree=3,
    )


def make_model(N=12):
    """Harden-regime scene: near-opaque (alpha clamps at 0.99) tilted overlapping
    flat-top beta_scaled surfels with interleaved depths, plus one ISOLATED surfel
    (pixels where F is the only contributor -> pass 2 blends nothing)."""
    from scene.gaussian_model import GaussianModel
    g = GaussianModel(3)
    g.feature_mode = 'sh'
    g.kernel_type = 'beta_scaled'
    g.kernel_type2 = None
    g.is_gestex = True
    g.active_sh_degree = 3
    g.max_sh_degree = 3
    M = N - 1  # overlapping cluster; last surfel isolated
    xs = torch.linspace(-0.30, 0.30, M, device=DEV)
    # interleaved depths so center-depth order != intersection order
    zs = torch.tensor([0.00, -0.06, 0.05, -0.03, 0.07, -0.08,
                       0.04, -0.05, 0.06, -0.02, 0.03][:M], device=DEV)
    xyz = torch.stack([xs, 0.1 * torch.sin(xs * 7), zs], dim=1)
    # isolated surfel far to the side (still on screen)
    iso = torch.tensor([[0.9, 0.9, 0.0]], device=DEV)
    xyz = torch.cat([xyz, iso], dim=0)
    g._xyz = torch.nn.Parameter(xyz.contiguous())
    g._scaling = torch.nn.Parameter(torch.full((N, 2), math.log(0.25), device=DEV))
    base_rots = [quat([0, 1, 0], 0.8), quat([0, 1, 0], -0.9), quat([1, 0, 0], 0.7),
                 quat([1, 1, 0], -0.8), quat([0, 1, 1], 0.85), quat([1, 0, 1], -0.75)]
    rots = torch.tensor((base_rots * 2)[:N], device=DEV, dtype=torch.float32)
    g._rotation = torch.nn.Parameter(rots.contiguous())
    # sigmoid(8) ~ 0.99966 -> alpha hits the 0.99 clamp over the flat top
    g._opacity = torch.nn.Parameter(torch.full((N, 1), 8.0, device=DEV))
    # beta_scaled shape: sigmoid(s)*4+0.001 ~ 0.10 -> flat-top disc (dG pole regime)
    g._shape = torch.nn.Parameter(torch.full((N, 1), -3.67, device=DEV))
    fdc = (0.2 + 0.4 * torch.rand((N, 1, 3), device=DEV))
    frest = 0.05 * torch.randn((N, (3 + 1) ** 2 - 1, 3), device=DEV)
    g._features_dc = torch.nn.Parameter(fdc.contiguous())
    g._features_rest = torch.nn.Parameter(frest.contiguous())
    g._appearance_level = torch.nn.Parameter(24.0 * torch.ones((N, 1), device=DEV),
                                             requires_grad=False)
    g._is_textured = torch.ones(N, dtype=torch.bool, device=DEV)
    g.max_radii2D = torch.zeros(N, device=DEV)
    return g


def run_leg(promo, path):
    assert path in ('collab', 'scalar')
    if path == 'scalar':
        os.environ['DISABLE_COLLABORATIVE_GEMM'] = '1'
    torch.manual_seed(0)

    from hash_encoder.config import Config
    from hash_encoder.modules import INGP
    from gaussian_renderer import render
    import diff_surfel_3D_sh_res_harden as HD

    cfg = Config(YAML)
    args = build_args()
    ingp = INGP(cfg, args=args).to(DEV)
    ingp.set_active_levels(current_iter=12000)
    HD.set_residual_mode(2)
    ingp.is_mixed_deferred_relu_mode = True
    HD.set_activation_bias(0.5, 0.0)
    HD.set_lru_slope(0.01)
    ingp.lru_slope = 0.01
    # NOTE: the renderer maps ingp.first_int_sort -> set_tile_depth_sort (the SORT)
    # and ingp.frontmost_on -> set_frontmost_first (the 2-pass PROMOTION). The
    # promotion is what we are stress-testing here.
    ingp.first_int_sort = False
    ingp.frontmost_on = bool(promo)
    ingp.first_int_tilekey = False

    cam = build_camera()
    pipe = Namespace(debug=False, skip_aux_normal_dist=True, compute_cov3D_python=False,
                     convert_SHs_python=False, depth_ratio=0.0)
    g = make_model()
    bg = torch.zeros(3, device=DEV)

    pkg = render(cam, g, pipe, bg, ingp=ingp, iteration=12000, cfg=cfg,
                 lowpass=True, is_training=True)
    img = pkg['render']
    assert torch.isfinite(img).all(), "FORWARD IMAGE NOT FINITE"
    # exercise the aux/mask path too if present
    loss = (img * (0.3 + 0.7 * torch.sin(
        torch.arange(img.numel(), device=DEV, dtype=torch.float32) * 0.37
    ).abs().view_as(img))).sum()
    if pkg.get('rend_alpha') is not None:
        loss = loss + 0.1 * pkg['rend_alpha'].sum()
    loss.backward()
    torch.cuda.synchronize()

    grads = {'img': img.detach().cpu()}
    for name, p in [('xyz', g._xyz), ('opacity', g._opacity), ('scaling', g._scaling),
                    ('rotation', g._rotation), ('f_dc', g._features_dc),
                    ('f_rest', g._features_rest), ('shape', g._shape)]:
        if p.grad is not None:
            grads[name] = p.grad.detach().cpu()
    for name, p in ingp.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            grads[f'ingp.{name}'] = p.grad.detach().cpu()
    mg = HD.get_mlp_grads()
    if mg is not None:
        for i, t in enumerate(mg):
            grads[f'mlp_W{i+1}'] = t.detach().cpu()

    bad = [k for k, v in grads.items() if not torch.isfinite(v).all()]
    huge = [(k, v.abs().max().item()) for k, v in grads.items()
            if k != 'img' and v.abs().max().item() > 1e7]
    os.makedirs(OUT_DIR, exist_ok=True)
    f = f'{OUT_DIR}/{("on" if promo else "off")}_{path}.pt'
    torch.save(grads, f)
    print(f"saved {f}")
    print("  NONFINITE:", bad if bad else "none")
    print("  >1e7 grads:", huge if huge else "none")
    print("  " + ", ".join(f"{k}={v.norm():.3e}" for k, v in grads.items() if k != 'img'))
    if bad:
        sys.exit(2)


def compare():
    ok_all = True
    for promo in ('off', 'on'):
        a = torch.load(f'{OUT_DIR}/{promo}_collab.pt')
        b = torch.load(f'{OUT_DIR}/{promo}_scalar.pt')
        print(f"\n=== promotion {promo.upper()}: collab vs scalar ===")
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                print(f"  {k:26s} MISSING in one leg"); ok_all = False; continue
            d = (a[k] - b[k]).abs().max().item()
            n = max(a[k].abs().max().item(), b[k].abs().max().item(), 1e-12)
            rel = d / n
            flag = 'OK' if rel < 1e-3 else ('warn' if rel < 1e-2 else 'MISMATCH')
            if flag == 'MISMATCH':
                ok_all = False
            print(f"  {k:26s} max|d|={d:.3e} rel={rel:.3e} [{flag}]")
    i_on = torch.load(f'{OUT_DIR}/on_collab.pt')['img']
    i_off = torch.load(f'{OUT_DIR}/off_collab.pt')['img']
    print(f"\nforward max|ON-OFF| = {(i_on - i_off).abs().max().item():.5f} (want > 0)")
    print("\nRESULT:", "PASS — near-opaque regime finite + collab==scalar"
          if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'compare'
    if mode == 'run':
        run_leg(promo=(sys.argv[2] == 'on'), path=sys.argv[3])
    elif mode == 'compare':
        compare()

def _build_leg(promo):
    torch.manual_seed(0)
    from hash_encoder.config import Config
    from hash_encoder.modules import INGP
    from gaussian_renderer import render
    import diff_surfel_3D_sh_res_harden as HD
    cfg = Config(YAML)
    args = build_args()
    ingp = INGP(cfg, args=args).to(DEV)
    ingp.set_active_levels(current_iter=12000)
    HD.set_residual_mode(2)
    ingp.is_mixed_deferred_relu_mode = True
    HD.set_activation_bias(0.5, 0.0)
    HD.set_lru_slope(0.01)
    ingp.lru_slope = 0.01
    ingp.first_int_sort = False
    ingp.frontmost_on = bool(promo)
    ingp.first_int_tilekey = False
    cam = build_camera()
    pipe = Namespace(debug=False, skip_aux_normal_dist=True, compute_cov3D_python=False,
                     convert_SHs_python=False, depth_ratio=0.0)
    g = make_model()
    bg = torch.zeros(3, device=DEV)

    def fwd_loss():
        pkg = render(cam, g, pipe, bg, ingp=ingp, iteration=12000, cfg=cfg,
                     lowpass=True, is_training=True)
        img = pkg['render']
        w = (0.3 + 0.7 * torch.sin(torch.arange(img.numel(), device=DEV,
                                                dtype=torch.float32) * 0.37).abs()).view_as(img)
        return (img * w).sum()
    return g, fwd_loss


def gradcheck(promo):
    """Central-difference check of promotion gradients through the REAL pipeline.
    f_dc (color chain incl. accum_rec folds) + xyz (geometry/T chain)."""
    g, fwd_loss = _build_leg(promo)
    loss = fwd_loss()
    loss.backward()
    torch.cuda.synchronize()
    checks = [('f_dc', g._features_dc, 1e-3), ('xyz', g._xyz, 2e-4)]
    print(f"\n=== FD gradcheck (promotion {'ON' if promo else 'OFF'}) ===")
    worst = 0.0
    for name, p, eps in checks:
        an = p.grad.detach().clone().flatten()
        flat = p.data.flatten()
        idxs = torch.linspace(0, flat.numel() - 1, steps=min(24, flat.numel())).long()
        rows = []
        for i in idxs.tolist():
            orig = flat[i].item()
            with torch.no_grad():
                flat[i] = orig + eps; lp = fwd_loss().item()
                flat[i] = orig - eps; lm = fwd_loss().item()
                flat[i] = orig
            fd = (lp - lm) / (2 * eps)
            a = an[i].item()
            denom = max(abs(a), abs(fd), 1e-3)
            rows.append(abs(a - fd) / denom)
        r = max(rows)
        worst = max(worst, r)
        print(f"  {name:6s} worst rel err over {len(idxs)} probes: {r:.3e}")
    print("  RESULT:", "PASS" if worst < 0.05 else "FAIL (check FD noise at splat edges)")


if __name__ == '__main__' and sys.argv[1:2] == ['gradcheck']:
    gradcheck(promo=(sys.argv[2] == 'on'))

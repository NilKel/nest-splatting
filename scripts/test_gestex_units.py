"""
Unit tests for `--method GEStex` joint-stage rasterizer (diff_surfel_gestex).

Isolates the fw+bw pass on a TINY hand-built set of surfels (no Scene/dataset), for
three configurations, to pinpoint which one triggers the joint-stage backward crash:

  1. UNTEXTURED-only : a few 3D-EWA "Gaussian" surfels (is_textured=False, scaling_z)
  2. TEXTURED-only   : a few flat 2D surfels WITH an RGB atlas (is_textured=True)
  3. MIXED           : both halves together

Each case runs render() -> loss -> backward and reports success / grad norms. For the
textured case it also finite-difference-checks the atlas gradient.

Run:  conda run -n nest_splatting python scripts/test_gestex_units.py
"""
import sys, math, traceback
from argparse import Namespace
import numpy as np
import torch

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')
from hash_encoder.config import Config
from hash_encoder.modules import INGP
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

torch.manual_seed(0)
DEV = 'cuda'
YAML = './configs/himalaya_2d.yaml'
R_ATLAS = 8


def build_args():
    return Namespace(
        method='res_switch', is_gestex=True, hybrid_levels=2, disable_c2f=True,
        freeze_mlp=False, hash_lr_scale=1.0, res_lr_scale=1.0, ste=False, lru=0.01,
        adaptive_cat_inference=False, adaptive_gate_inference=False,
        adaptive_zero_inference=False, params=None, freeze_mlp_from=None,
        feature='sh', kernel='gaussian', kernel2=None, activation_bias=[0.5, 0.0],
        sh_degree=3,
    )


def build_camera(W=96, H=96):
    fovx = fovy = 0.9
    R = np.eye(3)
    T = np.array([0.0, 0.0, 4.0])            # camera 4 units back → surfels at origin in front
    znear, zfar = 0.01, 100.0
    Rt = getWorld2View2(R, T)
    wvt = torch.tensor(Rt, dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    cc = wvt.inverse()[3, :3]
    return Namespace(FoVx=fovx, FoVy=fovy, image_width=W, image_height=H,
                     world_view_transform=wvt, full_proj_transform=fpt,
                     camera_center=cc, znear=znear, zfar=zfar, clip_plane=None)


def make_model(n_tex, n_untex):
    """Build a tiny GaussianModel: first n_tex textured surfels + n_untex untextured."""
    N = n_tex + n_untex
    g = GaussianModel(3)
    g.feature_mode = 'sh'
    g.kernel_type = 'gaussian'
    g.kernel_type2 = None
    g.is_gestex = True
    g.surfel_opac = 255.0
    g.ges_atlas_res = R_ATLAS
    g.active_sh_degree = 3
    g.max_sh_degree = 3

    # positions on a small grid in the z=0 plane, jittered
    xs = torch.linspace(-0.3, 0.3, N, device=DEV)
    xyz = torch.stack([xs, 0.1 * torch.sin(xs * 7), torch.zeros_like(xs)], dim=1)
    g._xyz = torch.nn.Parameter(xyz.contiguous())
    # 2D log-scales (fairly large so they cover pixels)
    g._scaling = torch.nn.Parameter(torch.full((N, 2), math.log(0.15), device=DEV))
    # 3rd axis: untextured get a real (isotropic) axis; textured a flat placeholder
    sz = torch.full((N, 1), math.log(0.15), device=DEV)
    sz[:n_tex] = math.log(0.05) + math.log(0.15)   # flat placeholder for surfels
    g._scaling_z = torch.nn.Parameter(sz.contiguous())
    rot = torch.zeros((N, 4), device=DEV); rot[:, 0] = 1.0
    g._rotation = torch.nn.Parameter(rot.contiguous())
    g._opacity = torch.nn.Parameter(torch.full((N, 1), 2.0, device=DEV))  # sigmoid(2)~0.88
    # SH: DC = mid-gray-ish, rest = 0
    fdc = 0.3 * torch.ones((N, 1, 3), device=DEV)
    frest = torch.zeros((N, (3 + 1) ** 2 - 1, 3), device=DEV)
    g._features_dc = torch.nn.Parameter(fdc.contiguous())
    g._features_rest = torch.nn.Parameter(frest.contiguous())
    g._appearance_level = torch.nn.Parameter(24.0 * torch.ones((N, 1), device=DEV), requires_grad=False)
    is_tex = torch.zeros(N, dtype=torch.bool, device=DEV)
    is_tex[:n_tex] = True
    g._is_textured = is_tex
    g.max_radii2D = torch.zeros(N, device=DEV)
    # RGB atlas (unbounded) — random small values so grads are non-degenerate.
    # GES_ATLAS_FILL: diagnostic — set a large constant so the forward image visibly
    # reflects the atlas (confirms the forward atlas path is actually taken).
    import os as _os
    _fill = _os.environ.get('GES_ATLAS_FILL')
    if _fill is not None:
        g._tex_atlas = torch.nn.Parameter(float(_fill) * torch.ones((N, R_ATLAS, R_ATLAS, 3), device=DEV))
    else:
        g._tex_atlas = torch.nn.Parameter(0.1 * torch.randn((N, R_ATLAS, R_ATLAS, 3), device=DEV))
    return g


def run_case(name, ingp, cam, pipe, cfg, n_tex, n_untex, use_atlas):
    print(f"\n{'='*72}\n[CASE] {name}: n_tex={n_tex}, n_untex={n_untex}, atlas={use_atlas}\n{'='*72}")
    g = make_model(n_tex, n_untex)
    bg = torch.zeros(3, device=DEV)
    try:
        pkg = render(cam, g, pipe, bg, ingp=ingp, iteration=25000, cfg=cfg,
                     lowpass=True, is_training=True)
        img = pkg['render']
        print(f"  forward OK: image {tuple(img.shape)} range=[{img.min().item():.4f},{img.max().item():.4f}] "
              f"mean={img.mean().item():.4f}")
        loss = img.mean()
        loss.backward()
        torch.cuda.synchronize()
        print("  backward OK.")
        # grads
        for nm, p in [('xyz', g._xyz), ('opacity', g._opacity), ('scaling', g._scaling),
                      ('scaling_z', g._scaling_z), ('rotation', g._rotation),
                      ('f_dc', g._features_dc)]:
            gr = p.grad
            print(f"    grad[{nm:9s}] {'None' if gr is None else f'norm={gr.norm().item():.3e}'}")
        # atlas grad (device-global buffer stashed on the model by the renderer)
        ag = getattr(g, '_ges_atlas_grad', None)
        if ag is not None:
            nz = int((ag.abs() > 0).sum().item())
            print(f"    atlas grad: norm={ag.norm().item():.3e}, nonzero_texels={nz}/{ag.numel()}")
        return True, g
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, g


def atlas_gradcheck(ingp, cam, pipe, cfg, n_tex=3):
    """Finite-difference check of the atlas gradient (textured-only case)."""
    print(f"\n{'='*72}\n[GRADCHECK] atlas bilinear backward (n_tex={n_tex})\n{'='*72}")
    g = make_model(n_tex, 0)
    bg = torch.zeros(3, device=DEV)
    pkg = render(cam, g, pipe, bg, ingp=ingp, iteration=25000, cfg=cfg, lowpass=True, is_training=True)
    loss = pkg['render'].mean()
    loss.backward()
    torch.cuda.synchronize()
    ag = getattr(g, '_ges_atlas_grad', None)
    if ag is None:
        print("  no atlas grad buffer — SKIP"); return
    # pick a few texels with nonzero analytic grad and FD-check them
    eps = 1e-3
    flat = ag.reshape(-1)
    idxs = torch.topk(flat.abs(), min(6, flat.numel())).indices.tolist()
    print(f"  checking {len(idxs)} highest-grad texels (eps={eps}):")
    ok = 0
    for fi in idxs:
        analytic = flat[fi].item()
        with torch.no_grad():
            base = g._tex_atlas.data.reshape(-1)
            orig = base[fi].item()
            base[fi] = orig + eps
            lp = render(cam, g, pipe, bg, ingp=ingp, iteration=25000, cfg=cfg, lowpass=True, is_training=False)['render'].mean().item()
            base[fi] = orig - eps
            lm = render(cam, g, pipe, bg, ingp=ingp, iteration=25000, cfg=cfg, lowpass=True, is_training=False)['render'].mean().item()
            base[fi] = orig
        fd = (lp - lm) / (2 * eps)
        rel = abs(analytic - fd) / (abs(fd) + 1e-8)
        flag = 'OK' if rel < 0.05 else 'MISMATCH'
        if rel < 0.05:
            ok += 1
        print(f"    texel[{fi:6d}] analytic={analytic:+.5e} fd={fd:+.5e} rel={rel:.3e} [{flag}]")
    print(f"  gradcheck: {ok}/{len(idxs)} within 5% rel tol")


def main():
    cfg = Config(YAML)
    args = build_args()
    ingp = INGP(cfg, args=args).to(DEV)
    ingp.set_active_levels(current_iter=25000)
    ingp.is_gestex_joint = True
    ingp.is_mixed_deferred_relu_mode = True   # mode-2 post-blend LRU (joint stage)

    # CRITICAL: upload MLP weights + mode/bias/lru to the gestex module's device globals
    # (mirrors the joint-transition fix in train.py). Without this, gestex's d_mlp_W1..3
    # are NULL → the backward null-reads them → illegal access.
    import diff_surfel_gestex as _dg
    _mw = ingp.get_fused_mlp_weights()
    if _mw is not None:
        _dg.set_mlp_weights(_mw[0].contiguous(), _mw[1].contiguous(), _mw[2].contiguous())
    _dg.set_residual_mode(2)
    _dg.set_activation_bias(0.5, 0.0)
    _dg.set_lru_slope(0.01)
    cam = build_camera()
    pipe = Namespace(debug=False, skip_aux_normal_dist=True, compute_cov3D_python=False,
                     convert_SHs_python=False, depth_ratio=0.0)

    # One case per process — a single CUDA illegal-access poisons the context, so
    # subsequent cases can't run in the same process. Select via argv[1].
    case = sys.argv[1] if len(sys.argv) > 1 else 'textured'
    if case == 'textured':
        run_case('TEXTURED-only (atlas per GESTEX_NOATLAS env)', ingp, cam, pipe, cfg, 4, 0, True)
    elif case == 'untextured':
        run_case('UNTEXTURED-only (EWA)', ingp, cam, pipe, cfg, 0, 4, False)
    elif case == 'mixed':
        run_case('MIXED (tex + untex)', ingp, cam, pipe, cfg, 2, 2, True)
    elif case == 'gradcheck':
        atlas_gradcheck(ingp, cam, pipe, cfg, n_tex=3)
    else:
        print(f"unknown case '{case}'")


if __name__ == "__main__":
    main()

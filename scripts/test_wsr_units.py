"""FD gradcheck for --wsr (diff_surfel_3D_sh_res_probe_wsr, sort-free WSR composite).

Three phases, real render() pipeline on a tiny synthetic proberes scene:

  0. SORTED EQUIVALENCE: the wsr clone with ingp.wsr_sorted=True must render
     byte-identically to the base probe module (set_wsr(0) path untouched).
  1. Analytic sanity under WSR: occ / SV / opacity / probe-head / texture-field
     grads all nonzero and finite.
  2. Central-FD vs analytic for: _wsr_occ (the new occ chain incl. the manual
     sigmoid grad hand-off), opacity (coverage + weight terms), SV (f_dc),
     probe-head bias (dL/dprobes under WSR weights), texture-field embeddings
     (dL/dtex under WSR weights), xyz + rot (dL/duv chain, head detached),
     beta shape.

Same kink-bracketing caveats as test_proberes_units.py.

  conda run -n nest_splatting python scripts/test_wsr_units.py
"""
import sys, os, math
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

DEV = 'cuda'
YAML = './configs/himalaya.yaml'
EPS_PARAM = 1.0 / 64.0
EPS_HEAD = float(os.environ.get("WSR_EPS_HEAD", 1.0 / 256.0))
EPS_GEOM = float(os.environ.get("WSR_EPS_GEOM", 1.0 / 512.0))
EPS_OCC = float(os.environ.get("WSR_EPS_OCC", 1.0 / 128.0))


def build_args(wsr=True):
    return Namespace(
        method='proberes', hybrid_levels=2, disable_c2f=True,
        freeze_mlp=False, hash_lr_scale=1.0, res_lr_scale=1.0, ste=False, lru=0.0,
        adaptive_cat_inference=False, adaptive_gate_inference=False,
        adaptive_zero_inference=False, params=None, freeze_mlp_from=None,
        feature='sh', kernel='gaussian', kernel2=None, activation_bias=[0.5, 0.0],
        sh_degree=3,
        probe_tex_res=64, probe_patch_px=12.0, probe_c2f_interval=0,
        wsr=wsr,
        # WSR_COMPOSITE=1: gradcheck the ht=1-style operator (wsr_mode 2,
        # exact frontmost + occ-weighted tail) instead of pure WSR.
        wsr_composite=os.environ.get('WSR_COMPOSITE') == '1',
        # WSR_GATE=<tau>: arm the 2-pass transmittance gate. On this tiny
        # 6-surfel scene the depths span <1 bin unless tau is aggressive;
        # FD flips at the gate boundary are an operator discontinuity, not a
        # VJP error (same caveat as the mode-2 depth-argmin).
        wsr_gate_tau=float(os.environ.get('WSR_GATE', '0') or 0.0),
        # WSR_DGATE=<margin>: arm the mean-depth gate instead (?wsr=3).
        wsr_dgate_margin=float(os.environ.get('WSR_DGATE', '0') or 0.0),
    )


def build_camera(W=96, H=96):
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    fovx = fovy = 0.9
    R = np.eye(3)
    T = np.array([0.0, 0.0, 4.0])
    znear, zfar = 0.01, 100.0
    Rt = getWorld2View2(R, T)
    wvt = torch.tensor(Rt, dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    cc = wvt.inverse()[3, :3]
    return Namespace(FoVx=fovx, FoVy=fovy, image_width=W, image_height=H,
                     world_view_transform=wvt, full_proj_transform=fpt,
                     camera_center=cc, znear=znear, zfar=zfar, clip_plane=None)


def quat(ax, ang):
    s = math.sin(ang / 2)
    n = math.sqrt(sum(a * a for a in ax))
    return [math.cos(ang / 2)] + [a / n * s for a in ax]


def make_model(N=6):
    from scene.gaussian_model import GaussianModel
    g = GaussianModel(3)
    g.feature_mode = 'sh'
    g.kernel_type = os.environ.get('WSR_TEST_KERNEL', 'gaussian')
    g.kernel_type2 = None
    g.active_sh_degree = 3
    g.max_sh_degree = 3
    xs = torch.linspace(-0.25, 0.25, N, device=DEV)
    zs = torch.tensor([0.00, -0.06, 0.05, -0.03, 0.07, -0.08][:N], device=DEV)
    if os.environ.get('WSR_GATE') or os.environ.get('WSR_DGATE'):
        # Two depth groups (~1 log-bin apart: depth ≈4 vs ≈5) so the gate
        # actually fires: the front trio saturates central pixels and the
        # back trio gets discarded there. Perturbing FRONT geometry/opacity
        # moves tbin → back-fragment gate flips → expected FD flags.
        zs = torch.tensor([0.00, -0.06, 0.05, -1.00, -0.95, -1.05][:N], device=DEV)
    xyz = torch.stack([xs, 0.1 * torch.sin(xs * 7), zs], dim=1)
    g._xyz = torch.nn.Parameter(xyz.contiguous())
    g._scaling = torch.nn.Parameter(torch.full((N, 2), math.log(0.22), device=DEV))
    rots = torch.tensor([
        quat([0, 1, 0], 0.8), quat([0, 1, 0], -0.9), quat([1, 0, 0], 0.7),
        quat([1, 1, 0], -0.8), quat([0, 1, 1], 0.85), quat([1, 0, 1], -0.75),
    ][:N], device=DEV, dtype=torch.float32)
    g._rotation = torch.nn.Parameter(rots.contiguous())
    g._opacity = torch.nn.Parameter(torch.full((N, 1), 1.2, device=DEV))
    fdc = (0.2 + 0.4 * torch.rand((N, 1, 3), device=DEV))
    frest = 0.05 * torch.randn((N, (3 + 1) ** 2 - 1, 3), device=DEV)
    g._features_dc = torch.nn.Parameter(fdc.contiguous())
    g._features_rest = torch.nn.Parameter(frest.contiguous())
    g._appearance_level = torch.nn.Parameter(24.0 * torch.ones((N, 1), device=DEV),
                                             requires_grad=False)
    g.max_radii2D = torch.zeros(N, device=DEV)
    if g.kernel_type in ('beta', 'beta_scaled'):
        g._shape = torch.nn.Parameter(torch.zeros((N, 1), device=DEV))
    # WSR occlusion logits — deliberately varied, none saturated.
    occ0 = torch.tensor([0.85, 0.35, 0.6, 0.5, 0.75, 0.45][:N], device=DEV)
    g._wsr_occ = torch.nn.Parameter(torch.log(occ0 / (1 - occ0)).view(N, 1).contiguous())
    return g


def _config_setters(mod):
    mod.set_residual_mode(0)
    mod.set_activation_bias(0.5, 0.0)
    mod.set_lru_slope(0.0)
    mod.set_contrib_thresh(0.0)
    mod.set_count_thresh(0)
    mod.set_opacity_thresh(0.0)
    mod.set_dropout(0.0, 0)


def setup():
    torch.manual_seed(0)
    from hash_encoder.config import Config
    from hash_encoder.modules import INGP
    from gaussian_renderer import render
    import diff_surfel_3D_sh_res_probe_wsr as PM
    import diff_surfel_3D_sh_res_probe as PM_BASE

    cfg = Config(YAML)
    args = build_args(wsr=True)
    ingp = INGP(cfg, args=args).to(DEV)
    assert ingp.probe_head is not None and ingp.probe_field is not None
    assert ingp.is_wsr_mode, "INGP did not pick up args.wsr"
    ingp.set_active_levels(current_iter=12000)
    ingp.hashgrid_disabled = False

    with torch.no_grad():
        for m in ingp.probe_field.mlp:
            if isinstance(m, torch.nn.Linear):
                m.weight.copy_(torch.round(torch.randn_like(m.weight) * 16.0) / 64.0)
                if m.bias is not None:
                    m.bias.copy_(torch.round(torch.randn_like(m.bias) * 8.0) / 64.0)
        ingp.probe_field.enc.embeddings.copy_(
            torch.round(torch.randn_like(ingp.probe_field.enc.embeddings) * 16.0) / 64.0)
        ingp.probe_head.mlp[-1].weight.copy_(
            torch.round(torch.randn_like(ingp.probe_head.mlp[-1].weight) * 4.0) / 256.0)
        ingp.probe_head.mlp[-1].bias.copy_(
            torch.tensor([0.05, -0.04, 0.30, 0.10], device=DEV))

    _config_setters(PM)
    _config_setters(PM_BASE)

    cam = build_camera()
    pipe = Namespace(debug=False, skip_aux_normal_dist=True, compute_cov3D_python=False,
                     convert_SHs_python=False, depth_ratio=0.0)
    g = make_model()
    bg = torch.zeros(3, device=DEV)

    def fwd_loss(fd=False):
        pkg = render(cam, g, pipe, bg, ingp=ingp, iteration=12000, cfg=cfg,
                     lowpass=True, is_training=True)
        img = pkg['render']
        w = torch.arange(img.numel(), device=DEV, dtype=torch.float32)
        w = (0.3 + 0.7 * torch.sin(w * 0.37).abs()).view_as(img)
        if fd:
            return (img.double() * w.double()).sum(), img
        return (img * w).sum(), img

    return g, ingp, fwd_loss, render, cam, pipe, bg, cfg, PM


def fd_check(label, param, index, analytic, fwd_loss, eps, l0=None):
    with torch.no_grad():
        base = param.data[index].item()
        if l0 is None:
            l0, _ = fwd_loss(fd=True); l0 = l0.item()
        param.data[index] = base + eps
        lp, _ = fwd_loss(fd=True); lp = lp.item()
        param.data[index] = base - eps
        lm, _ = fwd_loss(fd=True); lm = lm.item()
        param.data[index] = base
    fd = (lp - lm) / (2 * eps)
    sp = (lp - l0) / eps
    sm = (l0 - lm) / eps
    denom = max(abs(analytic), abs(fd), 1e-8)
    rel = abs(analytic - fd) / denom
    pad = 0.05 * max(abs(sp), abs(sm), 1e-8)
    kink_ok = (min(sp, sm) - pad) <= analytic <= (max(sp, sm) + pad)
    flag = ('OK' if (rel < 0.03 or abs(analytic - fd) < 3e-4)
            else ('kink' if kink_ok
                  else ('warn' if rel < 0.10 else 'FAIL')))
    print(f"  {label:24s}: analytic={analytic:+.6e} fd={fd:+.6e} rel={rel:.3e} [{flag}]"
          + (f" (one-sided [{min(sp,sm):+.4e}, {max(sp,sm):+.4e}])" if flag in ('kink', 'FAIL') else ""))
    return flag


def backward_with_occ_chain(g, loss):
    """loss.backward() + the manual occ sigmoid chain (mirrors train.py)."""
    loss.backward()
    torch.cuda.synchronize()
    st = getattr(g, '_wsr_render_state', None)
    assert st is not None, "renderer did not arm WSR (no _wsr_render_state)"
    occ_act, occ_grad = st[0], st[1]
    g_raw = (occ_grad * occ_act * (1.0 - occ_act)).view(-1, 1)
    if g._wsr_occ.grad is None:
        g._wsr_occ.grad = g_raw.clone()
    else:
        g._wsr_occ.grad += g_raw
    g._wsr_render_state = None


def main():
    g, ingp, fwd_loss, render, cam, pipe, bg, cfg, PM = setup()

    # ---- Phase 0: sorted equivalence (wsr clone set_wsr(0) vs base module) ----
    with torch.no_grad():
        ingp.wsr_sorted = True
        _, img_sorted_wsrmod = fwd_loss(fd=True)
        ingp.wsr_sorted = False
        ingp.is_wsr_mode = False          # routes to the base probe module
        _, img_base = fwd_loss(fd=True)
        ingp.is_wsr_mode = True
    d = (img_sorted_wsrmod - img_base).abs().max().item()
    print(f"phase 0 sorted-equivalence: max|Δ| = {d:.3e} "
          f"[{'OK' if d < 1e-6 else 'FAIL'}]")
    assert d < 1e-6, "wsr clone in sorted mode diverges from base probe module"

    # ---- Phase 1: analytic backward under WSR ----
    loss, img = fwd_loss()
    backward_with_occ_chain(g, loss)
    assert torch.isfinite(img).all(), "non-finite image"

    head_bias = ingp.probe_head.mlp[-1].bias
    field_emb = ingp.probe_field.enc.embeddings
    assert head_bias.grad is not None, "no grad on probe head bias"
    assert field_emb.grad is not None, "no grad on texture field embeddings"
    assert g._xyz.grad is not None and g._xyz.grad.abs().max() > 0, "no xyz grad"
    assert g._features_dc.grad is not None and g._features_dc.grad.abs().max() > 0, "no SV grad"
    assert g._opacity.grad is not None and g._opacity.grad.abs().max() > 0, "no opacity grad"
    assert g._wsr_occ.grad is not None and g._wsr_occ.grad.abs().max() > 0, \
        "no occ grad (device-global accumulator not flowing)"
    nz_emb = int((field_emb.grad.abs().sum(dim=1) > 0).sum())
    print(f"phase 1 ok: |occ.grad|max={g._wsr_occ.grad.abs().max():.3e}  "
          f"|opa.grad|max={g._opacity.grad.abs().max():.3e}  "
          f"|head_bias.grad|={head_bias.grad.abs().max():.3e}  emb rows={nz_emb}  "
          f"|xyz.grad|max={g._xyz.grad.abs().max():.3e}")

    occ_grad = g._wsr_occ.grad.detach().clone()
    opa_grad = g._opacity.grad.detach().clone()
    fdc_grad = g._features_dc.grad.detach().clone()
    hb_grad = head_bias.grad.detach().clone()
    emb_grad = field_emb.grad.detach().clone()

    fails = 0
    with torch.no_grad():
        l0, _ = fwd_loss(fd=True); l0 = l0.item()

    print("\n=== WSR occlusion logits (device-global grad + sigmoid chain) ===")
    for n in range(g._wsr_occ.shape[0]):
        f = fd_check(f'wsr_occ[{n}]', g._wsr_occ, (n, 0), occ_grad[n, 0].item(), fwd_loss, EPS_OCC, l0=l0)
        fails += (f == 'FAIL')

    print("\n=== opacity (WSR weight + coverage terms) ===")
    for n in [0, 2, 5]:
        f = fd_check(f'opacity[{n}]', g._opacity, (n, 0), opa_grad[n, 0].item(), fwd_loss, EPS_OCC, l0=l0)
        fails += (f == 'FAIL')

    print("\n=== SV base color (f_dc) ===")
    for n, ch in [(0, 0), (3, 1), (5, 2)]:
        f = fd_check(f'f_dc[{n},{ch}]', g._features_dc, (n, 0, ch), fdc_grad[n, 0, ch].item(), fwd_loss, EPS_HEAD, l0=l0)
        fails += (f == 'FAIL')

    print("\n=== probe-head bias (du, dv, theta_res, dlogrho) ===")
    for i, name in enumerate(['du', 'dv', 'theta', 'dlogrho']):
        f = fd_check(f'head_bias.{name}', head_bias, (i,), hb_grad[i].item(), fwd_loss, EPS_HEAD, l0=l0)
        fails += (f == 'FAIL')

    print("\n=== texture-field hash embeddings (dL/dtex under WSR) ===")
    rows = torch.nonzero(emb_grad.abs().sum(dim=1) > 1e-9).flatten()
    sel = rows[torch.linspace(0, len(rows) - 1, steps=min(4, len(rows))).long()]
    for r in sel.tolist():
        f = fd_check(f'field_emb[{r},0]', field_emb, (r, 0), emb_grad[r, 0].item(), fwd_loss, EPS_PARAM, l0=l0)
        fails += (f == 'FAIL')

    # Geometry FD with the head frozen to constants (same rationale as
    # test_proberes_units.py — validates the CUDA dL/duv → dL_ds chain).
    _cached_probes = ingp.probe_head(g.get_xyz, g.get_rotation, g.get_scaling).detach()
    ingp.probe_head.forward = lambda *a, **k: _cached_probes
    for p in [g._xyz, g._rotation, g._features_dc, g._opacity, g._wsr_occ] + \
             ([g._shape] if hasattr(g, '_shape') and isinstance(g._shape, torch.nn.Parameter) else []):
        p.grad = None
    loss2, _ = fwd_loss()
    backward_with_occ_chain(g, loss2)
    xyz_grad = g._xyz.grad.detach().clone()
    rot_grad = g._rotation.grad.detach().clone()
    with torch.no_grad():
        l0d, _ = fwd_loss(fd=True); l0d = l0d.item()

    print("\n=== surfel xyz (rasterizer + dL/duv chain; head detached) ===")
    for n in [0, 2, 4]:
        for dd in range(3):
            f = fd_check(f'xyz[{n},{dd}]', g._xyz, (n, dd), xyz_grad[n, dd].item(), fwd_loss, EPS_GEOM, l0=l0d)
            fails += (f == 'FAIL')

    if g.kernel_type in ('beta', 'beta_scaled'):
        print("\n=== beta shape logits (head detached) ===")
        shape_grad = g._shape.grad.detach().clone()
        for n in [0, 3]:
            f = fd_check(f'shape[{n}]', g._shape, (n, 0), shape_grad[n, 0].item(), fwd_loss, EPS_GEOM, l0=l0d)
            fails += (f == 'FAIL')

    print("\n=== surfel rotation quats (head detached) ===")
    for n, dd in [(0, 1), (1, 2), (3, 3)]:
        f = fd_check(f'rot[{n},{dd}]', g._rotation, (n, dd), rot_grad[n, dd].item(), fwd_loss, EPS_GEOM, l0=l0d)
        fails += (f == 'FAIL')
    del ingp.probe_head.forward

    print(f"\nRESULT: {'PASS' if fails == 0 else f'{fails} FAILURES'}")
    sys.exit(0 if fails == 0 else 1)


if __name__ == '__main__':
    main()

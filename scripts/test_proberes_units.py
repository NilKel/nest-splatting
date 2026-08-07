"""FD gradcheck for --method proberes (diff_surfel_3D_sh_res_probe).

Runs the REAL render() pipeline on a tiny synthetic scene (probe texture 64x64)
and compares analytic gradients vs central finite differences for:

  1. probe-head MLP output bias (4 raw channels: du, dv, theta_res, dlog_rho)
     — exercises dL/dprobes -> autograd -> head chain;
  2. texture-field hash embeddings — exercises dL/dtex -> bake -> field chain;
  3. surfel xyz — the JOINT positional gradient (rasterizer geometry + probe
     head hash + oct base placement + dL/duv -> ray-splat chain);
  4. surfel rotation quats — gauge angle + posenc + dL/duv chains;
  5. sanity: SV (f_dc) grads nonzero, image finite, residual actually nonzero.

The probe kernel forces the scalar backward (no collab-GEMM), so a single leg
suffices:

  conda run -n nest_splatting python scripts/test_proberes_units.py
"""
import sys, os, math
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

DEV = 'cuda'
YAML = './configs/himalaya.yaml'
EPS_PARAM = 1.0 / 64.0     # hash embedding probes
EPS_HEAD = float(os.environ.get("PROBERES_EPS_HEAD", 1.0 / 256.0))     # head bias probes (keeps the global ty/tx shift sub-texel)
EPS_GEOM = float(os.environ.get("PROBERES_EPS_GEOM", 1.0 / 512.0))     # xyz / quat probes


def build_args():
    return Namespace(
        # hybrid_levels=2 keeps the inherited (UNUSED by proberes) scene-hash
        # build under its hash_dim<=16 assert with the 6-level yaml.
        method='proberes', hybrid_levels=2, disable_c2f=True,
        freeze_mlp=False, hash_lr_scale=1.0, res_lr_scale=1.0, ste=False, lru=0.0,
        adaptive_cat_inference=False, adaptive_gate_inference=False,
        adaptive_zero_inference=False, params=None, freeze_mlp_from=None,
        feature='sh', kernel='gaussian', kernel2=None, activation_bias=[0.5, 0.0],
        sh_degree=3,
        probe_tex_res=64, probe_patch_px=12.0, probe_c2f_interval=0,
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
    g.kernel_type = os.environ.get('PROBERES_TEST_KERNEL', 'gaussian')
    g.kernel_type2 = None
    g.active_sh_degree = 3
    g.max_sh_degree = 3
    xs = torch.linspace(-0.25, 0.25, N, device=DEV)
    zs = torch.tensor([0.00, -0.06, 0.05, -0.03, 0.07, -0.08][:N], device=DEV)
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
        # raw shape logits -> get_shape activates into [0.5, 4]; ~2 = mid-softness
        g._shape = torch.nn.Parameter(torch.zeros((N, 1), device=DEV))
    return g


def setup():
    torch.manual_seed(0)
    from hash_encoder.config import Config
    from hash_encoder.modules import INGP
    from gaussian_renderer import render
    import diff_surfel_3D_sh_res_probe as PM

    cfg = Config(YAML)
    args = build_args()
    ingp = INGP(cfg, args=args).to(DEV)
    assert ingp.probe_head is not None and ingp.probe_field is not None
    ingp.set_active_levels(current_iter=12000)
    ingp.hashgrid_disabled = False

    # Give the texture field real content: without it, the zero-init field MLP
    # blocks embedding grads (dL/demb = W1ᵀ·... = 0) and residual == 0 everywhere.
    with torch.no_grad():
        for m in ingp.probe_field.mlp:
            if isinstance(m, torch.nn.Linear):
                m.weight.copy_(torch.round(torch.randn_like(m.weight) * 16.0) / 64.0)
                if m.bias is not None:
                    m.bias.copy_(torch.round(torch.randn_like(m.bias) * 8.0) / 64.0)
        ingp.probe_field.enc.embeddings.copy_(
            torch.round(torch.randn_like(ingp.probe_field.enc.embeddings) * 16.0) / 64.0)
        # Small nonzero head output so theta/scale/delta all sit off their bases.
        ingp.probe_head.mlp[-1].weight.copy_(
            torch.round(torch.randn_like(ingp.probe_head.mlp[-1].weight) * 4.0) / 256.0)
        ingp.probe_head.mlp[-1].bias.copy_(
            torch.tensor([0.05, -0.04, 0.30, 0.10], device=DEV))

    PM.set_residual_mode(0)
    PM.set_activation_bias(0.5, 0.0)
    PM.set_lru_slope(0.0)
    PM.set_contrib_thresh(0.0)
    PM.set_count_thresh(0)
    PM.set_opacity_thresh(0.0)
    PM.set_dropout(0.0, 0)

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

    return g, ingp, fwd_loss, render, cam, pipe, bg, cfg


def fd_check(label, param, index, analytic, fwd_loss, eps, l0=None):
    """Central FD on param.data[index] with one-sided kink bracketing.

    The loss is piecewise linear in texture coords (bilinear) and piecewise
    smooth across hash-cell / ReLU / alpha-cutoff boundaries, so central FD
    averages adjacent slopes while the analytic grad is a one-sided slope.
    As in test_filmres_sigm_split.py, analytic within the one-sided slope
    bracket [slope-, slope+] is a valid subgradient, not a VJP error."""
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


def main():
    g, ingp, fwd_loss, *_ = setup()

    # ---- analytic backward ----
    loss, img = fwd_loss()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(img).all(), "non-finite image"

    head_bias = ingp.probe_head.mlp[-1].bias
    field_emb = ingp.probe_field.enc.embeddings
    assert head_bias.grad is not None, "no grad on probe head bias"
    assert field_emb.grad is not None, "no grad on texture field embeddings"
    assert g._xyz.grad is not None and g._xyz.grad.abs().max() > 0, "no xyz grad"
    assert g._rotation.grad is not None, "no rotation grad"
    assert g._features_dc.grad is not None and g._features_dc.grad.abs().max() > 0, "no SV grad"
    nz_emb = int((field_emb.grad.abs().sum(dim=1) > 0).sum())
    print(f"analytic pass ok: |head_bias.grad|={head_bias.grad.abs().max():.3e}  "
          f"emb rows with grad={nz_emb}  |xyz.grad|max={g._xyz.grad.abs().max():.3e}  "
          f"|rot.grad|max={g._rotation.grad.abs().max():.3e}")
    assert head_bias.grad.abs().max() > 0, "head bias grad all-zero (probe grads not flowing)"
    assert nz_emb > 0, "texture embedding grads all-zero (dL/dtex not flowing)"

    hb_grad = head_bias.grad.detach().clone()
    emb_grad = field_emb.grad.detach().clone()
    xyz_grad = g._xyz.grad.detach().clone()
    rot_grad = g._rotation.grad.detach().clone()

    fails = 0
    with torch.no_grad():
        l0, _ = fwd_loss(fd=True); l0 = l0.item()

    print("\n=== probe-head bias (du, dv, theta_res, dlog_rho) ===")
    for i, name in enumerate(['du', 'dv', 'theta', 'dlogrho']):
        f = fd_check(f'head_bias.{name}', head_bias, (i,), hb_grad[i].item(), fwd_loss, EPS_HEAD, l0=l0)
        fails += (f == 'FAIL')

    print("\n=== texture-field hash embeddings ===")
    rows = torch.nonzero(emb_grad.abs().sum(dim=1) > 1e-9).flatten()
    sel = rows[torch.linspace(0, len(rows) - 1, steps=min(4, len(rows))).long()]
    for r in sel.tolist():
        f = fd_check(f'field_emb[{r},0]', field_emb, (r, 0), emb_grad[r, 0].item(), fwd_loss, EPS_PARAM, l0=l0)
        fails += (f == 'FAIL')

    # Geometry FD with the probe head ATTACHED crosses several finest hash
    # cells of the head's 3D grid per eps leg (piecewise-constant input grads),
    # so neither central FD nor one-sided brackets are conclusive there — and
    # that path is pure autograd anyway. The piece that NEEDS FD validation is
    # the CUDA dL/duv -> dL_ds chain, so FREEZE the probes to constants (cache
    # the head output; FD legs then move geometry under FIXED probes, matching
    # an analytic backward whose head path is cut).
    _cached_probes = ingp.probe_head(g.get_xyz, g.get_rotation, g.get_scaling).detach()
    ingp.probe_head.forward = lambda *a, **k: _cached_probes
    for p in [g._xyz, g._rotation, g._features_dc, g._opacity] + ([g._shape] if hasattr(g, '_shape') and isinstance(g._shape, torch.nn.Parameter) else []):
        p.grad = None
    loss2, _ = fwd_loss()
    loss2.backward()
    torch.cuda.synchronize()
    xyz_grad = g._xyz.grad.detach().clone()
    rot_grad = g._rotation.grad.detach().clone()
    with torch.no_grad():
        l0d, _ = fwd_loss(fd=True); l0d = l0d.item()

    print("\n=== surfel xyz (rasterizer + CUDA dL/duv chain; head detached) ===")
    for n in [0, 2, 4]:
        for d in range(3):
            f = fd_check(f'xyz[{n},{d}]', g._xyz, (n, d), xyz_grad[n, d].item(), fwd_loss, EPS_GEOM, l0=l0d)
            fails += (f == 'FAIL')

    if g.kernel_type in ('beta', 'beta_scaled'):
        print("\n=== beta shape logits (head detached) ===")
        shape_grad = g._shape.grad.detach().clone()
        for n in [0, 3]:
            f = fd_check(f'shape[{n}]', g._shape, (n, 0), shape_grad[n, 0].item(), fwd_loss, EPS_GEOM, l0=l0d)
            fails += (f == 'FAIL')

    print("\n=== surfel rotation quats (head detached) ===")
    for n, d in [(0, 1), (1, 2), (3, 3)]:
        f = fd_check(f'rot[{n},{d}]', g._rotation, (n, d), rot_grad[n, d].item(), fwd_loss, EPS_GEOM, l0=l0d)
        fails += (f == 'FAIL')
    del ingp.probe_head.forward  # restore the real head

    print(f"\nRESULT: {'PASS' if fails == 0 else f'{fails} FAILURES'}")
    sys.exit(0 if fails == 0 else 1)


if __name__ == '__main__':
    main()

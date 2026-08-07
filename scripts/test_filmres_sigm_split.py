"""FD gradcheck + collab-vs-scalar equivalence for --film_act gamma_sigm_split (mode 7)
in diff_surfel_3D_sh_filmres.

Mode 7: mlp_input[i] = sigmoid(gamma_l)*hash[i] + beta[i], one gamma per hash LEVEL
(l = i // l_dim). gamma_0 = _film_params col 0; gamma_1..3 = _film_params cols 22..24
(film_beta cols 21..23). Their grads flow through the existing dL_dfilm_gamma /
dL_dfilm_beta outputs.

Verifies, through the REAL render() filmres mode-5 pipeline on a tiny synthetic scene:
  1. analytic d(loss)/d(_film_params[n, col]) vs central finite differences for
     cols {0, 22, 23, 24} (the four gamma levels) and a spread of beta cols;
  2. all four gamma levels have NONZERO gradient (the known frozen-gamma symptom);
  3. collab-GEMM backward == scalar backward (DISABLE_COLLABORATIVE_GEMM=1 leg).

DISABLE_COLLABORATIVE_GEMM is read once per process, so each leg runs in its own process:

  python scripts/test_filmres_sigm_split.py run collab
  python scripts/test_filmres_sigm_split.py run scalar
  python scripts/test_filmres_sigm_split.py compare
"""
import sys, os, math
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

DEV = 'cuda'
YAML = './configs/himalaya.yaml'  # dim=4 per level -> 4 hash levels x 4D = 16D (canonical l_dim=4)
OUT_DIR = '/tmp/claude-1000/filmres_split_check'
EPS = 0.03125  # exactly representable in FP16 (gamma/beta are staged as half in-kernel);
               # small enough to limit MLP-ReLU kink crossings that bias central FD

# FD probe set: (label, column). gamma_0 = col 0; gamma_1..3 = cols 22..24;
# beta dim i = col 1+i (levels: beta0->l0, beta5->l1, beta10->l2, beta15->l3).
PROBE_COLS = [('gamma_l0', 0), ('gamma_l1', 22), ('gamma_l2', 23), ('gamma_l3', 24),
              ('beta_d0', 1), ('beta_d5', 6), ('beta_d10', 11), ('beta_d15', 16)]
PROBE_GAUSS = [0, 2, 4]


def build_args():
    return Namespace(
        method='3D_SH_filmres', hybrid_levels=2, disable_c2f=True,
        freeze_mlp=False, hash_lr_scale=1.0, res_lr_scale=1.0, ste=False, lru=0.0,
        adaptive_cat_inference=False, adaptive_gate_inference=False,
        adaptive_zero_inference=False, params=None, freeze_mlp_from=None,
        feature='sh', kernel='gaussian', kernel2=None, activation_bias=[0.5, 0.0],
        sh_degree=3, film_act='gamma_sigm_split',
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
    g.kernel_type = 'gaussian'
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
    g._opacity = torch.nn.Parameter(torch.full((N, 1), 1.2, device=DEV))  # sigmoid~0.77
    fdc = (0.2 + 0.4 * torch.rand((N, 1, 3), device=DEV))
    frest = 0.05 * torch.randn((N, (3 + 1) ** 2 - 1, 3), device=DEV)
    g._features_dc = torch.nn.Parameter(fdc.contiguous())
    g._features_rest = torch.nn.Parameter(frest.contiguous())
    g._appearance_level = torch.nn.Parameter(24.0 * torch.ones((N, 1), device=DEV),
                                             requires_grad=False)
    g.max_radii2D = torch.zeros(N, device=DEV)
    # FiLM params [N, 25]. Values chosen as exact multiples of 1/64 (exact in FP16, so
    # base and +/-EPS probes survive the in-kernel half staging without quantization).
    fp = torch.zeros((N, 25), device=DEV)
    grid = torch.arange(N * 25, device=DEV, dtype=torch.float32).view(N, 25)
    fp = (torch.round(torch.sin(grid * 0.7) * 16.0) / 64.0)  # in [-0.25, 0.25], 1/64 grid
    # distinct per-level gammas around 0: col 0 (gamma_0) and cols 22..24 (gamma_1..3)
    for n in range(N):
        fp[n, 0] = (8 + 3 * n % 16) / 64.0
        fp[n, 22] = (-12 + 5 * n % 16) / 64.0
        fp[n, 23] = (4 + 7 * n % 16) / 64.0
        fp[n, 24] = (-6 + 2 * n % 16) / 64.0
    g._film_params = torch.nn.Parameter(fp.contiguous())
    return g


def setup(path):
    assert path in ('collab', 'scalar')
    if path == 'scalar':
        os.environ['DISABLE_COLLABORATIVE_GEMM'] = '1'
    torch.manual_seed(0)

    from hash_encoder.config import Config
    from hash_encoder.modules import INGP
    from gaussian_renderer import render
    import diff_surfel_3D_sh_filmres as FM

    cfg = Config(YAML)
    args = build_args()
    ingp = INGP(cfg, args=args).to(DEV)
    ingp.set_active_levels(current_iter=12000)
    # Bump the hash table so the residual path carries real signal (default init ~1e-4
    # would leave the FD signal in the FP16 noise floor).
    with torch.no_grad():
        for name, p in ingp.named_parameters():
            if 'encoding' in name or 'grid' in name or 'embeddings' in name:
                p.copy_(torch.round(torch.randn_like(p) * 32.0) / 64.0)  # 1/64 grid, std 0.5
    FM.set_residual_mode(0)
    FM.set_activation_bias(0.5, 0.0)
    FM.set_lru_slope(0.0)
    FM.set_film_gamma_act(7)   # gamma_sigm_split under test

    cam = build_camera()
    pipe = Namespace(debug=False, skip_aux_normal_dist=True, compute_cov3D_python=False,
                     convert_SHs_python=False, depth_ratio=0.0)
    g = make_model()
    bg = torch.zeros(3, device=DEV)

    def fwd_loss(fd=False):
        pkg = render(cam, g, pipe, bg, ingp=ingp, iteration=12000, cfg=cfg,
                     lowpass=True, is_training=True)
        img = pkg['render']
        R_w = torch.arange(img.numel(), device=DEV, dtype=torch.float32)
        R_w = (0.3 + 0.7 * torch.sin(R_w * 0.37).abs()).view_as(img)
        if fd:
            # FD legs: accumulate in float64 — the float32 sum of the ~512-magnitude
            # loss quantizes (lp - lm) at ulp≈6e-5, i.e. an FD-grad lattice of ~5e-4.
            return (img.double() * R_w.double()).sum(), img
        return (img * R_w).sum(), img

    return g, ingp, fwd_loss, FM


def run_leg(path):
    g, ingp, fwd_loss, FM = setup(path)

    # ---- analytic backward ----
    loss, img = fwd_loss()
    loss.backward()
    torch.cuda.synchronize()
    assert g._film_params.grad is not None, "no grad on _film_params"
    fp_grad = g._film_params.grad.detach().clone()

    grads = {
        'img': img.detach().cpu(),
        'film_params': fp_grad.cpu(),
        'xyz': g._xyz.grad.detach().cpu(),
        'opacity': g._opacity.grad.detach().cpu(),
        'f_dc': g._features_dc.grad.detach().cpu(),
    }
    for name, p in ingp.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            grads[f'ingp.{name}'] = p.grad.detach().cpu()
    mg = FM.get_mlp_grads()
    if mg is not None:
        for i, t in enumerate(mg):
            grads[f'mlp_W{i+1}'] = t.detach().cpu()

    # ---- central finite differences on _film_params probes ----
    # Also record one-sided slopes: at an MLP-ReLU kink the loss is piecewise linear in
    # the probe, central FD = (slope+ + slope-)/2 while the analytic grad is the slope of
    # one side — analytic within [slope-, slope+] is then correct, not a VJP error.
    fd = {}
    with torch.no_grad():
        base = g._film_params.data.clone()
        l0, _ = fwd_loss(fd=True); l0 = l0.item()
        for label, col in PROBE_COLS:
            for n in PROBE_GAUSS:
                g._film_params.data[n, col] = base[n, col] + EPS
                lp, _ = fwd_loss(fd=True); lp = lp.item()
                g._film_params.data[n, col] = base[n, col] - EPS
                lm, _ = fwd_loss(fd=True); lm = lm.item()
                g._film_params.data[n, col] = base[n, col]
                fd[(label, n, col)] = ((lp - lm) / (2 * EPS),
                                       (lp - l0) / EPS, (l0 - lm) / EPS)
        g._film_params.data.copy_(base)
    torch.cuda.synchronize()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({'grads': grads, 'fd': fd}, f'{OUT_DIR}/{path}.pt')

    # ---- report analytic vs FD for this leg ----
    print(f"\n=== leg {path}: analytic vs central FD (eps={EPS}) ===")
    ok = True
    per_level = {}
    for (label, n, col), (fd_val, sp, sm) in fd.items():
        an = fp_grad[n, col].item()
        denom = max(abs(an), abs(fd_val), 1e-8)
        rel = abs(an - fd_val) / denom
        per_level.setdefault(label, []).append((an, fd_val, rel))
        # abs floor: per-pixel float32 image ulp accumulated over 27k pixels leaves
        # ~1e-4 FD noise in grad units — below that, rel is meaningless.
        # kink bracket: analytic within the one-sided slope interval = valid subgradient.
        pad = 0.05 * max(abs(sp), abs(sm), 1e-8)
        kink_ok = (min(sp, sm) - pad) <= an <= (max(sp, sm) + pad)
        flag = ('OK' if (rel < 0.03 or abs(an - fd_val) < 2e-4)
                else ('kink' if (rel < 0.5 and kink_ok)
                      else ('warn' if rel < 0.10 else 'FAIL')))
        if flag == 'FAIL':
            ok = False
        print(f"  {label:9s} g{n} col{col:2d}: analytic={an:+.6e} fd={fd_val:+.6e} "
              f"rel={rel:.3e} [{flag}]" +
              (f" (one-sided [{min(sp,sm):+.4e}, {max(sp,sm):+.4e}])" if flag == 'kink' else ""))
    # per-gamma-level summary: relerr + cosine + nonzero check
    print("  -- per-probe summary --")
    for label, rows in per_level.items():
        a = torch.tensor([r[0] for r in rows]); f = torch.tensor([r[1] for r in rows])
        cos = torch.nn.functional.cosine_similarity(a, f, dim=0).item()
        relmax = max(r[2] for r in rows)
        nz = a.abs().max().item()
        print(f"  {label:9s}: relmax={relmax:.3e} cos={cos:.6f} max|analytic|={nz:.3e}"
              + ("  <-- ZERO GRAD (frozen!)" if nz < 1e-12 else ""))
        if label.startswith('gamma') and nz < 1e-12:
            ok = False
    # every gamma level must be nonzero over the WHOLE tensor too
    for label, col in PROBE_COLS[:4]:
        full = fp_grad[:, col]
        print(f"  full-tensor {label}: |grad|_max={full.abs().max().item():.3e} "
              f"nonzero_rows={int((full.abs() > 0).sum())}/{full.numel()}")
        if full.abs().max().item() == 0.0:
            ok = False
    print(f"LEG {path}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def compare():
    a = torch.load(f'{OUT_DIR}/collab.pt')
    b = torch.load(f'{OUT_DIR}/scalar.pt')
    ok = True
    print("\n=== collab vs scalar (gamma_sigm_split ON) ===")
    keys = sorted(set(a['grads']) | set(b['grads']))
    worst = 0.0
    for k in keys:
        if k not in a['grads'] or k not in b['grads']:
            print(f"  {k:28s} MISSING in one leg"); ok = False; continue
        d = (a['grads'][k] - b['grads'][k]).abs().max().item()
        n = max(a['grads'][k].abs().max().item(), b['grads'][k].abs().max().item(), 1e-12)
        rel = d / n
        worst = max(worst, rel) if k != 'img' else worst
        flag = 'OK' if rel < 1e-3 else ('warn' if rel < 1e-2 else 'MISMATCH')
        if flag == 'MISMATCH':
            ok = False
        print(f"  {k:28s} max|d|={d:.3e} rel={rel:.3e} [{flag}]")
    print(f"\nworst grad rel diff (excl. img): {worst:.3e}")
    print("RESULT:", "COLLAB == SCALAR — mode-7 collab backward OK" if ok
          else "COLLAB PATH MISMATCH — mode-7 bug in the MODE-5 backward")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'compare'
    if mode == 'run':
        sys.exit(run_leg(sys.argv[2]))
    else:
        compare()

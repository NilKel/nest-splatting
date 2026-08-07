"""Gradient audit for diff_surfel_3D_sh_res_trunc (the --trunc post-blend
truncation barrier), through the REAL render() 3D_SH_res mode-5 pipeline on a
tiny synthetic scene of depth-stacked overlapping surfels.

Checks, per leg (collab / scalar backward — DISABLE_COLLABORATIVE_GEMM=1):
  1. IDENTITY   exit_T=1e-4: trunc module forward+backward bit-identical to the
                base diff_surfel_3D_sh_res (loss, image, alpha, all grads).
  2. FD         exit_T=0.5: analytic d(loss)/d{opacity, xyz, f_dc} vs central
                finite differences of the truncated loss. Loss includes BOTH
                the image term and a rend_alpha term (the channel that carries
                the Python-side noise composite gradient in real training).
                Probes whose +/-eps flips a fragment's inclusion (detected via
                the summed per-pixel contributor count) are reported as
                BOUNDARY — there the loss is genuinely discontinuous in the
                parameter and FD is meaningless (objective property, not a VJP
                bug).
  3. DEAD       a surfel fully behind every truncation front must have exactly
                zero analytic grad on all its parameters.
  4. compare    collab backward == scalar backward at exit_T=0.5.

DISABLE_COLLABORATIVE_GEMM is read once per process → each leg in its own run:
  python scripts/test_trunc_units.py run collab
  python scripts/test_trunc_units.py run scalar
  python scripts/test_trunc_units.py compare
"""
import sys, os, math
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

DEV = 'cuda'
YAML = './configs/himalaya.yaml'
OUT_DIR = '/tmp/claude-1000/trunc_units_check'
EPS = 0.03125          # color probes (can't flip fragment inclusion)
EPS_GEO = 1.0 / 512.0  # opacity/xyz probes: small enough that most probes
                       # don't move a ray across the truncation threshold
EXIT_T = 0.5
N_SURF = 8          # 0..5 stacked at pixel center; 6 offset; 7 = deep "dead" surfel


def build_args():
    return Namespace(
        method='3D_SH_res', hybrid_levels=2, disable_c2f=True,
        freeze_mlp=False, hash_lr_scale=1.0, res_lr_scale=1.0, ste=False, lru=0.0,
        adaptive_cat_inference=False, adaptive_gate_inference=False,
        adaptive_zero_inference=False, params=None, freeze_mlp_from=None,
        feature='sh', kernel='gaussian', kernel2=None, activation_bias=[0.5, 0.0],
        sh_degree=3, trunc=False,
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


def make_model():
    """Depth stack: surfels 0..5 near the optical axis at increasing depth with
    moderate opacities (sigmoid(0.0)=0.5) so T crosses 0.5 at the 1st-2nd
    fragment; surfel 6 offset laterally (partial overlap); surfel 7 deep behind
    everything (z=+1.5, behind 6 half-opacity layers → T << 0.5 at its depth on
    every covered pixel) = the DEAD probe."""
    from scene.gaussian_model import GaussianModel
    g = GaussianModel(3)
    g.feature_mode = 'sh'
    g.kernel_type = 'gaussian'
    g.kernel_type2 = None
    g.active_sh_degree = 3
    g.max_sh_degree = 3
    N = N_SURF
    xyz = torch.tensor([
        [0.00, 0.00, -0.30], [0.03, -0.02, -0.18], [-0.02, 0.03, -0.05],
        [0.02, 0.02, 0.08], [-0.03, -0.01, 0.20], [0.01, -0.03, 0.33],
        [0.30, 0.25, 0.00],
        [0.00, 0.00, 1.50],
    ], device=DEV)
    g._xyz = torch.nn.Parameter(xyz.contiguous())
    # g7 (the DEAD probe) is TINY (scale 0.05) and centered: its whole footprint
    # projects inside the dense middle of the front stack, where T crosses 0.5
    # within the first two fragments on every pixel — so it must never record.
    # (A large g7 reaches front-fringe pixels whose kernel-falloff alphas leave
    # T > 0.5 → it legitimately records there; bitten in test v1.)
    sc = torch.full((N, 2), math.log(0.30), device=DEV)
    sc[N - 1] = math.log(0.05)
    g._scaling = torch.nn.Parameter(sc.contiguous())
    rots = torch.tensor([
        quat([0, 1, 0], 0.35), quat([1, 0, 0], -0.30), quat([0, 1, 0], -0.25),
        quat([1, 1, 0], 0.30), quat([0, 1, 1], -0.35), quat([1, 0, 1], 0.28),
        quat([0, 1, 0], 0.2), quat([1, 0, 0], 0.1),
    ], device=DEV, dtype=torch.float32)
    g._rotation = torch.nn.Parameter(rots.contiguous())
    # logits: mostly 0.0 (alpha 0.5); surfel 2 stronger (0.8 -> ~0.69)
    opac = torch.zeros((N, 1), device=DEV)
    opac[2, 0] = 0.8
    g._opacity = torch.nn.Parameter(opac.contiguous())
    torch.manual_seed(3)
    fdc = (0.2 + 0.5 * torch.rand((N, 1, 3), device=DEV))
    frest = 0.05 * torch.randn((N, (3 + 1) ** 2 - 1, 3), device=DEV)
    g._features_dc = torch.nn.Parameter(fdc.contiguous())
    g._features_rest = torch.nn.Parameter(frest.contiguous())
    g._appearance_level = torch.nn.Parameter(24.0 * torch.ones((N, 1), device=DEV),
                                             requires_grad=False)
    g.max_radii2D = torch.zeros(N, device=DEV)
    return g


def setup(path):
    assert path in ('collab', 'scalar')
    if path == 'scalar':
        os.environ['DISABLE_COLLABORATIVE_GEMM'] = '1'
    torch.manual_seed(0)

    from hash_encoder.config import Config
    from hash_encoder.modules import INGP
    from gaussian_renderer import render
    import diff_surfel_3D_sh_res as BASE
    import diff_surfel_3D_sh_res_trunc as TR

    cfg = Config(YAML)
    args = build_args()
    ingp = INGP(cfg, args=args).to(DEV)
    ingp.set_active_levels(current_iter=12000)
    with torch.no_grad():
        for name, p in ingp.named_parameters():
            if 'encoding' in name or 'grid' in name or 'embeddings' in name:
                p.copy_(torch.round(torch.randn_like(p) * 32.0) / 64.0)
    # Mirror device-global installs into BOTH modules (module-local globals).
    for M in (BASE, TR):
        M.set_residual_mode(0)
        M.set_activation_bias(0.5, 0.0)
        M.set_lru_slope(0.0)
    TR.set_exit_T(1e-4)

    cam = build_camera()
    pipe = Namespace(debug=False, skip_aux_normal_dist=True, compute_cov3D_python=False,
                     convert_SHs_python=False, depth_ratio=0.0)
    g = make_model()
    bg = torch.zeros(3, device=DEV)

    def fwd(fd=False):
        pkg = render(cam, g, pipe, bg, ingp=ingp, iteration=12000, cfg=cfg,
                     lowpass=True, is_training=True)
        img, alp = pkg['render'], pkg['rend_alpha']
        gnum = pkg.get('gaussian_num')
        Rw = torch.arange(img.numel(), device=DEV, dtype=torch.float32)
        Rw = (0.3 + 0.7 * torch.sin(Rw * 0.37).abs()).view_as(img)
        Aw = torch.arange(alp.numel(), device=DEV, dtype=torch.float32)
        Aw = (0.4 + 0.6 * torch.cos(Aw * 0.23).abs()).view_as(alp)
        if fd:
            loss = (img.double() * Rw.double()).sum() + (alp.double() * Aw.double()).sum()
        else:
            loss = (img * Rw).sum() + (alp * Aw).sum()
        ncontrib = int(gnum.sum().item()) if gnum is not None else -1
        gmap = gnum.detach().clone() if gnum is not None else None
        return loss, img, alp, ncontrib, gmap

    return g, ingp, fwd, BASE, TR


def grads_of(g, ingp, mod):
    out = {
        'xyz': g._xyz.grad.detach().cpu(),
        'opacity': g._opacity.grad.detach().cpu(),
        'f_dc': g._features_dc.grad.detach().cpu(),
        'scaling': g._scaling.grad.detach().cpu(),
        'rotation': g._rotation.grad.detach().cpu(),
    }
    for name, p in ingp.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            out[f'ingp.{name}'] = p.grad.detach().cpu()
    mg = mod.get_mlp_grads()
    if mg is not None:
        for i, t in enumerate(mg):
            out[f'mlp_W{i+1}'] = t.detach().cpu()
    return out


def zero_grads(g, ingp):
    for p in [g._xyz, g._opacity, g._features_dc, g._scaling, g._rotation]:
        p.grad = None
    for _, p in ingp.named_parameters():
        p.grad = None


def run_leg(path):
    g, ingp, fwd, BASE, TR = setup(path)
    ok = True

    # ---------- 1. IDENTITY @ default threshold ----------
    # Backward grad accumulation uses atomicAdd → run-to-run reordering noise at
    # float rounding scale. Calibrate the noise floor with a base-vs-base rerun
    # and require trunc-vs-base to sit within 4× of it.
    print(f"\n=== [{path}] 1. IDENTITY: trunc(exit_T=1e-4) vs base ===")
    ingp.is_trunc_mode = False
    loss_b, img_b, alp_b, nc_b, _ = fwd()
    loss_b.backward(); torch.cuda.synchronize()
    gb = grads_of(g, ingp, BASE)
    zero_grads(g, ingp)
    loss_b2, img_b2, alp_b2, _, _ = fwd()
    loss_b2.backward(); torch.cuda.synchronize()
    gb2 = grads_of(g, ingp, BASE)
    zero_grads(g, ingp)
    noise_floor = {k: max((gb[k] - gb2[k]).abs().max().item(), 1e-12) for k in gb}
    print(f"  fwd determinism: max|dimg|={(img_b2 - img_b).abs().max().item():.3e}")

    ingp.is_trunc_mode = True
    TR.set_exit_T(1e-4)
    loss_t, img_t, alp_t, nc_t, _ = fwd()
    loss_t.backward(); torch.cuda.synchronize()
    gt = grads_of(g, ingp, TR)
    zero_grads(g, ingp)

    d_img = (img_t - img_b).abs().max().item()
    d_alp = (alp_t - alp_b).abs().max().item()
    print(f"  fwd: max|dimg|={d_img:.3e} max|dalpha|={d_alp:.3e} "
          f"ncontrib {nc_b} vs {nc_t}")
    if d_img != 0.0 or d_alp != 0.0 or nc_b != nc_t:
        ok = False
    for k in sorted(set(gb) | set(gt)):
        if k not in gb or k not in gt:
            print(f"  grad {k:28s} MISSING in one leg"); ok = False; continue
        d = (gb[k] - gt[k]).abs().max().item()
        nf = noise_floor.get(k, 1e-12)
        good = d <= max(4.0 * nf, 1e-10)
        print(f"  grad {k:28s} max|d|={d:.3e} noise_floor={nf:.3e} "
              f"[{'OK' if good else 'MISMATCH'}]")
        if not good:
            ok = False

    # ---------- 2. FD @ exit_T = 0.5 ----------
    print(f"\n=== [{path}] 2. FD gradcheck @ exit_T={EXIT_T} (eps={EPS}) ===")
    TR.set_exit_T(EXIT_T)
    loss5, img5, alp5, nc5, _ = fwd()
    loss5.backward(); torch.cuda.synchronize()
    g5 = grads_of(g, ingp, TR)
    an_op = g._opacity.grad.detach().clone()
    an_xyz = g._xyz.grad.detach().clone()
    an_fdc = g._features_dc.grad.detach().clone()
    zero_grads(g, ingp)
    print(f"  truncated ncontrib={nc5}  (base full ncontrib={nc_b})")

    probes = ([('opacity', n, 0, g._opacity, an_op, EPS_GEO) for n in range(N_SURF)]
              + [('xyz.z', n, 2, g._xyz, an_xyz, EPS_GEO) for n in (0, 2, 4, 6)]
              + [('f_dc.r', n, 0, g._features_dc, an_fdc, EPS) for n in (0, 2, 5)])
    n_fail = 0
    with torch.no_grad():
        for label, n, c, param, an_t, eps in probes:
            base_v = param.data.view(N_SURF, -1)[n, c].item()
            param.data.view(N_SURF, -1)[n, c] = base_v + eps
            lp, _, _, ncp, gm_p = fwd(fd=True); lp = lp.item()
            param.data.view(N_SURF, -1)[n, c] = base_v - eps
            lm, _, _, ncm, gm_m = fwd(fd=True); lm = lm.item()
            param.data.view(N_SURF, -1)[n, c] = base_v
            fd_val = (lp - lm) / (2 * eps)
            an = an_t.view(N_SURF, -1)[n, c].item()
            dnc_px = int((gm_p != gm_m).sum().item()) if gm_p is not None else abs(ncp - ncm)
            boundary = (dnc_px != 0)
            denom = max(abs(an), abs(fd_val), 1e-8)
            rel = abs(an - fd_val) / denom
            if boundary:
                flag = 'BOUNDARY (inclusion flipped — FD invalid, discontinuity site)'
            elif rel < 0.03 or abs(an - fd_val) < 2e-4:
                flag = 'OK'
            elif rel < 0.10:
                flag = 'warn'
            else:
                flag = 'FAIL'; n_fail += 1
            print(f"  {label:8s} g{n} : analytic={an:+.6e} fd={fd_val:+.6e} "
                  f"rel={rel:.3e} flip_px={dnc_px} [{flag}]")
    if n_fail:
        ok = False

    # ---------- 3. DEAD surfel ----------
    print(f"\n=== [{path}] 3. DEAD surfel (g{N_SURF-1}, behind every front) ===")
    for name, t in (('opacity', an_op), ('xyz', an_xyz), ('f_dc', an_fdc)):
        v = t.view(N_SURF, -1)[N_SURF - 1].abs().max().item()
        print(f"  |grad {name}| = {v:.3e} [{'OK' if v == 0.0 else 'NONZERO?!'}]")
        if v != 0.0:
            ok = False

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({'g5': g5, 'img5': img5.detach().cpu()}, f'{OUT_DIR}/{path}.pt')
    print(f"\nLEG {path}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def compare():
    a = torch.load(f'{OUT_DIR}/collab.pt', weights_only=False)
    b = torch.load(f'{OUT_DIR}/scalar.pt', weights_only=False)
    ok = True
    print("\n=== collab vs scalar backward @ exit_T=0.5 ===")
    keys = sorted(set(a['g5']) | set(b['g5']))
    for k in keys:
        if k not in a['g5'] or k not in b['g5']:
            print(f"  {k:28s} MISSING in one leg"); ok = False; continue
        d = (a['g5'][k] - b['g5'][k]).abs().max().item()
        n = max(a['g5'][k].abs().max().item(), b['g5'][k].abs().max().item(), 1e-12)
        rel = d / n
        flag = 'OK' if rel < 1e-3 else ('warn' if rel < 1e-2 else 'MISMATCH')
        if flag == 'MISMATCH':
            ok = False
        print(f"  {k:28s} max|d|={d:.3e} rel={rel:.3e} [{flag}]")
    print("RESULT:", "COLLAB == SCALAR under truncation" if ok
          else "COLLAB PATH MISMATCH under truncation")
    sys.exit(0 if ok else 1)


def run_boundary():
    """Surgical off-by-one check at the truncation boundary.

    5 identical surfels stacked coaxially (same x,y; z spaced), all facing the
    camera, opacity sigmoid(=0.32) and projected size (~15 px sigma) chosen so
    that within a central 4x4-pixel loss window every ray has the SAME alpha
    sequence and therefore the SAME crossing index k:

      exit_T=0.5 : T = 1 -> .68 -> .4624            => k=1 (frags 0,1 blend)
      exit_T=0.3 : T = 1 -> .68 -> .4624 -> .3144 -> .2138 => k=3 (frags 0..3)

    (worst-case window-corner alphas keep >=0.014 margin to the threshold, vs
    FD perturbations of ~7e-4 — no inclusion flips inside the window.)

    Loss is masked to the window, so gradients can only flow from window
    pixels. Checks per threshold:
      A. per-pixel contributor count inside the window == k+1 exactly;
      B. analytic grads of frags 0..k are NONZERO; frags k+1.. are EXACTLY 0
         (a reverse walk starting one fragment too deep would light up frag
         k+1; one too shallow would zero frag k);
      C. FD-exactness for the crossing fragment k, its predecessor k-1, and
         the first excluded fragment k+1 (0 == 0), on opacity and f_dc.
    """
    torch.manual_seed(0)
    from hash_encoder.config import Config
    from hash_encoder.modules import INGP
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    import diff_surfel_3D_sh_res as BASE
    import diff_surfel_3D_sh_res_trunc as TR

    cfg = Config(YAML)
    args = build_args()
    ingp = INGP(cfg, args=args).to(DEV)
    ingp.set_active_levels(current_iter=12000)
    with torch.no_grad():
        for name, p in ingp.named_parameters():
            if 'encoding' in name or 'grid' in name or 'embeddings' in name:
                p.copy_(torch.round(torch.randn_like(p) * 32.0) / 64.0)
    for M in (BASE, TR):
        M.set_residual_mode(0)
        M.set_activation_bias(0.5, 0.0)
        M.set_lru_slope(0.0)

    N = 5
    g = GaussianModel(3)
    g.feature_mode = 'sh'
    g.kernel_type = 'gaussian'
    g.kernel_type2 = None
    g.active_sh_degree = 3
    g.max_sh_degree = 3
    zs = torch.tensor([-0.20, -0.10, 0.00, 0.10, 0.20], device=DEV)
    xyz = torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), zs], dim=1)
    g._xyz = torch.nn.Parameter(xyz.contiguous())
    g._scaling = torch.nn.Parameter(torch.full((N, 2), math.log(0.60), device=DEV))
    g._rotation = torch.nn.Parameter(torch.tensor(
        [[1.0, 0, 0, 0]] * N, device=DEV))
    logit = math.log(0.32 / 0.68)
    g._opacity = torch.nn.Parameter(torch.full((N, 1), logit, device=DEV))
    torch.manual_seed(5)
    g._features_dc = torch.nn.Parameter((0.2 + 0.5 * torch.rand((N, 1, 3), device=DEV)).contiguous())
    g._features_rest = torch.nn.Parameter((0.05 * torch.randn((N, 15, 3), device=DEV)).contiguous())
    g._appearance_level = torch.nn.Parameter(24.0 * torch.ones((N, 1), device=DEV),
                                             requires_grad=False)
    g.max_radii2D = torch.zeros(N, device=DEV)

    cam = build_camera()
    pipe = Namespace(debug=False, skip_aux_normal_dist=True, compute_cov3D_python=False,
                     convert_SHs_python=False, depth_ratio=0.0)
    bg = torch.zeros(3, device=DEV)
    ingp.is_trunc_mode = True

    # central 4x4 loss window
    win = torch.zeros(1, 96, 96, device=DEV)
    win[:, 46:50, 46:50] = 1.0

    def fwd(fd=False):
        pkg = render(cam, g, pipe, bg, ingp=ingp, iteration=12000, cfg=cfg,
                     lowpass=True, is_training=True)
        img, alp = pkg['render'], pkg['rend_alpha']
        gnum = pkg.get('gaussian_num')
        loss_t = (img * win).sum() + (alp * win).sum()
        if fd:
            loss_t = (img.double() * win.double()).sum() + (alp.double() * win.double()).sum()
        win_counts = gnum.squeeze()[46:50, 46:50] if gnum is not None else None
        return loss_t, win_counts

    all_ok = True
    for exit_T, k in ((0.5, 1), (0.3, 3)):
        print(f"\n=== boundary: exit_T={exit_T} -> expected crossing k={k} "
              f"(frags 0..{k} blend, {k+1}.. excluded) ===")
        TR.set_exit_T(exit_T)
        for p in [g._xyz, g._opacity, g._features_dc]:
            p.grad = None
        loss, wc = fwd()
        loss.backward(); torch.cuda.synchronize()

        # A. contributor count in window
        if wc is not None:
            u = wc.unique().tolist()
            okA = (u == [k + 1])
            print(f"  A. window contributor counts: {u} "
                  f"[{'OK' if okA else 'FAIL (expected ' + str(k+1) + ')'}]")
            all_ok &= okA
        # B. aliveness pattern
        for n in range(N):
            go = g._opacity.grad[n].abs().max().item()
            gc = g._features_dc.grad[n].abs().max().item()
            gx = g._xyz.grad[n].abs().max().item()
            included = n <= k
            alive = (go != 0.0) or (gc != 0.0) or (gx != 0.0)
            okB = (alive == included)
            print(f"  B. frag {n}: |g_op|={go:.3e} |g_fdc|={gc:.3e} |g_xyz|={gx:.3e} "
                  f"{'included' if included else 'EXCLUDED'} "
                  f"[{'OK' if okB else 'FAIL'}]")
            all_ok &= okB
        # C. FD on k-1, k, k+1
        an_op = g._opacity.grad.detach().clone()
        an_fdc = g._features_dc.grad.detach().clone()
        with torch.no_grad():
            for label, param, an_t, eps, c in (
                    ('opacity', g._opacity, an_op, EPS_GEO, 0),
                    ('f_dc.r', g._features_dc, an_fdc, EPS, 0)):
                for n in (k - 1, k, k + 1):
                    if n < 0 or n >= N:
                        continue
                    bv = param.data.view(N, -1)[n, c].item()
                    param.data.view(N, -1)[n, c] = bv + eps
                    lp, _ = fwd(fd=True); lp = lp.item()
                    param.data.view(N, -1)[n, c] = bv - eps
                    lm, _ = fwd(fd=True); lm = lm.item()
                    param.data.view(N, -1)[n, c] = bv
                    fd_val = (lp - lm) / (2 * eps)
                    an = an_t.view(N, -1)[n, c].item()
                    denom = max(abs(an), abs(fd_val), 1e-8)
                    rel = abs(an - fd_val) / denom
                    okC = rel < 0.03 or abs(an - fd_val) < 2e-4
                    tag = ('crossing' if n == k else
                           ('pre-crossing' if n < k else 'first-excluded'))
                    print(f"  C. {label:8s} frag {n} ({tag:14s}): "
                          f"analytic={an:+.6e} fd={fd_val:+.6e} rel={rel:.3e} "
                          f"[{'OK' if okC else 'FAIL'}]")
                    all_ok &= okC
    print(f"\nBOUNDARY: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'compare'
    if mode == 'run':
        sys.exit(run_leg(sys.argv[2]))
    elif mode == 'boundary':
        sys.exit(run_boundary())
    else:
        compare()

"""Collab-vs-scalar backward equivalence check for the GES frontmost-first promotion
in diff_surfel_3D_sh_res_harden (the GEStex harden rasterizer).

The promotion's std (scalar) backward path is FD-verified; the MODE-5 collab-GEMM
backward is a separate code path (the historical silent-bug spot). This test renders
the SAME tiny tilted-surfel scene through the REAL render() mode-5 pipeline and
compares every gradient between the two backward paths — they must agree to float
precision. DISABLE_COLLABORATIVE_GEMM is read once per process, so each leg runs in
its own process:

  python scripts/test_frontmost_collab.py run  on  collab   # promotion ON,  collab GEMM
  python scripts/test_frontmost_collab.py run  on  scalar   # promotion ON,  scalar path
  python scripts/test_frontmost_collab.py run  off collab   # control pair
  python scripts/test_frontmost_collab.py run  off scalar
  python scripts/test_frontmost_collab.py compare
"""
import sys, os, math
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

DEV = 'cuda'
YAML = './configs/himalaya_2d.yaml'
OUT_DIR = '/tmp/claude-1000/fm_collab_check'


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
    """Tilted, overlapping surfels with interleaved depths — the promotion-relevant
    geometry (center-depth order != per-pixel intersection order)."""
    from scene.gaussian_model import GaussianModel
    g = GaussianModel(3)
    g.feature_mode = 'sh'
    g.kernel_type = 'gaussian'
    g.kernel_type2 = None
    g.is_gestex = True
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
    # Harden config: mode-2 signed residual + post-blend LRU (matches training post-5k)
    HD.set_residual_mode(2)
    ingp.is_mixed_deferred_relu_mode = True
    HD.set_activation_bias(0.5, 0.0)
    HD.set_lru_slope(0.01)
    ingp.lru_slope = 0.01
    # Promotion toggle consumed by the renderer hook (routes to the harden clone).
    # NOTE the renderer's flag split: ingp.frontmost_on -> set_frontmost_first (the
    # GES 2-pass PROMOTION under test); ingp.first_int_sort -> set_tile_depth_sort
    # (the independent tile-depth SORT — keep OFF here). This test used to set
    # first_int_sort, silently exercising the sort with the promotion OFF.
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
    R_w = torch.arange(img.numel(), device=DEV, dtype=torch.float32)
    R_w = (0.3 + 0.7 * torch.sin(R_w * 0.37).abs()).view_as(img)  # deterministic weights
    loss = (img * R_w).sum()
    loss.backward()
    torch.cuda.synchronize()

    grads = {
        'img': img.detach().cpu(),
        'xyz': g._xyz.grad.detach().cpu(),
        'opacity': g._opacity.grad.detach().cpu(),
        'scaling': g._scaling.grad.detach().cpu(),
        'rotation': g._rotation.grad.detach().cpu(),
        'f_dc': g._features_dc.grad.detach().cpu(),
        'f_rest': g._features_rest.grad.detach().cpu(),
    }
    # hash grad (autograd tensor input)
    for name, p in ingp.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            grads[f'ingp.{name}'] = p.grad.detach().cpu()
    # MLP weight grads live in the harden module's buffer
    mg = HD.get_mlp_grads()
    if mg is not None:
        for i, t in enumerate(mg):
            grads[f'mlp_W{i+1}'] = t.detach().cpu()

    os.makedirs(OUT_DIR, exist_ok=True)
    f = f'{OUT_DIR}/{("on" if promo else "off")}_{path}.pt'
    torch.save(grads, f)
    print(f"saved {f}: " + ", ".join(f"{k}={v.norm():.4e}" for k, v in grads.items()
                                     if k != 'img'))


def compare():
    ok_all = True
    for promo in ('off', 'on'):
        a = torch.load(f'{OUT_DIR}/{promo}_collab.pt')
        b = torch.load(f'{OUT_DIR}/{promo}_scalar.pt')
        print(f"\n=== promotion {promo.upper()}: collab vs scalar ===")
        keys = sorted(set(a) | set(b))
        for k in keys:
            if k not in a or k not in b:
                print(f"  {k:24s} MISSING in one leg"); ok_all = False; continue
            d = (a[k] - b[k]).abs().max().item()
            n = max(a[k].abs().max().item(), b[k].abs().max().item(), 1e-12)
            rel = d / n
            flag = 'OK' if rel < 1e-3 else ('warn' if rel < 1e-2 else 'MISMATCH')
            if flag == 'MISMATCH':
                ok_all = False
            print(f"  {k:24s} max|d|={d:.3e} rel={rel:.3e} [{flag}]")
    # promotion must change the image (scene is engineered for it)
    i_on = torch.load(f'{OUT_DIR}/on_collab.pt')['img']
    i_off = torch.load(f'{OUT_DIR}/off_collab.pt')['img']
    print(f"\nforward max|ON-OFF| = {(i_on - i_off).abs().max().item():.5f} (want > 0)")
    print("\nRESULT:", "COLLAB == SCALAR (both promo states) — collab path OK"
          if ok_all else "COLLAB PATH MISMATCH — bug localized to the MODE-5 backward")
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'compare'
    if mode == 'run':
        run_leg(promo=(sys.argv[2] == 'on'), path=sys.argv[3])
    else:
        compare()

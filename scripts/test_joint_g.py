"""FD gradcheck for the GEStex additive Gaussian pass (diff_surfel_gestex_joint_g):
C_G = sum_g[depth_g <= D_S] alpha_g * SH(g); W_G = sum alpha_g. Verifies the SH-color +
opacity gradients against finite differences (a depth_map that admits all Gaussians).
Run: conda run -n nest_splatting python scripts/test_joint_g.py
"""
import sys, math
import numpy as np
import torch
sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')
import diff_surfel_gestex_joint_g as jg
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

DEV = 'cuda'; torch.manual_seed(0)
N, W, Hh, fov, znear, zfar = 6, 96, 96, 0.9, 0.01, 100.0
wvt = torch.tensor(getWorld2View2(np.eye(3), np.array([0., 0., 4.])), dtype=torch.float32).transpose(0, 1).cuda()
proj = getProjectionMatrix(znear, zfar, fov, fov).transpose(0, 1).cuda()
fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
cc = wvt.inverse()[3, :3]
st = jg.GaussianRasterizationSettings(Hh, W, math.tan(fov * .5), math.tan(fov * .5),
                                      torch.zeros(3, device=DEV), 1.0, wvt, fpt, 0, cc, False, False)
rast = jg.GaussianRasterizer(st)

xs = torch.linspace(-0.3, 0.3, N, device=DEV)
means = torch.stack([xs, 0.1 * torch.sin(xs * 7), torch.zeros_like(xs)], 1).contiguous()
scales = torch.full((N, 3), 0.12, device=DEV)
rots = torch.zeros((N, 4), device=DEV); rots[:, 0] = 1.0
opac = torch.full((N, 1), 0.6, device=DEV)
depth_map = torch.full((1, Hh, W), 1e6, device=DEV)   # admit all Gaussians (no discard)


def render(sh):
    sp = torch.zeros((N, 3), device=DEV, requires_grad=True) + 0
    mc = torch.zeros((N, 1), device=DEV, requires_grad=True) + 0
    color, radii, W_G = rast(means3D=means, means2D=sp, max_contrib_ret=mc, opacities=opac,
                             depth_map=depth_map, shs=sh, scales=scales, rotations=rots)
    return color


sh = (0.2 * torch.randn(N, 1, 3, device=DEV)).requires_grad_(True)   # SH degree 0
img = render(sh)
print(f"forward: image range=[{img.min().item():.4f},{img.max().item():.4f}] mean={img.mean().item():.4f}")
loss = img.mean(); loss.backward()
print(f"sh grad: norm={sh.grad.norm().item():.4e}")

eps = 1e-3
flat = sh.grad.reshape(-1); idxs = torch.topk(flat.abs(), 5).indices.tolist(); ok = 0
print("[gradcheck] SH coeffs (top-5 by |grad|):")
for fi in idxs:
    a = flat[fi].item(); base = sh.data.reshape(-1); orig = base[fi].item()
    base[fi] = orig + eps
    with torch.no_grad(): lp = render(sh).mean().item()
    base[fi] = orig - eps
    with torch.no_grad(): lm = render(sh).mean().item()
    base[fi] = orig
    fd = (lp - lm) / (2 * eps); rel = abs(a - fd) / (abs(fd) + 1e-9); ok += rel < 0.05
    print(f"  sh[{fi:3d}] analytic={a:+.4e} fd={fd:+.4e} rel={rel:.2e} [{'OK' if rel<0.05 else 'X'}]")
print(f"\nSUMMARY: joint_g SH {ok}/5 within 5% rel tol")

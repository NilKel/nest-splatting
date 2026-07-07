"""Unit test + FD gradcheck for the GEStex sort-free surfel z-buffer pass
(diff_surfel_gestex_joint_s): C_S = colors_precomp(SV) + bilinear(atlas) at the
frontmost surfel. Verifies the atlas + colors_precomp backward against finite diff.
Run: conda run -n nest_splatting python scripts/test_joint_s.py
"""
import sys, math
import numpy as np
import torch
sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')
import diff_surfel_gestex_joint_s as js
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

DEV = 'cuda'; torch.manual_seed(0)
R, N, W, Hh = 8, 6, 96, 96
fov, znear, zfar = 0.9, 0.01, 100.0

wvt = torch.tensor(getWorld2View2(np.eye(3), np.array([0., 0., 4.])), dtype=torch.float32).transpose(0, 1).cuda()
proj = getProjectionMatrix(znear, zfar, fov, fov).transpose(0, 1).cuda()
fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
cc = wvt.inverse()[3, :3]
st = js.GaussianRasterizationSettings(Hh, W, math.tan(fov * .5), math.tan(fov * .5),
                                      torch.zeros(3, device=DEV), 1.0, wvt, fpt, 3, cc, False, False)
rast = js.GaussianRasterizer(st)

xs = torch.linspace(-0.3, 0.3, N, device=DEV)
means = torch.stack([xs, 0.1 * torch.sin(xs * 7), torch.zeros_like(xs)], 1).contiguous()
scales = torch.full((N, 2), 0.15, device=DEV)          # activated 2D scales
rots = torch.zeros((N, 4), device=DEV); rots[:, 0] = 1.0
opac = torch.full((N, 1), 255.0, device=DEV)           # opaque discs (z-buffer)
mod_depth = 5.0 * scales.mean(-1, keepdim=True)


def render(colors, atlas, atlas_grad):
    js.set_gestex_atlas(atlas, atlas_grad, R, 4.0)
    color, radii, others = rast(means3D=means, opacities=opac, colors_precomp=colors,
                                mod_depth=mod_depth, scales=scales, rotations=rots)
    return color


colors = (0.3 + 0.1 * torch.randn(N, 3, device=DEV)).requires_grad_(True)
atlas = (0.1 * torch.randn(N, R, R, 3, device=DEV))
atlas_grad = torch.zeros_like(atlas)

img = render(colors, atlas, atlas_grad)
print(f"forward: image {tuple(img.shape)} range=[{img.min().item():.4f},{img.max().item():.4f}] mean={img.mean().item():.4f}")
loss = img.mean(); loss.backward()
print(f"colors_precomp(SV) grad: norm={colors.grad.norm().item():.4e}")
print(f"atlas grad: norm={atlas_grad.norm().item():.4e} nonzero={(atlas_grad.abs()>0).sum().item()}/{atlas_grad.numel()}")

# ---- FD gradcheck (atlas + colors) ----
eps = 1e-3
def loss_only(colors_, atlas_):
    with torch.no_grad():
        return render(colors_, atlas_, torch.zeros_like(atlas_)).mean().item()

print("\n[gradcheck] atlas texels (top-6 by |grad|):")
flat = atlas_grad.reshape(-1); idxs = torch.topk(flat.abs(), 6).indices.tolist(); ok = 0
for fi in idxs:
    a = flat[fi].item(); base = atlas.reshape(-1); orig = base[fi].item()
    base[fi] = orig + eps; lp = loss_only(colors, atlas)
    base[fi] = orig - eps; lm = loss_only(colors, atlas); base[fi] = orig
    fd = (lp - lm) / (2 * eps); rel = abs(a - fd) / (abs(fd) + 1e-9)
    ok += rel < 0.05
    print(f"  atlas[{fi:5d}] analytic={a:+.4e} fd={fd:+.4e} rel={rel:.2e} [{'OK' if rel<0.05 else 'X'}]")

print("[gradcheck] colors_precomp (top-4 by |grad|):")
cflat = colors.grad.reshape(-1); cidx = torch.topk(cflat.abs(), 4).indices.tolist(); okc = 0
for fi in cidx:
    a = cflat[fi].item(); base = colors.data.reshape(-1); orig = base[fi].item()
    base[fi] = orig + eps; lp = loss_only(colors, atlas)
    base[fi] = orig - eps; lm = loss_only(colors, atlas); base[fi] = orig
    fd = (lp - lm) / (2 * eps); rel = abs(a - fd) / (abs(fd) + 1e-9)
    okc += rel < 0.05
    print(f"  colors[{fi:3d}] analytic={a:+.4e} fd={fd:+.4e} rel={rel:.2e} [{'OK' if rel<0.05 else 'X'}]")

print(f"\nSUMMARY: atlas {ok}/6, colors {okc}/4 within 5% rel tol")

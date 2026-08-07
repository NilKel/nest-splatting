"""Texture-GS-style UV field stage for --method proberes.

Trains a global surface parameterization on a trained 3D_SH_res checkpoint,
then bakes BOTH the atlas content and the per-surfel probes analytically:

  phi     : R^3 -> [0,1]^2   smooth MLP (NO positional encoding), maps surface
            points to the texture domain.
  phi_inv : [0,1]^2 -> R^3   hash-encoded inverse, used only in the training
            losses (cycle consistency + Chamfer) to shape phi. (Texture-GS
            Sec. 4.2: hash on the inverse keeps the FORWARD map smooth.)

  T[p]    = teacher_residual( phi_inv(p) )        <- atlas starts FULL of the
            trained INGP's residual content (no learning from scratch)
  probe_i = ( phi(c_i) * R,  J_phi(c_i) . [s_u*u_hat | s_v*v_hat] * R )
            <- overlapping surfels share texels BY CONSTRUCTION (phi takes the
            3D point), giving the gradient pooling proberes lacked.

Usage:
  conda run -n nest_splatting python scripts/probe_uv_field.py train -m <ckpt_dir>
  conda run -n nest_splatting python scripts/probe_uv_field.py bake  -m <ckpt_dir> --tex_res 2048
  # then finetune:
  python train.py --method proberes ... --init_ply <ckpt>/point_cloud/iteration_*/point_cloud.ply \
      --probe_init_dir <ckpt>/uv_field --probe_freeze_head --densify_until_iter 0 --probe_nosh_lambda 1.0

Outputs under <ckpt_dir>/uv_field/: uv_field.pt, tex_init.pt, probes.pt, previews.
"""
import os, sys, json, glob, math, pickle, argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gridencoder import GridEncoder

DEV = 'cuda'


# ---------------------------------------------------------------- checkpoint io
def load_args(model_path):
    p = os.path.join(model_path, 'args.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    with open(os.path.join(model_path, 'args.json')) as f:
        return argparse.Namespace(**json.load(f))


class Ckpt:
    """3D_SH_res checkpoint: INGP teacher + per-surfel tensors from the PLY."""

    def __init__(self, model_path, iteration=-1):
        from hash_encoder.config import Config
        from hash_encoder.modules import INGP
        from utils.general_utils import build_rotation
        from plyfile import PlyData

        self.model_path = model_path
        self.args = load_args(model_path)
        assert self.args.method in ('3D_SH_res', 'res_switch', '3D_SH_add'), \
            f'teacher expects a 3D_SH_res-family checkpoint, got {self.args.method}'
        cfg_yaml = os.path.join(model_path, 'config.yaml')
        self.cfg = Config(cfg_yaml if os.path.exists(cfg_yaml) else self.args.yaml)
        if iteration == -1:
            ngps = glob.glob(os.path.join(model_path, 'ngp_*.pth'))
            assert ngps, f'no ngp_*.pth in {model_path}'
            iteration = max(int(os.path.basename(f)[4:-4]) for f in ngps)
        self.iteration = iteration
        self.ingp = INGP(self.cfg, args=self.args).to(DEV)
        self.ingp.load_model(model_path, iteration)
        self.ingp.set_active_levels(iteration)
        self.hash_dim = self.ingp.active_hashgrid_levels * self.ingp.level_dim
        self.mlp = self.ingp.mlp_fused.float()

        ply = PlyData.read(os.path.join(model_path, 'point_cloud',
                                        f'iteration_{iteration}', 'point_cloud.ply'))
        el = ply.elements[0]

        def col(n):
            return torch.tensor(np.asarray(el.data[n], dtype=np.float32), device=DEV)

        self.center = torch.stack([col('x'), col('y'), col('z')], 1)
        s = torch.stack([col('scale_0'), col('scale_1')], 1).exp()
        q = torch.stack([col('rot_0'), col('rot_1'), col('rot_2'), col('rot_3')], 1)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        R = build_rotation(q)
        self.axis_u = R[..., 0] * s[:, 0:1]        # world-scaled tangent (1 sigma)
        self.axis_v = R[..., 1] * s[:, 1:2]
        self.opacity = torch.sigmoid(col('opacity'))
        self.N = self.center.shape[0]

        # scene normalization for phi's input
        med = self.center.median(dim=0).values
        rad = (self.center - med).norm(dim=-1).quantile(0.95).clamp_min(1e-6) * 1.2
        self.norm_c, self.norm_r = med, float(rad)
        print(f'[CKPT] {model_path} iter={iteration} N={self.N:,} hash_dim={self.hash_dim} '
              f'norm_r={self.norm_r:.3f}')

    @torch.no_grad()
    def residual(self, xyz):
        """Teacher residual RGB at world points [M,3] (identity activation, signed).
        INGP(points_3D=...) matches the CUDA hash query (verified in the distill work)."""
        H = self.ingp(points_3D=xyz).float()[:, :self.hash_dim]
        mlp_in = torch.zeros(xyz.shape[0], 16, device=DEV)
        mlp_in[:, :self.hash_dim] = H
        return self.mlp(mlp_in)[:, :3]

    def surface_samples(self, n):
        """Random on-disc surface points: c + u*axis_u + v*axis_v, |uv| ~ N(0,1) clipped."""
        idx = torch.randint(0, self.N, (n,), device=DEV)
        uv = torch.randn(n, 2, device=DEV).clamp_(-2.0, 2.0)
        return (self.center[idx] + uv[:, 0:1] * self.axis_u[idx]
                + uv[:, 1:2] * self.axis_v[idx])


# ---------------------------------------------------------------- networks
class Phi(nn.Module):
    """Forward UV map: smooth by construction (no positional encoding)."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2))

    def forward(self, xn):
        return torch.sigmoid(self.net(xn))


class PhiInv(nn.Module):
    """Inverse map with hash encoding (capacity lives here, per Texture-GS)."""

    def __init__(self, hidden=128, levels=8):
        super().__init__()
        scale = (512 / 16) ** (1.0 / (levels - 1))
        self.enc = GridEncoder(input_dim=2, num_levels=levels, level_dim=2,
                               per_level_scale=scale, base_resolution=16,
                               log2_hashmap_size=19)
        self.net = nn.Sequential(
            nn.Linear(levels * 2, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 3))

    def forward(self, u):
        return self.net(self.enc(u))


def fps(points, k):
    """Farthest-point sampling on GPU. points [N,3] -> indices [k]."""
    N = points.shape[0]
    idx = torch.zeros(k, dtype=torch.long, device=points.device)
    dist = torch.full((N,), 1e10, device=points.device)
    idx[0] = torch.randint(0, N, (1,))
    for i in range(1, k):
        dist = torch.minimum(dist, (points - points[idx[i - 1]]).pow(2).sum(-1))
        idx[i] = dist.argmax()
    return idx


def chamfer(a, b):
    d = torch.cdist(a, b)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


# ---------------------------------------------------------------- stages
def train(ck, out_dir, steps=3000, batch=16384, n_uv=8192, n_fps=8192,
          tex_res=2048, patch_px=12.0, w_area=1.0, w_cov=0.0):
    phi, phi_inv = Phi().to(DEV), PhiInv().to(DEV)
    opt = torch.optim.Adam([
        {'params': phi.parameters(), 'lr': 1e-3},
        {'params': phi_inv.enc.parameters(), 'lr': 1e-2},
        {'params': phi_inv.net.parameters(), 'lr': 1e-3},
    ], betas=(0.9, 0.99), eps=1e-15)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)

    print('[TRAIN] FPS-subsampling pseudo-GT point cloud...')
    p_fps = ((ck.center[fps(ck.center, n_fps)] - ck.norm_c) / ck.norm_r)

    for it in range(steps):
        x = (ck.surface_samples(batch) - ck.norm_c) / ck.norm_r
        u = torch.rand(n_uv, 2, device=DEV)
        l_3d = (x - phi_inv(phi(x))).norm(dim=-1).mean()
        inv_u = phi_inv(u)
        l_cd = chamfer(inv_u, p_fps)
        l_2d = (u - phi(inv_u)).norm(dim=-1).mean()
        # Coverage loss on PHI'S RANGE: chamfer in UV space between phi(surface)
        # and uniform samples. Without it phi's image can collapse into a small
        # blob (observed: all probes in ~340x600px of 2048^2) while phi_inv
        # covers the full square -> fragments read texels baked for OTHER
        # surface points -> structured garbage residuals.
        l_cov = chamfer(phi(x[:4096]), u[:4096]) if w_cov > 0 else torch.zeros((), device=DEV)
        # Area regularizer: per-surfel patch size (6*sqrt|det A| texels, where
        # A = J_phi . [axis_u|axis_v]/r * tex_res) should match patch_px —
        # otherwise phi compresses the surface into a corner of the UV square
        # and probes collapse to ~1 texel (observed: p50=1.1 without this).
        cidx = torch.randint(0, ck.N, (2048,), device=DEV)
        cn = ((ck.center[cidx] - ck.norm_c) / ck.norm_r).detach().requires_grad_(True)
        uvc = phi(cn)
        gx = torch.autograd.grad(uvc[:, 0].sum(), cn, create_graph=True)[0]
        gy = torch.autograd.grad(uvc[:, 1].sum(), cn, create_graph=True)[0]
        J = torch.stack([gx, gy], dim=1)                                   # [B,2,3]
        tang = torch.stack([ck.axis_u[cidx], ck.axis_v[cidx]], dim=2) / ck.norm_r
        Apx = torch.bmm(J, tang) * tex_res
        det = (Apx[:, 0, 0] * Apx[:, 1, 1] - Apx[:, 0, 1] * Apx[:, 1, 0]).abs().clamp_min(1e-8)
        l_area = w_area * ((0.5 * det.log() + math.log(6.0) - math.log(patch_px)) ** 2).mean()
        loss = l_3d + l_cd + l_2d + l_area + w_cov * l_cov
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it > 0 and it % 1000 == 0:
            os.makedirs(out_dir, exist_ok=True)
            pr_now = phi_probes(ck, phi, ck.norm_c, ck.norm_r, tex_res)
            torch.save({'probes': pr_now.cpu(), 'centers': ck.center.cpu()},
                       os.path.join(out_dir, 'probes.pt'))
            T_now = scatter_bake(ck, out_dir, tex_res=tex_res, grid=8, save=False)
            render_check(ck, pr_now, T_now, out_dir, tag=f'step{it}')
        if it % 250 == 0 or it == steps - 1:
            print(f'[TRAIN] {it:5d}  L={loss.item():.4f}  3d_cycle={l_3d.item():.4f} '
                  f'chamfer={l_cd.item():.4f}  2d_cycle={l_2d.item():.4f}  area={l_area.item():.4f}  cov={l_cov.item():.4f}')

    os.makedirs(out_dir, exist_ok=True)
    torch.save({'phi': phi.state_dict(), 'phi_inv': phi_inv.state_dict(),
                'norm_c': ck.norm_c.cpu(), 'norm_r': ck.norm_r,
                'iteration': ck.iteration}, os.path.join(out_dir, 'uv_field.pt'))
    print(f'[TRAIN] saved {out_dir}/uv_field.pt')


def phi_probes(ck, phi, norm_c, norm_r, tex_res):
    """probes = (phi(c)*R, J_phi(c) . [axis_u|axis_v]/r * R) -> [N,6]."""
    cn = ((ck.center - norm_c) / norm_r).detach().requires_grad_(True)
    uv = phi(cn)
    gx = torch.autograd.grad(uv[:, 0].sum(), cn, retain_graph=True)[0]
    gy = torch.autograd.grad(uv[:, 1].sum(), cn)[0]
    with torch.no_grad():
        J = torch.stack([gx, gy], dim=1)
        tang = torch.stack([ck.axis_u, ck.axis_v], dim=2) / norm_r
        A = torch.bmm(J, tang) * tex_res
        t0 = uv.detach() * tex_res
        return torch.stack([A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1],
                            t0[:, 0], t0[:, 1]], dim=-1).contiguous()


_RC = {}


def render_check(ck, probes, T, out_dir, tag):
    """Render TEST VIEW 0 residual-only (tex_only) through the REAL CUDA
    proberes kernel with the given probes + atlas: proves texture readback
    without train.py. Saves <out_dir>/check_<tag>_testview0_tex.png (+GT once).
    Residuals pass through UNBOUNDED; only the final image is clamped for PNG."""
    import copy
    from argparse import Namespace
    if not _RC:
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        from scene.gaussian_model import GaussianModel
        from hash_encoder.modules import INGP
        _src = getattr(ck.args, 'source_path', None) or getattr(ck.args, 'source', None)
        if not _src or not os.path.exists(_src):
            _parts = os.path.normpath(ck.model_path).split(os.sep)
            _i = _parts.index('outputs')
            _src = os.path.join(os.path.expanduser('~/Projects/data'), _parts[_i + 1], _parts[_i + 2])
        # Load test camera 0 through the repo's OWN dataset readers, so the
        # train/test split and intrinsics match exactly what train.py's Scene()
        # saw. (Hand-parsing transforms_test.json only handled nerf-synthetic
        # and raised FileNotFoundError on every COLMAP scene, e.g. mip-360.)
        from PIL import Image as _Image
        from scene.dataset_readers import sceneLoadTypeCallbacks
        if os.path.exists(os.path.join(_src, 'sparse')):
            _si = sceneLoadTypeCallbacks['Colmap'](
                _src, getattr(ck.args, 'images', None) or 'images', True)
        else:
            _si = sceneLoadTypeCallbacks['Blender'](
                _src, bool(getattr(ck.args, 'white_background', False)), True)
        _ci0 = (_si.test_cameras or _si.train_cameras)[0]
        # Mirror utils/camera_utils.loadCam EXACTLY, so the check renders at the
        # resolution the model actually trained at. Two traps here:
        #  - size comes from cam_info.image.size (the pixels on disk, honouring
        #    `-i images_N`), NOT cam_info.width/height, which on COLMAP scenes are
        #    the FULL-res intrinsics (room: 3114x2075 vs the 1557x1038 it trained on).
        #  - `-r -1` is not "no scaling": 3DGS caps the long side at 1600.
        # Rendering at 2x training resolution makes every surfel cover 4x the pixels
        # and magnifies the ~12-texel probe patches, so the result looks far softer
        # than it is — and benchmark_baked.py gets this right via Scene(), which made
        # the two visually incomparable.
        _img = _ci0.image
        _W, _H = _img.size
        _r = int(getattr(ck.args, 'resolution', -1) or -1)
        if _r in (1, 2, 4, 8):
            _W, _H = round(_W / _r), round(_H / _r)
        elif _W > 1600:
            _d = _W / 1600.0
            _W, _H = int(_W / _d), int(_H / _d)
        if (_W, _H) != _img.size:
            _img = _img.resize((_W, _H), _Image.LANCZOS)
        print(f'[CHECK] test cam "{_ci0.image_name}" (-r {_r}, images={getattr(ck.args,"images",None)}) '
              f'-> rendering at {_W}x{_H}')
        class _CI: pass
        ci = _CI(); ci.R, ci.T, ci.FovX, ci.FovY = _ci0.R, _ci0.T, _ci0.FovX, _ci0.FovY
        ci.width, ci.height, ci.image = _W, _H, _img
        ci.clip_plane = getattr(_ci0, 'clip_plane', None)
        wvt = torch.tensor(getWorld2View2(ci.R, ci.T), dtype=torch.float32).transpose(0, 1).cuda()
        proj = getProjectionMatrix(0.01, 100.0, ci.FovX, ci.FovY).transpose(0, 1).cuda()
        cam = Namespace(FoVx=ci.FovX, FoVy=ci.FovY, image_width=ci.width, image_height=ci.height,
                        world_view_transform=wvt,
                        full_proj_transform=(wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0),
                        camera_center=wvt.inverse()[3, :3], znear=0.01, zfar=100.0,
                        clip_plane=ci.clip_plane)
        g = GaussianModel(3)
        # Set feature_mode BEFORE load_ply so the loader restores sv_site_*/sv_col_*
        # columns (that's the gate in GaussianModel.load_ply) — otherwise the SV
        # base is silently dropped and composites render SH-only.
        g.feature_mode = 'SV' if getattr(ck.args, 'feature', 'sh') == 'SV' else 'sh'
        g.kernel_type = getattr(ck.args, 'kernel', 'gaussian')
        g.kernel_type2 = None
        g.load_ply(os.path.join(ck.model_path, 'point_cloud',
                                f'iteration_{ck.iteration}', 'point_cloud.ply'), args=ck.args)
        g.active_sh_degree = g.max_sh_degree
        a2 = copy.deepcopy(ck.args)
        a2.method = 'proberes'
        a2.probe_tex_res = T.shape[0]
        a2.probe_no_pixels = False
        a2.disable_c2f = True
        ingp2 = INGP(ck.cfg, args=a2).to(DEV)
        ingp2.set_active_levels(current_iter=10 ** 6)
        ingp2.hashgrid_disabled = False
        try:
            ci.image.save(os.path.join(out_dir, 'check_GT_testview0.png'))
        except Exception:
            pass
        _RC.update(cam=cam, g=g, ingp=ingp2,
                   pipe=Namespace(debug=False, skip_aux_normal_dist=True,
                                  compute_cov3D_python=False, convert_SHs_python=False,
                                  depth_ratio=0.0))
    from gaussian_renderer import render
    ingp2 = _RC['ingp']
    ingp2.probe_head.fixed_probes = probes.to(DEV).contiguous()
    ingp2.probe_head.fixed_centers = ck.center
    outs = {}
    with torch.no_grad():
        ingp2.probe_field.pixels.data.copy_(T.to(DEV))
        for _name, _dm in (('tex', 'tex_only'), ('full', None), ('sv', 'sh_only')):
            pkg = render(_RC['cam'], _RC['g'], _RC['pipe'], torch.zeros(3, device=DEV),
                         ingp=ingp2, iteration=10 ** 6, cfg=ck.cfg,
                         lowpass=getattr(ck.args, 'lowpass', True), is_training=False,
                         decompose_mode=_dm)
            outs[_name] = pkg['render']
    try:
        from PIL import Image
        for _name, _im in outs.items():
            Image.fromarray((_im.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
                            .astype('uint8')).save(
                os.path.join(out_dir, f'check_{tag}_testview0_{_name}.png'))
    except Exception as e:
        print(f'[CHECK {tag}] save failed: {e}')
    print(f'[CHECK {tag}] testview0  tex mean={outs["tex"].mean():.4f}  '
          f'sv mean={outs["sv"].mean():.4f}  full mean={outs["full"].mean():.4f} '
          f'max={outs["full"].max():.3f}  (SV+tex saturating if full >> sv)')


def bake(ck, out_dir, tex_res=2048, chunk=1 << 20):
    st = torch.load(os.path.join(out_dir, 'uv_field.pt'), map_location=DEV)
    phi, phi_inv = Phi().to(DEV), PhiInv().to(DEV)
    phi.load_state_dict(st['phi'])
    phi_inv.load_state_dict(st['phi_inv'])
    phi.eval(); phi_inv.eval()
    norm_c, norm_r = st['norm_c'].to(DEV), st['norm_r']

    # ---- atlas: T[p] = teacher_residual(phi_inv(p)) ----
    t = (torch.arange(tex_res, device=DEV, dtype=torch.float32) + 0.5) / tex_res
    gy, gx = torch.meshgrid(t, t, indexing='ij')
    P = torch.stack([gx, gy], dim=-1).view(-1, 2)          # x fast: matches tex[y, x]
    T = torch.empty(P.shape[0], 3, device=DEV)
    with torch.no_grad():
        for i in range(0, P.shape[0], chunk):
            x3 = phi_inv(P[i:i + chunk]) * norm_r + norm_c
            T[i:i + chunk] = ck.residual(x3)
    T = T.view(tex_res, tex_res, 3)
    torch.save(T.cpu(), os.path.join(out_dir, 'tex_init.pt'))

    probes = phi_probes(ck, phi, norm_c, norm_r, tex_res)
    torch.save({'probes': probes.cpu(), 'centers': ck.center.cpu()},
               os.path.join(out_dir, 'probes.pt'))
    A = probes[:, :4].view(-1, 2, 2)
    t0 = probes[:, 4:6]
    # diagnostics + previews
    rho = (A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]).abs().sqrt()
    q = torch.quantile(6.0 * rho, torch.tensor([0.1, 0.5, 0.9], device=DEV))
    print(f'[BAKE] tex {tex_res}x{tex_res}  T range [{T.min():.3f},{T.max():.3f}] '
          f'std={T.std():.3f} | probe patch texels p10={q[0]:.1f} p50={q[1]:.1f} p90={q[2]:.1f}')
    try:
        from PIL import Image
        img = ((T + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
        Image.fromarray(img).save(os.path.join(out_dir, 'tex_init_preview.png'))
        cov = torch.zeros(tex_res, tex_res, device=DEV)
        xs = t0[:, 0].round().long().clamp(0, tex_res - 1)
        ys = t0[:, 1].round().long().clamp(0, tex_res - 1)
        cov[ys, xs] = 1.0
        Image.fromarray((cov * 255).byte().cpu().numpy()).save(
            os.path.join(out_dir, 'probe_centers_preview.png'))
    except Exception as e:
        print('[BAKE] preview skipped:', e)
    print(f'[BAKE] saved tex_init.pt + probes.pt in {out_dir}')
    render_check(ck, probes, T, out_dir, tag='final')


def oct_probes(ck, tex_res=2048, patch_px=12.0, aniso=True, contracted=False):
    """Analytic octahedral probes (ProbeHead3D's frozen base): uniform spread
    by construction — no phi needed once the bake scatters through the same
    probes the renderer reads with.

    `aniso=True` (default) gives the patch INDEPENDENT per-axis extents
    (rho_u ∝ su, rho_v ∝ sv) instead of one isotropic rho from sqrt(su*sv), so a
    10:1 surfel gets a 10:1 atlas rect rather than a square that over-serves its
    short axis and under-serves its long one. This is what the BC7 bake does with
    its (res_x, res_y) pairs, and what phi probes get for free from
    J_phi . [axis_u|axis_v]. TOTAL texel demand is UNCHANGED —
    sum (patch*su/s)(patch*sv/s) == sum (patch*sqrt(su*sv)/s)^2 — so this
    redistributes each patch's shape at constant storage. aniso=False reproduces
    the previous similarity-transform probes bit-for-bit.

    Nyquist note: footprint_u = patch_px * su/s_med texels vs a requirement of
    12*su/cell_size — both linear in scale, so ONE patch_px Nyquist-matches every
    surfel at patch_px = 12*s_med/cell_size (21.9 on chair). See §4 of the doc for
    why that is unaffordable at tex_res 2048 (18.8x oversubscribed).
    """
    from hash_encoder.probe_modules import _oct_encode
    d = ck.center - ck.norm_c
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    base = _oct_encode(d)
    t0 = (base * 0.96 + 0.02) * tex_res
    su = ck.axis_u.norm(dim=-1); sv = ck.axis_v.norm(dim=-1)
    if contracted:
        # Size footprints by CONTRACTED-space scale, not world scale. The teacher's
        # hash queries mip-NeRF-360-contracted coords (modules.py _encode_3D):
        #   x_n = (x - vmid)/vsize ;  r>1 -> r' = 2 - 1/r  (direction preserved)
        # so a far surfel's TANGENTIAL extent is compressed by (2-1/r)/r. Its
        # residual detail is band-limited by that metric — sizing probes in world
        # units gives the far field 79x the texel area per surfel for content the
        # contraction already flattened (bicycle: far band = 87% of texels, 33% of
        # variation -> the p32 background artifacts). factor=1 inside the unit
        # ball, so bounded scenes are unchanged.
        if not getattr(ck.ingp, 'contract', False):
            # Teacher trained WITHOUT contraction: its hash metric is linear
            # everywhere, so world scale IS the right sizing — the factor would
            # wrongly shrink far surfels. No-op with a loud note.
            print('[OCT] --oct_contracted requested but teacher has contract=False; '
                  'world-scale sizing kept (factor would not match the hash metric).')
            contracted = False
    if contracted:
        vol_min, vol_max = ck.ingp.voxel_range
        vmid, vsize = (vol_min + vol_max) * 0.5, (vol_max - vol_min) * 0.5
        r = ((ck.center - vmid) / vsize).norm(dim=-1).clamp_min(1e-6)
        cfac = torch.where(r <= 1.0, torch.ones_like(r), (2.0 - 1.0 / r) / r)
        su = su * cfac; sv = sv * cfac
        print(f'[OCT] contracted sizing: r p10/50/90 = '
              f'{r.quantile(0.1):.2f}/{r.quantile(0.5):.2f}/{r.quantile(0.9):.2f}, '
              f'factor p10/50/90 = {cfac.quantile(0.1):.3f}/{cfac.quantile(0.5):.3f}/{cfac.quantile(0.9):.3f}')
    s_med = torch.sqrt(su * sv).clamp_min(1e-9).log().median().exp()
    if aniso:
        rho_u = (patch_px / 6.0) * su / s_med
        rho_v = (patch_px / 6.0) * sv / s_med
    else:
        rho_u = rho_v = (patch_px / 6.0) * torch.sqrt(su * sv) / s_med
    uh = ck.axis_u / su[:, None]; vh = ck.axis_v / sv[:, None]
    up = torch.tensor([0.0, 0.0, 1.0], device=DEV).expand_as(ck.center)
    n = torch.cross(uh, vh, dim=-1)
    c = up - (up * n).sum(-1, keepdim=True) * n
    alt = torch.tensor([1.0, 0.0, 0.0], device=DEV).expand_as(ck.center)
    alt = alt - (alt * n).sum(-1, keepdim=True) * n
    c = torch.where(c.norm(dim=-1, keepdim=True) > 1e-3, c, alt)
    phi_g = torch.atan2((c * vh).sum(-1), (c * uh).sum(-1))
    ct, st = torch.cos(-phi_g), torch.sin(-phi_g)
    # A = R(-phi_g) . diag(rho_u, rho_v): scale in the surfel's own uv axes, THEN
    # rotate into the atlas frame. Reduces exactly to rho*R when rho_u == rho_v.
    probes = torch.stack([rho_u * ct, -rho_v * st, rho_u * st, rho_v * ct,
                          t0[:, 0], t0[:, 1]], dim=-1).contiguous()
    return probes


def nyquist_grid(ck, uv_extent=3.0, grid_min=8, grid_max=128):
    """Per-surfel, per-axis sampling resolution that resolves the FINEST hash
    level over the surfel's +-uv_extent sigma span (Nyquist: 2 samples/cell).

    Mirrors `compute_adaptive_resolution` in scripts/benchmark_baked.py, which
    sizes the per-surfel BC7 atlas the same way. Returns [N,2] int (gx, gy).

    Sampling below this ALIASES the teacher into the atlas: on chair, p50 is
    25.2 samples/axis but p90 is 67.5, so the old fixed grid=24 band-limited
    only the smaller half of the surfels.
    """
    hp = ck.ingp.hash_encoding.get_params()
    num_levels, per_level_scale, base_resolution = hp[2], hp[3], hp[4]
    finest = base_resolution * (per_level_scale ** (num_levels - 1))
    cell = (ck.ingp.voxel_range[1] - ck.ingp.voxel_range[0]) / finest
    s2 = torch.stack([ck.axis_u.norm(dim=-1), ck.axis_v.norm(dim=-1)], 1)   # 1-sigma
    nyq = 2.0 * (2.0 * uv_extent * s2 / cell)                               # samples/axis
    g = (2.0 ** torch.ceil(torch.log2(nyq.clamp(min=1.0)))).int()
    return g.clamp(min=grid_min, max=grid_max), cell, nyq


def scatter_bake(ck, out_dir, tex_res=2048, grid=0, use_oct=False, patch_px=12.0,
                 save=True, grid_min=8, grid_max=128, pt_budget=4_000_000,
                 oct_aniso=True, oct_contracted=False):
    """Bake T by SCATTERING teacher content through the PROBES (the same map
    fragments read with). Read/write consistency by construction: phi's cycle
    error affects only layout/collisions, never value correctness. For each
    surfel, a grid of uv points on its disc writes teacher(3D point) into
    T[A.uv + t] with kernel + bilinear weights; overlaps average.

    `grid=0` (default) sizes the sample grid per surfel by `nyquist_grid` so no
    teacher detail is aliased away. `grid=N` forces the old uniform N x N.

    NOTE this changes only SAMPLING, not STORAGE: samples still land in a
    ~patch_px footprint of the shared tex_res^2 atlas, so denser sampling buys
    a correct prefilter (band-limited downsample), not more stored detail.
    """
    if use_oct:
        pr = oct_probes(ck, tex_res, patch_px, aniso=oct_aniso, contracted=oct_contracted)
        torch.save({'probes': pr.cpu(), 'centers': ck.center.cpu()},
                   os.path.join(out_dir, 'probes.pt'))
        print('[SCATTER] using ANALYTIC OCT probes (phi bypassed); probes.pt overwritten')
    else:
        pr = torch.load(os.path.join(out_dir, 'probes.pt'), map_location=DEV)
        pr = pr['probes'] if isinstance(pr, dict) else pr
    A = pr[:, :4].view(-1, 2, 2)
    t0 = pr[:, 4:6]
    num = torch.zeros(tex_res * tex_res, 3, device=DEV)
    den = torch.zeros(tex_res * tex_res, device=DEV)

    if grid > 0:
        g = torch.full((ck.N, 2), int(grid), dtype=torch.int32, device=DEV)
        print(f'[SCATTER] uniform grid {grid}x{grid} (Nyquist sizing DISABLED)')
    else:
        g, cell, nyq = nyquist_grid(ck, 3.0, grid_min, grid_max)
        clamped = (nyq > grid_max).any(1).float().mean().item()
        q = torch.tensor([0.5, 0.9, 0.99], device=DEV)
        p50, p90, p99 = torch.quantile(nyq.flatten().float(), q).tolist()
        print(f'[SCATTER] Nyquist sampling: cell_size={cell:.6f}  required/axis '
              f'p50={p50:.1f} p90={p90:.1f} p99={p99:.1f}  '
              f'cap={grid_max} -> {clamped*100:.1f}% clamped')

    # Group by (gx, gy) so each group shares one uv grid; batch by POINT count
    # (a 128x128 group is 16384 samples/surfel — batching by surfel would OOM).
    key = g[:, 0].long() * 100000 + g[:, 1].long()
    total_pts = 0
    for k in key.unique().tolist():
        sel = (key == k).nonzero(as_tuple=True)[0]
        gx, gy = k // 100000, k % 100000
        lu = torch.linspace(-3.0, 3.0, gx, device=DEV)
        lv = torch.linspace(-3.0, 3.0, gy, device=DEV)
        gvv, guu = torch.meshgrid(lv, lu, indexing='ij')
        uvg = torch.stack([guu, gvv], -1).view(-1, 2)               # [G,2] sigma units
        G = uvg.shape[0]
        # Per-sample AREA element: makes the num/den average invariant to grid
        # density, so a 128x128 surfel does not outvote an 8x8 one 256:1 where
        # they collide. (Constant, hence a no-op, in the old uniform path.)
        dA = (6.0 / gx) * (6.0 / gy)
        w_k = torch.exp(-0.5 * (uvg ** 2).sum(-1)) * dA             # gaussian falloff
        total_pts += sel.numel() * G
        B = max(1, pt_budget // G)
        for i in range(0, sel.numel(), B):
            gi = sel[i:i + B]
            b = gi.numel()
            x3 = (ck.center[gi, None, :] + uvg[None, :, 0:1] * ck.axis_u[gi, None, :]
                  + uvg[None, :, 1:2] * ck.axis_v[gi, None, :]).view(-1, 3)
            val = ck.residual(x3)                                   # [b*G,3]
            tc = torch.einsum('bij,gj->bgi', A[gi], uvg) + t0[gi, None, :]
            x = (tc[..., 0] - 0.5).clamp(0, tex_res - 2).view(-1)
            y = (tc[..., 1] - 0.5).clamp(0, tex_res - 2).view(-1)
            x0f, y0f = x.floor(), y.floor()
            fx, fy = x - x0f, y - y0f
            x0, y0 = x0f.long(), y0f.long()
            wk = (w_k * ck.opacity[gi].view(b, 1)).view(-1)
            for dx, dy, wb in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                               (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                idx = (y0 + dy) * tex_res + (x0 + dx)
                w = wb * wk
                num.index_add_(0, idx, val * w[:, None])
                den.index_add_(0, idx, w)
    print(f'[SCATTER] {total_pts/1e6:.1f}M teacher samples over '
          f'{key.unique().numel()} (gx,gy) groups')
    T = torch.where(den[:, None] > 1e-6, num / den.clamp_min(1e-6)[:, None],
                    torch.zeros_like(num)).view(tex_res, tex_res, 3)
    cov = (den > 1e-6).float().mean().item()
    if not save:
        print(f'[SCATTER quick] coverage={cov:.1%}')
        return T
    torch.save(T.cpu(), os.path.join(out_dir, 'tex_init.pt'))
    print(f'[SCATTER] T range [{T.min():.3f},{T.max():.3f}] std={T.std():.3f} '
          f'texel coverage={cov:.1%}  -> tex_init.pt overwritten')
    _pr_final = torch.load(os.path.join(out_dir, 'probes.pt'), map_location=DEV)
    _pr_final = _pr_final['probes'] if isinstance(_pr_final, dict) else _pr_final
    render_check(ck, _pr_final.to(DEV), T, out_dir, tag='final')
    try:
        from PIL import Image
        Image.fromarray(((T + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()).save(
            os.path.join(out_dir, 'tex_init_preview.png'))
    except Exception:
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['train', 'bake', 'scatter', 'all'])
    ap.add_argument('-m', '--model_path', required=True)
    ap.add_argument('--iteration', type=int, default=-1)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--tex_res', type=int, default=2048)
    ap.add_argument('--patch_px', type=float, default=12.0)
    ap.add_argument('--oct', action='store_true', help='scatter: analytic octahedral probes instead of phi probes')
    ap.add_argument('--oct_contracted', action='store_true',
                    help='scatter --oct: size footprints by CONTRACTED-space surfel scale '
                         '(tangential factor (2-1/r)/r for r>1) instead of world scale — '
                         'matches the hash metric the teacher residual is band-limited by; '
                         'deflates far-field footprints on unbounded scenes, no-op inside the unit ball')
    ap.add_argument('--oct_iso', action='store_true',
                    help='scatter --oct: isotropic rho (pre-2026-07-31 similarity probes) '
                         'instead of per-axis rho_u/rho_v; same total texels either way')
    ap.add_argument('--out_tag', type=str, default='uv_field',
                    help='output subdir under <model_path> (lets variants coexist)')
    ap.add_argument('--w_cov', type=float, default=0.0, help='phi-range coverage loss weight (0 = off, pre-coverage behavior)')
    ap.add_argument('--seed', type=int, default=0,
                    help='RNG seed for phi/phi_inv init + sampling (verify collapse-axis reproducibility)')
    ap.add_argument('--scatter_grid', type=int, default=0,
                    help='scatter: uv samples/axis per surfel. 0 (default) = per-surfel '
                         'Nyquist vs the finest hash level; >0 forces a uniform grid (old behavior was 24)')
    ap.add_argument('--scatter_grid_max', type=int, default=128,
                    help='scatter: cap on the Nyquist sample grid (128 leaves ~1.4%% of chair surfels clamped)')
    ap.add_argument('--scatter_grid_min', type=int, default=8,
                    help='scatter: floor on the Nyquist sample grid')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ck = Ckpt(a.model_path, a.iteration)
    out = os.path.join(a.model_path, a.out_tag)
    if a.stage in ('train', 'all'):
        train(ck, out, steps=a.steps, tex_res=a.tex_res, patch_px=a.patch_px, w_cov=a.w_cov)
    if a.stage in ('bake', 'all'):
        bake(ck, out, tex_res=a.tex_res)
    if a.stage in ('scatter', 'all'):
        scatter_bake(ck, out, tex_res=a.tex_res, use_oct=a.oct, patch_px=a.patch_px,
                     grid=a.scatter_grid, grid_min=a.scatter_grid_min,
                     grid_max=a.scatter_grid_max, oct_aniso=not a.oct_iso,
                     oct_contracted=a.oct_contracted)

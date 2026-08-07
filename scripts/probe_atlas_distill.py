#!/usr/bin/env python3
"""Probe-atlas distillation for a trained 3D_SH_filmres checkpoint.

Distills the scene residual teacher
    residual_i(u,v) = MLP( gamma_eff * H(xyz_i(u,v)) + beta_i ),   xyz_i(u,v) = c_i + u*sx_i*R_i[:,0] + v*sy_i*R_i[:,1]
(gamma_eff = --lock_gamma = 1.0 for the target run; identity film_act; signed residual, NO outer
ReLU — the mode-2 convention) into ONE shared learnable RGB image + per-surfel probes (5-param
affine maps into the image), optimized purely in PyTorch (no rasterizer in the loop).

Subcommands (all state under <model_path>/probe_atlas/):
    verify  — end-to-end teacher check against the REAL diff_surfel_3D_sh_filmres CUDA renderer:
              render isolated checkpoint surfels twice (full vs hash+beta zeroed, residual_mode 2)
              so (C_A - C_B)/alpha == CUDA residual per pixel; compare vs the Python teacher at
              analytically ray-traced uv. Writes verify.json.
    bake    — teacher textures at 16x16 uv per surfel (fp16) -> teacher_tex.pt, plus 8x8
              descriptors [N,192] -> desc.pt.
    pack    — GPU k-means (K=(img_res/patch)^2, L2 on descriptors) -> assignment + centroids;
              atlas image init from upsampled cluster-mean textures; identity probe init.
              Reports cluster histogram + init distill PSNR. -> pack.pt
    train   — Adam distillation of image (lr 1e-2) + probes (lr 1e-3) with kernel-weighted L1
              and mip annealing (level 5 -> 0). Restartable from train_ckpt.pt. -> train_final.pt
    export  — final PNG + u8 .pt + probes .pt, per-surfel PSNR distribution, init-vs-final table,
              storage comparison, 8 preview crops (teacher|init|final).
    all     — verify -> bake -> pack -> train -> export.

`--texture neural` (variant): the atlas CONTENT is a pure-PyTorch 2D multires-hash + MLP
neural texture T(p), p in [0,1]^2, instead of a learnable image. NO k-means/packing: probes
init on a regular G x G grid (G = ceil(sqrt(N)), non-overlapping cells) and field + probes
co-train from scratch under the same kernel-weighted L1 distillation loss (no mip pyramid —
the coarse hash levels provide long-range gradients natively; optional coarse-to-fine on
hash levels via --c2f_interval). Export bakes T on the pixel grid to a plain image and
reports BOTH field-PSNR and baked-image-PSNR. State under probe_atlas/<--out_tag>/
(default v_neural_l8f2). The default --texture image path is untouched.

Usage:
    conda run -n nest_splatting python scripts/probe_atlas_distill.py all \
        -m outputs/mip_360/bicycle/3D_SH_filmres/b0_SV_30thr_005w25gLP_N2F_Jac
"""

import os
import sys
import json
import glob
import math
import time
import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV = 'cuda'
UV_EXTENT = 4.0          # bake texture domain: uv in [-4,4] sigma (texel centers)
BAKE_RES = 16            # teacher texture resolution per surfel
DESC_RES = 8             # descriptor resolution (downsampled bake)
KSQ = 9.0                # beta_scaled compact support: rho^2 < 9 (3 sigma)
UV_SAMPLE = 3.0          # training uv sample box (support region)
PSNR_CAP = 80.0


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def load_training_args(model_path):
    p = os.path.join(model_path, 'args.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    with open(os.path.join(model_path, 'args.json')) as f:
        return argparse.Namespace(**json.load(f))


def detect_iteration(model_path, iteration=-1):
    if iteration != -1:
        return iteration
    ngp = glob.glob(os.path.join(model_path, 'ngp_*.pth'))
    assert ngp, f'no ngp_*.pth in {model_path}'
    return max(int(os.path.basename(f)[4:-4]) for f in ngp)


class Run:
    """Checkpoint bundle: INGP (hash + fused MLP) + per-surfel tensors from the PLY."""

    def __init__(self, model_path, iteration=-1):
        from hash_encoder.config import Config
        from hash_encoder.modules import INGP

        self.model_path = model_path
        self.iteration = detect_iteration(model_path, iteration)
        self.args = load_training_args(model_path)
        cfg_yaml = os.path.join(model_path, 'config.yaml')
        self.cfg = Config(cfg_yaml if os.path.exists(cfg_yaml) else self.args.yaml)

        assert self.args.method == '3D_SH_filmres', f'method={self.args.method} (expected 3D_SH_filmres)'
        self.lock_gamma = getattr(self.args, 'lock_gamma', None)
        self.film_act = getattr(self.args, 'film_act', 'identity')
        assert self.film_act == 'identity', f'film_act={self.film_act} not supported by this teacher'

        self.ingp = INGP(self.cfg, args=self.args).to(DEV)
        self.ingp.load_model(model_path, self.iteration)
        self.ingp.set_active_levels(self.iteration)
        self.ingp.eval()
        assert self.ingp.active_hashgrid_levels == self.ingp.hashgrid_levels, 'c2f not fully active?'
        self.hash_dim = self.ingp.hashgrid_levels * self.ingp.level_dim
        assert self.hash_dim <= 16

        # ---- PLY ----
        from plyfile import PlyData
        ply_path = os.path.join(model_path, 'point_cloud', f'iteration_{self.iteration}', 'point_cloud.ply')
        ply = PlyData.read(ply_path)
        el = ply.elements[0]
        names = {p.name for p in el.properties}
        N = len(el.data)

        def col(name):
            return torch.tensor(np.asarray(el.data[name], dtype=np.float32), device=DEV)

        self.center = torch.stack([col('x'), col('y'), col('z')], dim=1)                    # [N,3]
        self.scale = torch.stack([col('scale_0'), col('scale_1')], dim=1).exp()             # [N,2] world sigma
        quat = torch.stack([col('rot_0'), col('rot_1'), col('rot_2'), col('rot_3')], dim=1)  # w-first
        from utils.general_utils import build_rotation
        self.R = build_rotation(quat)                                                       # [N,3,3]
        self.axis_u = self.R[:, :, 0] * self.scale[:, 0:1]                                  # sx * R[:,0]
        self.axis_v = self.R[:, :, 1] * self.scale[:, 1:2]                                  # sy * R[:,1]
        self.quat_raw = quat

        fp = torch.stack([col(f'film_{i}') for i in range(25)], dim=1)                      # [N,25]
        self.film_params = fp
        # CUDA stages beta in shared memory as FP16 — mirror that quantization exactly.
        self.beta = fp[:, 1:1 + self.hash_dim].half().float().contiguous()                  # [N,hash_dim]
        self.gamma_eff = float(self.lock_gamma) if self.lock_gamma is not None else None
        assert self.gamma_eff is not None, 'teacher assumes --lock_gamma (gamma_eff constant)'

        assert 'shape' in names, 'no beta-kernel shape column in PLY'
        self.kernel = getattr(self.args, 'kernel', 'gaussian')
        self.shape_raw = col('shape').view(-1, 1)
        self.kbeta = torch.sigmoid(self.shape_raw) * 5.0                                    # activated beta_scaled shape
        self.opacity = torch.sigmoid(col('opacity')).view(-1, 1)
        self.N = N

        # fused MLP in fp32 (bias-free 16->16->16->16, rows 0..2 of the output = RGB residual)
        self.mlp = self.ingp.mlp_fused.float()

        self.out_dir = os.path.join(model_path, 'probe_atlas')
        os.makedirs(self.out_dir, exist_ok=True)

        print(f'[RUN] {model_path} iter={self.iteration} N={N:,} hash_dim={self.hash_dim} '
              f'levels={self.ingp.resolutions} gamma_eff={self.gamma_eff} kernel={self.kernel} '
              f'voxel_range={self.ingp.voxel_range} contract={self.ingp.contract}')

    # ---- teacher ----
    @torch.no_grad()
    def teacher_xyz(self, xyz, beta_rows):
        """residual RGB at world points xyz [M,3] with per-point beta [M,hash_dim]."""
        H = self.ingp(points_3D=xyz).float()[:, :self.hash_dim]
        mlp_in = torch.zeros(xyz.shape[0], 16, device=DEV)
        mlp_in[:, :self.hash_dim] = self.gamma_eff * H + beta_rows
        return self.mlp(mlp_in)[:, :3]

    @torch.no_grad()
    def teacher_uv(self, sid, uv):
        """residual RGB for surfels sid [B] at local uv [B,S,2] (sigma units). Returns [B,S,3]."""
        B, S = uv.shape[0], uv.shape[1]
        xyz = (self.center[sid, None, :]
               + uv[..., 0:1] * self.axis_u[sid, None, :]
               + uv[..., 1:2] * self.axis_v[sid, None, :]).reshape(-1, 3)
        beta_rows = self.beta[sid].repeat_interleave(S, dim=0)
        return self.teacher_xyz(xyz, beta_rows).view(B, S, 3)

    def kernel_weight(self, sid, uv):
        """beta_scaled falloff weight = clip(1 - rho^2/9, 0)^kbeta  [B,S]."""
        rho2 = (uv ** 2).sum(-1)
        base = (1.0 - rho2 / KSQ).clamp_min(0.0)
        return base ** self.kbeta[sid]


def bake_uv_grid(res=BAKE_RES, extent=UV_EXTENT, device=DEV):
    """Texel-center uv grid [(res*res), 2], row-major (v outer, u inner)."""
    step = 2.0 * extent / res
    t = (torch.arange(res, device=device, dtype=torch.float32) + 0.5) * step - extent
    v, u = torch.meshgrid(t, t, indexing='ij')
    return torch.stack([u, v], dim=-1).view(-1, 2)


# --------------------------------------------------------------------------------------
# Probes / atlas sampling (shared by pack/train/export)
# --------------------------------------------------------------------------------------

def rect_center(assign, img_res, patch):
    """Continuous pixel-space center of each surfel's cluster rect. [N,2] (x,y)."""
    cells = img_res // patch
    row = (assign // cells).float()
    col = (assign % cells).float()
    # grid_sample(align_corners=False): px = ((g+1)*W - 1)/2 puts pixel i's center at px=i,
    # so rect pixels [r0 .. r0+patch-1] have continuous center r0 + (patch-1)/2 and the
    # identity probe (n=+-1 -> +-patch/2) spans exactly [r0-0.5, r0+patch-0.5] = the rect.
    cx = col * patch + (patch - 1) / 2.0
    cy = row * patch + (patch - 1) / 2.0
    return torch.stack([cx, cy], dim=1)


def probe_grid_coords(probes, centers, uv, img_res, patch):
    """Map local uv [B,S,2] through per-surfel probes [B,5] -> grid_sample coords [B,S,2].

    probe params: (du, dv, log_su, log_sv, theta). Identity (all zero) maps uv [-4,4]
    exactly onto the surfel's patch^2 rect.
    """
    du, dv = probes[:, 0:1], probes[:, 1:2]
    su, sv = probes[:, 2:3].exp(), probes[:, 3:4].exp()
    th = probes[:, 4:5]
    half = patch / 2.0
    n = uv / UV_EXTENT                                     # [-1,1]
    qx = n[..., 0] * su * half
    qy = n[..., 1] * sv * half
    c, s = torch.cos(th), torch.sin(th)
    rx = c * qx - s * qy
    ry = s * qx + c * qy
    px = centers[:, 0:1] + rx + du
    py = centers[:, 1:2] + ry + dv
    gx = (2.0 * px + 1.0) / img_res - 1.0
    gy = (2.0 * py + 1.0) / img_res - 1.0
    return torch.stack([gx, gy], dim=-1)


def sample_atlas(image, grid):
    """image [3,R,R] fp32, grid [B,S,2] -> [B,S,3]."""
    out = F.grid_sample(image[None], grid[None], mode='bilinear',
                        padding_mode='border', align_corners=False)
    return out[0].permute(1, 2, 0)


# --------------------------------------------------------------------------------------
# Neural texture (--texture neural): 2D multires hash encoder + MLP over p in [0,1]^2
# --------------------------------------------------------------------------------------

class NeuralTexture2D(torch.nn.Module):
    """Instant-NGP-style 2D hash field: L levels, geometric resolutions base->img_res,
    F features/level (concat), bilinear interp per level, dense grids while they fit the
    table, spatial hash (primes XOR) beyond. Decoder: L*F -> 32 -> 32 -> 3 (ReLU,ReLU,lin)."""

    PRIME = 2654435761

    def __init__(self, img_res=4096, levels=8, feat=2, log2_table=19, base=16, hidden=32):
        super().__init__()
        self.img_res, self.levels, self.feat = img_res, levels, feat
        self.log2_table, self.base, self.hidden = log2_table, base, hidden
        self.table_size = 2 ** log2_table
        g = (img_res / base) ** (1.0 / max(levels - 1, 1))
        self.res = [int(round(base * g ** l)) for l in range(levels)]
        self.dense, sizes = [], []
        for r in self.res:
            nv = (r + 1) ** 2
            self.dense.append(nv <= self.table_size)
            sizes.append(nv if nv <= self.table_size else self.table_size)
        self.sizes = sizes
        self.offsets = np.cumsum([0] + sizes).tolist()
        self.tables = torch.nn.Parameter((torch.rand(self.offsets[-1], feat) * 2 - 1) * 1e-4)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(levels * feat, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 3))

    def describe(self):
        return (f'levels={self.levels} feat={self.feat} res={self.res} '
                f'dense={self.dense} table=2^{self.log2_table} hidden={self.hidden} '
                f'entries={self.offsets[-1]:,} ({self.tables.numel():,} params, '
                f'{self.tables.numel() * 4 / 1e6:.1f} MB fp32)')

    def encode(self, p, active_levels=None):
        """p [M,2] in [0,1] -> [M, L*F]. Levels >= active_levels contribute zeros (c2f)."""
        M = p.shape[0]
        act = self.levels if active_levels is None else min(active_levels, self.levels)
        feats = []
        for l in range(self.levels):
            if l >= act:
                feats.append(torch.zeros(M, self.feat, device=p.device, dtype=p.dtype))
                continue
            r = self.res[l]
            x = (p * r).clamp(0, r - 1e-4)
            i0 = x.floor()
            f = x - i0
            i0 = i0.long()
            acc = None
            for dx in (0, 1):
                for dy in (0, 1):
                    ix = i0[:, 0] + dx
                    iy = i0[:, 1] + dy
                    if self.dense[l]:
                        idx = iy * (r + 1) + ix
                    else:
                        idx = torch.bitwise_xor(ix, iy * self.PRIME) % self.table_size
                    wx = f[:, 0] if dx else 1.0 - f[:, 0]
                    wy = f[:, 1] if dy else 1.0 - f[:, 1]
                    term = (wx * wy)[:, None] * self.tables[self.offsets[l] + idx]
                    acc = term if acc is None else acc + term
            feats.append(acc)
        return torch.cat(feats, dim=-1)

    def forward(self, p, active_levels=None):
        return self.mlp(self.encode(p, active_levels))


class ProbeHead(torch.nn.Module):
    """--probe_mode latent: probes predicted by an MLP from a FROZEN appearance anchor
    F_i = [H3D(center_i) (16D teacher hash feature) | beta_i (16D)] (+ optional geometry)
    plus a small learnable per-surfel latent z_i (8D). The MLP head is zero-init so the
    step-0 probe distribution is IDENTICAL to free mode's regular-grid init (residuals
    around the same grid base). Optional per-surfel free residual on the 5 params."""

    def __init__(self, run, geom_inputs=False, latent_dim=8, free_residual=True, geom_pe=0):
        super().__init__()
        with torch.no_grad():
            H = run.ingp(points_3D=run.center).float()[:, :run.hash_dim]      # teacher hash @ centers
            F_anchor = torch.cat([H, run.beta], dim=1)                        # [N,32]
            if geom_inputs:
                if geom_pe > 0:
                    # Fourier features replace the raw geometry (log-scales normalized to
                    # ~[-1,1] by /6 first): sin/cos(2^k pi c), k=0..L-1  -> 8*2*L D
                    geom = torch.cat([run.scale.log() / 6.0,
                                      run.R[:, :, 0], run.R[:, :, 1]], dim=1)  # 8 components
                    pe = []
                    for k in range(geom_pe):
                        ang = (2.0 ** k) * math.pi * geom
                        pe += [torch.sin(ang), torch.cos(ang)]
                    geom = torch.cat(pe, dim=1)
                else:
                    geom = torch.cat([run.scale.log(),
                                      run.R[:, :, 0], run.R[:, :, 1]], dim=1)  # raw 8D (unchanged)
                F_anchor = torch.cat([F_anchor, geom], dim=1)
        self.register_buffer('anchor', F_anchor.contiguous())
        self.latent_dim = latent_dim
        # latent_dim == 0: no learnable latent at all (anchor-only probe MLP)
        self.z = torch.nn.Parameter(torch.randn(run.N, latent_dim) * 0.01) if latent_dim > 0 else None
        in_dim = F_anchor.shape[1] + latent_dim
        hidden = 128 if geom_pe > 0 else 64
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 5))
        torch.nn.init.zeros_(self.mlp[-1].weight)                             # step 0 == grid init
        torch.nn.init.zeros_(self.mlp[-1].bias)
        self.resid = torch.nn.Parameter(torch.zeros(run.N, 5)) if free_residual else None

    def probes(self, sid=None):
        a = self.anchor if sid is None else self.anchor[sid]
        if self.z is not None:
            z = self.z if sid is None else self.z[sid]
            a = torch.cat([a, z], dim=1)
        out = self.mlp(a)
        if self.resid is not None:
            out = out + (self.resid if sid is None else self.resid[sid])
        return out

    def param_groups(self, cli):
        g = [{'params': self.mlp.parameters(), 'lr': 1e-3}]
        if self.z is not None:
            g.append({'params': [self.z], 'lr': 1e-2})
        if self.resid is not None:
            g.append({'params': [self.resid], 'lr': 1e-4})
        return g


def grid_probe_centers(N, img_res):
    """Regular-grid probe init (no k-means): surfel i -> cell (i mod G, i // G),
    G = ceil(sqrt(N)). Returns (centers [N,2] continuous px, cell_size float)."""
    G = math.ceil(math.sqrt(N))
    cell = img_res / G
    i = torch.arange(N, device=DEV, dtype=torch.float32)
    col = torch.remainder(i, G)
    row = torch.div(i, G, rounding_mode='floor')
    cx = (col + 0.5) * cell - 0.5
    cy = (row + 0.5) * cell - 0.5
    return torch.stack([cx, cy], dim=1), cell


@torch.no_grad()
def eval_wpsnr_fn(run, student_fn, probes, centers, img_res, patch, sids, chunk=4096):
    """Kernel-weighted distill PSNR (peak=1.0) over the bake uv grid for surfels sids.
    student_fn(grid [B,S,2] in [-1,1]) -> [B,S,3]. Returns (overall, per_surfel [len(sids)])."""
    grid_uv = bake_uv_grid()
    per = []
    se_sum, w_sum = 0.0, 0.0
    for i in range(0, len(sids), chunk):
        sid = sids[i:i + chunk]
        uv = grid_uv[None].expand(len(sid), -1, -1)
        t = run.teacher_uv(sid, uv)
        g = probe_grid_coords(probes[sid], centers[sid], uv, img_res, patch)
        s = student_fn(g)
        w = run.kernel_weight(sid, uv)
        se = ((t - s) ** 2).mean(-1) * w
        mse_i = se.sum(1) / w.sum(1).clamp_min(1e-8)
        per.append((10.0 * torch.log10(1.0 / mse_i.clamp_min(1e-12))).clamp(max=PSNR_CAP))
        se_sum += se.sum().item()
        w_sum += w.sum().item()
    overall = 10.0 * math.log10(1.0 / max(se_sum / max(w_sum, 1e-8), 1e-12))
    return min(overall, PSNR_CAP), torch.cat(per)


def eval_wpsnr(run, image, probes, centers, img_res, patch, sids, chunk=4096):
    """Image-atlas convenience wrapper (identical protocol to eval_wpsnr_fn)."""
    return eval_wpsnr_fn(run, lambda g: sample_atlas(image, g),
                         probes, centers, img_res, patch, sids, chunk)


def field_sampler(field, active_levels=None):
    """grid_sample-convention sampler over the neural field: grid [-1,1] -> p [0,1]."""
    def fn(grid):
        B, S = grid.shape[0], grid.shape[1]
        p = ((grid.reshape(-1, 2) + 1.0) * 0.5).clamp(0.0, 1.0)
        return field(p, active_levels).view(B, S, 3)
    return fn


def image_bilinear_sampler(field, img_res, active_levels=None, pixels=None):
    """Deployment-faithful student (--student image): the value at a query is the
    BILINEAR interpolation of the baked image's 4 surrounding texel centers, with the
    texels generated lazily from the field (the full image is never materialized).
    Matches grid_sample(align_corners=False, padding_mode='border') semantics, i.e.
    exactly what GPU texture hardware does with the exported image. Gradients flow to
    the field through the texel values AND to the probes through the bilinear weights'
    dependence on the query position (piecewise-linear, like grid_sample)."""
    def fn(grid):
        B, S = grid.shape[0], grid.shape[1]
        px = (grid.reshape(-1, 2) + 1.0) * (img_res / 2.0) - 0.5      # pixel space, texel i center at px=i
        x0f = torch.floor(px)
        f = px - x0f                                                   # frac in [0,1); d f/d px = 1
        x0 = x0f.long()
        nx = torch.stack([x0[:, 0], x0[:, 0] + 1, x0[:, 0], x0[:, 0] + 1], 1).clamp_(0, img_res - 1)
        ny = torch.stack([x0[:, 1], x0[:, 1], x0[:, 1] + 1, x0[:, 1] + 1], 1).clamp_(0, img_res - 1)
        flat = (ny * img_res + nx).view(-1)                            # [4M]
        uniq, inv = torch.unique(flat, return_inverse=True)            # dedup shared texels
        pt = torch.stack([(uniq % img_res).float(), torch.div(uniq, img_res, rounding_mode='floor').float()], 1)
        texel = field((pt + 0.5) / img_res, active_levels)             # [U,3] lazy bake at texel centers
        if pixels is not None:                                         # hybrid: + explicit residual image
            texel = texel + pixels.view(-1, 3)[uniq]
        v = texel[inv].view(-1, 4, 3)
        wx = torch.stack([1 - f[:, 0], f[:, 0], 1 - f[:, 0], f[:, 0]], 1)
        wy = torch.stack([1 - f[:, 1], 1 - f[:, 1], f[:, 1], f[:, 1]], 1)
        return (v * (wx * wy).unsqueeze(-1)).sum(1).view(B, S, 3)
    return fn


def sample_uv(B, S, gen, mode='uniform'):
    """Training uv samples over [-UV_SAMPLE, UV_SAMPLE]^2. 'stratified' = jittered
    sqrt(S) x sqrt(S) grid per surfel: guaranteed footprint coverage per draw AND
    unbiased (uniform jitter within each stratum) — no lattice aliasing. Falls back
    to uniform when S is not a perfect square."""
    R = int(math.isqrt(S))
    if mode != 'stratified' or R * R != S:
        return (torch.rand(B, S, 2, device=DEV, generator=gen) * 2.0 - 1.0) * UV_SAMPLE
    j = torch.rand(B, R, R, 2, device=DEV, generator=gen)
    iy, ix = torch.meshgrid(torch.arange(R, device=DEV, dtype=torch.float32),
                            torch.arange(R, device=DEV, dtype=torch.float32), indexing='ij')
    uv01 = (torch.stack([ix, iy], -1) + j) / R
    return (uv01.view(B, S, 2) * 2.0 - 1.0) * UV_SAMPLE


# --------------------------------------------------------------------------------------
# verify — teacher vs the real CUDA renderer
# --------------------------------------------------------------------------------------

def _build_camera(C, Rc2w, W=400, H=400, fov=0.8):
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    Rn = Rc2w.cpu().numpy().astype(np.float64)
    t = (-Rn.T @ C.cpu().numpy().astype(np.float64))
    wvt = torch.tensor(getWorld2View2(Rn, t), dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(0.01, 100.0, fov, fov).transpose(0, 1).cuda()
    fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    cc = wvt.inverse()[3, :3]
    return argparse.Namespace(FoVx=fov, FoVy=fov, image_width=W, image_height=H,
                              world_view_transform=wvt, full_proj_transform=fpt,
                              camera_center=cc, znear=0.01, zfar=100.0, clip_plane=None)


def _single_surfel_model(run, i):
    """Tiny GaussianModel holding checkpoint surfel i alone (SH base, constant dc)."""
    from scene.gaussian_model import GaussianModel
    g = GaussianModel(3)
    g.feature_mode = 'sh'
    g.kernel_type = run.kernel
    g.kernel_type2 = None
    g.active_sh_degree = 3
    g.max_sh_degree = 3
    g._xyz = torch.nn.Parameter(run.center[i:i + 1].clone())
    g._scaling = torch.nn.Parameter(run.scale[i:i + 1].log().clone())
    g._rotation = torch.nn.Parameter(run.quat_raw[i:i + 1].clone())
    g._opacity = torch.nn.Parameter(torch.full((1, 1), 8.0, device=DEV))   # sigmoid ~ 0.9997
    g._features_dc = torch.nn.Parameter(torch.full((1, 1, 3), 1.0, device=DEV))
    g._features_rest = torch.nn.Parameter(torch.zeros(1, 15, 3, device=DEV))
    g._appearance_level = torch.nn.Parameter(24.0 * torch.ones(1, 1, device=DEV), requires_grad=False)
    g._shape = torch.nn.Parameter(run.shape_raw[i:i + 1].clone())
    fp = run.film_params[i:i + 1].clone()
    if run.gamma_eff is not None:
        fp[:, 0] = run.gamma_eff
    g._film_params = torch.nn.Parameter(fp)
    g.max_radii2D = torch.zeros(1, device=DEV)
    return g


def cmd_verify(run, cli):
    """Per-pixel CUDA-vs-Python teacher comparison on isolated real surfels."""
    from gaussian_renderer import render
    import diff_surfel_3D_sh_filmres as FM

    t0 = time.time()
    FM.set_residual_mode(2)                   # signed residual, no per-Gauss ReLU -> exact subtraction
    ab = getattr(run.args, 'activation_bias', [0.5, 0.0])
    FM.set_activation_bias(float(ab[0]), float(ab[1]))
    FM.set_lru_slope(0.0)
    FM.set_film_gamma_act(0)                  # identity
    FM.set_film_lock_gamma(run.gamma_eff)

    # pick 3 surfels: opaque, decently sized, with nonzero beta (so the residual has signal)
    smax = run.scale.max(1).values
    q80 = torch.quantile(smax, 0.80)
    q98 = torch.quantile(smax, 0.98)
    cand = ((run.opacity[:, 0] > 0.9) & (smax > q80) & (smax < q98)
            & (run.beta.norm(dim=1) > run.beta.norm(dim=1).median()))
    ids = torch.nonzero(cand)[:, 0]
    g_ = torch.Generator(device='cpu').manual_seed(1234)
    pick = ids[torch.randperm(len(ids), generator=g_)[:3]].tolist()
    print(f'[VERIFY] surfels {pick} (of {int(cand.sum())} candidates)')

    pipe = argparse.Namespace(debug=False, skip_aux_normal_dist=True, compute_cov3D_python=False,
                              convert_SHs_python=False, depth_ratio=0.0)
    bg = torch.zeros(3, device=DEV)
    W = H = 400
    fov = 0.8
    results = []
    all_ok = True

    for i in pick:
        g = _single_surfel_model(run, i)
        pk = run.center[i]
        R0 = run.R[i, :, 0]
        R1 = run.R[i, :, 1]
        n = run.R[i, :, 2]
        sx, sy = run.scale[i, 0].item(), run.scale[i, 1].item()
        d = 12.0 * max(sx, sy)
        C = pk - d * n                                       # camera behind the surfel, looking along +n
        Rc2w = torch.stack([R0, R1, n], dim=1)               # cam axes: x=u, y=v, z=normal
        cam = _build_camera(C, Rc2w, W, H, fov)

        def do_render():
            with torch.no_grad():
                pkg = render(cam, g, pipe, bg, ingp=run.ingp, iteration=run.iteration,
                             cfg=run.cfg, lowpass=bool(getattr(run.args, 'lowpass', False)),
                             is_training=True)
            return pkg['render'].clone(), pkg['rend_alpha'].clone()

        img_a, alpha = do_render()
        emb = run.ingp.hash_encoding.embeddings
        with torch.no_grad():
            emb_saved = emb.data.clone()
            fp_saved = g._film_params.data.clone()
            emb.data.zero_()
            g._film_params.data[:, 1:] = 0.0
        img_b, _ = do_render()
        with torch.no_grad():
            emb.data.copy_(emb_saved)
            g._film_params.data.copy_(fp_saved)

        # analytic per-pixel uv via ray-plane intersection (matches the CUDA ray-splat math)
        iy, ix = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                                torch.arange(W, device=DEV, dtype=torch.float32), indexing='ij')
        tx = math.tan(fov / 2)
        xn = (2.0 * ix + 1.0) / W - 1.0
        yn = (2.0 * iy + 1.0) / H - 1.0
        dir_cam = torch.stack([xn * tx, yn * tx, torch.ones_like(xn)], dim=-1)   # [H,W,3]
        dir_w = dir_cam @ Rc2w.T
        denom = (dir_w * n).sum(-1)
        tpar = ((pk - C) * n).sum() / denom
        X = C + tpar[..., None] * dir_w
        u = ((X - pk) * R0).sum(-1) / sx
        v = ((X - pk) * R1).sum(-1) / sy
        rho2 = u * u + v * v

        mask = (alpha[0] > 0.5) & (rho2 < 8.0)               # inside support, away from the edge
        npix = int(mask.sum())
        res_cuda = ((img_a - img_b) / alpha.clamp_min(1e-6)).permute(1, 2, 0)[mask]   # [M,3]
        xyz = X[mask]
        beta_rows = run.beta[i][None].expand(xyz.shape[0], -1)
        res_py = run.teacher_xyz(xyz, beta_rows)

        diff = (res_cuda - res_py).abs()
        mag = res_py.abs()
        rel = diff / (mag + 1e-2)
        cos = F.cosine_similarity(res_cuda.reshape(1, -1), res_py.reshape(1, -1)).item()
        # spatial-variation check of the teacher on the 16x16 grid
        tex = run.teacher_uv(torch.tensor([i], device=DEV),
                             bake_uv_grid()[None]).view(BAKE_RES, BAKE_RES, 3)
        r = dict(surfel=i, n_pixels=npix,
                 teacher_min=tex.min().item(), teacher_max=tex.max().item(),
                 teacher_std_spatial=tex.std(dim=(0, 1)).mean().item(),
                 abs_err_mean=diff.mean().item(), abs_err_p95=diff.quantile(0.95).item(),
                 abs_err_max=diff.max().item(),
                 rel_err_median=rel.median().item(), rel_err_p95=rel.quantile(0.95).item(),
                 res_rms_cuda=res_cuda.pow(2).mean().sqrt().item(),
                 res_rms_teacher=res_py.pow(2).mean().sqrt().item(), cosine=cos)
        ok = (npix > 1000 and r['abs_err_p95'] < 0.02 and cos > 0.995
              and r['teacher_std_spatial'] > 1e-4)
        r['pass'] = bool(ok)
        all_ok &= ok
        results.append(r)
        print(f"  surfel {i}: {npix}px  teacher[min/max/std]={r['teacher_min']:+.4f}/"
              f"{r['teacher_max']:+.4f}/{r['teacher_std_spatial']:.4f}  "
              f"abs_err[mean/p95/max]={r['abs_err_mean']:.2e}/{r['abs_err_p95']:.2e}/"
              f"{r['abs_err_max']:.2e}  rel[med/p95]={r['rel_err_median']:.3f}/{r['rel_err_p95']:.3f}  "
              f"rms(cuda/teacher)={r['res_rms_cuda']:.4f}/{r['res_rms_teacher']:.4f}  "
              f"cos={cos:.6f}  [{'OK' if ok else 'FAIL'}]")

    out = dict(iteration=run.iteration, surfels=results, all_pass=bool(all_ok),
               wall_s=time.time() - t0)
    with open(os.path.join(run.out_dir, 'verify.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[VERIFY] {'PASS' if all_ok else 'FAIL'}  ({out['wall_s']:.1f}s)  -> verify.json")
    if not all_ok:
        raise SystemExit('TEACHER VERIFICATION FAILED — do not train on this teacher. '
                         'See verify.json / stdout for the discrepancy.')


# --------------------------------------------------------------------------------------
# bake
# --------------------------------------------------------------------------------------

def cmd_bake(run, cli):
    t0 = time.time()
    path_tex = os.path.join(run.out_dir, 'teacher_tex.pt')
    grid_uv = bake_uv_grid()
    tex = torch.empty(run.N, BAKE_RES * BAKE_RES, 3, dtype=torch.float16, device=DEV)
    chunk = 16384
    for i in range(0, run.N, chunk):
        sid = torch.arange(i, min(i + chunk, run.N), device=DEV)
        uv = grid_uv[None].expand(len(sid), -1, -1)
        tex[i:i + len(sid)] = run.teacher_uv(sid, uv).half()
        if (i // chunk) % 3 == 0:
            print(f'  bake {i + len(sid):,}/{run.N:,}', flush=True)
    tex = tex.view(run.N, BAKE_RES, BAKE_RES, 3)
    stats = dict(min=tex.min().item(), max=tex.max().item(),
                 mean=tex.float().mean().item(), std=tex.float().std().item(),
                 rms=tex.float().pow(2).mean().sqrt().item())
    torch.save({'tex': tex.cpu(), 'bake_res': BAKE_RES, 'uv_extent': UV_EXTENT,
                'iteration': run.iteration, 'stats': stats}, path_tex)

    # descriptors: 8x8 avg-pool, flatten [N,192]
    desc = F.avg_pool2d(tex.float().permute(0, 3, 1, 2), BAKE_RES // DESC_RES)
    desc = desc.permute(0, 2, 3, 1).reshape(run.N, -1).contiguous()
    torch.save({'desc': desc.cpu()}, os.path.join(run.out_dir, 'desc.pt'))
    print(f'[BAKE] N={run.N:,} tex 16x16 fp16 ({tex.numel() * 2 / 1e6:.0f} MB) '
          f'stats={stats}  desc [N,{desc.shape[1]}]  ({time.time() - t0:.1f}s)')


# --------------------------------------------------------------------------------------
# pack
# --------------------------------------------------------------------------------------

@torch.no_grad()
def gpu_kmeans(x, K, iters=25, seed=0, chunk=16384):
    """Plain Lloyd L2 k-means on GPU, chunked over rows. x [N,D] fp32. Returns (assign, centroids)."""
    N, D = x.shape
    gen = torch.Generator(device='cpu').manual_seed(seed)
    cent = x[torch.randperm(N, generator=gen)[:K].to(x.device)].clone()
    assign = torch.zeros(N, dtype=torch.long, device=x.device)
    for it in range(iters):
        c2 = (cent ** 2).sum(1)
        for i in range(0, N, chunk):
            xi = x[i:i + chunk]
            d = xi @ cent.T
            d = (xi ** 2).sum(1, keepdim=True) - 2 * d + c2[None]
            assign[i:i + xi.shape[0]] = d.argmin(1)
        new = torch.zeros_like(cent)
        cnt = torch.zeros(K, device=x.device)
        new.index_add_(0, assign, x)
        cnt.index_add_(0, assign, torch.ones(N, device=x.device))
        empty = cnt == 0
        n_empty = int(empty.sum())
        new[~empty] /= cnt[~empty, None]
        if n_empty > 0:  # respawn empty clusters at random points
            ridx = torch.randperm(N, generator=gen)[:n_empty].to(x.device)
            new[empty] = x[ridx]
        shift = (new - cent).norm(dim=1).mean().item()
        cent = new
        if it % 5 == 0 or it == iters - 1:
            print(f'  kmeans it {it}: mean shift {shift:.5f}, empty {n_empty}', flush=True)
    # final assignment
    c2 = (cent ** 2).sum(1)
    for i in range(0, N, chunk):
        xi = x[i:i + chunk]
        d = (xi ** 2).sum(1, keepdim=True) - 2 * (xi @ cent.T) + c2[None]
        assign[i:i + xi.shape[0]] = d.argmin(1)
    return assign, cent


def cmd_pack(run, cli):
    t0 = time.time()
    img_res, patch = cli.img_res, cli.patch
    K = (img_res // patch) ** 2
    tex = torch.load(os.path.join(run.out_dir, 'teacher_tex.pt'))['tex'].to(DEV)
    desc = torch.load(os.path.join(run.out_dir, 'desc.pt'))['desc'].to(DEV)

    print(f'[PACK] k-means K={K:,} on desc [{desc.shape[0]:,},{desc.shape[1]}]')
    assign, cent = gpu_kmeans(desc, K, iters=cli.kmeans_iters, seed=0)

    cnt = torch.bincount(assign, minlength=K)
    hist = torch.bincount(cnt.clamp(max=50), minlength=51)
    used = int((cnt > 0).sum())
    print(f'[PACK] clusters used {used:,}/{K:,}  size p50={cnt.float().median():.0f} '
          f'p90={cnt.float().quantile(0.9):.0f} max={cnt.max()}')
    print('       size histogram (0..20):', hist[:21].tolist())

    # cluster mean textures -> atlas init
    mean_tex = torch.zeros(K, BAKE_RES, BAKE_RES, 3, device=DEV)
    mean_tex.view(K, -1).index_add_(0, assign, tex.float().view(run.N, -1))
    mean_tex /= cnt.clamp_min(1)[:, None, None, None]
    up = F.interpolate(mean_tex.permute(0, 3, 1, 2), size=(patch, patch),
                       mode='bilinear', align_corners=False)         # [K,3,32,32]
    cells = img_res // patch
    image = up.view(cells, cells, 3, patch, patch).permute(2, 0, 3, 1, 4) \
              .reshape(3, img_res, img_res).contiguous()

    probes = torch.zeros(run.N, 5, device=DEV)                        # identity probes
    centers = rect_center(assign, img_res, patch)

    eval_ids = torch.arange(run.N, device=DEV)[torch.randperm(run.N,
                generator=torch.Generator().manual_seed(7))[:4096].to(DEV)]
    init_psnr, per = eval_wpsnr(run, image, probes, centers, img_res, patch, eval_ids)
    wd = per.sort().values[:max(1, len(per) // 10)].mean().item()
    print(f'[PACK] init distill PSNR (4096-surfel eval): overall={init_psnr:.2f} dB '
          f'worst-decile={wd:.2f} dB')

    torch.save({'image': image.half().cpu(), 'probes': probes.cpu(), 'assign': assign.cpu(),
                'centroids': cent.half().cpu(), 'img_res': img_res, 'patch': patch, 'K': K,
                'init_psnr': init_psnr, 'init_psnr_worst_decile': wd,
                'cluster_counts': cnt.cpu()},
               os.path.join(run.out_dir, 'pack.pt'))
    print(f'[PACK] saved pack.pt  ({time.time() - t0:.1f}s)')


# --------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------

def mip_level(step, steps):
    """5 -> 0 annealing; one level drop / 3k steps; last 5k at level 0 (for steps=20000)."""
    lvl = 5 - step // 3000
    return max(0, min(5, lvl))


def cmd_train(run, cli):
    t0 = time.time()
    pack = torch.load(os.path.join(run.out_dir, 'pack.pt'))
    img_res, patch = pack['img_res'], pack['patch']
    assign = pack['assign'].to(DEV)
    centers = rect_center(assign, img_res, patch)

    ckpt_path = os.path.join(run.out_dir, 'train_ckpt.pt')
    start = 0
    if os.path.exists(ckpt_path) and not cli.restart:
        ck = torch.load(ckpt_path)
        image = ck['image'].to(DEV).float().requires_grad_(True)
        probes = ck['probes'].to(DEV).float().requires_grad_(True)
        opt_img = torch.optim.Adam([image], lr=cli.lr_image)
        opt_prb = torch.optim.Adam([probes], lr=cli.lr_probes)
        opt_img.load_state_dict(ck['opt_img'])
        opt_prb.load_state_dict(ck['opt_prb'])
        start = ck['step']
        print(f'[TRAIN] resumed from step {start}')
    else:
        image = pack['image'].to(DEV).float().requires_grad_(True)
        probes = pack['probes'].to(DEV).float().requires_grad_(True)
        opt_img = torch.optim.Adam([image], lr=cli.lr_image)
        opt_prb = torch.optim.Adam([probes], lr=cli.lr_probes)

    eval_ids = torch.arange(run.N, device=DEV)[torch.randperm(run.N,
                generator=torch.Generator().manual_seed(7))[:2048].to(DEV)]
    log_path = os.path.join(run.out_dir, 'train_log.txt')
    logf = open(log_path, 'a')
    B, S = cli.batch_surfels, cli.uv_per_surfel
    steps = cli.steps
    gen = torch.Generator(device=DEV).manual_seed(1000 + start)

    for step in range(start, steps):
        m = mip_level(step, steps)
        sid = torch.randint(0, run.N, (B,), device=DEV, generator=gen)
        uv = (torch.rand(B, S, 2, device=DEV, generator=gen) * 2.0 - 1.0) * UV_SAMPLE
        with torch.no_grad():
            t = run.teacher_uv(sid, uv)
            w = run.kernel_weight(sid, uv)
        grid = probe_grid_coords(probes[sid], centers[sid], uv, img_res, patch)
        img_m = image if m == 0 else F.avg_pool2d(image[None], 2 ** m)[0]
        s = sample_atlas(img_m, grid)
        loss = ((t - s).abs().mean(-1) * w).sum() / w.sum().clamp_min(1e-8)
        opt_img.zero_grad(set_to_none=True)
        opt_prb.zero_grad(set_to_none=True)
        loss.backward()
        opt_img.step()
        opt_prb.step()

        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                psnr, per = eval_wpsnr(run, image.detach(), probes.detach(), centers,
                                       img_res, patch, eval_ids)
                wd = per.sort().values[:len(per) // 10].mean().item()
            msg = (f'step {step:6d}  mip {m}  loss {loss.item():.5f}  '
                   f'evalPSNR {psnr:.2f} dB  worst-dec {wd:.2f} dB  '
                   f'({time.time() - t0:.0f}s)')
            print(f'[TRAIN] {msg}', flush=True)
            logf.write(msg + '\n')
            logf.flush()
        if (step + 1) % 1000 == 0 or step == steps - 1:
            torch.save({'image': image.detach().cpu(), 'probes': probes.detach().cpu(),
                        'opt_img': opt_img.state_dict(), 'opt_prb': opt_prb.state_dict(),
                        'step': step + 1, 'img_res': img_res, 'patch': patch}, ckpt_path)

    torch.save({'image': image.detach().cpu(), 'probes': probes.detach().cpu(),
                'img_res': img_res, 'patch': patch, 'steps': steps,
                'wall_s': time.time() - t0},
               os.path.join(run.out_dir, 'train_final.pt'))
    logf.close()
    print(f'[TRAIN] done ({time.time() - t0:.1f}s) -> train_final.pt')


# --------------------------------------------------------------------------------------
# train / export — neural texture variant (--texture neural)
# --------------------------------------------------------------------------------------

def neural_dir(run, cli):
    d = os.path.join(run.out_dir, cli.out_tag)
    os.makedirs(d, exist_ok=True)
    return d


def build_field(cli):
    return NeuralTexture2D(img_res=cli.img_res, levels=cli.tex_levels, feat=cli.tex_feat,
                           log2_table=cli.hash2d_log2, base=cli.tex_base,
                           hidden=cli.tex_hidden).to(DEV)


def neural_active(step, cli):
    if cli.c2f_interval <= 0:
        return cli.tex_levels
    return min(cli.tex_levels, 4 + step // cli.c2f_interval)


def cmd_train_neural(run, cli):
    t0 = time.time()
    out = neural_dir(run, cli)
    img_res = cli.img_res
    centers, cell = grid_probe_centers(run.N, img_res)
    field = build_field(cli)
    print(f'[TRAIN-NEURAL] field: {field.describe()}')
    print(f'[TRAIN-NEURAL] probe grid G={math.ceil(math.sqrt(run.N))} cell={cell:.2f}px '
          f'(non-overlapping init), c2f_interval={cli.c2f_interval}, probe_mode={cli.probe_mode}')

    ckpt_path = os.path.join(out, 'train_ckpt.pt')
    start = 0
    groups = [
        {'params': [field.tables], 'lr': cli.lr_hash},
        {'params': field.mlp.parameters(), 'lr': cli.lr_mlp},
    ]
    if cli.probe_mode == 'latent':
        head = ProbeHead(run, geom_inputs=cli.probe_geom_inputs,
                         latent_dim=cli.probe_latent_dim,
                         free_residual=cli.probe_free_residual,
                         geom_pe=cli.geom_pe).to(DEV)
        groups += head.param_groups(cli)
        probes = None
        hidden = 128 if cli.geom_pe > 0 else 64
        print(f'[TRAIN-NEURAL] latent probes: anchor {head.anchor.shape[1]}D + z {cli.probe_latent_dim}D '
              f'-> {hidden} -> {hidden} -> 5 (zero-init head), free_residual={cli.probe_free_residual}, '
              f'geom_inputs={cli.probe_geom_inputs}, geom_pe={cli.geom_pe}')
    else:
        head = None
        probes = torch.zeros(run.N, 5, device=DEV, requires_grad=True)
        groups.append({'params': [probes], 'lr': cli.lr_probes})
    opt = torch.optim.Adam(groups)
    # --lr_decay cosine: per-step factor 1.0 -> 0.1 on ALL groups, recomputed from `step`
    # each iteration from the construction-time base lrs (survives checkpoint resume —
    # opt.load_state_dict restores stale decayed lrs, which we overwrite every step).
    base_lrs = [g['lr'] for g in opt.param_groups]
    # --student image: train THROUGH the deployed read path bilinear(bake(field)).
    # Texels are generated lazily (4 per query, deduped); export's baked image is then
    # exactly the optimized object — no bake-time quality drop. (Built BEFORE the resume
    # block: warm-start hybrids resume a pixel-less ckpt with fresh zero pixels.)
    pixels, opt_px = None, None
    if cli.student == 'image':
        if cli.explicit_image:
            # Hybrid: zero-init explicit residual image summed with the field at texel
            # centers. Field carries coarse structure + probe gradients; pixels remove
            # the hash-collision capacity ceiling. Separate Adam (its 2x805MB moments
            # stay OUT of the checkpoint; resume restarts them — pixels themselves are
            # checkpointed fp16).
            pixels = torch.zeros(img_res, img_res, 3, device=DEV, requires_grad=True)
            opt_px = torch.optim.Adam([pixels], lr=cli.lr_pixels)
            print(f'[TRAIN-NEURAL] explicit_image ON: +{img_res}^2x3 fp32 residual pixels '
                  f'({img_res * img_res * 3 * 4 / 1e9:.2f} GB train mem), lr={cli.lr_pixels}')
        student_of = lambda act: image_bilinear_sampler(field, img_res, act, pixels)
        print(f'[TRAIN-NEURAL] student=image: bilinear from lazily-baked {img_res}^2 texel '
              f'centers (grid_sample/BC7-TMU semantics), uv_sampling={cli.uv_sampling}')
    else:
        student_of = lambda act: field_sampler(field, act)
    if os.path.exists(ckpt_path) and not cli.restart:
        ck = torch.load(ckpt_path)
        field.load_state_dict(ck['field'])
        if head is not None:
            head.load_state_dict(ck['head'])
        else:
            with torch.no_grad():
                probes.copy_(ck['probes'].to(DEV))
        opt.load_state_dict(ck['opt'])
        if pixels is not None and ck.get('pixels') is not None:
            with torch.no_grad():
                pixels.copy_(ck['pixels'].to(DEV).float())
        start = ck['step']
        print(f'[TRAIN-NEURAL] resumed from step {start}')

    eval_ids = torch.arange(run.N, device=DEV)[torch.randperm(run.N,
                generator=torch.Generator().manual_seed(7))[:2048].to(DEV)]
    logf = open(os.path.join(out, 'train_log.txt'), 'a')
    B, S = cli.batch_surfels, cli.uv_per_surfel
    grad_on = cli.grad_loss_w > 0.0
    if grad_on:
        S = max(1, S // 2)   # 3 evals per sample (uv, uv+eps_u, uv+eps_v) — keep step cost similar
        eps = cli.grad_eps
        print(f'[TRAIN-NEURAL] grad-domain loss ON: W={cli.grad_loss_w}, eps={eps} sigma, '
              f'uv/surfel {cli.uv_per_surfel} -> {S}')
    steps = cli.steps
    gen = torch.Generator(device=DEV).manual_seed(1000 + start)
    area_w = None      # --sample_by_area: cached footprint weights, refreshed every 100 steps
    if cli.sample_by_area:
        print('[TRAIN-NEURAL] sample_by_area ON: surfel batch ~ probe footprint area '
              '(weights exp(log su+log sv), clamped [0.25x,16x] median, cached 100 steps)')

    if cli.lr_decay == 'cosine':
        print(f'[TRAIN-NEURAL] lr_decay=cosine: factor 0.1 + 0.45*(1+cos(pi*step/{steps})) '
              f'(1.0 -> 0.1) on all {len(base_lrs)} groups')

    for step in range(start, steps):
        act = neural_active(step, cli)
        if cli.lr_decay == 'cosine':
            fac = 0.1 + 0.45 * (1.0 + math.cos(math.pi * step / steps))
            for g, b in zip(opt.param_groups, base_lrs):
                g['lr'] = b * fac
            if opt_px is not None:
                opt_px.param_groups[0]['lr'] = cli.lr_pixels * fac
        if cli.sample_by_area:
            if area_w is None or step % 100 == 0:
                with torch.no_grad():
                    pr = head.probes() if head is not None else probes
                    area = (pr[:, 2] + pr[:, 3]).exp()               # su * sv
                    med = area.median().clamp_min(1e-12)
                    area_w = area.clamp(0.25 * med, 16.0 * med)
                    area_w = area_w / area_w.sum()
            sid = torch.multinomial(area_w, B, replacement=True, generator=gen)
        else:
            sid = torch.randint(0, run.N, (B,), device=DEV, generator=gen)
        uv = sample_uv(B, S, gen, cli.uv_sampling)
        probes_b = head.probes(sid) if head is not None else probes[sid]
        if grad_on:
            # base + u-offset + v-offset points in one batched eval each (teacher & student)
            e_u = torch.tensor([eps, 0.0], device=DEV)
            e_v = torch.tensor([0.0, eps], device=DEV)
            uv3 = torch.cat([uv, uv + e_u, uv + e_v], dim=1)          # [B, 3S, 2]
            with torch.no_grad():
                t3 = run.teacher_uv(sid, uv3)
                w = run.kernel_weight(sid, uv)
            grid3 = probe_grid_coords(probes_b, centers[sid], uv3, img_res, cell)
            s3 = student_of(act)(grid3)
            t, tu, tv = t3[:, :S], t3[:, S:2 * S], t3[:, 2 * S:]
            s, su, sv = s3[:, :S], s3[:, S:2 * S], s3[:, 2 * S:]
            loss_color = ((t - s).abs().mean(-1) * w).sum() / w.sum().clamp_min(1e-8)
            dgrad = ((tu - t) - (su - s)).abs().mean(-1) + ((tv - t) - (sv - s)).abs().mean(-1)
            loss_grad = ((dgrad / eps) * 0.5 * w).sum() / w.sum().clamp_min(1e-8)
            if step == start:
                print(f'[TRAIN-NEURAL] step {step} raw terms: color={loss_color.item():.5f} '
                      f'grad={loss_grad.item():.5f} (W*grad={cli.grad_loss_w * loss_grad.item():.5f}, '
                      f'ratio={cli.grad_loss_w * loss_grad.item() / max(loss_color.item(), 1e-8):.2f})',
                      flush=True)
            loss = loss_color + cli.grad_loss_w * loss_grad
        else:
            with torch.no_grad():
                t = run.teacher_uv(sid, uv)
                w = run.kernel_weight(sid, uv)
            grid = probe_grid_coords(probes_b, centers[sid], uv, img_res, cell)
            s = student_of(act)(grid)
            loss = ((t - s).abs().mean(-1) * w).sum() / w.sum().clamp_min(1e-8)
        opt.zero_grad(set_to_none=True)
        if opt_px is not None:
            opt_px.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if opt_px is not None:
            opt_px.step()

        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                probes_all = (head.probes() if head is not None else probes).detach()
                psnr, per = eval_wpsnr_fn(run, student_of(act), probes_all,
                                          centers, img_res, cell, eval_ids)
                wd = per.sort().values[:len(per) // 10].mean().item()
            msg = (f'step {step:6d}  act {act}/{cli.tex_levels}  loss {loss.item():.5f}  '
                   f'evalPSNR {psnr:.2f} dB  worst-dec {wd:.2f} dB  ({time.time() - t0:.0f}s)')
            print(f'[TRAIN-NEURAL] {msg}', flush=True)
            logf.write(msg + '\n')
            logf.flush()
        ck_every = 5000 if pixels is not None else 1000   # pixel ckpts are ~400MB fp16
        if (step + 1) % ck_every == 0 or step == steps - 1:
            ck = {'field': field.state_dict(), 'opt': opt.state_dict(), 'step': step + 1,
                  'img_res': img_res, 'levels': cli.tex_levels, 'feat': cli.tex_feat,
                  'hash2d_log2': cli.hash2d_log2, 'tex_base': cli.tex_base, 'tex_hidden': cli.tex_hidden,
                  'probe_mode': cli.probe_mode, 'student': cli.student,
                  'pixels': pixels.detach().half().cpu() if pixels is not None else None}
            if head is not None:
                ck['head'] = head.state_dict()
            else:
                ck['probes'] = probes.detach().cpu()
            torch.save(ck, ckpt_path)

    with torch.no_grad():
        probes_final = (head.probes() if head is not None else probes).detach().cpu()
    fin = {'field': field.state_dict(), 'probes': probes_final,
           'img_res': img_res, 'levels': cli.tex_levels, 'feat': cli.tex_feat,
           'hash2d_log2': cli.hash2d_log2, 'tex_base': cli.tex_base, 'tex_hidden': cli.tex_hidden,
           'steps': steps, 'wall_s': time.time() - t0, 'probe_mode': cli.probe_mode,
           'probe_geom_inputs': cli.probe_geom_inputs,
           'probe_latent_dim': cli.probe_latent_dim,
           'probe_free_residual': cli.probe_free_residual,
           'geom_pe': cli.geom_pe, 'grad_loss_w': cli.grad_loss_w,
           'student': cli.student, 'uv_sampling': cli.uv_sampling,
           'pixels': pixels.detach().half().cpu() if pixels is not None else None}
    if head is not None:
        fin['head'] = head.state_dict()
    torch.save(fin, os.path.join(out, 'train_final.pt'))
    logf.close()
    print(f'[TRAIN-NEURAL] done ({time.time() - t0:.1f}s) -> {out}/train_final.pt')


@torch.no_grad()
def bake_field_image(field, img_res, chunk=2 ** 21):
    """Evaluate T(p) at every pixel center -> [3,R,R] fp32."""
    img = torch.empty(3, img_res, img_res, device=DEV)
    t = (torch.arange(img_res, device=DEV, dtype=torch.float32) + 0.5) / img_res
    for y0 in range(0, img_res, max(1, chunk // img_res)):
        y1 = min(y0 + max(1, chunk // img_res), img_res)
        yy, xx = torch.meshgrid(t[y0:y1], t, indexing='ij')
        p = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        img[:, y0:y1, :] = field(p).permute(1, 0).reshape(3, y1 - y0, img_res)
    return img


def cmd_export_neural(run, cli):
    import imageio.v2 as imageio
    t0 = time.time()
    out = neural_dir(run, cli)
    fin = torch.load(os.path.join(out, 'train_final.pt'))
    img_res = fin['img_res']
    field = NeuralTexture2D(img_res=img_res, levels=fin['levels'], feat=fin['feat'],
                            log2_table=fin['hash2d_log2'], base=fin['tex_base'],
                            hidden=fin.get('tex_hidden', 32)).to(DEV)
    field.load_state_dict(fin['field'])
    field.eval()
    probes = fin['probes'].to(DEV).float()
    centers, cell = grid_probe_centers(run.N, img_res)

    # bake the field to a plain RGB image (the deployable artifact)
    baked = bake_field_image(field, img_res)
    lo, hi = baked.min().item(), baked.max().item()
    u8 = ((baked - lo) / max(hi - lo, 1e-8) * 255.0).round().to(torch.uint8)
    torch.save({'u8': u8.cpu(), 'lo': lo, 'hi': hi, 'img_res': img_res}, os.path.join(out, 'atlas_u8.pt'))
    imageio.imwrite(os.path.join(out, 'atlas.png'), _to_png_u8(baked).permute(1, 2, 0).cpu().numpy())
    torch.save({'probes': probes.cpu(), 'centers': centers.cpu(), 'img_res': img_res,
                'cell': cell, 'uv_extent': UV_EXTENT}, os.path.join(out, 'probes.pt'))

    # PSNR: the live field AND the baked image (bilinear resample) — the artifact's true quality
    all_ids = torch.arange(run.N, device=DEV)
    f_overall, f_per = eval_wpsnr_fn(run, field_sampler(field), probes, centers,
                                     img_res, cell, all_ids)
    b_overall, b_per = eval_wpsnr(run, baked, probes, centers, img_res, cell, all_ids)

    def stats(overall, per):
        return dict(overall=overall, worst_decile=per.sort().values[:run.N // 10].mean().item(),
                    p10=per.quantile(0.10).item(), p50=per.quantile(0.50).item(),
                    p90=per.quantile(0.90).item())

    table = dict(field=stats(f_overall, f_per), baked_image=stats(b_overall, b_per))
    for k, v in table.items():
        print(f"[EXPORT-NEURAL] {k:12s}: overall {v['overall']:.2f} dB  "
              f"worst-decile {v['worst_decile']:.2f}  "
              f"p10/p50/p90 {v['p10']:.2f}/{v['p50']:.2f}/{v['p90']:.2f}")

    # storage
    tbl_fp32 = field.tables.numel() * 4
    tbl_fp16 = field.tables.numel() * 2
    mlp_bytes = sum(p.numel() for p in field.mlp.parameters()) * 4
    probe_bytes = run.N * 5 * 2
    baked_u8 = img_res * img_res * 3
    stor = dict(hash_tables_fp32_bytes=tbl_fp32, hash_tables_fp16_bytes=tbl_fp16,
                mlp_fp32_bytes=mlp_bytes, probes_fp16_bytes=probe_bytes,
                neural_total_fp32=tbl_fp32 + mlp_bytes + probe_bytes,
                neural_total_fp16=tbl_fp16 + mlp_bytes // 2 + probe_bytes,
                baked_image_u8_bytes=baked_u8,
                image_variant_u8_bytes=cli.img_res * cli.img_res * 3)
    print(f'[EXPORT-NEURAL] storage: hash tables {tbl_fp32 / 1e6:.1f} MB fp32 '
          f'({tbl_fp16 / 1e6:.1f} MB fp16) + MLP {mlp_bytes / 1e3:.1f} KB + '
          f'probes {probe_bytes / 1e6:.2f} MB = {stor["neural_total_fp32"] / 1e6:.1f} MB fp32 '
          f'({stor["neural_total_fp16"] / 1e6:.1f} MB fp16)  |  baked u8 image {baked_u8 / 1e6:.1f} MB '
          f'(image-variant atlas: {stor["image_variant_u8_bytes"] / 1e6:.1f} MB)')

    # previews: 4 worst + 4 best by field PSNR — teacher | field | baked
    order = f_per.argsort()
    pick = torch.cat([order[:4], order[-4:]]).tolist()
    grid_uv = bake_uv_grid()
    tiles = []
    for i in pick:
        sid = torch.tensor([i], device=DEV)
        uv = grid_uv[None]
        g = probe_grid_coords(probes[sid], centers[sid], uv, img_res, cell)
        t = run.teacher_uv(sid, uv).view(BAKE_RES, BAKE_RES, 3)
        s_f = field_sampler(field)(g).view(BAKE_RES, BAKE_RES, 3)
        s_b = sample_atlas(baked, g).view(BAKE_RES, BAKE_RES, 3)
        row = torch.cat([t, s_f, s_b], dim=1)
        row = F.interpolate(row.permute(2, 0, 1)[None], scale_factor=6,
                            mode='nearest')[0].permute(1, 2, 0)
        tiles.append(row)
        tag = 'worst' if len(tiles) <= 4 else 'best'
        imageio.imwrite(os.path.join(out, f'preview_{tag}_{i}_psnr{f_per[i]:.1f}.png'),
                        _to_png_u8(row).cpu().numpy())
    imageio.imwrite(os.path.join(out, 'previews.png'),
                    _to_png_u8(torch.cat(tiles, dim=0)).cpu().numpy())

    # ---- latent probe mode extras: storage terms + probe-clustering evidence ----
    clustering = None
    if fin.get('probe_mode') == 'latent':
        z_bytes = run.N * fin.get('probe_latent_dim', 8) * 2
        head_params = 0
        for k, v in fin['head'].items():
            if k.startswith('mlp.'):
                head_params += v.numel()
        head_bytes = head_params * 4
        resid_bytes = run.N * 5 * 2 if fin.get('probe_free_residual') else 0
        stor.update(latents_fp16_bytes=z_bytes, probe_mlp_fp32_bytes=head_bytes,
                    probe_resid_fp16_bytes=resid_bytes)
        stor['neural_total_fp32'] = tbl_fp32 + mlp_bytes + z_bytes + head_bytes + resid_bytes
        stor['neural_total_fp16'] = tbl_fp16 + mlp_bytes // 2 + z_bytes + head_bytes // 2 + resid_bytes
        stor['materialized_probes_fp16_bytes'] = probe_bytes  # alt deploy: drop head+latents
        print(f'[EXPORT-NEURAL] latent-mode storage: +latents {z_bytes / 1e6:.2f} MB fp16 '
              f'+ probeMLP {head_bytes / 1e3:.1f} KB ({head_params:,} params) '
              f'+ free-resid {resid_bytes / 1e6:.2f} MB -> total {stor["neural_total_fp32"] / 1e6:.1f} MB fp32')

        # Overlap-sharing evidence: probe-center distance for high-anchor-similarity pairs
        # vs random pairs. Hypothesis: cos(F_i,F_j) > 0.95 pairs land much closer in the atlas.
        with torch.no_grad():
            Hc = run.ingp(points_3D=run.center).float()[:, :run.hash_dim]
            Fa = torch.cat([Hc, run.beta], dim=1)
            Fn = Fa / Fa.norm(dim=1, keepdim=True).clamp_min(1e-8)
            pcent = centers + probes[:, 0:2]                     # probe centers (px)
            gen = torch.Generator(device=DEV).manual_seed(3)
            n_pairs = 8_000_000
            ia = torch.randint(0, run.N, (n_pairs,), device=DEV, generator=gen)
            ib = torch.randint(0, run.N, (n_pairs,), device=DEV, generator=gen)
            keep = ia != ib
            ia, ib = ia[keep], ib[keep]
            cos = (Fn[ia] * Fn[ib]).sum(1)
            dist = (pcent[ia] - pcent[ib]).norm(dim=1)
            hi = cos > 0.95
            bins = torch.tensor([0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                                dtype=torch.float32, device=DEV)
            hist_hi = torch.bucketize(dist[hi], bins).bincount(minlength=len(bins) + 1)
            hist_rand = torch.bucketize(dist, bins).bincount(minlength=len(bins) + 1)
            clustering = dict(
                n_pairs=int(ia.numel()), n_high_sim=int(hi.sum()),
                dist_bins_px=bins.tolist(),
                hist_high_sim=hist_hi.tolist(), hist_random=hist_rand.tolist(),
                dist_high_sim=dict(median=dist[hi].median().item(), mean=dist[hi].mean().item(),
                                   p10=dist[hi].quantile(0.10).item()) if hi.any() else None,
                dist_random=dict(median=dist.median().item(), mean=dist.mean().item(),
                                 p10=dist.quantile(0.10).item()))
            if hi.any():
                print(f"[EXPORT-NEURAL] probe clustering: {int(hi.sum()):,} high-sim pairs "
                      f"(cos>0.95) of {int(ia.numel()):,}  dist median {dist[hi].median():.0f}px "
                      f"vs random {dist.median():.0f}px  (mean {dist[hi].mean():.0f} vs {dist.mean():.0f})")
            else:
                print('[EXPORT-NEURAL] probe clustering: no pairs with cos>0.95 found')

    report = dict(iteration=run.iteration, N=run.N, texture='neural', img_res=img_res,
                  field=field.describe(), cell_px=cell, psnr=table, storage=stor,
                  probe_mode=fin.get('probe_mode', 'free'), probe_clustering=clustering,
                  preview_ids=pick, train_wall_s=fin.get('wall_s'), export_wall_s=time.time() - t0)
    with open(os.path.join(out, 'export_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[EXPORT-NEURAL] wrote atlas.png / atlas_u8.pt / probes.pt / previews.png / '
          f'export_report.json under {out}  ({time.time() - t0:.1f}s)')


# --------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------

def _to_png_u8(x):
    """signed residual -> viewable u8 (0.5 + x/2, clipped)."""
    return ((x * 0.5 + 0.5).clamp(0, 1) * 255.0).round().to(torch.uint8)


def cmd_export(run, cli):
    import imageio.v2 as imageio
    t0 = time.time()
    pack = torch.load(os.path.join(run.out_dir, 'pack.pt'))
    fin = torch.load(os.path.join(run.out_dir, 'train_final.pt'))
    img_res, patch = fin['img_res'], fin['patch']
    assign = pack['assign'].to(DEV)
    centers = rect_center(assign, img_res, patch)
    image = fin['image'].to(DEV).float()
    probes = fin['probes'].to(DEV).float()
    image0 = pack['image'].to(DEV).float()
    probes0 = pack['probes'].to(DEV).float()

    # final image PNG + u8 .pt (empirical min/max quantization — no sigma clamp)
    lo, hi = image.min().item(), image.max().item()
    u8 = ((image - lo) / max(hi - lo, 1e-8) * 255.0).round().to(torch.uint8)
    torch.save({'u8': u8.cpu(), 'lo': lo, 'hi': hi, 'img_res': img_res, 'patch': patch},
               os.path.join(run.out_dir, 'atlas_u8.pt'))
    imageio.imwrite(os.path.join(run.out_dir, 'atlas.png'),
                    _to_png_u8(image).permute(1, 2, 0).cpu().numpy())
    torch.save({'probes': probes.cpu(), 'assign': assign.cpu(), 'centers': centers.cpu(),
                'img_res': img_res, 'patch': patch, 'uv_extent': UV_EXTENT},
               os.path.join(run.out_dir, 'probes.pt'))

    # per-surfel PSNR distributions (all N)
    all_ids = torch.arange(run.N, device=DEV)
    f_overall, f_per = eval_wpsnr(run, image, probes, centers, img_res, patch, all_ids)
    i_overall, i_per = eval_wpsnr(run, image0, probes0, centers, img_res, patch, all_ids)

    def decile(per):
        return per.sort().values[:run.N // 10].mean().item()

    q = lambda p, x: x.quantile(p).item()
    table = dict(
        init=dict(overall=i_overall, worst_decile=decile(i_per),
                  p10=q(0.10, i_per), p50=q(0.50, i_per), p90=q(0.90, i_per)),
        final=dict(overall=f_overall, worst_decile=decile(f_per),
                   p10=q(0.10, f_per), p50=q(0.50, f_per), p90=q(0.90, f_per)),
    )
    print('[EXPORT] distill PSNR (kernel-weighted, peak=1.0, all surfels):')
    print(f"  init : overall {i_overall:.2f} dB  worst-decile {table['init']['worst_decile']:.2f}"
          f"  p10/p50/p90 {table['init']['p10']:.2f}/{table['init']['p50']:.2f}/{table['init']['p90']:.2f}")
    print(f"  final: overall {f_overall:.2f} dB  worst-decile {table['final']['worst_decile']:.2f}"
          f"  p10/p50/p90 {table['final']['p10']:.2f}/{table['final']['p50']:.2f}/{table['final']['p90']:.2f}")

    # storage
    atlas_bytes = img_res * img_res * 3
    classic_texels = run.N * patch * patch
    classic_bytes = classic_texels * 3
    probe_bytes = run.N * 5 * 2  # fp16 probes
    stor = dict(atlas_u8_bytes=atlas_bytes, probes_fp16_bytes=probe_bytes,
                classic_rect_texels=classic_texels, classic_rect_u8_bytes=classic_bytes,
                ratio_vs_classic=classic_bytes / (atlas_bytes + probe_bytes))
    print(f'[EXPORT] storage: atlas u8 {atlas_bytes / 1e6:.1f} MB + probes {probe_bytes / 1e6:.2f} MB '
          f'vs classic per-surfel {patch}x{patch} rects = {classic_texels / 1e6:.1f}M texels '
          f'({classic_bytes / 1e6:.1f} MB u8) -> {stor["ratio_vs_classic"]:.1f}x smaller')

    # previews: 4 worst + 4 best by final PSNR
    order = f_per.argsort()
    pick = torch.cat([order[:4], order[-4:]]).tolist()
    grid_uv = bake_uv_grid()
    tiles = []
    for i in pick:
        sid = torch.tensor([i], device=DEV)
        uv = grid_uv[None]
        t = run.teacher_uv(sid, uv).view(BAKE_RES, BAKE_RES, 3)
        s0 = sample_atlas(image0, probe_grid_coords(probes0[sid], centers[sid], uv,
                          img_res, patch)).view(BAKE_RES, BAKE_RES, 3)
        s1 = sample_atlas(image, probe_grid_coords(probes[sid], centers[sid], uv,
                          img_res, patch)).view(BAKE_RES, BAKE_RES, 3)
        row = torch.cat([t, s0, s1], dim=1)                       # 16 x 48 x 3
        row = F.interpolate(row.permute(2, 0, 1)[None], scale_factor=6,
                            mode='nearest')[0].permute(1, 2, 0)   # 96 x 288
        tiles.append(row)
        tag = 'worst' if len(tiles) <= 4 else 'best'
        imageio.imwrite(os.path.join(run.out_dir, f'preview_{tag}_{i}_psnr{f_per[i]:.1f}.png'),
                        _to_png_u8(row).cpu().numpy())
    sheet = torch.cat(tiles, dim=0)
    imageio.imwrite(os.path.join(run.out_dir, 'previews.png'), _to_png_u8(sheet).cpu().numpy())

    report = dict(iteration=run.iteration, N=run.N, img_res=img_res, patch=patch,
                  psnr=table, storage=stor, preview_ids=pick, wall_s=time.time() - t0)
    with open(os.path.join(run.out_dir, 'export_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[EXPORT] wrote atlas.png / atlas_u8.pt / probes.pt / previews.png / '
          f'export_report.json  ({time.time() - t0:.1f}s)')


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='probe-atlas distillation (3D_SH_filmres)')
    ap.add_argument('cmd', choices=['verify', 'bake', 'pack', 'train', 'export', 'all'])
    ap.add_argument('--model_path', '-m', required=True)
    ap.add_argument('--iteration', type=int, default=-1)
    ap.add_argument('--img_res', type=int, default=4096)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch_surfels', type=int, default=4096)
    ap.add_argument('--uv_per_surfel', type=int, default=64)
    ap.add_argument('--lr_image', type=float, default=1e-2)
    ap.add_argument('--lr_probes', type=float, default=1e-3)
    ap.add_argument('--kmeans_iters', type=int, default=25)
    ap.add_argument('--restart', action='store_true', help='ignore train_ckpt.pt')
    # --texture neural (2D hash+MLP atlas content, no k-means/packing)
    ap.add_argument('--texture', choices=['image', 'neural'], default='image')
    ap.add_argument('--out_tag', default='v_neural_l8f2',
                    help='neural mode: subdir of probe_atlas/ for all state')
    ap.add_argument('--tex_levels', type=int, default=8)
    ap.add_argument('--tex_feat', type=int, default=2)
    ap.add_argument('--tex_base', type=int, default=16)
    ap.add_argument('--tex_hidden', type=int, default=32,
                    help='neural texture decoder hidden width (default 32 = current)')
    ap.add_argument('--hash2d_log2', type=int, default=19)
    ap.add_argument('--c2f_interval', type=int, default=1500,
                    help='neural c2f: +1 hash level every N steps from 4 coarsest (0 = all active)')
    ap.add_argument('--lr_hash', type=float, default=1e-2)
    ap.add_argument('--lr_mlp', type=float, default=1e-3)
    # --probe_mode latent: probes = MLP(frozen anchor [H3D(center)|beta] + learnable z_i)
    ap.add_argument('--probe_mode', choices=['free', 'latent'], default='free')
    ap.add_argument('--probe_geom_inputs', action='store_true',
                    help='latent mode: append [log sx, log sy, tangent frame 6D] to the anchor')
    ap.add_argument('--probe_latent_dim', type=int, default=8,
                    help='latent mode: per-surfel learnable latent dim (0 = anchor-only, no z)')
    ap.add_argument('--grad_loss_w', type=float, default=0.0,
                    help='gradient-domain distillation weight (0 = off; halves uv_per_surfel)')
    ap.add_argument('--grad_eps', type=float, default=0.05,
                    help='finite-difference step for --grad_loss_w (sigma units)')
    ap.add_argument('--geom_pe', type=int, default=0,
                    help='latent mode: Fourier-encode geometry anchor inputs with L bands (0 = raw)')
    ap.add_argument('--lr_decay', choices=['none', 'cosine'], default='none',
                    help='neural train: per-step cosine LR decay 1.0->0.1 on all optimizer groups')
    ap.add_argument('--student', choices=['field', 'image'], default='field',
                    help='image: supervise bilinear(bake(field)) — the exact deployed '
                         'read path (GPU-TMU/BC7 semantics); texels lazily generated')
    ap.add_argument('--uv_sampling', choices=['uniform', 'stratified'], default='uniform',
                    help='stratified: jittered sqrt(S)^2 grid per surfel (coverage-'
                         'guaranteed + unbiased); uniform: plain Monte Carlo')
    ap.add_argument('--explicit_image', action='store_true',
                    help='(student=image) hybrid: zero-init explicit residual pixel image '
                         'summed with the field at texel centers — removes the hash-'
                         'collision capacity ceiling; field keeps probe gradients')
    ap.add_argument('--lr_pixels', type=float, default=1e-2)
    ap.add_argument('--sample_by_area', action='store_true',
                    help='draw the surfel batch ~ current probe footprint area (exp(log su+log sv), '
                         'clamped [0.25x,16x] median, cached 100 steps). INTENTIONALLY reweights the '
                         'distill objective toward large-footprint probes (fixes big-probe '
                         'under-supervision); eval protocol unchanged.')
    ap.add_argument('--probe_free_residual', action=argparse.BooleanOptionalAction, default=True,
                    help='latent mode: small per-surfel free residual on the 5 probe params (lr 1e-4)')
    cli = ap.parse_args()

    run = Run(cli.model_path, cli.iteration)
    if cli.texture == 'neural':
        if cli.cmd == 'pack':
            raise SystemExit('--texture neural has no pack stage (no k-means; regular-grid probe init)')
        stages = [cli.cmd] if cli.cmd != 'all' else ['verify', 'train', 'export']
        table = {'verify': cmd_verify, 'bake': cmd_bake,
                 'train': cmd_train_neural, 'export': cmd_export_neural}
    else:
        stages = [cli.cmd] if cli.cmd != 'all' else ['verify', 'bake', 'pack', 'train', 'export']
        table = {'verify': cmd_verify, 'bake': cmd_bake, 'pack': cmd_pack,
                 'train': cmd_train, 'export': cmd_export}
    for s in stages:
        print(f'\n===== {s.upper()}{" (neural)" if cli.texture == "neural" and s in ("train", "export") else ""} =====')
        table[s](run, cli)


if __name__ == '__main__':
    main()

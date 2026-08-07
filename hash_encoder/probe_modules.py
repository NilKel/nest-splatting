"""--method proberes: probe-mapped shared-texture residual (Python side).

Two learnable components, both plain PyTorch (autograd end-to-end; the CUDA
rasterizer `diff_surfel_3D_sh_res_probe` only consumes their *outputs* and
returns dL/dprobes + dL/dtex which autograd chains back here):

  ProbeTexField2D — 2D multires-hash + MLP field T(p), p in [0,1]^2, baked
      differentiably to a [R, R, 3] image every render call. Coarse-to-fine on
      hash levels for long-range gradients.

  ProbeHead3D — per-surfel probe predictor: 3D multires hash on the (normalized)
      surfel centers ⊕ Fourier posenc of the tangent axes ⊕ normalized
      log-scales → MLP → 4 raw outputs (Δu, Δv, θ_res, Δlog ρ), zero-init last
      layer. Reparameterized so the raw outputs are SMOOTH functions of
      geometry:
        translation = octahedral map of the direction from the scene center
                      (+ learned Δ)                       [world-anchored]
        rotation    = θ_res − φ_gauge, φ_gauge = in-plane angle of world-up
                      projected onto the surfel plane      [gauge-cancelled]
        scale       = (patch_px / 6) · exp(Δlog ρ) · sqrt(sx·sy)/s_med
                      (≈ ±3σ of a median surfel covers patch_px texels)
                                                           [metric-consistent]
      Output: probes [N, 6] = [A00, A01, A10, A11, tx, ty] with
      texcoord = A · uv + t (uv in surfel σ units, texcoord in texture pixels).

Positional gradients: probes are recomputed from pc.get_xyz / get_rotation /
get_scaling every render call, so dL/dprobes flows through the 3D hash +
posenc into surfel positions AND rotations/scales, joining the rasterizer's
own geometry gradients.
"""

import math
import torch
import torch.nn as nn

from gridencoder import GridEncoder
from utils.general_utils import build_rotation


def _fourier_encode(x, bands):
    """x [N, D] -> [N, D * 2 * bands] with sin/cos(2^k pi x)."""
    outs = []
    for k in range(bands):
        ang = (2.0 ** k) * math.pi * x
        outs.append(torch.sin(ang))
        outs.append(torch.cos(ang))
    return torch.cat(outs, dim=-1)


def _oct_encode(d):
    """Octahedral map: unit direction [N,3] -> [0,1]^2 (equal-area-ish, smooth
    away from the ±z seam). Standard octahedral wrap for the lower hemisphere."""
    ad = d.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    p = d / ad                                  # project onto L1 sphere
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    wrap_x = (1.0 - y.abs()) * torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    wrap_y = (1.0 - x.abs()) * torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
    ox = torch.where(z < 0, wrap_x, x)
    oy = torch.where(z < 0, wrap_y, y)
    return torch.stack([ox, oy], dim=-1) * 0.5 + 0.5


class ProbeTexField2D(nn.Module):
    def __init__(self, tex_res=2048, levels=16, level_dim=2, log2_table=19,
                 base_res=8, hidden=64, pixels=True):
        super().__init__()
        self.tex_res = int(tex_res)
        self.levels = int(levels)
        self.level_dim = int(level_dim)
        self.active_levels = int(levels)   # c2f ramp target; set externally
        per_level_scale = (self.tex_res / base_res) ** (1.0 / max(levels - 1, 1))
        self.enc = GridEncoder(input_dim=2, num_levels=levels, level_dim=level_dim,
                               per_level_scale=per_level_scale, base_resolution=base_res,
                               log2_hashmap_size=log2_table)
        # Learnable per-texel pixel image riding on the field:
        #   tex = MLP(hash(p)) + pixels
        # The pixels get DIRECT per-texel gradients — no MLP bottleneck, no hash
        # collisions — guaranteeing the texture can absorb whatever dL/dtex asks
        # for even where the field underfits. Zero-init (no bootstrap issue: it
        # is a leaf addend, its gradient is dL/dtex itself). This IS the final
        # deployable artifact anyway (the shipped texture is the baked sum).
        self.pixels = nn.Parameter(torch.zeros(self.tex_res, self.tex_res, 3)) if pixels else None
        self.bake_interval = 1      # --probe_bake_interval: field re-eval cadence
        self.no_field = False       # --probe_no_field: texture IS the pixel image (no 2D hash+MLP at all)
        self._fcache = None
        self._fcache_iter = -10 ** 9
        self.mlp = nn.Sequential(
            nn.Linear(levels * level_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 3))
        # Residual field starts SMALL but not zero: an exactly-zero last layer
        # blocks ALL upstream gradients (dL/dh2 = W3ᵀ·g = 0 → embeddings never
        # learn until W3 crawls off zero — observed as "no textures by 5k").
        # std 0.01 → residual starts as ~0.05-amplitude noise, harmless under
        # the ReLU cascade, and the whole field trains from iteration 1.
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, p01):
        """p01 [M, 2] in [0,1] -> RGB residual [M, 3]."""
        f = self.enc(p01)
        if self.active_levels < self.levels:
            mask = f.new_zeros(self.levels * self.level_dim)
            mask[: self.active_levels * self.level_dim] = 1.0
            f = f * mask
        return self.mlp(f)

    def _bake_dense(self, chunk=1 << 21):
        """Evaluate the field at every texel center -> [R, R, 3] image
        (texture pixels, texel centers at integer+0.5). Builds the full
        autograd graph when grad is enabled."""
        R = self.tex_res
        dev = self.enc.embeddings.device
        t = (torch.arange(R, device=dev, dtype=torch.float32) + 0.5) / R
        gy, gx = torch.meshgrid(t, t, indexing='ij')
        p = torch.stack([gx, gy], dim=-1).view(-1, 2)   # x fast, matches tex[y, x]
        outs = []
        for i in range(0, p.shape[0], chunk):
            outs.append(self.forward(p[i:i + chunk]))
        img = torch.cat(outs, dim=0).view(R, R, 3)
        if self.pixels is not None:
            img = img + self.pixels
        return img

    def bake_cached(self, iteration, chunk=1 << 21):
        """Speed path: re-evaluate the 2D field only every `bake_interval`
        iterations, caching the field image; the pixel image is added fresh
        each call so it keeps EXACT per-texel gradients every iteration. The
        field receives gradients only on re-bake iterations (periodic-freeze
        semantics). interval=1 -> identical to bake()."""
        if self.no_field and self.pixels is not None:
            return self.pixels
        iv = int(getattr(self, 'bake_interval', 1))
        if iv <= 1:
            return self.bake(chunk=chunk)
        if (self._fcache is None or iteration is None
                or iteration - self._fcache_iter >= iv):
            full = self.bake(chunk=chunk)                 # field + pixels, with grads
            with torch.no_grad():
                self._fcache = (full.detach() - self.pixels.detach()
                                if self.pixels is not None else full.detach())
            self._fcache_iter = -1 if iteration is None else int(iteration)
            return full
        return self._fcache + self.pixels if self.pixels is not None else self._fcache

    def bake(self, chunk=1 << 21, sparse_bw=True):
        """Bake the texture image. With sparse_bw (default), the forward runs
        graph-free and the backward RECOMPUTES the field only at texels with
        nonzero dL/dtex — numerically identical gradients, but no 4M-point
        activation graph and backward cost proportional to touched texels."""
        if self.no_field and self.pixels is not None:
            # --probe_no_field: the texture IS the pixel image. Zero per-iter
            # work; dL_dtex flows straight into the leaf parameter.
            return self.pixels
        if not (sparse_bw and torch.is_grad_enabled()):
            return self._bake_dense(chunk)
        return _SparseBake.apply(self.enc.embeddings, self, chunk)


class _SparseBake(torch.autograd.Function):
    """Bake with sparse-recompute backward. Forward evaluates the field with
    no graph; backward selects texels with nonzero incoming gradient, re-runs
    the field on just those points under enable_grad, and accumulates param
    grads directly (torch.autograd.backward inside — checkpoint-style). The
    `anchor` input (any field parameter) only exists so the output is marked
    requires_grad and the backward is invoked; it receives no gradient here
    because the internal backward already accumulated into every parameter."""

    @staticmethod
    def forward(ctx, anchor, field, chunk):
        ctx.field = field
        ctx.chunk = chunk
        with torch.no_grad():
            img = field._bake_dense(chunk)
        return img

    @staticmethod
    def backward(ctx, grad_img):
        field, chunk = ctx.field, ctx.chunk
        R = field.tex_res
        g = grad_img.reshape(-1, 3)
        rows = g.abs().sum(dim=-1).nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            t = (torch.arange(R, device=g.device, dtype=torch.float32) + 0.5) / R
            ys, xs = rows // R, rows % R
            p = torch.stack([t[xs], t[ys]], dim=-1)
            with torch.enable_grad():
                for i in range(0, p.shape[0], chunk):
                    out = field.forward(p[i:i + chunk])
                    if field.pixels is not None:
                        out = out + field.pixels.view(-1, 3)[rows[i:i + chunk]]
                    torch.autograd.backward(out, g[rows[i:i + chunk]])
        return None, None, None


class ProbeHead3D(nn.Module):
    def __init__(self, tex_res=2048, patch_px=12.0, levels=8, level_dim=2,
                 log2_table=19, base_res=16, fine_res=512, pe_bands=2, hidden=64,
                 abs_placement=False):
        super().__init__()
        # abs_placement: NO analytic base — position = sigmoid(raw)*tex_res,
        # rotation = raw angle. Pure "the hash-MLP decides" ablation; expect
        # heavy early contention (no placement prior).
        self.abs_placement = bool(abs_placement)
        # Fixed probes loaded from a UV-field bake (probe_uv_field.py): forward
        # returns these verbatim (N must stay constant -> run with densify off).
        self.fixed_probes = None
        self.fixed_centers = None   # bake-time centers, for nearest-center remap
        self.tex_res = int(tex_res)
        self.patch_px = float(patch_px)
        self.levels = int(levels)
        self.level_dim = int(level_dim)
        self.active_levels = int(levels)   # c2f ramp target; set externally
        per_level_scale = (fine_res / base_res) ** (1.0 / max(levels - 1, 1))
        self.enc = GridEncoder(input_dim=3, num_levels=levels, level_dim=level_dim,
                               per_level_scale=per_level_scale, base_resolution=base_res,
                               log2_hashmap_size=log2_table)
        in_dim = levels * level_dim + 6 * 2 * pe_bands + 2   # hash | posenc(u,v axes) | log-scales
        self.pe_bands = int(pe_bands)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 4))
        # Near-zero (NOT exactly zero) last layer: step-0 probes sit within a few
        # px / a few % of the analytic base placement, but an exactly-zero W
        # would block ALL gradient into the head's encoder + hidden layers
        # (dL/dh2 = Wᵀg = 0) — the head then can't learn to separate surfels
        # whose octahedral base placements collide (same direction from scene
        # center, different radii), which blurs the texture toward DC.
        if self.abs_placement:
            # Absolute mode: STANDARD (kaiming) init on the last layer — no
            # near-zero shrink. There is no analytic base to preserve, and a
            # centered init would collapse every probe at sigmoid(0)=0.5 (image
            # center). Full-spread outputs from step 0: positions across the
            # image, rotations ~+-1 rad, scales ~e^+-0.5 around the metric base.
            self.mlp[-1].reset_parameters()
            # Kaiming alone leaves raw outputs at ~0.06 std -> sigmoid keeps all
            # probes within +-40px of image center. Gain the position/rotation
            # rows so raw std ~ 1.5: positions spread over ~[0.1, 0.9]*tex_res,
            # driven by the (spatially smooth) hash features — nearby surfels
            # still land near each other, a soft learned substitute for the
            # octahedral prior. Scale row keeps standard init.
            with torch.no_grad():
                self.mlp[-1].weight[0:3].mul_(25.0)
        else:
            nn.init.normal_(self.mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.mlp[-1].bias)
        # Ablation/debug: when True, probes FOLLOW geometry but do not DRIVE it —
        # the head's inputs are detached so dL/dprobes stops at the head (no
        # positional/rotational gradient through the probe path; the
        # rasterizer's own geometry gradients are unaffected).
        self.detach_geom = False
        # Scene normalization buffers, filled on first forward (or restored from ckpt).
        self.register_buffer('scene_center', torch.zeros(3))
        self.register_buffer('scene_radius', torch.ones(1))
        self.register_buffer('log_smed', torch.zeros(1))
        self.register_buffer('norm_inited', torch.zeros(1))
        # log_smed tracks the RUNNING median surfel size until frozen (train.py
        # flips this at --probe_smed_freeze_iter via set_active_levels). A
        # first-forward-only snapshot goes stale as densification shrinks
        # surfels (cold-init scales are ~4x too big) -> probe patches collapse
        # to ~1-2 texels -> one flat color per surfel. Frozen late for texture
        # stability (rescaling probes after content forms would smear it).
        self.smed_frozen = False

    @torch.no_grad()
    def _maybe_init_norm(self, xyz, scales):
        first = self.norm_inited.item() < 0.5
        if first:
            med = xyz.median(dim=0).values
            rad = (xyz - med).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
            self.scene_center.copy_(med)
            self.scene_radius.fill_(float(rad))
            self.norm_inited.fill_(1.0)
        # Running s_med: only during training forwards (grad enabled), so eval
        # renders / FD probes never shift the calibration between passes.
        if first or (not self.smed_frozen and torch.is_grad_enabled()):
            smed = torch.sqrt(scales[:, 0] * scales[:, 1]).clamp_min(1e-9).log().median()
            self.log_smed.fill_(float(smed))

    def forward(self, xyz, rot_q, scales):
        """xyz [N,3], rot_q [N,4] (normalized quats), scales [N,2] (activated).
        Returns probes [N,6] = [A00, A01, A10, A11, tx, ty] (texture pixels)."""
        if self.fixed_probes is not None:
            if self.fixed_probes.shape[0] != xyz.shape[0]:
                # N changed (e.g. startup opacity prune after --init_ply, or a
                # later prune event): remap by NEAREST BAKE-TIME CENTER — each
                # surviving/new surfel adopts the probe of the closest surfel
                # from the bake. Probes are a smooth function of position, so
                # nearby surfels' probes are interchangeable to first order.
                assert self.fixed_centers is not None, (
                    "fixed probes N mismatch and no bake-time centers saved — "
                    "re-run scripts/probe_uv_field.py bake with the current version")
                with torch.no_grad():
                    idx = torch.empty(xyz.shape[0], dtype=torch.long, device=xyz.device)
                    for i in range(0, xyz.shape[0], 8192):
                        d = torch.cdist(xyz[i:i + 8192], self.fixed_centers)
                        idx[i:i + 8192] = d.argmin(dim=1)
                    if isinstance(self.fixed_probes, torch.nn.Parameter):
                        # keep the Parameter OBJECT (it's in the optimizer);
                        # remap fires before any step, so Adam state is empty.
                        self.fixed_probes.data = self.fixed_probes.data[idx].contiguous()
                    else:
                        self.fixed_probes = self.fixed_probes[idx].contiguous()
                    self.fixed_centers = self.fixed_centers[idx].contiguous()
                print(f"[PROBERES] fixed probes remapped by nearest center -> N={xyz.shape[0]}")
            return self.fixed_probes
        self._maybe_init_norm(xyz, scales)
        if self.detach_geom:
            xyz, rot_q, scales = xyz.detach(), rot_q.detach(), scales.detach()
        R = build_rotation(rot_q)                    # [N,3,3], columns = axes
        u_ax, v_ax, n_ax = R[..., 0], R[..., 1], R[..., 2]

        # --- head inputs ---
        x01 = ((xyz - self.scene_center) / (2.0 * self.scene_radius) + 0.5).clamp(0.0, 1.0)
        f = self.enc(x01)
        if self.active_levels < self.levels:
            mask = f.new_zeros(self.levels * self.level_dim)
            mask[: self.active_levels * self.level_dim] = 1.0
            f = f * mask
        pe = _fourier_encode(torch.cat([u_ax, v_ax], dim=-1), self.pe_bands)
        logs = (torch.log(scales.clamp_min(1e-9)) - self.log_smed) / 3.0
        raw = self.mlp(torch.cat([f, pe, logs], dim=-1))   # [N,4]

        if self.abs_placement:
            # --- pure-MLP placement: no octahedral base, no gauge angle ---
            tx = torch.sigmoid(raw[:, 0]) * self.tex_res
            ty = torch.sigmoid(raw[:, 1]) * self.tex_res
            theta = raw[:, 2]
        else:
            # --- base translation: octahedral placement, world-anchored ---
            d = xyz - self.scene_center
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            base = _oct_encode(d)                              # [N,2] in [0,1]
            margin = 0.02
            base_px = (base * (1.0 - 2.0 * margin) + margin) * self.tex_res
            delta_scale = self.tex_res / 8.0
            tx = base_px[:, 0] + raw[:, 0] * delta_scale
            ty = base_px[:, 1] + raw[:, 1] * delta_scale

            # --- gauge-cancelled rotation ---
            up = xyz.new_tensor([0.0, 0.0, 1.0]).expand_as(xyz)
            c = up - (up * n_ax).sum(-1, keepdim=True) * n_ax
            c_norm = c.norm(dim=-1, keepdim=True)
            alt = xyz.new_tensor([1.0, 0.0, 0.0]).expand_as(xyz)
            alt = alt - (alt * n_ax).sum(-1, keepdim=True) * n_ax
            c = torch.where(c_norm > 1e-3, c, alt)
            phi = torch.atan2((c * v_ax).sum(-1), (c * u_ax).sum(-1))
            theta = raw[:, 2] - phi

        # --- metric-consistent scale (texels per σ) ---
        s_world = torch.sqrt(scales[:, 0].clamp_min(1e-9) * scales[:, 1].clamp_min(1e-9))
        rho = (self.patch_px / 6.0) * torch.exp(raw[:, 3] + (torch.log(s_world) - self.log_smed))

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        probes = torch.stack([rho * cos_t, -rho * sin_t,
                              rho * sin_t,  rho * cos_t,
                              tx, ty], dim=-1)
        return probes

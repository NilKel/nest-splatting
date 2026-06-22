"""
ClipRelightHead — a clip-conditioned per-Gaussian deform + relight head.

This is an **add-on layer on top of `--method 3D_SH_res`**. It runs once over ALL
surfels at the start of the forward pass (Lagrangian, Python-side, no CUDA), and
emits *adjusted* per-Gaussian tensors that are then handed to the normal
`3D_SH_res` rasterizer.

Design contract (agreed in the design discussion):

  * The TEXTURE residual (`texgrid`: the per-pixel hash+MLP field of 3D_SH_res)
    is an INVARIANT 3D material function f(x). This head NEVER touches it and is
    not conditioned by it as a field — the clip plane is a *reveal + reshade*
    operator, not a texture editor. The texgrid is added downstream in the
    rasterizer, unchanged, no matter where the cut is.

  * This head owns a SEPARATE small hashgrid (`clipgrid`) used purely as a
    positional conditioning feature for predicting the clip-conditioned
    corrections. Reading a feature != making the texture a function of (d, n).

  * Color composition follows 3D_SH_res mode-2 (deferred per-pixel ReLU):
        image = ReLU( sv_exposed + texgrid(x) )
    so the head's `sv_exposed` is deliberately LEFT SIGNED (not re-clamped here)
    — that is what lets the clip-conditioned exposure SUBTRACT to correct an
    over-/under-exposed invariant texgrid for a given cut.

Per-Gaussian outputs of the head (all gated to the cut by `influence`):
  - Δμ          [N, 3]   position snap
  - Δscale      [N, 2]   2DGS in-plane log-scale snap   (mult: scale * exp(infl·Δs))
  - Δrot        [N, 4]   quaternion residual (add + renormalize)
  - exposure    m [N,3] (mult, sigmoid·m_max) and a [N,3] (add, tanh) on the SV base
  - cull_mask   [N, 1]   smooth visibility: sigmoid(-k · signed_distance)

Clip-plane convention (matches data/clipping/watermelon): plane is
`a·x + b·y + c·z + d = 0` with **kept (solid) half  n·x + d <= 0**, n=[a,b,c]
(unit). Signed distance of a Gaussian:  s_g = μ·n + d  (<= 0 means kept).

Zero-init: the final MLP layer is zero-initialised, so at init m=m_max·σ(0)=1,
a=0, Δμ=Δscale=Δrot=0 → the head is an exact IDENTITY and the scene renders as
plain SV 2DGS. It then learns the clip-conditioned corrections.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gridencoder import GridEncoder


# Output channel layout of the head MLP (15 total).
_OUT_M = 3       # SV multiplier (sigmoid · m_max)
_OUT_A = 3       # SV additive  (tanh)
_OUT_DMU = 3     # position delta
_OUT_DROT = 4    # quaternion delta
_OUT_DSCALE = 2  # 2DGS in-plane log-scale delta
_OUT_TOTAL = _OUT_M + _OUT_A + _OUT_DMU + _OUT_DROT + _OUT_DSCALE


@dataclass
class ClipRelightOutput:
    """Adjusted per-Gaussian tensors + diagnostics. Drop the adjusted tensors
    straight into the rasterizer (or render() override)."""
    mu: torch.Tensor          # [N, 3]   adjusted positions
    scaling: torch.Tensor     # [N, 2]   adjusted (activated) in-plane scales
    rotation: torch.Tensor    # [N, 4]   adjusted (normalized) quaternions
    opacity: torch.Tensor     # [N, 1]   opacity * cull_mask
    sv_exposed: torch.Tensor  # [N, 3]   SIGNED exposed SV color (no re-clamp)
    # diagnostics (detached-friendly; carry grads if you need them)
    signed_dist: torch.Tensor  # [N, 1]  s_g = μ·n + d
    influence: torch.Tensor    # [N, 1]  exp(-s_g^2 / sigma^2)
    cull_mask: torch.Tensor    # [N, 1]  sigmoid(-k · s_g)
    m: torch.Tensor            # [N, 3]
    a: torch.Tensor            # [N, 3]


class ClipRelightHead(nn.Module):
    """Clip-conditioned deform + relight head for 3D_SH_res surfels.

    Args:
        scene_bound: μ is normalised by this before the clipgrid query
                     (watermelon lives in the unit sphere; texgrid voxel range
                     is [-1.5, 1.5], so 1.5 keeps the clipgrid query in-range).
        clip_levels / clip_level_dim: clipgrid size (F = clip_levels·clip_level_dim).
        hidden / n_hidden_layers: head MLP shape.
        sigma: gating bandwidth (world units). Smaller => corrections hug the cut.
        cull_k: steepness of the differentiable culling sigmoid.
        m_max: SV multiplier ceiling (m = m_max · sigmoid(·); m_max=2 => init m=1).
        enable_geometry: emit Δμ/Δscale/Δrot (set False to ablate to relight-only).
    """

    def __init__(
        self,
        scene_bound: float = 1.5,
        clip_levels: int = 8,
        clip_level_dim: int = 2,
        clip_base_res: int = 16,
        clip_per_level_scale: float = 1.5,
        clip_log2_hashmap: int = 19,
        hidden: int = 64,
        n_hidden_layers: int = 2,
        sigma: float = 0.1,
        cull_k: float = 50.0,
        m_max: float = 2.0,
        enable_geometry: bool = True,
    ):
        super().__init__()
        self.scene_bound = float(scene_bound)
        self.sigma = float(sigma)
        self.cull_k = float(cull_k)
        self.m_max = float(m_max)
        self.enable_geometry = bool(enable_geometry)

        # clipgrid: positional conditioning feature (NOT the texgrid).
        self.clipgrid = GridEncoder(
            num_levels=clip_levels,
            level_dim=clip_level_dim,
            per_level_scale=clip_per_level_scale,
            base_resolution=clip_base_res,
            log2_hashmap_size=clip_log2_hashmap,
        )
        feat_dim = clip_levels * clip_level_dim

        # MLP input: [clipgrid(μ) F | signed_dist 1 | normal 3].  View-INDEPENDENT
        # (no view dir) so geometry deltas don't change with the camera; SV stays
        # view-dependent because m/a ride on the per-view SV eval downstream.
        in_dim = feat_dim + 1 + 3
        layers = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, _OUT_TOTAL)

        # Zero-init the head => identity at init (m=1, a=0, all deltas=0).
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ------------------------------------------------------------------ #
    @staticmethod
    def signed_distance(mu: torch.Tensor, clip_plane: torch.Tensor):
        """s_g = μ·n + d, with n = unit(clip_plane[:3]), d = clip_plane[3].

        Returns (s_g [N,1], n_unit [3], d float-tensor). Kept half: s_g <= 0.
        """
        n = clip_plane[:3]
        n = n / (n.norm() + 1e-8)
        d = clip_plane[3]
        s_g = (mu * n.unsqueeze(0)).sum(-1, keepdim=True) + d  # [N,1]
        return s_g, n, d

    def forward(
        self,
        mu: torch.Tensor,        # [N, 3]   base xyz (pc.get_xyz)
        scaling: torch.Tensor,   # [N, 2]   base activated in-plane scale (pc.get_scaling)
        rotation: torch.Tensor,  # [N, 4]   base normalized quaternion (pc.get_rotation)
        opacity: torch.Tensor,   # [N, 1]   base activated opacity (pc.get_opacity)
        sv_rgb: torch.Tensor,    # [N, 3]   base CLAMPED SV color, ReLU(feat+0.5)
        clip_plane: torch.Tensor,  # [4]    [a,b,c,d] world, kept half n·x+d<=0
    ) -> ClipRelightOutput:
        N = mu.shape[0]
        assert scaling.shape == (N, 2), f"2DGS scale must be [N,2], got {tuple(scaling.shape)}"
        assert rotation.shape == (N, 4)
        assert opacity.shape == (N, 1)
        assert sv_rgb.shape == (N, 3)
        assert clip_plane.shape == (4,)

        # 1) signed distance + plane normal
        s_g, n, _ = self.signed_distance(mu, clip_plane)                  # [N,1], [3]

        # 2) differentiable culling: keep half is s_g <= 0 -> mask -> 1
        cull_mask = torch.sigmoid(-self.cull_k * s_g)                     # [N,1]
        opacity_out = opacity * cull_mask                                 # [N,1]

        # 3) proximity gate: only correct near the cut
        influence = torch.exp(-(s_g * s_g) / (self.sigma ** 2))           # [N,1]

        # 4) clipgrid conditioning feature at (normalised) μ
        clipfeat = self.clipgrid(mu / self.scene_bound, bound=1.0)        # [N,F]
        n_exp = n.unsqueeze(0).expand(N, 3)                               # [N,3]
        mlp_in = torch.cat([clipfeat, s_g, n_exp], dim=-1)                # [N,F+4]
        raw = self.out(self.trunk(mlp_in))                               # [N,15]

        m_raw, a_raw, dmu, drot, dscale = torch.split(
            raw, [_OUT_M, _OUT_A, _OUT_DMU, _OUT_DROT, _OUT_DSCALE], dim=-1)

        m = self.m_max * torch.sigmoid(m_raw)                            # [N,3] in [0,m_max]; init 1
        a = torch.tanh(a_raw)                                            # [N,3] in (-1,1); init 0

        # 5) exposure on the CLAMPED SV base, LEFT SIGNED (mode-2 composition).
        #    lerp from base->exposed by influence so far-from-cut == base.
        sv_target = m * sv_rgb + a                                       # [N,3] signed
        sv_exposed = sv_rgb + influence * (sv_target - sv_rgb)           # [N,3] signed

        # 6) geometry snap (gated). Identity at init (deltas=0).
        if self.enable_geometry:
            mu_out = mu + influence * dmu                                # [N,3]
            # multiplicative log-scale => positivity preserved, dscale=0 => identity
            scaling_out = scaling * torch.exp(influence * dscale)        # [N,2]
            rotation_out = F.normalize(rotation + influence * drot, dim=-1)  # [N,4]
        else:
            mu_out, scaling_out, rotation_out = mu, scaling, rotation

        return ClipRelightOutput(
            mu=mu_out, scaling=scaling_out, rotation=rotation_out,
            opacity=opacity_out, sv_exposed=sv_exposed,
            signed_dist=s_g, influence=influence, cull_mask=cull_mask, m=m, a=a,
        )

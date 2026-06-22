#
# `--deform` — per-surfel deformation field (Deformable-3DGS-style), adapted to
# this repo's hybrid Lagrangian/Eulerian pipeline.
#
# Goal: model the small movements a subject makes during capture. The Gaussians
# are stored in a CANONICAL pose; for each frame (timestep) every surfel is
# nudged by a (Δposition, Δrotation) before rasterizing, so the canonical scene
# explains every frame once the per-frame motion is factored out.
#
# Parametrization (the user's choice): a genuine PER-SURFEL latent
# `z_i ∈ R^D` (stored on the GaussianModel as `_deform_latent`, threaded through
# every clone/split/prune/PLY path like any per-Gauss tensor) decoded together
# with a per-frame TIME code by this small MLP:
#
#     (Δx_i, Δq_i) = MLP( z_i ⊕ γ(t) )
#     x'_i = x_i + Δx_i                      # canonical -> time-t position
#     q'_i = normalize(q_i + Δq_i)           # canonical -> time-t rotation (D-3DGS)
#
# γ(t) is a Fourier encoding of the scalar normalized frame time t∈[0,1], which
# gives temporal smoothness for free (adjacent frames deform similarly).
#
# The last MLP layer is ZERO-initialized → Δx=Δq=0 at iter 0 → byte-identical to
# no deformation at the start. The deformation is applied in render() (Python,
# before the rasterizer call), so the CUDA hash query — reconstructed from the
# means we pass in — samples at the DEFORMED position automatically. No CUDA
# rebuild; composes with --3rgs (deform per-surfel, then per-camera rigid).

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TimeEncoder(nn.Module):
    """Fourier (positional) encoding of a scalar time t∈[0,1] → [1 + 2L]."""

    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = 1 + 2 * num_freqs
        # frequencies 2^0..2^(L-1) * pi
        freqs = (2.0 ** torch.arange(num_freqs).float()) * torch.pi
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [...] (scalar or batch). Returns [..., 1 + 2L].
        t = t.reshape(-1, 1)                       # [B, 1]
        ang = t * self.freqs[None, :]              # [B, L]
        return torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, 1+2L]


class DeformModel(nn.Module):
    """Per-surfel latent + time → (Δposition, Δrotation)."""

    def __init__(self, latent_dim: int, width: int = 128, depth: int = 4,
                 num_time_freqs: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_enc = _TimeEncoder(num_time_freqs)
        in_dim = latent_dim + self.time_enc.out_dim
        layers = [nn.Linear(in_dim, width), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU(inplace=True)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, 7)            # 3 Δpos + 4 Δrot(quat residual)
        # Zero-init the output head ⇒ identity deformation at iter 0.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, latent: torch.Tensor, t):
        """latent: [N, D]; t: python float or scalar tensor (frame time in [0,1]).
        Returns (d_xyz [N,3], d_rot [N,4])."""
        N = latent.shape[0]
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=latent.device)
        tcode = self.time_enc(t.to(latent.device)).expand(N, -1)   # [N, 1+2L]
        h = self.backbone(torch.cat([latent, tcode], dim=-1))      # [N, width]
        out = self.head(h)                                         # [N, 7]
        d_xyz = out[:, :3]
        d_rot = out[:, 3:]
        return d_xyz, d_rot


def apply_deform(means3D, rotations, d_xyz, d_rot):
    """Apply a deformation (Δpos, Δquat-residual) to canonical means/rotations.
    Mirrors Deformable-3DGS: position adds, rotation adds-then-renormalizes.
    rotations may be None (compute_cov3D_python path) → only means are moved."""
    means3D = means3D + d_xyz
    if rotations is not None:
        rotations = F.normalize(rotations + d_rot, dim=-1)
    return means3D, rotations

#
# MCMC Relocation Utilities
# Based on "3D Gaussian Splatting as Markov Chain Monte Carlo"
#

from diff_surfel_rasterization import compute_relocation
import torch
import math

# Maximum value for relocation ratio
N_MAX = 51

# Precompute binomial coefficients on CUDA
BINOMS = None

def _init_binoms():
    """Initialize binomial coefficients tensor on CUDA."""
    global BINOMS
    if BINOMS is None:
        BINOMS = torch.zeros((N_MAX, N_MAX)).float().cuda()
        for n in range(N_MAX):
            for k in range(n + 1):
                BINOMS[n, k] = math.comb(n, k)
    return BINOMS

def compute_relocation_cuda(opacity_old, scale_old, N):
    """
    Compute new opacities and scales using the MCMC relocation kernel.
    
    This implements Equation (9) from the 3DGS-MCMC paper, which computes
    how to split a Gaussian into N children while preserving the overall
    contribution to the rendered image.
    
    Args:
        opacity_old (torch.Tensor): [M] tensor of opacities (after sigmoid activation)
        scale_old (torch.Tensor): [M, 2] tensor of 2D scales (after exp activation)
        N (torch.Tensor): [M] tensor of relocation ratios (how many children each Gaussian produces)
    
    Returns:
        new_opacity (torch.Tensor): [M] tensor of updated opacities
        new_scale (torch.Tensor): [M, 2] tensor of updated scales
    """
    binoms = _init_binoms()
    
    # Ensure N is in valid range
    N = N.clone()
    N.clamp_(min=1, max=N_MAX - 1)
    
    # Make tensors contiguous
    opacity_old = opacity_old.contiguous()
    scale_old = scale_old.contiguous()
    N = N.int().contiguous()
    
    new_opacity, new_scale = compute_relocation(
        opacity_old, scale_old, N, binoms, N_MAX
    )
    
    return new_opacity, new_scale


#
# diff_surfel_bake: Baking submodule for MLP + hashgrid → SH coefficients
# Cloned from diff_surfel_3D_sh, stripped to baking-only interface.
#

import torch
from . import _C


def set_mlp_weights(W1, W2, W3, is_sh_mode=True):
    """
    Copy MLP weights to CUDA global memory for bake kernel.

    Args:
        W1: [32, 32] Layer 1 weights (WMMA-padded, col 24 = implicit bias)
        W2: [32, 32] Layer 2 weights
        W3: [48, 32] Layer 3 weights (48 SH coefficients)
        is_sh_mode: True (always True for baking)
    """
    _C.set_mlp_weights(W1.contiguous(), W2.contiguous(), W3.contiguous(), is_sh_mode)


def bake_gaussians(centers, quats, scales, gauss_features,
                   hash_features, level_offsets,
                   voxel_min, voxel_max, l_scale, Base,
                   align_corners, interp, if_contract,
                   active_hashgrid_levels, appearance_levels,
                   grid_size=8, uv_extent=4.0):
    """
    Bake MLP → SH coefficients at NxN grid per Gaussian.

    Samples the hash encoding + MLP at UV grid points on each surfel's
    tangent plane over [-uv_extent, +uv_extent] in parametric space.
    uv_extent should match the AABB cutoff used during training (4.0 for Gaussian kernel).

    Returns:
        [N, grid_size*grid_size, 48] float tensor of SH coefficients
    """
    return _C.bake_gaussians(
        centers, quats, scales, gauss_features,
        hash_features, level_offsets,
        voxel_min, voxel_max, l_scale, Base,
        align_corners, interp, if_contract,
        active_hashgrid_levels, appearance_levels,
        grid_size, uv_extent
    )

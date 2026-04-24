"""
Baked rendering submodule — forward-only 2DGS rasterizer with residual texture support.

Usage:
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    settings = GaussianRasterizationSettings(...)
    rasterizer = GaussianRasterizer(settings)
    color, radii = rasterizer(
        means3D=..., means2D=..., opacities=..., shs=...,
        scales=..., rotations=...,
        residual_textures=residual_flat,  # [N, 192] FP16 or None
    )
"""

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

# Module-level persistent CUDA buffers for allocation reuse.
# After the first frame, geom/binning/img buffers are already the right size
# and resizeFunctional becomes a no-op (no cudaMalloc).
_buffer_cache = {}

def _get_buffers(device):
    key = str(device)
    if key not in _buffer_cache:
        dev = torch.device(device)
        _buffer_cache[key] = {
            'geom': torch.empty(0, dtype=torch.uint8, device=dev),
            'binning': torch.empty(0, dtype=torch.uint8, device=dev),
            'img': torch.empty(0, dtype=torch.uint8, device=dev),
        }
    return _buffer_cache[key]


def rasterize_gaussians(
    means3D, means2D, sh, colors_precomp, opacities,
    scales, rotations, settings,
    shapes=None, kernel_type=0, residual_textures=None,
    atlas_texture=None, atlas_rects=None, atlas_width=0,
    sb_params=None, sb_number=0,
):
    return _RasterizeGaussians.apply(
        means3D, means2D, sh, colors_precomp, opacities,
        scales, rotations, settings,
        shapes, kernel_type, residual_textures,
        atlas_texture, atlas_rects, atlas_width,
        sb_params, sb_number,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means3D, means2D, sh, colors_precomp, opacities,
        scales, rotations, settings,
        shapes, kernel_type, residual_textures,
        atlas_texture, atlas_rects, atlas_width,
        sb_params, sb_number,
    ):
        if shapes is None:
            shapes = torch.Tensor([]).cuda()
        if residual_textures is None:
            residual_textures = torch.Tensor([]).half().cuda()
        if atlas_texture is None:
            atlas_texture = torch.Tensor([]).half().cuda()
        if atlas_rects is None:
            atlas_rects = torch.Tensor([]).float().cuda()
        if sb_params is None:
            sb_params = torch.Tensor([]).float().cuda()

        buffers = _get_buffers(means3D.device)

        args = (
            settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            settings.scale_modifier,
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            settings.image_height,
            settings.image_width,
            sh,
            settings.sh_degree,
            settings.campos,
            settings.prefiltered,
            settings.debug,
            settings.beta,
            shapes,
            kernel_type,
            residual_textures,
            atlas_texture,
            atlas_rects,
            atlas_width,
            settings.aabb_mode,
            sb_params,
            sb_number,
            buffers['geom'],
            buffers['binning'],
            buffers['img'],
        )

        num_rendered, color, radii, buffers['geom'], buffers['binning'], buffers['img'] = _C.rasterize_gaussians(*args)

        return color, radii

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Baked rendering is inference-only, no backward pass")


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    beta: float
    aabb_mode: int = 3  # 0=square, 1=square+AdR, 2=rect, 3=rect+AdR


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None,
                scales=None, rotations=None, shapes=None, kernel_type=0,
                residual_textures=None,
                atlas_texture=None, atlas_rects=None, atlas_width=0,
                sb_params=None, sb_number=0):

        settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Provide exactly one of either SHs or precomputed colors!')

        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()
        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()

        return rasterize_gaussians(
            means3D, means2D, shs, colors_precomp, opacities,
            scales, rotations, settings,
            shapes, kernel_type, residual_textures,
            atlas_texture, atlas_rects, atlas_width,
            sb_params, sb_number,
        )


def set_activation_bias(sh_bias=0.5, res_bias=0.0):
    """Mirror diff_surfel_3D_sh_res.set_activation_bias on the baked renderer."""
    _C.set_activation_bias(float(sh_bias), float(res_bias))


def set_compact_mult(val=1.0):
    """FastGS Compact Box multiplier on the baked AdR cutoff. val=1.0 disables the shrink."""
    _C.set_compact_mult(float(val))


def clear_atlas_cache():
    """Release all cached uint8 atlases and their cudaTextureObject_t handles.

    Call this if you load a new/different atlas between renders, or at shutdown.
    Safe to call repeatedly.
    """
    _C.clear_atlas_cache()


def set_atlas_use_uint8(val=True):
    """Switch atlas hardware-texture encoding.

    True (default)  → uint8-quantized (~±6σ), ¼ memory, hw bilinear.
    False           → half4 (lossless vs training FP16 storage), larger memory.

    Flushes the atlas cache so the next render rebuilds with the chosen format.
    """
    _C.set_atlas_use_uint8(bool(val))


def set_use_atlas_tex_object(val=True):
    """Switch between hardware texture object path (default) and the pre-texture
    software path (raw FP16 global-memory reads + manual bilinear math in the
    render kernel).

    True  → hw texture (fast, uses texture cache + hw bilinear).
    False → software bilinear (legacy path; useful for A/B benchmarking).
    """
    _C.set_use_atlas_tex_object(bool(val))

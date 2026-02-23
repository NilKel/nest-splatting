"""
Baked rendering submodule — forward-only 2DGS rasterizer with residual texture support.

Usage:
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    settings = GaussianRasterizationSettings(...)
    rasterizer = GaussianRasterizer(settings)
    color, radii, depth = rasterizer(
        means3D=..., means2D=..., opacities=..., shs=...,
        scales=..., rotations=...,
        residual_textures=residual_flat,  # [N, 192] FP16 or None
    )
"""

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def rasterize_gaussians(
    means3D, means2D, sh, colors_precomp, opacities,
    scales, rotations, settings,
    shapes=None, kernel_type=0, residual_textures=None,
    atlas_texture=None, atlas_rects=None, atlas_width=0,
):
    return _RasterizeGaussians.apply(
        means3D, means2D, sh, colors_precomp, opacities,
        scales, rotations, settings,
        shapes, kernel_type, residual_textures,
        atlas_texture, atlas_rects, atlas_width,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means3D, means2D, sh, colors_precomp, opacities,
        scales, rotations, settings,
        shapes, kernel_type, residual_textures,
        atlas_texture, atlas_rects, atlas_width,
    ):
        if shapes is None:
            shapes = torch.Tensor([]).cuda()
        if residual_textures is None:
            residual_textures = torch.Tensor([]).half().cuda()
        if atlas_texture is None:
            atlas_texture = torch.Tensor([]).half().cuda()
        if atlas_rects is None:
            atlas_rects = torch.Tensor([]).float().cuda()

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
        )

        num_rendered, color, others, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # No backward needed — inference only
        return color, radii, others

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


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None,
                scales=None, rotations=None, shapes=None, kernel_type=0,
                residual_textures=None,
                atlas_texture=None, atlas_rects=None, atlas_width=0):

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
        )

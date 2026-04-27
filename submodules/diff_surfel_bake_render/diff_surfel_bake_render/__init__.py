"""
Baked rendering submodule — forward-only 2DGS rasterizer + atlas residual.

Inference-only: no backward, no densification gradients. All fixed-shape
buffers are cached at module level and reused frame-to-frame.

Atlas mode is the only supported residual path. The legacy per-Gaussian
8×8 shared `residual_textures` and the 48D-SH residual code paths are gone.

Usage:
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    settings = GaussianRasterizationSettings(...)
    rasterizer = GaussianRasterizer(settings)
    color, radii = rasterizer(
        means3D=..., opacities=..., shs=...,
        scales=..., rotations=...,
        atlas_texture=..., atlas_rects=..., atlas_width=4096,
    )

    # Pre-activated tensors fast-path (skip Python property accessors per frame):
    pkg = prepare_gaussian_inputs(gaussians, sh_degree=3, kernel_type=kernel_type)
    color, _ = rasterizer(**pkg, atlas_texture=..., atlas_rects=..., atlas_width=...)
"""

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

# ---------------------------------------------------------------------------
# Persistent buffer caches.
# ---------------------------------------------------------------------------
_scratch_cache = {}     # (device,) -> {'geom', 'binning', 'img'} uint8 scratch buffers
_radii_cache = {}       # (device,) -> int32[P]
_out_color_cache = {}   # (device, H, W) -> float32[3, H, W]
_empty_cache = {}       # (device, dtype_key) -> empty tensor singleton


def _scratch_buffers(device):
    key = str(device)
    if key not in _scratch_cache:
        dev = torch.device(device)
        _scratch_cache[key] = {
            'geom': torch.empty(0, dtype=torch.uint8, device=dev),
            'binning': torch.empty(0, dtype=torch.uint8, device=dev),
            'img': torch.empty(0, dtype=torch.uint8, device=dev),
        }
    return _scratch_cache[key]


def _get_radii(device, P):
    key = str(device)
    buf = _radii_cache.get(key)
    if buf is None or buf.numel() < P:
        buf = torch.empty(P, dtype=torch.int32, device=device)
        _radii_cache[key] = buf
    return buf


def _get_out_color(device, H, W):
    key = (str(device), int(H), int(W))
    buf = _out_color_cache.get(key)
    if buf is None:
        # Zero-init once; the kernel overwrites every inside pixel each frame.
        buf = torch.zeros(3, H, W, dtype=torch.float32, device=device)
        _out_color_cache[key] = buf
    return buf


def _empty(device, dtype_key, dtype):
    key = (str(device), dtype_key)
    t = _empty_cache.get(key)
    if t is None:
        t = torch.empty(0, dtype=dtype, device=device)
        _empty_cache[key] = t
    return t


def rasterize_gaussians(
    means3D, sh, colors_precomp, opacities,
    scales, rotations, settings,
    shapes=None, kernel_type=0,
    atlas_texture=None, atlas_rects=None, atlas_width=0,
    sb_params=None, sb_number=0,
):
    return _RasterizeGaussians.apply(
        means3D, sh, colors_precomp, opacities,
        scales, rotations, settings,
        shapes, kernel_type,
        atlas_texture, atlas_rects, atlas_width,
        sb_params, sb_number,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means3D, sh, colors_precomp, opacities,
        scales, rotations, settings,
        shapes, kernel_type,
        atlas_texture, atlas_rects, atlas_width,
        sb_params, sb_number,
    ):
        device = means3D.device

        # Optional tensors → cached empty singletons.
        if shapes is None:
            shapes = _empty(device, 'f32', torch.float32)
        if atlas_texture is None:
            atlas_texture = _empty(device, 'f16', torch.float16)
        if atlas_rects is None:
            atlas_rects = _empty(device, 'f32', torch.float32)
        if sb_params is None:
            sb_params = _empty(device, 'f32', torch.float32)

        scratch = _scratch_buffers(device)
        H = settings.image_height
        W = settings.image_width
        P = means3D.shape[0]
        out_color = _get_out_color(device, H, W)
        radii = _get_radii(device, P)

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
            H,
            W,
            sh,
            settings.sh_degree,
            settings.campos,
            settings.prefiltered,
            settings.debug,
            settings.beta,
            shapes,
            kernel_type,
            atlas_texture,
            atlas_rects,
            atlas_width,
            settings.aabb_mode,
            sb_params,
            sb_number,
            scratch['geom'],
            scratch['binning'],
            scratch['img'],
            out_color,
            radii,
        )

        scratch['geom'], scratch['binning'], scratch['img'] = _C.rasterize_gaussians(*args)
        return out_color, radii

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

    def forward(self, means3D, opacities, shs=None, colors_precomp=None,
                scales=None, rotations=None, shapes=None, kernel_type=0,
                atlas_texture=None, atlas_rects=None, atlas_width=0,
                sb_params=None, sb_number=0,
                # Backward-compat: old callers passed `means2D` / `residual_textures`,
                # both unused now. Accept silently.
                means2D=None, residual_textures=None):

        settings = self.raster_settings
        device = means3D.device

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Provide exactly one of either SHs or precomputed colors!')

        if shs is None:
            shs = _empty(device, 'f32', torch.float32)
        if colors_precomp is None:
            colors_precomp = _empty(device, 'f32', torch.float32)
        if scales is None:
            scales = _empty(device, 'f32', torch.float32)
        if rotations is None:
            rotations = _empty(device, 'f32', torch.float32)

        return rasterize_gaussians(
            means3D, shs, colors_precomp, opacities,
            scales, rotations, settings,
            shapes, kernel_type,
            atlas_texture, atlas_rects, atlas_width,
            sb_params, sb_number,
        )


# ---------------------------------------------------------------------------
# Pre-activation helper.
# Property accessors on a GaussianModel apply activations (exp, sigmoid,
# quat-normalize, etc.) every time they're called. For inference these are
# CONSTANT — call once after load_ply() and reuse the activated tensors
# directly instead of paying the activation cost per frame.
# ---------------------------------------------------------------------------
def prepare_gaussian_inputs(gaussians, sh_degree=3, kernel_type=0):
    """Snapshot post-activation tensors from a GaussianModel for fast inference.

    Returns a dict with:
      means3D, opacities, scales, rotations, shs, shapes (or None),
      sb_params (flat [N*K*6] or None), sb_number (int).

    All tensors are made `.contiguous()` so the C++ side can skip the per-call
    contiguity check. Pass the dict as **kwargs to `GaussianRasterizer.forward`.
    """
    pkg = {
        'means3D':    gaussians.get_xyz.contiguous(),
        'opacities':  gaussians.get_opacity.contiguous(),
        'scales':     gaussians.get_scaling.contiguous(),
        'rotations':  gaussians.get_rotation.contiguous(),
        'shs':        gaussians.get_features.contiguous(),
    }
    # Beta-kernel shapes (only when actually using a beta variant).
    shapes = None
    if kernel_type > 0 and hasattr(gaussians, '_shape') \
       and gaussians._shape is not None and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape.contiguous()
    pkg['shapes'] = shapes
    pkg['kernel_type'] = kernel_type
    return pkg


# ---------------------------------------------------------------------------
# Cached rasterizer: one GaussianRasterizer instance per (H, W). Settings
# tuple is rebuilt per frame (cheap), but the wrapper nn.Module is reused.
# Works for interactive renderers where camera changes per frame.
# ---------------------------------------------------------------------------
_rasterizer_cache = {}  # (H, W, sh_degree, beta, aabb_mode) -> GaussianRasterizer


def get_rasterizer(image_height, image_width, tanfovx, tanfovy, bg,
                   viewmatrix, projmatrix, campos,
                   sh_degree=3, beta=0.0, aabb_mode=3,
                   scale_modifier=1.0, prefiltered=False, debug=False):
    """Build (or reuse) a GaussianRasterizer with fresh per-frame camera params.

    The `nn.Module` wrapper is cached keyed by (H, W, sh_degree, beta, aabb_mode).
    The settings NamedTuple is rebuilt every call (microsecond-cheap) so
    viewmatrix / projmatrix / campos / tanfov can change per frame.
    """
    key = (int(image_height), int(image_width), int(sh_degree),
           float(beta), int(aabb_mode))
    settings = GaussianRasterizationSettings(
        image_height=image_height, image_width=image_width,
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg, scale_modifier=scale_modifier,
        viewmatrix=viewmatrix, projmatrix=projmatrix,
        sh_degree=sh_degree, campos=campos,
        prefiltered=prefiltered, debug=debug,
        beta=beta, aabb_mode=aabb_mode,
    )
    rasterizer = _rasterizer_cache.get(key)
    if rasterizer is None:
        rasterizer = GaussianRasterizer(raster_settings=settings)
        _rasterizer_cache[key] = rasterizer
    else:
        rasterizer.raster_settings = settings
    return rasterizer


def set_activation_bias(sh_bias=0.5, res_bias=0.0):
    _C.set_activation_bias(float(sh_bias), float(res_bias))


def set_compact_mult(val=1.0):
    _C.set_compact_mult(float(val))


def clear_atlas_cache():
    _C.clear_atlas_cache()


def set_atlas_use_uint8(val=True):
    _C.set_atlas_use_uint8(bool(val))


def set_use_atlas_tex_object(val=True):
    _C.set_use_atlas_tex_object(bool(val))


def clear_buffer_cache():
    """Release all cached scratch / output / radii / rasterizer buffers."""
    _scratch_cache.clear()
    _radii_cache.clear()
    _out_color_cache.clear()
    _empty_cache.clear()
    _rasterizer_cache.clear()

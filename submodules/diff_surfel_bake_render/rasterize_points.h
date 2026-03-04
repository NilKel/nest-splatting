/*
 * Baked rendering submodule — forward-only, no backward.
 * Supports render_mode=6 (SH base + residual texture).
 */
#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

// Forward-only rendering: returns (num_rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer)
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,    // precomputed per-Gaussian colors (empty if using SH)
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const float beta,
	const torch::Tensor& shapes,     // beta kernel shape parameters (empty if not using)
	const int kernel_type,
	const torch::Tensor& residual_textures,  // [N, 192] FP16 residual textures (empty if SH-only)
	const torch::Tensor& atlas_texture,      // [H*W*3] FP16 atlas (empty if not atlas mode)
	const torch::Tensor& atlas_rects,        // [N, 4] float atlas UV rects (empty if not atlas mode)
	const int atlas_width,                   // atlas dimension (0 if not atlas mode)
	const int aabb_mode,                     // 0=square, 1=square+AdR, 2=rect, 3=rect+AdR
	// Persistent buffers — pass empty on first call, reused on subsequent calls
	torch::Tensor geomBuffer,
	torch::Tensor binningBuffer,
	torch::Tensor imgBuffer);

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix);

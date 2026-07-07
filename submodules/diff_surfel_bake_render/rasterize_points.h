/*
 * Baked rendering submodule — forward-only, no backward.
 * Supports render_mode=6 (SH base + residual texture).
 */
#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

// Forward-only rendering. All output tensors (out_color, radii) and scratch
// buffers are caller-owned and persistent. We return only the (possibly
// resized) scratch buffers so Python can hold their new pointers.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
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
	const torch::Tensor& atlas_texture,      // [H*W*3] FP16 atlas (required — only mode supported)
	const torch::Tensor& atlas_rects,        // [N, 4] float atlas UV rects (required)
	const int atlas_width,                   // atlas dimension
	const int aabb_mode,                     // 0=square, 1=square+AdR, 2=rect, 3=rect+AdR
	// Optional Spherical-Beta params [N, K, 6], K=sb_number (empty if SB disabled)
	const torch::Tensor& sb_params,
	const int sb_number,
	// Optional Spherical-Voronoi state (--feature SV). Pre-activated tensors:
	//   voronoi_sites  [N, K, 3]  unit vectors (caller does F.normalize(_sv_sites))
	//   voronoi_tau    [N, K]     post-exp scalars (caller does torch.exp(_sv_tau))
	//   voronoi_colors [N, K, 3]  raw RGB (no activation; ReLU + sh_bias in CUDA)
	// When voronoi_K > 0 AND colors_precomp is empty, preprocessCUDA replaces
	// computeColorFromSH with computeColorFromVoronoi (per-Gaussian fused).
	// Bit-equivalent to nest's torch eval_voronoi_sv → fake-SH-DC path.
	const torch::Tensor& voronoi_sites,
	const torch::Tensor& voronoi_tau,
	const torch::Tensor& voronoi_colors,
	const int voronoi_K,
	// Persistent scratch buffers (resized in place when growth is needed).
	torch::Tensor geomBuffer,
	torch::Tensor binningBuffer,
	torch::Tensor imgBuffer,
	// Persistent caller-owned outputs (Python pre-allocates and reuses).
	torch::Tensor out_color,
	torch::Tensor radii,
	const int sort_mode = 0,                 // 0 = legacy 64-bit single sort, 1 = FastGS two-stage
	// `--method mixed_3d`: per-Gauss textured flag [P] + activated 3rd-axis
	// scale [P]. Empty → pure 2DGS bake (unchanged).
	const torch::Tensor& is_textured = torch::Tensor(),
	const torch::Tensor& scaling_z = torch::Tensor());

// Device-global setters mirroring training-time setters.
void SetActivationBiasBakeCUDA(float sh_bias, float res_bias);
void SetCompactMultBakeCUDA(float val);
void SetBetaMultBakeCUDA(float val);
void SetDropLowpassBakeCUDA(bool val);
void SetOpacityAwareBetaBakeCUDA(bool val);
void SetResidualModeBakeCUDA(int mode);
void SetUntexKernelBakeCUDA(int v);

// BC7 atlas (Phase 2). Pass empty tensor + zeros to clear.
void SetAtlasBC7CUDA(torch::Tensor bc7_bytes, int W, int H, float offset, float scale);
void ClearAtlasBC7CUDA();

// RVQ atlas — codebooks + surfel-major indices + per-surfel cumulative
// block-offset. After this, render kernel does L-stage codebook decode
// at each atlas-sample site. Call SetAtlasRVQCUDA with empty tensors or
// ClearAtlasRVQCUDA() to revert.
void SetAtlasRVQCUDA(torch::Tensor codebooks_fp16,
                     torch::Tensor indices_u8,
                     torch::Tensor surfel_offsets_i64,
                     int B,
                     float atlas_scale,
                     float atlas_offset);
void ClearAtlasRVQCUDA();
void SetAtlasRVQBilinearCUDA(bool val);
void SetAtlasRVQUseSharedCBCUDA(bool val);
void SetAtlasRVQUseTexCBCUDA(bool val);
void SetAtlasRVQUseTexIdxCUDA(bool val);

// Frees the cached cudaArrays + cudaTextureObjects for all atlases seen so far.
void ClearAtlasCacheCUDA();

// Switch atlas encoding: true = uint8 quantized (smaller, hw bilinear), false = half4 (lossless).
// Flushes the cache so next render rebuilds with the chosen format.
void SetAtlasUseUint8CUDA(bool val);

// Runtime toggle: hardware texture object (true, default) vs pre-texture
// software path (false — raw FP16 global reads + manual bilinear in kernel).
void SetUseAtlasTexObjectCUDA(bool val);

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix);

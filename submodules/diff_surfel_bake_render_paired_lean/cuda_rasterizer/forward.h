/*
 * Baked rendering submodule — forward-only.
 */
#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>  // cudaTextureObject_t

namespace FORWARD
{
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		int* radii_x,
		int* radii_y,
		float2* points_xy_image,
		float* depths,
		float* transMats,
		__half* colors,
		float4* normal_opacity,
		float4* conic_t,
		const dim3 grid,
		uint32_t* tiles_touched,        // both modes: per-primitive tile count
		uint32_t* depth_keys_compact,   // FastGS (sort_mode==1)
		uint32_t* prim_idx_compact,     // FastGS
		uint32_t* n_visible_atomic,     // FastGS
		uint32_t* n_instances_atomic,   // FastGS
		const int sort_mode,            // 0 = legacy, 1 = FastGS two-stage
		bool prefiltered,
		const float* shapes,
		const int kernel_type,
		const int aabb_mode = 3,
		// --feature SV: optional per-Gaussian Spherical-Voronoi state. When
		// voronoi_K > 0 AND colors_precomp == nullptr, preprocessCUDA calls
		// computeColorFromVoronoi instead of computeColorFromSH. Pre-activated:
		// sites = unit vectors, tau = post-exp scalars, colors = raw RGB.
		const float* voronoi_sites = nullptr,
		const float* voronoi_tau = nullptr,
		const float* voronoi_colors = nullptr,
		const int voronoi_K = 0,
		// SB precompute hook: when sb_number > 0, preprocessCUDA also calls
		// `eval_sb` and writes per-Gauss RGB to `sb_rgb_out` (P*3 fp16). The
		// render kernel then reads from there instead of recomputing per-pixel.
		const float* sb_params = nullptr,
		const int sb_number = 0,
		__half* sb_rgb_out = nullptr,
		// `--method mixed_3d`: per-Gauss textured flag + activated 3rd-axis
		// scale; ewa_conic scratch [P] (preprocess writes untextured rows).
		const bool* is_textured = nullptr,
		const float* scaling_z = nullptr,
		float4* ewa_conic = nullptr,
		// LEAN_CONIC Option A: [P*6] fp32 per-Gauss (u0,v0,J⁻¹) precomputed at AABB center.
		float* conic_uv = nullptr);

	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const float beta,
		int W, int H,
		const float2* points_xy_image,
		const __half* features,
		const float* transMats,
		const float* depths,
		const float4* normal_opacity,
		const float* bg_color,
		float* out_color,
		const float* shapes,
		const int kernel_type,
		const float* means3D,
		const float* cam_pos,
		const __half* atlas_texture,
		const float* atlas_rects,
		const int atlas_width,
		const float* sb_params = nullptr,
		const int sb_number = 0,
		// Per-Gauss SB RGB precomputed in preprocessCUDA. Read when sb_number > 0.
		const __half* sb_rgb_in = nullptr,
		cudaTextureObject_t atlas_tex_obj = 0,
		float atlas_offset = 0.0f,
		float atlas_scale = 1.0f,
		// `--method mixed_3d`: per-Gauss textured flag + EWA conic [P].
		const bool* is_textured = nullptr,
		const float4* ewa_conic = nullptr,
		// LEAN_CONIC Option A: [P*6] per-Gauss (u0,v0,J⁻¹) cache from preprocess.
		const float* conic_uv = nullptr);

	// Device-global setters (mirror diff_surfel_3D_sh_res training-time setters).
	void setActivationBias(float sh_bias, float res_bias);
	void setCompactMult(float val);
	// EXPERIMENT: beta_scaled footprint mult + drop low-pass (alpha max-pool + filter_r).
	void setBetaMult(float val);
	void setUntexMult(float val);
	void setDropLowpass(bool val);
	void setOpacityAwareBeta(bool val);
	// 0 = 3D_SH_res (default outer-ReLU). 1 = 3D_SH_add (separate ReLUs).
	void setResidualMode(int mode);
	// `--method mixed_3d --kernel2`: untextured-EWA kernel override (-1 = unset).
	void setUntexKernel(int v);

	// RVQ atlas decode: install codebooks + surfel-major indices + per-surfel
	// block-offset cumulative count. When set, the render kernel does an
	// L-stage codebook lookup at each fragment instead of tex2D / FP16-gather.
	// Pass nullptrs or call clearAtlasRVQ() to revert to the BC7/FP16 path.
	void setAtlasRVQ(const __half* codebooks, const uint8_t* indices,
	                 const int64_t* surfel_offsets,
	                 int L, int K, int B, unsigned long long N_used);
	void clearAtlasRVQ();
	// 1 = 4-tap bilinear (default), 0 = nearest (~4× fewer codebook reads).
	void setAtlasRVQBilinear(int v);
	// 1 = load codebook to dynamic __shared__ at kernel start (faster reads,
	// reduces occupancy if codebook is big). Caller must also set the
	// per-launch dynamic-shared byte count via setAtlasRVQSharedBytes.
	void setAtlasRVQUseSharedCB(int v);
	// Opt-in to > 48 KB dynamic shared for the render kernel. Returns true
	// on success, false if the device doesn't support `bytes` shared/block.
	bool optInRVQShared(int bytes);
	// Install codebook + indices texture objects + bool toggles for whether
	// to use the texture or fall back to global. Tex object handles are
	// owned by the caller (rasterize_points.cu).
	void setAtlasRVQTex(cudaTextureObject_t cb, cudaTextureObject_t idx,
	                    int use_cb, int use_idx);
	// Dequant params for the uint8 codebook texture. Codewords share the
	// atlas's atlas_scale/atlas_offset (same float space).
	void setAtlasRVQCBDequant(float scale, float offset);
}

#endif

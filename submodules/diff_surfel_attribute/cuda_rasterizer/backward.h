/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "rgb_type.h"  // rgb_t typedef (FP16/FP32 via FP16_RGB)

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const float beta,
		int W, int H,
		uint32_t c_dim, uint32_t level, uint32_t l_dim, float l_scale, uint32_t Base,
		bool align_corners, uint32_t interp,
		const bool if_contract,
		float focal_x, float focal_y,
		const glm::vec2* scales,
		const float* other_maps,
		const int* out_index,
		const float* bg_color,
		const float2* means2D,
		const float4* normal_opacity,
		const rgb_t* colors,  // FP16 SH baseline (geomState.rgb) — see config.h::FP16_RGB
		const float* transMats,
		const float* homotrans,
		const float* ap_level,
		const __half* hash_features,
		const int* level_offsets,
		const float* gridrange,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_depths,
		float* dL_dfeatures,
		float* dL_dtransMat,
		float* dL_dhomoMat,
		float4* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_gradsum,
		const glm::vec3* cam_pos,
		const uint32_t D_diffuse = 0,
		const float* hash_features_diffuse = nullptr,
		const int* level_offsets_diffuse = nullptr,
		const float* gridrange_diffuse = nullptr,
		float* dL_dfeatures_diffuse = nullptr,
		const int render_mode = 0,
		const float* shapes = nullptr,
		const int kernel_type = 0,
		float* dL_dshapes = nullptr,
		const bool detach_hash_grad = false,
		// MLP gradient buffers for 3D_SH_res (render_mode=5, bias-free, all [16×16])
		float* dL_dmlp_W1 = nullptr,    // [16 * 16] = 256 floats
		float* dL_dmlp_W2 = nullptr,    // [16 * 16] = 256 floats
		float* dL_dmlp_W3 = nullptr,    // [16 * 16] = 256 floats
		// DC SH features for 3D_SH_cat (render_mode=6)
		const float* dc_features = nullptr,
		// `--method mixed` per-Gauss bool flag [P] (nullptr → all-textured behavior).
		const bool* is_textured = nullptr,
		// `--method mixed_3d`: per-Gauss EWA conic [P] + activated 3rd-axis
		// scale [P]. scaling_z != nullptr forces the std (non-GEMM) backward
		// and enables the untextured EWA render-backward branch.
		const float4* ewa_conic = nullptr,
		const float* scaling_z = nullptr);

	// Set backward's own threshold copies (extern __device__ doesn't work across .cu files)
	void setContribThresh(float val);
	void setCountThresh(int val);
	void setOverdrawLambda(float val);
	void setWeightRegLambda(float val);
	void setResBias(float val);
	// 0 = 3D_SH_res (stacked outer ReLU), 1 = 3D_SH_add (separate ReLUs).
	void setResidualMode(int mode);
	// `--ste`: straight-through estimator on the per-Gauss outer ReLU.
	void setSteRelu(int v);
	void setAaKernelSize(float val);
	// Periodic-freeze flag: when true, the mode 5 backward skips all
	// hash/MLP gradient work (weight-grad GEMMs, input-chain backprop,
	// query_feature<true>, and the tile dL_dW flush). Geometry backward
	// runs unchanged. Default false restores exact pre-flag behavior.
	void setSkipMlpGrad(bool val);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* transMats,
		const float* view,
		const float* proj,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const glm::vec3* campos,
		float4* dL_dmean2D,
		const float* dL_dnormal3D,
		float* dL_dtransMat,
		float* dL_dhomoMat,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dmeans,
		glm::vec2* dL_dscale,
		glm::vec4* dL_drot,
		const bool pixel_center = false,
		// `--method mixed_3d`: untextured EWA rows VJP + dL_dscaling_z output.
		const bool* is_textured = nullptr,
		const float* scaling_z = nullptr,
		float* dL_dscaling_z = nullptr);
}

// Unified backward kernel for 3D mode that reads transMat from geomBuffer
// Computes dL_dopacity, dL_dtransMat, and dL_dmean2D in one pass from dL_dweight
// Also accepts dL_duv from hash/xyz gradient path (like cat mode)
void backward_from_weight_grad(
    int num_pixels,
    int N,
    int W, int H,
    const float* dL_dweight,
    const int* gaussian_ids,
    const int* pixel_ids,
    const int* pixel_starts,
    const float* T_values,
    const float* G_values,
    const float* alpha_values,
    const float* opacity_values,
    const float* s_x_values,
    const float* s_y_values,
    const float* rho_flag,
    const float* dL_duv_x,      // [M] hash/xyz gradient contribution (can be nullptr)
    const float* dL_duv_y,      // [M] hash/xyz gradient contribution (can be nullptr)
    const float* transMat,
    const float* mean2D_precomp, // [N*2] pre-computed mean2D from forward (can be nullptr)
    float* dL_dopacity,
    float* dL_dtransMat,
    float* dL_dmean2D);

// Convert screen-space dL_dtransMat to world-space dL_dscale and dL_drotation
// This performs the proper coordinate transformation that the native backward does:
//   P = world2ndc * ndc2pix (includes image dimension scaling!)
//   dL_dM = P * transpose(dL_dT) + dL_dhomoMat (xyz gradient contribution!)
//   dL_dscale = [dot(dL_dM[0], R[0]), dot(dL_dM[1], R[1])]
//   dL_drot = quat_to_rotmat_vjp(rot, dL_dR)
// Also handles normal gradient from depth/normal loss via dL_dnormal3D
void transMat_to_scale_rot_grad(
    int N,
    int W, int H,                // Image dimensions for ndc2pix transformation
    const float* dL_dtransMat,   // [N, 9] screen-space transMat gradient
    const float* dL_dhomoMat,    // [N, 9] xyz gradient contribution (can be nullptr)
    const float* dL_dmean2D,     // [N, 2] 2D mean gradient (can be nullptr)
    const float* dL_dnormal3D,   // [N, 3] normal gradient from depth/normal loss (can be nullptr)
    const float* means3D,        // [N, 3] world-space positions (needed for dL_dmean2D)
    const float* transMat_precomp, // [N, 9] forward pass transMat (can be nullptr)
    const float* scales,         // [N, 2]
    const float* rotations,      // [N, 4] quaternions
    const float* projmatrix,     // [16] 4x4 projection matrix
    const float* viewmatrix,     // [16] 4x4 view matrix (for normal gradient transform)
    float* dL_dscales,           // [N, 2] output
    float* dL_drots,             // [N, 4] output
    float* dL_dmeans);           // [N, 3] output (mean position gradients)

// Backward kernel profiling (clock64 instrumentation)
void resetBackwardProfile();
void readBackwardProfile(unsigned long long* cycles, unsigned int* counts);

#endif

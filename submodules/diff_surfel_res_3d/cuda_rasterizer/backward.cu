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

#include "backward.h"
#include "auxiliary.h"
#include "hashgrid.h"
#include "forward.h"  // For FORWARD::getMlpWeightPointers
#include "mma_utils.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdlib>  // for getenv, atoi

namespace cg = cooperative_groups;

// Contribution threshold — backward's own copy (forward.cu has its own)
// extern __device__ does NOT work across .cu compilation units without -rdc=true
__device__ float d_contrib_thresh_bw = 0.0f;
__device__ int d_count_thresh_bw = 0;
__device__ float d_overdraw_lambda_bw = 0.0f;
__device__ float d_weight_reg_lambda_bw = 0.0f;  // Weight-squared reg: -lambda * 2 * w * T per Gaussian
__device__ float d_res_bias = 0.5f;  // Residual activation bias: ReLU(residual + d_res_bias)
// d_ste_relu: straight-through estimator on the per-Gauss outer ReLU
// (mode 0). When 1, backward bypasses the clamp gate → gradient passes as
// identity even at clamped activations. Default 0 = exact gradient.
__device__ int d_ste_relu = 0;
// `--lru`: leaky-ReLU slope α for the outer per-Gauss ReLU (mode 0). Backward
// uses this as the gate value at clamped sites (pre ≤ 0): α = 0 (default) →
// standard ReLU. α > 0 → grad scaled by α at clamped sites. STE overrides LRU
// at clamped sites where dL/dpixel < 0 (release-clamp direction).
__device__ float d_lru_slope = 0.0f;
// d_residual_mode: 0 = 3D_SH_res (stacked outer ReLU), 1 = 3D_SH_add (separate ReLUs).
// Mirrors forward.cu — the backward gradient routing differs between the two modes.
__device__ int d_residual_mode = 0;
// `--method res_3d_double`: backward mirror of FORWARD's d_textured_bias_gate.
// Default 1 → textured carriers force sh_color = 0 (matches plain `res_3d`).
// 0 → keep SH path active (matches `res_3d_double` design).
__device__ int d_textured_bias_gate = 1;
__device__ float d_aa_kernel_size = 0.0f;  // AA-2DGS Jacobian mip filter σ (0 = off)
// Periodic-freeze flag for the mode 5 (3D_SH_res) backward. When true, the
// kernel skips EVERYTHING hash/MLP-gradient-related: the 3 weight-grad WMMA
// GEMMs, the scalar W^T input-chain backprop (Phase 2/3/4), the
// query_feature<true> call (hash-table dL_dgrid + dL/dxyz-from-hash), and
// the tile-level dL_dW flush. Geometry backward (transMat, normals, alpha,
// opacity) is untouched. Default false — when flag is false, ALL gates below
// evaluate true and the kernel runs byte-for-byte identically to pre-flag.
// Controlled from Python via BACKWARD::setSkipMlpGrad().
__device__ bool d_skip_mlp_grad = false;

// ============================================================================
// BACKWARD KERNEL PROFILING (clock64 instrumentation)
// Measures cycle counts per phase of the mode 5 backward, thread 0 per block
// ============================================================================
__device__ unsigned long long d_bw_profile[6] = {0};
// [0] = Phase A: intersection + MLP forward + sigmoid bw + GEMM L3 + dL_dz2
// [1] = Phase B: GEMM L2 + dL_dz1
// [2] = Phase C: GEMM L1 + dL_dinput + feature/hash/geometry grads
// [3] = Tile flush cycles
// [4] = Total cycles (ballot to end, all Gaussians)
// [5] = Reserved
__device__ unsigned int d_bw_profile_counts[4] = {0};
// [0] = Total Gaussians processed (with n_active > 0)
// [1] = Total Gaussians skipped (ballot skip)
// [2] = Total tiles processed
// [3] = Total intersections (participating threads)

// ============================================================================
// MLP weight pointers are passed as kernel parameters (not via extern device)
// This avoids cross-compilation-unit issues without -rdc=true
// ============================================================================

// View direction encoding helper (matching forward.cu)
__device__ void encode_view_direction_bw(const float3& view_dir, float* view_enc) {
    const float pi = 3.14159265358979323846f;
    // Base direction (3D)
    view_enc[0] = view_dir.x;
    view_enc[1] = view_dir.y;
    view_enc[2] = view_dir.z;
    // Frequency band 1: sin/cos(pi * dir)
    view_enc[3] = sinf(pi * view_dir.x);
    view_enc[4] = cosf(pi * view_dir.x);
    view_enc[5] = sinf(pi * view_dir.y);
    view_enc[6] = cosf(pi * view_dir.y);
    view_enc[7] = sinf(pi * view_dir.z);
    view_enc[8] = cosf(pi * view_dir.z);
    // Frequency band 2: sin/cos(2*pi * dir)
    view_enc[9] = sinf(2.0f * pi * view_dir.x);
    view_enc[10] = cosf(2.0f * pi * view_dir.x);
    view_enc[11] = sinf(2.0f * pi * view_dir.y);
    view_enc[12] = cosf(2.0f * pi * view_dir.y);
    view_enc[13] = sinf(2.0f * pi * view_dir.z);
    view_enc[14] = cosf(2.0f * pi * view_dir.z);
    // Pad to 16D
    view_enc[15] = 0.0f;
}

// Include mode-specific implementations AFTER extern declarations and function definitions
// Define guard so mode_3d_direct_fused.cu skips duplicate declarations
#define BACKWARD_CU_INCLUDES_MODE
#include "modes/mode_3d_direct_fused.cu"

// mlp_forward_for_backward removed — use mlp_forward_inline from mode_3d_direct_fused.cu
// No pre-activation storage needed: ReLU derivative = (h_post > 0)

// MLP backward pass - computes gradients for weights and input (legacy path)
// Uses h_post > 0 for ReLU derivative (no pre-activation storage needed)
template <int IN_DIM, int HIDDEN_DIM, int OUT_DIM>
__device__ void mlp_backward(
    const float* input,
    const float* output,      // Forward pass output (after sigmoid)
    const float* dL_doutput,  // Gradient w.r.t. output
    const float* h1_post,
    const float* h2_post,
    float* dL_dinput,         // Gradient w.r.t. input [IN_DIM]
    float* dL_dW1,            // [HIDDEN_DIM * IN_DIM]
    float* dL_dW2,            // [HIDDEN_DIM * HIDDEN_DIM]
    float* dL_dW3,            // [OUT_DIM * HIDDEN_DIM]
    const MlpWeights& mlp,
    bool applied_sigmoid = true
) {
    float dL_dz3[OUT_DIM];
    #pragma unroll
    for (int o = 0; o < OUT_DIM; o++) {
        if (applied_sigmoid) {
            float sig = output[o];
            dL_dz3[o] = dL_doutput[o] * sig * (1.0f - sig);
        } else {
            dL_dz3[o] = dL_doutput[o];
        }
    }

    float dL_dh2_post[HIDDEN_DIM] = {0};
    #pragma unroll
    for (int o = 0; o < OUT_DIM; o++) {
        float dz = dL_dz3[o];
        #pragma unroll
        for (int h = 0; h < HIDDEN_DIM; h++) {
            atomicAdd(&dL_dW3[o * HIDDEN_DIM + h], dz * h2_post[h]);
            dL_dh2_post[h] += dz * __half2float(mlp.W3[o * HIDDEN_DIM + h]);
        }
    }

    // ReLU backward: h_post > 0 equivalent to h_pre > 0
    float dL_dz2[HIDDEN_DIM];
    #pragma unroll
    for (int h = 0; h < HIDDEN_DIM; h++) {
        dL_dz2[h] = (h2_post[h] > 0) ? dL_dh2_post[h] : 0.0f;
    }

    float dL_dh1_post[HIDDEN_DIM] = {0};
    #pragma unroll
    for (int h = 0; h < HIDDEN_DIM; h++) {
        float dz = dL_dz2[h];
        #pragma unroll
        for (int i = 0; i < HIDDEN_DIM; i++) {
            atomicAdd(&dL_dW2[h * HIDDEN_DIM + i], dz * h1_post[i]);
            dL_dh1_post[i] += dz * __half2float(mlp.W2[h * HIDDEN_DIM + i]);
        }
    }

    float dL_dz1[HIDDEN_DIM];
    #pragma unroll
    for (int h = 0; h < HIDDEN_DIM; h++) {
        dL_dz1[h] = (h1_post[h] > 0) ? dL_dh1_post[h] : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < IN_DIM; i++) {
        dL_dinput[i] = 0.0f;
    }
    #pragma unroll
    for (int h = 0; h < HIDDEN_DIM; h++) {
        float dz = dL_dz1[h];
        #pragma unroll
        for (int i = 0; i < IN_DIM; i++) {
            atomicAdd(&dL_dW1[h * IN_DIM + i], dz * input[i]);
            dL_dinput[i] += dz * __half2float(mlp.W1[h * IN_DIM + i]);
        }
    }
}

// Input-only backward: same as mlp_backward but skips weight gradient atomicAdds.
// Used for freeze_mlp mode where we only need dL/d_input for hash gradients.
template<int IN_DIM, int HIDDEN_DIM, int OUT_DIM>
__device__ void mlp_backward_input_only(
    const float* input,
    const float* output,
    const float* dL_doutput,
    const float* h1_post,
    const float* h2_post,
    float* dL_dinput,
    const MlpWeights& mlp,
    bool applied_sigmoid = true
) {
    float dL_dz3[OUT_DIM];
    #pragma unroll
    for (int o = 0; o < OUT_DIM; o++) {
        if (applied_sigmoid) {
            float sig = output[o];
            dL_dz3[o] = dL_doutput[o] * sig * (1.0f - sig);
        } else {
            dL_dz3[o] = dL_doutput[o];
        }
    }

    float dL_dh2_post[HIDDEN_DIM] = {0};
    #pragma unroll
    for (int o = 0; o < OUT_DIM; o++) {
        float dz = dL_dz3[o];
        #pragma unroll
        for (int h = 0; h < HIDDEN_DIM; h++) {
            dL_dh2_post[h] += dz * __half2float(mlp.W3[o * HIDDEN_DIM + h]);
        }
    }

    float dL_dz2[HIDDEN_DIM];
    #pragma unroll
    for (int h = 0; h < HIDDEN_DIM; h++) {
        dL_dz2[h] = (h2_post[h] > 0) ? dL_dh2_post[h] : 0.0f;
    }

    float dL_dh1_post[HIDDEN_DIM] = {0};
    #pragma unroll
    for (int h = 0; h < HIDDEN_DIM; h++) {
        float dz = dL_dz2[h];
        #pragma unroll
        for (int i = 0; i < HIDDEN_DIM; i++) {
            dL_dh1_post[i] += dz * __half2float(mlp.W2[h * HIDDEN_DIM + i]);
        }
    }

    float dL_dz1[HIDDEN_DIM];
    #pragma unroll
    for (int h = 0; h < HIDDEN_DIM; h++) {
        dL_dz1[h] = (h1_post[h] > 0) ? dL_dh1_post[h] : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < IN_DIM; i++) {
        dL_dinput[i] = 0.0f;
    }
    #pragma unroll
    for (int h = 0; h < HIDDEN_DIM; h++) {
        float dz = dL_dz1[h];
        #pragma unroll
        for (int i = 0; i < IN_DIM; i++) {
            dL_dinput[i] += dz * __half2float(mlp.W1[h * IN_DIM + i]);
        }
    }
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ other_maps,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float4* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	// Dead-code declarations kept in sync with renderCUDAsurfelBackward to
	// match the renamed load loop above. renderCUDA itself is not launched.
	float dL_dpixel_tex[C];
	float dL_dpixel_untex[C];
	// Alias so the unused renderCUDA body (which still references `dL_dpixel`)
	// compiles. Resolves to the tex array — no semantic change for dead code.
	float* const dL_dpixel = dL_dpixel_tex;
	// `--l2` is not plumbed into the unused kernel; pinning to nullptr makes the
	// shared load loop's `dL_dpixels_untex != nullptr` check fold to false.
	const float* const dL_dpixels_untex = nullptr;

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	// `--method mixed_3d`: slot 3 holds A_tex — the explicit textured-only
	// running sum of w_j = α_j · T_full_j (NOT 1 - T_tex). The 2DGS distortion
	// identity L_dist = Σ w_i (m_i² A − 2 m_i M1 + M2) requires A, M1, M2 all
	// to be partial sums over the same subset; for textured-only we must
	// accumulate A explicitly because `1 - T_tex ≠ Σ_textured w` when EWA
	// absorbers are interleaved (they scale subsequent w's via T_full but
	// don't move T_tex). For non-mixed runs (no untex) A_tex telescopes to
	// 1 - T, so this is a no-op there.
	const float final_A_tex = inside ? final_Ts[pix_id + 3 * H * W] : 0.0f;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++) {
			dL_dpixel_tex[i] = dL_dpixels[i * H * W + pix_id];
			// When dL_dpixels_untex is unset (no --l2), mirror tex so the per-
			// Gauss selector below resolves to the same array regardless of
			// is_textured — kernel becomes byte-identical to pre-flag behavior.
			dL_dpixel_untex[i] = (dL_dpixels_untex != nullptr)
				? dL_dpixels_untex[i * H * W + pix_id]
				: dL_dpixel_tex[i];
		}
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Early exit: all threads done (also serves as sync before shared memory loading)
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor) {
				// Once contributor wraps past 0 (uint32 underflow), all remaining
				// Gaussians will also be skipped. Mark done for tile-level early exit.
				if (contributor > toDo) { done = true; }
				continue;
			}

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y);
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

		// compute intersection and depth
		float rho = min(rho3d, rho2d);
		float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
		if (c_d < near_n) continue;
		float4 nor_o = collected_normal_opacity[j];
		float normal[3] = {nor_o.x, nor_o.y, nor_o.z};  // Already normalized in preprocessing
		float opa = nor_o.w;

		// accumulations

		float power = -0.5f * rho;
		if (power > 0.0f)
			continue;

		const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			float dL_dz = 0.0f;
			float dL_dweight = 0;
#if RENDER_AXUTILITY
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			// `--method mixed_3d`: final_A_tex excludes EWA occlusion (T_tex), making
// the dist gradient strictly textured-only. final_A_tex == final_A for
// non-mixed_3d so this is a no-op there.
dL_dweight += (final_D2 + m_d * m_d * final_A_tex - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			// `--method mixed_3d`: final_A_tex for textured-only dist gradient.
const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A_tex - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif

			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz); // propagate depth loss
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

// Backward version of the rendering procedure.
template <uint32_t C, uint32_t D_DIFFUSE = 0>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDAsurfelBackward(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float beta,
	int W, int H,
	uint32_t level, const uint32_t l_dim, float l_scale, uint32_t Base,
	bool align_corners, uint32_t interp,
	const bool if_contract,
	const glm::vec2* scales,
	float focal_x, float focal_y,
	const float* __restrict__ other_maps,
	const int* __restrict__ out_index,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ homotrans,
	const float* __restrict__ ap_level,
	const __half* __restrict__ hash_features,
	const int* __restrict__ level_offsets,
	const float* __restrict__ gridrange,
	const rgb_t* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	// `--l2` (mixed_3d only): per-Gauss image-grad routing — when non-null,
	// untextured Gauss read this array instead of `dL_dpixels`. nullptr →
	// every Gauss reads from `dL_dpixels` (byte-identical to pre-flag).
	const float* __restrict__ dL_dpixels_untex,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dfeatures,
	float * __restrict__ dL_dtransMat,
	float * __restrict__ dL_dhomoMat,
	float4* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_gradsum,
	const glm::vec3* __restrict__ cam_pos,
	const float* __restrict__ hash_features_diffuse = nullptr,
	const int* __restrict__ level_offsets_diffuse = nullptr,
	const float* __restrict__ gridrange_diffuse = nullptr,
	float* __restrict__ dL_dfeatures_diffuse = nullptr,
	const int render_mode = 0,
	const float* __restrict__ shapes = nullptr,
	const int kernel_type = 0,
	float* __restrict__ dL_dshapes = nullptr,
	const bool detach_hash_grad = false,
	// MLP gradient buffers for 3D_SH_res (render_mode=5, bias-free, all [16×16])
	float* __restrict__ dL_dmlp_W1 = nullptr,    // [16 * 16] = W1_SIZE
	float* __restrict__ dL_dmlp_W2 = nullptr,    // [16 * 16] = W2_SIZE
	float* __restrict__ dL_dmlp_W3 = nullptr,    // [16 * 16] = W3_SIZE
	// MLP weight pointers for 3D_SH_res (bias-free, FP16, all [16×16])
	const __half* __restrict__ mlp_W1_ptr = nullptr,
	const __half* __restrict__ mlp_W2_ptr = nullptr,
	const __half* __restrict__ mlp_W3_ptr = nullptr,
	// DC SH features for 3D_SH_cat (render_mode=6)
	const float* __restrict__ dc_features = nullptr,
	// `--method mixed` per-Gauss bool flag [P]. nullptr → all-textured behavior.
	const bool* __restrict__ is_textured = nullptr,
	// `--method mixed_3d`: per-Gauss EWA conic [P] (a,b,c,opacity). nullptr →
	// untextured rows fall back to 2DGS ray-splat backward.
	const float4* __restrict__ ewa_conic = nullptr,
	const float* __restrict__ scaling_z = nullptr)
{
	// Create MLP weights struct from parameters (bias-free, all [16×16])
	MlpWeights mlp_weights = {
		mlp_W1_ptr,
		mlp_W2_ptr,
		mlp_W3_ptr
	};

	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	// Warp partition for per-Gaussian gradient reductions (GEMM path).
	auto warp = cg::tiled_partition<32>(block);

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float pix_off = (render_mode & 0x800) ? 0.5f : 0.0f;
	const float2 pixf = {(float)pix.x + pix_off, (float)pix.y + pix_off};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float collected_size[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	__shared__ float3 collected_SuTu[BLOCK_SIZE];
	__shared__ float3 collected_SvTv[BLOCK_SIZE];
	__shared__ float3 collected_pk[BLOCK_SIZE];
	__shared__ uint32_t collected_ap_level[BLOCK_SIZE];
	__shared__ float2 collected_shapes[BLOCK_SIZE];  // Kernel shape: .x = primary, .y = nexel gamma_y
	__shared__ bool collected_is_textured_bw[BLOCK_SIZE];  // `--method mixed` per-Gauss flag
	__shared__ float4 collected_ewa_conic_bw[BLOCK_SIZE];  // `--method mixed_3d` EWA conic

	// Per-tile MLP gradient accumulators for 3D_SH_res mode (render_mode=5, bias-free)
	// All [16×16] = 256 floats each. Accumulate to shared memory first, flush to global once per tile
	__shared__ float tile_dL_dW1[W1_SIZE];   // 16*16 = 1KB
	__shared__ float tile_dL_dW2[W2_SIZE];   // 16*16 = 1KB
	__shared__ float tile_dL_dW3[W3_SIZE];   // 16*16 = 1KB

	// Shared memory cache for MLP weights (FP16, all [16×16], 1.5KB total)
	// Loaded once per tile, used for both MLP forward recomputation and per-pixel backward
	__shared__ __half smem_mlp_W1[W1_SIZE];   // 16*16 = 512 bytes
	__shared__ __half smem_mlp_W2[W2_SIZE];   // 16*16 = 512 bytes
	__shared__ __half smem_mlp_W3[W3_SIZE];   // 16*16 = 512 bytes

	// Shared memory for per-Gaussian baseline features (dual hashgrid mode)
	// NOTE: Disabled for baseline_double/baseline_blend_double due to shared memory limits  
	// We query on-demand instead (less efficient but fits in shared memory)
	// Legacy comment - this shared memory block is currently disabled
	// __shared__ float collected_feat_pk[BLOCK_SIZE][16 * 4];  // Max 16 levels × 4 features per Gaussian

	// get total rendered points number per pixel.
	const int render_number = other_maps[pix_id + NUM_OFFSET * H * W];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	// `--method res_3d` SINGLE-PASS dual-cascade backward state. Forward
	// persisted T_sv_aux_final / T_tex_aux_final to final_T slots 4/5. We
	// reverse-divide by (1 - α_j) AT THE TIME a Gauss j is processed, routed
	// by its is_textured flag, to recover T_aux at that contribution site.
	// Each cascade tracks its own `accum_rec` + `last_alpha` + `last_color`
	// so the α-grad formula `(c - accum_rec_post)·dL/dC` references the
	// correct downstream context for that cascade.
	//
	// Non-res_3d use (no untextured contributors → T_sv_aux_final = 1,
	// T_tex_aux_final = T_final, accum_rec_sv stays 0, accum_rec_tex tracks
	// joint accum_rec) → numerically equivalent to the joint backward, since
	// out_color = C_sv_aux + C_tex_aux = 0 + C[] and dL/dC[ch] = dL/dC_tex_aux[ch].
	float T_sv_back  = inside ? final_Ts[pix_id + 4 * H * W] : 1.0f;
	float T_tex_back = inside ? final_Ts[pix_id + 5 * H * W] : 1.0f;
	float accum_rec_sv [C] = { 0 };
	float accum_rec_tex[C] = { 0 };
	float last_alpha_sv  = 0.0f;
	float last_alpha_tex = 0.0f;
	float last_color_sv [C] = { 0 };
	float last_color_tex[C] = { 0 };

	// `--l2` (mixed_3d only): per-Gauss image-gradient routing. Two upstream
	// image gradients (one per loss); the per-Gauss `dL_dpixel` pointer set
	// inside each inner-loop iteration picks tex- vs untex-grad array based on
	// the Gauss's `is_textured` flag. When dL_dpixels_untex == nullptr (default,
	// no --l2), `dL_dpixel_untex` is initialised from `dL_dpixels` too so the
	// per-Gauss selector is a no-op — byte-identical to the pre-flag behavior.
	float dL_dpixel_tex[C];
	float dL_dpixel_untex[C];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	// Per-pixel gradient for overdraw_sum from Python-side autograd (e.g.,
	// --w_overdraw_reg's (w_r * od_map).mean()). Combined with the CUDA-side
	// global d_overdraw_lambda_bw below so --overdraw_reg and --w_overdraw_reg
	// can compose additively. Without this read, the autograd grad arriving in
	// dL_depths[OVERDRAW_OFFSET] would be silently discarded.
	float dL_doverdraw_px = 0.0f;

	// Per-pixel gradient for beta_sum (= Σ_i w_i · β_i) from Python-side autograd
	// (--w_lambda_perpix's (w_r * beta_sum).mean()). Injects a DIRECT gradient to
	// each contributor's shape (β_i): dL/dβ_i += dL_dbeta_sum_px · w_i. Does NOT
	// propagate through w_i back to α — by design, this reg only dampens β values,
	// not opacity/coverage. Detached from the α chain.
	float dL_dbeta_sum_px = 0.0f;

	if (inside) {
		// here dL_ddepth is dL_dD (blended depth value), so no change here.
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++)
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
		dL_doverdraw_px = dL_depths[OVERDRAW_OFFSET * H * W + pix_id];
		dL_dbeta_sum_px = dL_depths[BETA_SUM_OFFSET * H * W + pix_id];

	}
	
	int collec_offsets[16] = {0};
	// float feat[C] = {0};
	// float grad_feat[C] = {0};
	// float dL_dxyz[3] = {0};
	float voxel_min = 0.0f;
	float voxel_max = 0.0f;
	if(level > 0){
		// For cat mode (render_mode==1) and adaptive_zero (render_mode==2), level is encoded as:
		// (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
		// Decode to get actual hashgrid levels for offset copying
		int actual_levels = level;
		if(render_mode == 1){
			// cat mode: Extract active_hashgrid_levels from encoded value
			int active_hashgrid_levels = (level >> 8) & 0xFF;
			actual_levels = active_hashgrid_levels;  // Use ACTIVE hashgrid levels for coarse-to-fine
		} else if((render_mode & 0xFF) == 2){
			// adaptive_zero mode: level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			// Note: render_mode may have inference flag in upper bits, so mask to get base mode
			int active_hashgrid_levels = (level >> 8) & 0xFF;
			actual_levels = active_hashgrid_levels;
		} else if(render_mode == 3){
			// 3D mode: level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			// Hashgrid query happens in PyTorch, not CUDA - so actual_levels = 0
			actual_levels = 0;
		} else if((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6){
			// 3D_SH_res / 3D_SH_cat mode: level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			// Hash query happens in CUDA kernel (like cat mode), so use active_hashgrid_levels
			// Note: render_mode may have bit flags set, so mask to get base mode
			int active_hashgrid_levels = (level >> 8) & 0xFF;
			actual_levels = active_hashgrid_levels;
		} else if(level > 16){
			printf("Error: level %d  > 16.", level);
			return;
		}
		for(int l = 0; l <= actual_levels; l++) collec_offsets[l] = level_offsets[l];
		voxel_min = gridrange[0];
		voxel_max = gridrange[1];
	}
	
	// Setup baseline hashgrid offsets once (dual hashgrid mode)
	int collec_offsets_diffuse[16] = {0};
	float voxel_min_diffuse = 0.0f;
	float voxel_max_diffuse = 0.0f;
	// NOTE: Changed from compile-time D_DIFFUSE check to runtime check (for modes 4, 5, 12)
	if(level > 0 && level_offsets_diffuse != nullptr && 
	   (render_mode == 3)){
		for(int l = 0; l <= level; l++) collec_offsets_diffuse[l] = level_offsets_diffuse[l];
		voxel_min_diffuse = gridrange_diffuse[0];
		voxel_max_diffuse = gridrange_diffuse[1];
	}

	// NOTE: baseline_blend_double post-processing was removed during render mode cleanup
	// (it was old mode 3, now deleted)

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	// `--method mixed_3d`: slot 3 holds A_tex — the explicit textured-only
	// running sum of w_j = α_j · T_full_j (NOT 1 - T_tex). The 2DGS distortion
	// identity L_dist = Σ w_i (m_i² A − 2 m_i M1 + M2) requires A, M1, M2 all
	// to be partial sums over the same subset; for textured-only we must
	// accumulate A explicitly because `1 - T_tex ≠ Σ_textured w` when EWA
	// absorbers are interleaved (they scale subsequent w's via T_full but
	// don't move T_tex). For non-mixed runs (no untex) A_tex telescopes to
	// 1 - T, so this is a no-op there.
	const float final_A_tex = inside ? final_Ts[pix_id + 3 * H * W] : 0.0f;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++) {
			dL_dpixel_tex[i] = dL_dpixels[i * H * W + pix_id];
			// When dL_dpixels_untex is unset (no --l2), mirror tex so the per-
			// Gauss selector below resolves to the same array regardless of
			// is_textured — kernel becomes byte-identical to pre-flag behavior.
			dL_dpixel_untex[i] = (dL_dpixels_untex != nullptr)
				? dL_dpixels_untex[i * H * W + pix_id]
				: dL_dpixel_tex[i];
		}
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Overdraw regularization: running accumulator for sum_{j>i} dsig_j * w_j
	const float od_lambda = d_overdraw_lambda_bw;
	float overdraw_accum = 0.0f;
	const float OD_K = 10.0f;
	const float OD_THRESH = 1.0f / 255.0f;
	// dL/d(overdraw) combines two sources, both per-pixel-equivalent:
	//   1. CUDA-global --overdraw_reg via d_overdraw_lambda_bw (mean-reduced, so /(H*W))
	//   2. Python autograd dL_depths[OVERDRAW_OFFSET, pix] from --w_overdraw_reg's
	//      (w_r * od_map).mean() — PyTorch already baked in the /(H*W) factor here.
	const float dL_doverdraw_cuda = (od_lambda > 0.0f) ? od_lambda / (float)(H * W) : 0.0f;
	const float dL_doverdraw = dL_doverdraw_cuda + dL_doverdraw_px;

	// Weight-squared regularization: d(sum w²)/d(alpha_i) has direct + indirect terms
	// Direct: 2*w_i*T_i  (changing alpha_i changes w_i directly)
	// Indirect: -sum_{j>i}(2*w_j²) / (1-alpha_i)  (changing alpha_i changes T_j for all j>i)
	const float wr_lambda = d_weight_reg_lambda_bw;
	const float dL_dwr = (wr_lambda > 0.0f) ? -wr_lambda / (float)(H * W) : 0.0f;
	float wr_accum = 0.0f;  // Running sum of w_j² for j > i (back-to-front traversal)

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Note: 3D_SH_res does not use view direction encoding (MLP is view-independent)
	// View dependence comes from SH evaluation in preprocessing

	// Load MLP weights into shared memory and optionally zero gradient buffers
	// Note: (render_mode & 0xFF) masks out bit flags to get base mode
	if ((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) {
		const int tid = block.thread_rank();
		// Always load MLP weights (needed for both full backward and input-only backward)
		for (int idx = tid; idx < W1_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W1[idx] = mlp_weights.W1[idx];
		for (int idx = tid; idx < W2_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W2[idx] = mlp_weights.W2[idx];
		for (int idx = tid; idx < W3_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W3[idx] = mlp_weights.W3[idx];
		// Zero gradient buffers only when weight grads are needed (not freeze_mlp)
		if (dL_dmlp_W1 != nullptr) {
			for (int idx = tid; idx < W1_SIZE; idx += BLOCK_SIZE)
				tile_dL_dW1[idx] = 0.0f;
			for (int idx = tid; idx < W2_SIZE; idx += BLOCK_SIZE)
				tile_dL_dW2[idx] = 0.0f;
			for (int idx = tid; idx < W3_SIZE; idx += BLOCK_SIZE)
				tile_dL_dW3[idx] = 0.0f;
		}
		block.sync();
	}

	// Tile-level max last_contributor — enables batch-level early-out in the GEMM path.
	// If every Gaussian in a batch has contributor-index >= tile_max_last_contrib, no
	// pixel participates, so we skip the whole batch (SMEM load + inner loop). This is
	// the analog of FastGS's per-tile max_contrib bucket-skip, adapted to our layout.
	__shared__ int s_tile_max_last_contrib;
	if (block.thread_rank() == 0) s_tile_max_last_contrib = 0;
	block.sync();
	if (inside && last_contributor > 0) atomicMax(&s_tile_max_last_contrib, last_contributor);
	block.sync();
	const int tile_max_last_contrib = s_tile_max_last_contrib;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Early exit: all threads done (also serves as sync before shared memory loading)
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Batch-level early-out (GEMM path only): all Gaussians in this batch have
		// contributor index in [contributor - effective_toDo, contributor). If the
		// LOWEST of these is already >= tile_max_last_contrib, no pixel needs the batch.
		if ((render_mode & 0x100) && dL_dmlp_W1 != nullptr) {
			const int effective_toDo_check = min((int)BLOCK_SIZE, (int)toDo);
			const int min_current_contrib = (int)contributor - effective_toDo_check;
			if (min_current_contrib >= tile_max_last_contrib) {
				contributor -= effective_toDo_check;
				continue;
			}
		}

		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			
			collected_size[block.thread_rank()] =  PI * scales[coll_id].x * scales[coll_id].y;
			// Cache evaluated colors (RGB) in shared memory.
			// colors is FP16 (rgb_t) when FP16_RGB=1 — upcast on load; smem stays FP32.
			for (int ch = 0; ch < C; ch++)
				collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = RGB_TO_FLOAT(colors[coll_id * C + ch]);
			// from 2dgs eq.(5)
			if(homotrans != nullptr){
				collected_SuTu[block.thread_rank()] = {homotrans[16 * coll_id+0], homotrans[16 * coll_id+4], homotrans[16 * coll_id+8]};
				collected_SvTv[block.thread_rank()] = {homotrans[16 * coll_id+1], homotrans[16 * coll_id+5], homotrans[16 * coll_id+9]};
				collected_pk[block.thread_rank()] = {homotrans[16 * coll_id+3], homotrans[16 * coll_id+7], homotrans[16 * coll_id+11]};
			}
			if(ap_level != nullptr){
				collected_ap_level[block.thread_rank()] = floorf(ap_level[coll_id]);
			}
			// Collect shape for beta kernel (only when using beta kernel)
			if(shapes != nullptr){
				if (kernel_type == 5) {
				collected_shapes[block.thread_rank()] = {shapes[coll_id * 2], shapes[coll_id * 2 + 1]};
			} else {
				collected_shapes[block.thread_rank()] = {shapes[coll_id], 0.0f};
			}
			}
			// `--method mixed`: pull per-Gauss flag. Defaults to true (textured) when null.
			collected_is_textured_bw[block.thread_rank()] = (is_textured == nullptr) ? true : is_textured[coll_id];
			collected_ewa_conic_bw[block.thread_rank()] = (ewa_conic == nullptr) ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : ewa_conic[coll_id];

		// NOTE: Per-Gaussian feature caching disabled due to shared memory limits
		// Features are now queried on-demand in the per-pixel loop (cases 4, 5, 12)
		}
		block.sync();

		// ============================================================================
		// MODE 5 (3D_direct_fused): SYNCHRONIZED ITERATION WITH COLLABORATIVE GEMM
		// All threads must iterate together - no early exit based on 'done'
		// Bit 8 (0x100) indicates host enabled collaborative GEMM with dynamic smem
		// ============================================================================
		if ((render_mode & 0x100) && dL_dmlp_W1 != nullptr) {
			// Synchronized loop - ALL threads iterate through ALL Gaussians in this batch
			const int effective_toDo = min(BLOCK_SIZE, toDo);
			for (int j = 0; j < effective_toDo; j++)
			{
				// Determine participation: this pixel contributed to this Gaussian
				// (matches the contributor--; if(contributor >= last_contributor) continue; logic)
				const int current_contributor = contributor - j - 1;
				bool participates = inside && !done && (current_contributor < last_contributor);

				// Quick ballot: skip this Gaussian entirely if no thread participates
				// __syncthreads_count replaces the __syncthreads() before L3 GEMM
				int n_active = __syncthreads_count(participates);
				if (n_active == 0) {
					if (block.thread_rank() == 0) atomicAdd(&d_bw_profile_counts[1], 1);
					continue;
				}

				// Profiling: T0 at sync point (ballot check)
				unsigned long long _prof_t0 = clock64();
				if (block.thread_rank() == 0) {
					atomicAdd(&d_bw_profile_counts[0], 1);
					atomicAdd(&d_bw_profile_counts[3], (unsigned int)n_active);
				}

				// Load Gaussian data from shared memory
				const int global_id = collected_id[j];
				const float2 xy = collected_xy[j];
				const float3 Tu = collected_Tu[j];
				const float3 Tv = collected_Tv[j];
				const float3 Tw = collected_Tw[j];
				const float4 nor_o = collected_normal_opacity[j];
				const float opa = nor_o.w;
				float normal[3] = {nor_o.x, nor_o.y, nor_o.z};

				// Per-Gaussian gradient accumulators (register-local). Per-pixel
				// contributions accumulate into these; at the end of this j-iteration
				// we warp-reduce and perform a single atomicAdd per warp per field.
				// Non-participating threads contribute 0.
				float acc_dL_dcolors[3]    = {0.0f, 0.0f, 0.0f};
				float acc_dL_dnormal3D[3]  = {0.0f, 0.0f, 0.0f};
				float acc_dL_dshapes[2]    = {0.0f, 0.0f};
				float acc_dL_dhomoMat[9]   = {0};
				float acc_dL_dtransMat[9]  = {0};
				float acc_dL_dmean2D[4]    = {0.0f, 0.0f, 0.0f, 0.0f};
				float acc_dL_dopacity      = 0.0f;

				// Per-pixel intersection data
				float2 s = {0, 0};
				float rho3d = 0, rho2d = 0, rho = 0;
				float c_d = 0, alpha = 0, G = 0, w = 0;
				// `--method res_3d` dual cascade: per-cascade COLOR/MLP weight
				// (textured: alpha * T_tex_back, untextured: alpha * T_sv_back).
				// Assigned inside the `if (valid_alpha)` block alongside w.
				float w_color_collab = 0.0f;
				float3 xyz = {0, 0, 0};
				// Store k, l, p for geometry gradients later
				float3 k_stored = {0, 0, 0};
				float3 l_stored = {0, 0, 0};
				float3 p_stored = {0, 0, 0};

				// `--method mixed`: per-Gauss textured flag — UNIFORM across all threads
				// in the block since it's per-Gaussian, not per-pixel. Safe to branch
				// on without breaking warp-uniformity of collective GEMMs.
				const bool tex_j = collected_is_textured_bw[j];
				// `--l2`: per-Gauss image-grad routing. When no L2 split is active,
				// `dL_dpixel_untex` was mirrored from `dL_dpixel_tex` at load time, so
				// this selector resolves to the same array for every Gauss →
				// byte-identical to pre-flag behavior. Block-uniform branch (tex_j
				// is per-Gauss, all threads in the block see the same value).
				const float* const dL_dpixel = tex_j ? dL_dpixel_tex : dL_dpixel_untex;

				// ============================================================
				// `--method mixed_3d` — UNTEXTURED EWA backward, MODE-5 path.
				// Block-uniform branch (tex_j + ewa_conic are per-Gaussian
				// uniform across all 256 threads): the WHOLE block handles
				// this untextured Gaussian here and `continue`s, uniformly
				// skipping the collaborative GEMM (untextured has no MLP).
				// Per-thread gating uses `ok` (NOT `continue`) so no thread
				// diverges before the block-uniform continue — required since
				// the next j iterates a block barrier (__syncthreads_count).
				// Math is identical to the verified std-path EWA branch
				// (same recurrence + FastGS conic/mean2D/opacity grads).
				// ============================================================
				if (!tex_j && ewa_conic != nullptr) {
					bool ok = participates;
					const float4 con_o = collected_ewa_conic_bw[j];
					const int gid_e = collected_id[j];
					const float2 xy_e = collected_xy[j];
					const float2 de = { xy_e.x - pixf.x, xy_e.y - pixf.y };
					const float opa_e = con_o.w;
					// Untextured kernel = `--kernel2` (render_mode bits[16..19] =
					// kernel_type2+1; 0 ⇒ kernel_type). Same decode as the std
					// EWA backward + the forward EWA branch (block-uniform).
					const int _utk_n = (render_mode >> 16) & 0xF;
					const int ut_kt = _utk_n ? (_utk_n - 1) : kernel_type;
					const bool is_beta_e = (ut_kt == 1 || ut_kt == 4);
					float me = 0.0f, G_e = 0.0f, dG_dm = 0.0f, base_e = 0.0f, beta_e = 0.0f, alpha_e = 0.0f;
					if (ok && opa_e <= 0.0f) ok = false;     // culled in preprocess
					if (ok) {
						me = con_o.x * de.x * de.x + 2.0f * con_o.y * de.x * de.y + con_o.z * de.y * de.y;
						if (is_beta_e) {
							const float k_sq = (ut_kt == 4) ? 9.0f : 1.0f;
							if (me >= k_sq + 1e-6f) ok = false;   // compact support
							else {
								beta_e = collected_shapes[j].x;
								base_e = fmaxf(0.0f, 1.0f - me / k_sq);
								G_e = powf(base_e, beta_e);
								dG_dm = (base_e > 1e-9f) ? (-(beta_e / k_sq) * powf(base_e, beta_e - 1.0f)) : 0.0f;
							}
						} else {
							const float power = -0.5f * me;
							if (power > 0.0f) ok = false;         // me < 0 (degenerate)
							else { G_e = expf(power); dG_dm = -0.5f * G_e; }
						}
						if (ok) {
							alpha_e = fminf(0.99f, opa_e * G_e);
							if (alpha_e < 1.0f / 255.0f) ok = false;
						}
					}
					if (ok) {
						// ---- shared per-pixel recurrence (mirror textured) ----
						T = T / (1.f - alpha_e);
						// `--method res_3d` dual cascade: SV-aux only.
						T_sv_back = T_sv_back / (1.f - alpha_e);
						const float dchannel_dcolor = alpha_e * T_sv_back;
						const float w_e = alpha_e * T;
						float dL_dalpha_color = 0.0f;
						float dL_dalpha = 0.0f;
						for (int ch = 0; ch < C; ch++) {
							accum_rec_sv[ch] = last_alpha_sv * last_color_sv[ch] + (1.f - last_alpha_sv) * accum_rec_sv[ch];
							const float cval = RGB_TO_FLOAT(colors[gid_e * 3 + ch]);
							last_color_sv[ch] = cval;
							dL_dalpha_color += (cval - accum_rec_sv[ch]) * dL_dpixel[ch];
							atomicAdd(&(dL_dcolors[gid_e * 3 + ch]), dchannel_dcolor * dL_dpixel[ch]);
						}
#if RENDER_AXUTILITY
						const float c_d = depths[gid_e];
						// `--method mixed_3d`: EWA Gaussians are DETACHED from the
						// normal-consistency loss (--lambda_normal / --w_normal).
						// Forward keeps them in D (so surf_normal = depth_to_normal(D)
						// sees the full depth) and in the recurrence state (so
						// textured contributors get correct dL_dalpha via their own
						// (c_d_tex - accum_depth_rec) terms), but EWA's own dL_dalpha
						// does NOT pick up the depth/normal grads:
						//   • skip dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth
						//   • skip dL_dalpha += (0 - accum_normal_rec) * dL_dnormal2D
						// Distortion is already skipped (dL_dweight = 0). Color and
						// mask (dL_dpixel / dL_daccum) grads still flow into EWA so
						// it keeps being optimized for photometric + opacity tasks.
						float dL_dweight = 0.0f;
						dL_dalpha += dL_dweight - last_dL_dT;
						last_dL_dT = dL_dweight * alpha_e + (1 - alpha_e) * last_dL_dT;
						accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
						last_depth = c_d;
						// EWA detached from depth loss (see comment above).
						accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
						dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;
						for (int ch = 0; ch < 3; ch++) {
							accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
							last_normal[ch] = 0.0f;
							// EWA detached from normal loss (see comment above).
						}
#endif
						// `--method res_3d` dual cascade: color * T_sv_back, others * T.
						dL_dalpha = dL_dalpha_color * T_sv_back + dL_dalpha * T;
						float dL_dalpha_reg = 0.0f;
						// --method mixed/mixed_3d: overdraw is textured-only — this EWA
						// branch is `!tex_j`, so it neither receives an overdraw grad
						// nor advances `overdraw_accum` (forward also skipped its sig).
						// (w_square_sum / --weight_reg stays unmasked.)
						if (dL_dwr != 0.0f) {
							float safe_denom_wr = fmaxf(1.0f - alpha_e, 1e-7f);
							dL_dalpha_reg += dL_dwr * (2.0f * w_e * T - 2.0f * wr_accum / safe_denom_wr);
							wr_accum += w_e * w_e;
						}
						last_alpha = alpha_e;
						last_alpha_sv = alpha_e;
						dL_dalpha += dL_dalpha_reg;
						// Opacity clamp: skip opacity/conic/mean2D/shape grads
						// (dα/dopa = dα/dG = 0) but the recurrence above already ran.
						if (opa_e * G_e <= 0.99f) {
							atomicAdd(&(dL_dopacity[gid_e]), G_e * dL_dalpha);
							const float dL_dG = opa_e * dL_dalpha;
							const float dL_dm = dL_dG * dG_dm;
							// dL/dconic carried via dL_dtransMat[gid*9+0..2]
							// (off-diag in FastGS half convention: dx·dy, no 2×).
							atomicAdd(&dL_dtransMat[gid_e * 9 + 0], dL_dm * de.x * de.x);
							atomicAdd(&dL_dtransMat[gid_e * 9 + 1], dL_dm * de.x * de.y);
							atomicAdd(&dL_dtransMat[gid_e * 9 + 2], dL_dm * de.y * de.y);
							const float dL_dddx = dL_dm * (2.0f * con_o.x * de.x + 2.0f * con_o.y * de.y);
							const float dL_dddy = dL_dm * (2.0f * con_o.y * de.x + 2.0f * con_o.z * de.y);
							const float gx = dL_dddx * 0.5f * (float)W;
							const float gy = dL_dddy * 0.5f * (float)H;
							atomicAdd(&dL_dmean2D[gid_e].x, gx);
							atomicAdd(&dL_dmean2D[gid_e].y, gy);
							atomicAdd(&dL_dmean2D[gid_e].z, fabsf(gx));
							atomicAdd(&dL_dmean2D[gid_e].w, fabsf(gy));
							if (is_beta_e && dL_dshapes != nullptr && base_e > 1e-7f)
								atomicAdd(&dL_dshapes[gid_e], dL_dG * G_e * logf(base_e));
						}
					}
					continue;  // block-uniform → skip GEMM for this Gaussian
				}

				// Beta/flex/general kernel variables (needed across alpha + geometry gradient phases)
				float shape_val = 0.0f, base = 0.0f;
				float alpha_beta = 0.0f, alpha_lp = 0.0f;
				bool beta_wins = false;
				float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
				float per_gaussian_beta_val = 0.0f, G_raw = 0.0f, demon_val = 1.0f;
				float general_beta_val = 0.0f, general_pow_term_val = 0.0f, general_rho_safe_val = 0.0f;

				// Compute intersection (only for participating pixels) — IDENTICAL
				// 2DGS ray-splat geometry for textured AND untextured. The splat
				// shape is the rho3d ray-plane intersection; untextured differs
				// only in the color path (no hash/MLP), handled below.
				if (participates) {
					float3 k = {pixf.x * Tw.x - Tu.x, pixf.x * Tw.y - Tu.y, pixf.x * Tw.z - Tu.z};
					float3 l = {pixf.y * Tw.x - Tv.x, pixf.y * Tw.y - Tv.y, pixf.y * Tw.z - Tv.z};
					float3 p = cross(k, l);
					// Store for geometry gradients
					k_stored = k;
					l_stored = l;
					p_stored = p;

					if (p.z != 0.0f) {
						s = {p.x / p.z, p.y / p.z};
						rho3d = s.x * s.x + s.y * s.y;
						float2 d = {xy.x - pixf.x, xy.y - pixf.y};
						rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
						rho = min(rho3d, rho2d);
						c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;

						if (c_d >= near_n) {
							// Compute alpha based on kernel type (must match forward exactly)
							bool valid_alpha = false;
							// `--method mixed`: untextured uses the run's kernel (same
							// as textured); only the color path differs.
							if (kernel_type == 1 || kernel_type == 4) {
								if (rho3d < k_sq + 1e-6f) {
									shape_val = collected_shapes[j].x;
									base = fmaxf(0.0f, 1.0f - rho3d / k_sq);
									alpha_beta = powf(base, shape_val);
									alpha_lp = expf(-rho2d / 2.0f);
									beta_wins = (alpha_beta >= alpha_lp);
									G = beta_wins ? alpha_beta : alpha_lp;
									alpha = fminf(0.99f, opa * G);
									valid_alpha = true;
								}
							} else if (kernel_type == 2) {
								float power = -0.5f * rho;
								if (power <= 0.0f) {
									G_raw = expf(power);
									per_gaussian_beta_val = collected_shapes[j].x;
									G = G_raw; demon_val = 1.0f;
									if (per_gaussian_beta_val > 0.0f) {
										demon_val = 1.0f + per_gaussian_beta_val * G_raw;
										G = (1.0f + per_gaussian_beta_val) * G_raw / demon_val;
									}
									alpha = fminf(0.99f, opa * G);
									valid_alpha = true;
								}
							} else if (kernel_type == 3) {
								general_beta_val = collected_shapes[j].x;
								general_rho_safe_val = fmaxf(rho, 1e-8f);
								general_pow_term_val = powf(general_rho_safe_val, 0.5f * general_beta_val);
								float power = -0.5f * general_pow_term_val;
								if (power <= 0.0f) {
									G = expf(power);
									alpha = fminf(0.99f, opa * G);
									valid_alpha = true;
								}
							} else {
								float power = -0.5f * rho;
								if (power <= 0.0f) {
									G = expf(power);
									alpha = min(0.99f, opa * G);
									valid_alpha = true;
								}
							}
							if (valid_alpha && alpha >= 1.0f / 255.0f) {
								// CRITICAL: Recover T_before FIRST (matches 2DGS backward)
								T = T / (1.f - alpha);
								// `--method res_3d`: textured Gauss decay tex-aux.
								// (Block-uniform: tex_j is per-Gauss uniform across
								// the 256-thread block on this iteration.)
								if (tex_j) T_tex_back = T_tex_back / (1.f - alpha);
								else       T_sv_back  = T_sv_back  / (1.f - alpha);
								w = alpha * T;
								// Per-cascade weight for COLOR / MLP-residual grads.
								// Joint w stays for reg + skip_hash + geometry sites.
								w_color_collab = tex_j ? (alpha * T_tex_back) : (alpha * T_sv_back);
								// `--method mixed`: world-xyz reconstruction is needed ONLY for the
								// hash query (textured). Skip it + the two shared loads for untex.
								if (tex_j) {
									const float3 pk = collected_pk[j];
									if (rho3d <= rho2d) {
										const float3 sutu = collected_SuTu[j];
										const float3 svtv = collected_SvTv[j];
										xyz = {s.x * sutu.x + s.y * svtv.x + pk.x,
										       s.x * sutu.y + s.y * svtv.y + pk.y,
										       s.x * sutu.z + s.y * svtv.z + pk.z};
									} else {
										xyz = pk;
									}
								}
							} else participates = false;
						} else participates = false;
					} else participates = false;
				}

				// ======== INTERLEAVED MLP BACKWARD WITH COLLABORATIVE GEMM ========
				// 3D_SH_res: SH base color + hash MLP residual
				// MLP input: [hash(hash_dim) | pad(16-hash_dim)] = 16D
				// MLP output: 3D RGB residual (identity activation, NO sigmoid)
				// feat = SH_color + MLP_residual
				const int active_hashgrid_levels = (level >> 8) & 0xFF;

				// --- Phase 1: Forward recomputation + identity backward → dL_dz3 ---
				float my_input[TC_INPUT_DIM] = {0};
				float my_h1_post[TC_HIDDEN_DIM] = {0};
				float my_h2_post[TC_HIDDEN_DIM] = {0};
				float my_residual[ORIG_OUTPUT_DIM] = {0};
				float my_dL_dz3[TC_OUTPUT_DIM] = {0};

				bool skip_hash = (d_contrib_thresh_bw > 0.0f && w < d_contrib_thresh_bw)
				                 || (d_count_thresh_bw > 0 && current_contributor >= (uint32_t)d_count_thresh_bw);
				// `--method mixed` untextured: short-circuit the entire MLP-related
				// recomputation. my_input / my_h*/my_residual / my_dL_dz3 all stay 0,
				// so the L1/L2/L3 collective GEMMs run with zero contribution from
				// this Gaussian. Color grad is the simple 2DGS form: w * dL_dpixel.
				// `--method res_3d`: per-cascade w_color_collab (SV cascade for
				// !tex_j). This path is dead for res_3d (untex EWA already
				// short-circuited via `continue` at line ~1352), but kept correct
				// for any future `--method mixed` use of this submodule.
				if (participates && !tex_j) {
					for (int ch = 0; ch < 3; ch++) {
						acc_dL_dcolors[ch] += dL_dpixel[ch] * w_color_collab;
					}
				}
				if (participates && tex_j) {
					// 1. Query hash features — skip for tail pixels (matches forward)
					const int hash_dim_collab = active_hashgrid_levels * l_dim;
					if (!skip_hash && active_hashgrid_levels > 0 && l_dim == 4) {
						float hash_feat[16] = {0};
						uint32_t appearance_level = collected_ap_level[j];
						if (hash_dim_collab == 4)
							query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 8)
							query_feature<false, 8, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 12)
							query_feature<false, 12, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 16)
							query_feature<false, 16, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						for (int i = 0; i < hash_dim_collab && i < TC_INPUT_DIM; i++) my_input[i] = hash_feat[i];
					} else if (!skip_hash && active_hashgrid_levels > 0 && l_dim == 2) {
						// 2D per level — supports 1..8 hash levels (hash_dim ∈ {2,4,6,8,10,12,14,16}).
						float hash_feat[16] = {0};
						uint32_t appearance_level = collected_ap_level[j];
						if (hash_dim_collab == 2)
							query_feature<false, 2, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 4)
							query_feature<false, 4, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 6)
							query_feature<false, 6, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 8)
							query_feature<false, 8, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 10)
							query_feature<false, 10, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 12)
							query_feature<false, 12, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 14)
							query_feature<false, 14, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						else if (hash_dim_collab == 16)
							query_feature<false, 16, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							                           appearance_level, hash_features, active_hashgrid_levels,
							                           l_scale, Base, align_corners, interp, if_contract, false);
						for (int i = 0; i < hash_dim_collab && i < TC_INPUT_DIM; i++) my_input[i] = hash_feat[i];
					}
					// Remaining positions are zero (WMMA padding)

					// 2. Recompute MLP forward → residual (identity activation, no sigmoid)
					MlpWeights smem_mlp = {smem_mlp_W1, smem_mlp_W2, smem_mlp_W3};
					mlp_forward_inline(my_input, my_residual, my_h1_post, my_h2_post, false, smem_mlp);

					// 3. Load SH base color and compute feat = SH + residual.
					// `--method res_3d` per-Gauss bias gate (mirror of forward):
					// textured carriers post-split force sh_color = 0 so the
					// +sh_bias floor isn't part of their feat (pure residual).
					float sh_color[3];
					for (int ch = 0; ch < 3; ch++)
						sh_color[ch] = (d_textured_bias_gate && tex_j) ? 0.0f : RGB_TO_FLOAT(colors[global_id * 3 + ch]);

					// Activation gates depend on d_residual_mode (see forward.cu):
					//   0 (3D_SH_res): outer ReLU gates BOTH branches together.
					//   1 (3D_SH_add): separate ReLUs — residual gated by
					//                  (residual + res_bias > 0); SH path is
					//                  ungated here (SH's inner ReLU lives
					//                  upstream in preprocessCUDA).
					// Mode 2 (mixed): signed residual, no per-Gauss ReLU → both gates = 1.
					// (The per-pixel ReLU at the end is handled by torch.relu in Python,
					// so dL_dpixel already comes in with the correct clamp gating.)
					#pragma unroll
					for (int o = 0; o < ORIG_OUTPUT_DIM; o++) {
						float gate_res;
						if (d_residual_mode == 1) {
							gate_res = (my_residual[o] + d_res_bias > 0.0f) ? 1.0f : 0.0f;
						} else if (d_residual_mode == 2) {
							gate_res = 1.0f;
						} else {
							// `--ste`: SIGN-AWARE STE on the outer ReLU. Gradient
							// passes at clamped pixels ONLY when dL/dpixel < 0
							// (i.e. loss wants channel HIGHER → release-clamp
							// direction). Eliminates the deep-negative runaway
							// of naive STE. See diff_surfel_3D_sh_res for the
							// full reasoning.
							gate_res = ((sh_color[o] + my_residual[o] + d_res_bias > 0.0f) ||
							            (d_ste_relu && dL_dpixel[o] < 0.0f))
							           ? 1.0f : d_lru_slope;  // `--lru` α (0 = std ReLU)
						}
						// res_3d dual cascade: per-cascade MLP grad
						my_dL_dz3[o] = dL_dpixel[o] * w_color_collab * gate_res;
					}
					// Positions 3-15 of dL_dz3 stay zero (WMMA padding)

					for (int ch = 0; ch < 3; ch++) {
						float gate_sh;
						if (d_residual_mode == 1 || d_residual_mode == 2) {
							// SH always passes (mode 1: outer ReLU on residual only;
							// mode 2: no ReLU; SH's inner ReLU is upstream).
							gate_sh = 1.0f;
						} else {
							// `--ste` sign-aware (see gate_res above).
							gate_sh = ((sh_color[ch] + my_residual[ch] + d_res_bias > 0.0f) ||
							           (d_ste_relu && dL_dpixel[ch] < 0.0f))
							          ? 1.0f : d_lru_slope;  // `--lru` α (0 = std ReLU)
						}
						// res_3d dual cascade: SH color grad uses per-cascade weight
						acc_dL_dcolors[ch] += dL_dpixel[ch] * w_color_collab * gate_sh;
					}
				}

				// Note: no __syncthreads needed here - ballot check above provides sync
				extern __shared__ __half dynamic_smem[];
				// d_skip_mlp_grad: periodic-freeze flag. All threads see the same
				// device-global value, so each of the conditionals below is
				// uniform across the block — collective WMMA GEMMs are either
				// entered by everyone or skipped by everyone. Default false →
				// every branch evaluates true → identical to pre-flag path.
				// `--method mixed`: tex_j is per-Gauss (uniform across all 256 threads),
				// so this skip is warp-uniform — collective WMMA is safely bypassed
				// for untextured contributors. All my_* tensors are zero for untex
				// → GEMM would accumulate zero anyway; skipping reclaims the cycles.
				if (!d_skip_mlp_grad && tex_j) {
					wmma_gemm_layer3(my_dL_dz3, my_h2_post, tile_dL_dW3, dynamic_smem);
				}

				// --- Phase 2: Layer 3 backward → dL_dz2 ---
				float my_dL_dz2[TC_HIDDEN_DIM] = {0};
				if (participates && tex_j && !d_skip_mlp_grad) {
					// dL_dh2 = W3^T @ dL_dz3, then ReLU backward: dL_dz2 = dL_dh2 * (h2_post > 0)
					#pragma unroll
					for (int h = 0; h < TC_HIDDEN_DIM; h++) {
						float dL_dh2 = 0;
						#pragma unroll
						for (int o = 0; o < ORIG_OUTPUT_DIM; o++) {
							dL_dh2 += my_dL_dz3[o] * __half2float(smem_mlp_W3[o * TC_HIDDEN_DIM + h]);
						}
						my_dL_dz2[h] = (my_h2_post[h] > 0) ? dL_dh2 : 0;
					}
				}
				__syncthreads();
				// Profiling: T1 at sync before GEMM L2 (end of Phase A)
				unsigned long long _prof_t1 = clock64();
				if (!d_skip_mlp_grad && tex_j) {
					wmma_gemm_layer2(my_dL_dz2, my_h1_post, tile_dL_dW2, dynamic_smem);
				}

				// --- Phase 3: Layer 2 backward → dL_dz1 ---
				float my_dL_dz1[TC_HIDDEN_DIM] = {0};
				if (participates && tex_j && !d_skip_mlp_grad) {
					// dL_dh1 = W2^T @ dL_dz2, then ReLU backward: dL_dz1 = dL_dh1 * (h1_post > 0)
					#pragma unroll
					for (int h = 0; h < TC_HIDDEN_DIM; h++) {
						float dL_dh1 = 0;
						#pragma unroll
						for (int i = 0; i < TC_HIDDEN_DIM; i++) {
							dL_dh1 += my_dL_dz2[i] * __half2float(smem_mlp_W2[i * TC_HIDDEN_DIM + h]);
						}
						my_dL_dz1[h] = (my_h1_post[h] > 0) ? dL_dh1 : 0;
					}
				}
				__syncthreads();
				// Profiling: T2 at sync before GEMM L1 (end of Phase B)
				unsigned long long _prof_t2 = clock64();
				if (!d_skip_mlp_grad && tex_j) {
					wmma_gemm_layer1(my_dL_dz1, my_input, tile_dL_dW1, dynamic_smem);
				}

				// ======== Phase 4: dL_dinput → hash & geometry gradients ========
				if (participates) {
					// dL_dinput = W1^T @ dL_dz1 (first hash_dim elements = hash features)
					// `--method mixed`: skip entire MLP-input pull-back for untextured.
					// my_dL_dz1 is 0 for untextured anyway, but skipping the compute
					// saves 16*12 scalar mults per untextured contributor per pixel.
					const int hash_dim = active_hashgrid_levels * l_dim;
					float my_dL_dinput[12] = {0};  // Max 12D (3 levels × 4D)
					if (tex_j && !d_skip_mlp_grad) {
						for (int i = 0; i < hash_dim && i < 12; i++) {
							float sum = 0;
							for (int h = 0; h < TC_HIDDEN_DIM; h++) {
								sum += my_dL_dz1[h] * __half2float(smem_mlp_W1[h * TC_INPUT_DIM + i]);
							}
							my_dL_dinput[i] = sum;
						}
					}

					// Backprop to hash features - dL_dxyz flows to geometry.
					// Skipped entirely under d_skip_mlp_grad or `--method mixed`
					// untextured Gaussians: no hash-table dL_dgrid atomicAdds, no
					// dL/dxyz-from-hash accumulation. Geometry backward below still
					// runs for transMat / normals / alpha.
					float dL_dxyz[3] = {0, 0, 0};
					if (tex_j && !d_skip_mlp_grad && !skip_hash && active_hashgrid_levels > 0 && l_dim == 4) {
						float dL_dhash[16];
						for (int i = 0; i < hash_dim; i++) dL_dhash[i] = my_dL_dinput[i];
						float hash_feat_dummy[16];
						uint32_t appearance_level = collected_ap_level[j];
						if (hash_dim == 4) {
							query_feature<true, 4, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 8) {
							query_feature<true, 8, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 12) {
							query_feature<true, 12, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 16) {
							query_feature<true, 16, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						}
						if (detach_hash_grad) { dL_dxyz[0] = 0; dL_dxyz[1] = 0; dL_dxyz[2] = 0; }
					} else if (tex_j && !d_skip_mlp_grad && !skip_hash && active_hashgrid_levels > 0 && l_dim == 2) {
						// 2D per level — supports 1..8 hash levels (hash_dim ∈ {2,4,6,8,10,12,14,16}).
						float dL_dhash[16];
						for (int i = 0; i < hash_dim; i++) dL_dhash[i] = my_dL_dinput[i];
						float hash_feat_dummy[16];
						uint32_t appearance_level = collected_ap_level[j];
						if (hash_dim == 2) {
							query_feature<true, 2, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 4) {
							query_feature<true, 4, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 6) {
							query_feature<true, 6, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 8) {
							query_feature<true, 8, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 10) {
							query_feature<true, 10, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 12) {
							query_feature<true, 12, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 14) {
							query_feature<true, 14, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						} else if (hash_dim == 16) {
							query_feature<true, 16, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
							                          appearance_level, hash_features, active_hashgrid_levels,
							                          l_scale, Base, align_corners, interp, if_contract, false,
							                          dL_dhash, dL_dfeatures, dL_dxyz);
						}
						if (detach_hash_grad) { dL_dxyz[0] = 0; dL_dxyz[1] = 0; dL_dxyz[2] = 0; }
					}

					// ======== GEOMETRY GRADIENTS ========
					float dL_dalpha = 0.0f;
					// Recompute feat for alpha gradient — branched on d_residual_mode.
					// `--method mixed` untextured: forward uses `feat[ch] = rgb[gid*3+ch]`
					// directly (no residual, no outer ReLU), so the backward feat MUST
					// match that — otherwise dL_dalpha's accum-chain is wrong.
					float feat[C];
					float sh_color_recomp[3];
					for (int ch = 0; ch < 3; ch++) {
						// `--method res_3d` per-Gauss bias gate: tex carriers
						// force sh_color = 0 (mirror of forward) so feat is
						// pure residual.
						sh_color_recomp[ch] = (d_textured_bias_gate && tex_j) ? 0.0f : RGB_TO_FLOAT(colors[global_id * 3 + ch]);
						if (!tex_j) {
							feat[ch] = sh_color_recomp[ch];
						} else if (d_residual_mode == 1) {
							feat[ch] = sh_color_recomp[ch] + fmaxf(0.0f, my_residual[ch] + d_res_bias);
						} else if (d_residual_mode == 2) {
							// mixed: signed residual, no per-Gauss ReLU.
							feat[ch] = sh_color_recomp[ch] + my_residual[ch] + d_res_bias;
						} else {
							feat[ch] = fmaxf(0.0f, sh_color_recomp[ch] + my_residual[ch] + d_res_bias);
						}
					}

					// Update accumulators — `--method res_3d` dual cascade per-Gauss.
					// dL_dalpha_color (per-cascade T_color) split from dL_dalpha
					// (joint T for depth/dist/normal/mask) at the end of this iter.
					float dL_dalpha_color = 0.0f;
					for (int ch = 0; ch < C; ch++) {
						float* const _ar = tex_j ? accum_rec_tex  : accum_rec_sv;
						float* const _lc = tex_j ? last_color_tex : last_color_sv;
						const float _la  = tex_j ? last_alpha_tex : last_alpha_sv;
						_ar[ch] = _la * _lc[ch] + (1.f - _la) * _ar[ch];
						_lc[ch] = feat[ch];
						dL_dalpha_color += (feat[ch] - _ar[ch]) * dL_dpixel[ch];
					}

					float dL_dz = 0.0f;
					float dL_dweight = 0;
#if RENDER_AXUTILITY
					const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
					const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
					if (current_contributor == median_contributor-1) {
						dL_dz += dL_dmedian_depth;
					}
#if DETACH_WEIGHT
					dL_dweight += 0;
#else
					// `--method mixed_3d`: final_A_tex excludes EWA occlusion (T_tex), making
// the dist gradient strictly textured-only. final_A_tex == final_A for
// non-mixed_3d so this is a no-op there.
dL_dweight += (final_D2 + m_d * m_d * final_A_tex - 2 * m_d * final_D) * dL_dreg;
#endif
					dL_dalpha += dL_dweight - last_dL_dT;
					last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
					// `--method mixed_3d`: final_A_tex for textured-only dist gradient.
const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A_tex - final_D) * dL_dreg;
					dL_dz += dL_dmd * dmd_dd;

					accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
					last_depth = c_d;
					dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;

					accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
					dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

					for (int ch = 0; ch < 3; ch++) {
						accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
						last_normal[ch] = normal[ch];
						dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
						acc_dL_dnormal3D[ch] += alpha * T * dL_dnormal2D[ch];
					}
#endif

					// `--method res_3d` dual cascade: color * T_color (per-cascade),
					// other (depth/dist/normal/mask) * T (joint).
					const float T_color_collab = tex_j ? T_tex_back : T_sv_back;
					dL_dalpha = dL_dalpha_color * T_color_collab + dL_dalpha * T;

					// Regularization gradients: only affect opacity, NOT geometry
					float dL_dalpha_reg = 0.0f;

					// Overdraw regularization gradient — gated on `tex_j` so
					// --method mixed/mixed_3d's untextured 2DGS rows are excluded
					// (untextured = specular layer, not penalized for overdraw).
					// `is_textured == nullptr` ⇒ tex_j defaults true ⇒ byte-identical.
					if (dL_doverdraw > 0.0f && tex_j) {
						float sig_od = 1.0f / (1.0f + expf(-OD_K * (w - OD_THRESH)));
						float dsig_od = OD_K * sig_od * (1.0f - sig_od);
						// Direct: dsig * T. Indirect: -overdraw_accum / (1 - alpha)
						float safe_denom = fmaxf(1.0f - alpha, 1e-7f);
						dL_dalpha_reg += dL_doverdraw * (dsig_od * T - overdraw_accum / safe_denom);
						overdraw_accum += dsig_od * w;
					}

					// Weight-squared regularization gradient (direct + indirect)
					if (dL_dwr != 0.0f) {
						float safe_denom_wr = fmaxf(1.0f - alpha, 1e-7f);
						dL_dalpha_reg += dL_dwr * (2.0f * w * T - 2.0f * wr_accum / safe_denom_wr);
						wr_accum += w * w;
					}

					last_alpha = alpha;
					// `--method res_3d`: advance per-cascade last_alpha.
					if (tex_j) last_alpha_tex = alpha;
					else       last_alpha_sv  = alpha;

					// Geometry gradient: only from RGB loss (no reg)
					float dL_dG = opa * dL_dalpha;
					// Opacity gradient: RGB + reg
					dL_dalpha += dL_dalpha_reg;

					// Kernel-specific shape gradients and dL_dG adjustments.
					// `--method mixed`: untextured uses the run's kernel too, so its
					// shape gradients must flow exactly like textured.
					if (kernel_type == 1 || kernel_type == 4) {
						if (beta_wins && dL_dshapes != nullptr && base > 1e-7f) {
							float dL_dshape = dL_dalpha * opa * alpha_beta * logf(base);
							acc_dL_dshapes[0] += dL_dshape;
						}
					} else if (kernel_type == 2 && per_gaussian_beta_val > 0.0f) {
						const float dG_dg_raw = (1.0f + per_gaussian_beta_val) / (demon_val * demon_val);
						dL_dG *= dG_dg_raw;
						if (dL_dshapes != nullptr) {
							float dG_dbeta = G_raw * (1.0f - G_raw) / (demon_val * demon_val);
							float dL_dbeta = dL_dalpha * opa * dG_dbeta;
							acc_dL_dshapes[0] += dL_dbeta;
						}
					} else if (kernel_type == 3 && dL_dshapes != nullptr) {
						float log_rho = logf(general_rho_safe_val);
						float dG_dbeta = -0.25f * G * general_pow_term_val * log_rho;
						float dL_dbeta = dL_dalpha * opa * dG_dbeta;
						acc_dL_dshapes[0] += dL_dbeta;
					}

					// --w_lambda_perpix direct shape gradient.
					// Forward accumulates beta_sum_pix = Σ_i w_i · β_i  (BETA_SUM_OFFSET).
					// Python loss: (w_r · beta_sum).mean()  →  dL/d(beta_sum_pix) = dL_dbeta_sum_px.
					// We inject ONLY the direct coefficient: dL/dβ_i += dL_dbeta_sum_px · w_i.
					// The indirect path (β → α → w) is intentionally NOT followed — by design
					// this reg dampens β values only, never opacity/coverage.
					if (dL_dshapes != nullptr && dL_dbeta_sum_px != 0.0f &&
					    (kernel_type == 1 || kernel_type == 2 || kernel_type == 3 || kernel_type == 4)) {
						acc_dL_dshapes[0] += dL_dbeta_sum_px * w;
					}

					// Compute dL_duv from dL_dxyz (hash gradients flowing to geometry)
					float2 dL_duv = {0.0f, 0.0f};
					if (homotrans != nullptr && active_hashgrid_levels > 0) {
						const float dL_dpx = dL_dxyz[0];
						const float dL_dpy = dL_dxyz[1];
						const float dL_dpz = dL_dxyz[2];

						if (rho3d <= rho2d) {
							const float3 sutu = collected_SuTu[j];
							const float3 svtv = collected_SvTv[j];
							dL_duv = {
								dL_dpx * sutu.x + dL_dpy * sutu.y + dL_dpz * sutu.z,
								dL_dpx * svtv.x + dL_dpy * svtv.y + dL_dpz * svtv.z
							};

							// Backprop to homotrans matrix
							acc_dL_dhomoMat[0] += dL_dpx * s.x;
							acc_dL_dhomoMat[1] += dL_dpy * s.x;
							acc_dL_dhomoMat[2] += dL_dpz * s.x;
							acc_dL_dhomoMat[3] += dL_dpx * s.y;
							acc_dL_dhomoMat[4] += dL_dpy * s.y;
							acc_dL_dhomoMat[5] += dL_dpz * s.y;
						}
						// for both rho3d and rho2d
						acc_dL_dhomoMat[6] += dL_dxyz[0];
						acc_dL_dhomoMat[7] += dL_dxyz[1];
						acc_dL_dhomoMat[8] += dL_dxyz[2];
					}

					// Geometry gradients based on whether rho3d or rho2d was used
					if (rho3d <= rho2d) {
						float2 dL_ds;
						if (kernel_type == 5) {
							// Nexel kernel: anisotropic gamma exponents
							// G = exp(-0.5 * (pow(comp_x, gamma_x) + pow(comp_y, gamma_y)))
							// dG/ds_x = -G * gamma_x * pow(comp_x, gamma_x - 1) * s_x
							float gamma_x = collected_shapes[j].x;
							float gamma_y = collected_shapes[j].y;
							const float GAMMA_EPS = 1e-6f;
							float comp_x = fminf(s.x * s.x + GAMMA_EPS, powf(1000.0f, 1.0f / gamma_x));
							float comp_y = fminf(s.y * s.y + GAMMA_EPS, powf(1000.0f, 1.0f / gamma_y));
							dL_ds = {
								dL_dG * (-G) * gamma_x * powf(comp_x, gamma_x - 1.0f) * s.x + dL_dz * Tw.x,
								dL_dG * (-G) * gamma_y * powf(comp_y, gamma_y - 1.0f) * s.y + dL_dz * Tw.y
							};
							// dL_dgamma: -0.5 * dL_dG * G * log(comp) * pow(comp, gamma)
							float dL_dgamma_x = -0.5f * dL_dG * G * logf(comp_x) * powf(comp_x, gamma_x);
							float dL_dgamma_y = -0.5f * dL_dG * G * logf(comp_y) * powf(comp_y, gamma_y);
							acc_dL_dshapes[0] += dL_dgamma_x;
							acc_dL_dshapes[1] += dL_dgamma_y;
						} else {
							// Compute dG_factor based on kernel type
							float dG_factor;
							if (kernel_type == 1 || kernel_type == 4) {
								if (beta_wins && base > 1e-7f) {
									dG_factor = -shape_val * alpha_beta / (base * k_sq);
								} else {
									dG_factor = 0.0f;
								}
							} else if (kernel_type == 3) {
								dG_factor = -0.5f * general_beta_val * G * general_pow_term_val / general_rho_safe_val;
							} else {
								dG_factor = -G;  // Gaussian (and flex with adjusted dL_dG)
							}
							dL_ds = {
								dL_dG * dG_factor * s.x + dL_dz * Tw.x,
								dL_dG * dG_factor * s.y + dL_dz * Tw.y
							};
						}
						dL_ds.x += dL_duv.x;
						dL_ds.y += dL_duv.y;

						const float3 dz_dTw = {s.x, s.y, 1.0};
						const float dsx_pz = dL_ds.x / p_stored.z;
						const float dsy_pz = dL_ds.y / p_stored.z;
						const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
						const float3 dL_dk = cross(l_stored, dL_dp);
						const float3 dL_dl = cross(dL_dp, k_stored);

						const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
						const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
						const float3 dL_dTw = {
							pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x,
							pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y,
							pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};

						acc_dL_dtransMat[0] += dL_dTu.x;
						acc_dL_dtransMat[1] += dL_dTu.y;
						acc_dL_dtransMat[2] += dL_dTu.z;
						acc_dL_dtransMat[3] += dL_dTv.x;
						acc_dL_dtransMat[4] += dL_dTv.y;
						acc_dL_dtransMat[5] += dL_dTv.z;
						acc_dL_dtransMat[6] += dL_dTw.x;
						acc_dL_dtransMat[7] += dL_dTw.y;
						acc_dL_dtransMat[8] += dL_dTw.z;
						// AbsGS: accumulate absolute Tu.z / Tv.z for cancellation-free densification
						acc_dL_dmean2D[2] += fabsf(dL_dTu.z);
						acc_dL_dmean2D[3] += fabsf(dL_dTv.z);
					} else {
						// 2D fallback: gradient w.r.t. screen-space position
						float dG_factor_2d;
						if (kernel_type == 1 || kernel_type == 4) {
							if (!beta_wins) {
								dG_factor_2d = -0.5f * alpha_lp * FilterInvSquare;
							} else {
								dG_factor_2d = 0.0f;
							}
						} else if (kernel_type == 3) {
							dG_factor_2d = -0.5f * general_beta_val * G * general_pow_term_val / general_rho_safe_val * FilterInvSquare;
						} else {
							dG_factor_2d = -G * FilterInvSquare;
						}
						const float dG_ddelx = dG_factor_2d * (xy.x - pixf.x);
						const float dG_ddely = dG_factor_2d * (xy.y - pixf.y);
						acc_dL_dmean2D[0] += dL_dG * dG_ddelx;
						acc_dL_dmean2D[1] += dL_dG * dG_ddely;
						// AbsGS: abs gradient from low-pass filter path
						acc_dL_dmean2D[2] += fabsf(dL_dG * dG_ddelx);
						acc_dL_dmean2D[3] += fabsf(dL_dG * dG_ddely);
						if (render_mode & 0x400) {
							// --lowpass: propagate low-pass filter + depth gradient to transMat
							const float dL_dxy_x = dL_dG * dG_ddelx;
							const float dL_dxy_y = dL_dG * dG_ddely;
							const float inv_Tw_z = 1.0f / (Tw.z + 1e-7f);
							const float2 dL_ds_lp = {dL_dz * Tw.x, dL_dz * Tw.y};
							const float3 dz_dTw_lp = {s.x, s.y, 1.0f};
							const float dsx_pz_lp = dL_ds_lp.x / p_stored.z;
							const float dsy_pz_lp = dL_ds_lp.y / p_stored.z;
							const float3 dL_dp_lp = {dsx_pz_lp, dsy_pz_lp, -(dsx_pz_lp * s.x + dsy_pz_lp * s.y)};
							const float3 dL_dk_lp = cross(l_stored, dL_dp_lp);
							const float3 dL_dl_lp = cross(dL_dp_lp, k_stored);
							const float3 dL_dTu_lp = {-dL_dk_lp.x, -dL_dk_lp.y, -dL_dk_lp.z};
							const float3 dL_dTv_lp = {-dL_dl_lp.x, -dL_dl_lp.y, -dL_dl_lp.z};
							const float3 dL_dTw_lp = {
								pixf.x * dL_dk_lp.x + pixf.y * dL_dl_lp.x + dL_dz * dz_dTw_lp.x + dL_dxy_x * inv_Tw_z,
								pixf.x * dL_dk_lp.y + pixf.y * dL_dl_lp.y + dL_dz * dz_dTw_lp.y + dL_dxy_y * inv_Tw_z,
								pixf.x * dL_dk_lp.z + pixf.y * dL_dl_lp.z + dL_dz * dz_dTw_lp.z - (dL_dxy_x * xy.x + dL_dxy_y * xy.y) * inv_Tw_z};
							acc_dL_dtransMat[0] += dL_dTu_lp.x;
							acc_dL_dtransMat[1] += dL_dTu_lp.y;
							acc_dL_dtransMat[2] += dL_dTu_lp.z;
							acc_dL_dtransMat[3] += dL_dTv_lp.x;
							acc_dL_dtransMat[4] += dL_dTv_lp.y;
							acc_dL_dtransMat[5] += dL_dTv_lp.z;
							acc_dL_dtransMat[6] += dL_dTw_lp.x;
							acc_dL_dtransMat[7] += dL_dTw_lp.y;
							acc_dL_dtransMat[8] += dL_dTw_lp.z;
						} else {
							acc_dL_dtransMat[8] += dL_dz;
						}
					}

					acc_dL_dopacity += G * dL_dalpha;

					// NOTE: T was already updated at the start of this iteration
					// (T = T / (1-alpha) to recover T_before, matching 2DGS backward)
					// No T-based termination needed - loop runs through all contributors
				}

				// ====================================================================
				// Per-Gaussian gradient flush: warp-reduce register accumulators, then
				// one atomicAdd per warp per field. All 32 lanes in the warp must
				// participate in cg::reduce (non-participants carry 0 → benign).
				// This replaces up-to-256 colliding atomicAdds per Gaussian per field
				// with up-to-8 (one per warp). MLP weight grads (tile_dL_dW*) remain
				// on the collaborative-GEMM path — not touched here. Hash-table grads
				// inside query_feature<true> also stay as-is (each pixel hits distinct
				// cells, no locality to reduce across).
				// ====================================================================
				{
					// dL_dopacity: 1 slot
					{
						float s_ = cg::reduce(warp, acc_dL_dopacity, cg::plus<float>());
						if (warp.thread_rank() == 0) atomicAdd(&dL_dopacity[global_id], s_);
					}
					// dL_dcolors: 3 slots
					#pragma unroll
					for (int ch = 0; ch < 3; ch++) {
						float s_ = cg::reduce(warp, acc_dL_dcolors[ch], cg::plus<float>());
						if (warp.thread_rank() == 0) atomicAdd(&dL_dcolors[global_id * 3 + ch], s_);
					}
					// dL_dnormal3D: 3 slots
					#pragma unroll
					for (int ch = 0; ch < 3; ch++) {
						float s_ = cg::reduce(warp, acc_dL_dnormal3D[ch], cg::plus<float>());
						if (warp.thread_rank() == 0) atomicAdd(&dL_dnormal3D[global_id * 3 + ch], s_);
					}
					// dL_dtransMat: 9 slots
					#pragma unroll
					for (int k = 0; k < 9; k++) {
						float s_ = cg::reduce(warp, acc_dL_dtransMat[k], cg::plus<float>());
						if (warp.thread_rank() == 0) atomicAdd(&dL_dtransMat[global_id * 9 + k], s_);
					}
					// dL_dmean2D: 4 slots (.x, .y, .z, .w)
					{
						float sx = cg::reduce(warp, acc_dL_dmean2D[0], cg::plus<float>());
						float sy = cg::reduce(warp, acc_dL_dmean2D[1], cg::plus<float>());
						float sz = cg::reduce(warp, acc_dL_dmean2D[2], cg::plus<float>());
						float sw = cg::reduce(warp, acc_dL_dmean2D[3], cg::plus<float>());
						if (warp.thread_rank() == 0) {
							atomicAdd(&dL_dmean2D[global_id].x, sx);
							atomicAdd(&dL_dmean2D[global_id].y, sy);
							atomicAdd(&dL_dmean2D[global_id].z, sz);
							atomicAdd(&dL_dmean2D[global_id].w, sw);
						}
					}
					// dL_dhomoMat: 9 slots (only written when homotrans != nullptr)
					if (homotrans != nullptr) {
						#pragma unroll
						for (int k = 0; k < 9; k++) {
							float s_ = cg::reduce(warp, acc_dL_dhomoMat[k], cg::plus<float>());
							if (warp.thread_rank() == 0) atomicAdd(&dL_dhomoMat[global_id * 9 + k], s_);
						}
					}
					// dL_dshapes: 1 or 2 slots depending on kernel_type (guard on nullptr)
					if (dL_dshapes != nullptr) {
						if (kernel_type == 5) {
							float s0 = cg::reduce(warp, acc_dL_dshapes[0], cg::plus<float>());
							float s1 = cg::reduce(warp, acc_dL_dshapes[1], cg::plus<float>());
							if (warp.thread_rank() == 0) {
								atomicAdd(&dL_dshapes[global_id * 2 + 0], s0);
								atomicAdd(&dL_dshapes[global_id * 2 + 1], s1);
							}
						} else {
							float s0 = cg::reduce(warp, acc_dL_dshapes[0], cg::plus<float>());
							if (warp.thread_rank() == 0) atomicAdd(&dL_dshapes[global_id], s0);
						}
					}
				}

				// Profiling: T3 at end of Phase C (sync needed to measure wall time)
				__syncthreads();
				unsigned long long _prof_t3 = clock64();
				if (block.thread_rank() == 0) {
					atomicAdd(&d_bw_profile[0], _prof_t1 - _prof_t0);  // Phase A
					atomicAdd(&d_bw_profile[1], _prof_t2 - _prof_t1);  // Phase B
					atomicAdd(&d_bw_profile[2], _prof_t3 - _prof_t2);  // Phase C
					atomicAdd(&d_bw_profile[4], _prof_t3 - _prof_t0);  // Total
				}
			}
			// Update contributor after processing all Gaussians in this batch
			contributor -= effective_toDo;
			// Mark pixel done once all its contributing Gaussians have been processed
			// (contributor underflows past 0 for uint32, or explicitly reaches 0)
			if (inside && (contributor == 0 || contributor > toDo)) {
				done = true;
			}
		}
		// ============================================================================
		// OTHER MODES: ORIGINAL PER-PIXEL ITERATION (divergent)
		// ============================================================================
		else
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor) {
				// Once contributor wraps past 0 (uint32 underflow), all remaining
				// Gaussians will also be skipped. Mark done for tile-level early exit.
				if (contributor > toDo) { done = true; }
				continue;
			}

			// `--method mixed`: untextured Gaussians use the SAME 2DGS ray-splat
			// geometry as textured/2DGS. They differ ONLY in the color path
			// (skip hash/MLP — gated below). The splat shape IS rho3d.
			const bool tex_bw = collected_is_textured_bw[j];
			// `--l2`: per-Gauss image-grad routing — see MODE-5 site above for
			// the rationale and the no-op behavior when --l2 is inactive.
			const float* const dL_dpixel = tex_bw ? dL_dpixel_tex : dL_dpixel_untex;

			// ============================================================
			// `--method mixed_3d` — UNTEXTURED EWA 3D-ellipsoid backward.
			// Self-contained mirror of the textured per-pixel recurrence
			// (color/depth/alpha/normal/dist/reg state advances kept exactly
			// so textured neighbours in the same pixel stay consistent) with
			// the EWA alpha/G, and FastGS-verbatim conic/mean2D/opacity grads.
			// dL_dconic is carried in dL_dtransMat[gid*9+0..2] →
			// preprocessCUDA's untextured-EWA branch (ewa_backward_vjp).
			// Geometry depth/normal targets are not backpropagated for
			// untextured (simple-splat layer, normal≡0); recurrence STATE is
			// still advanced so cross-set occlusion grads remain correct.
			// `continue` past all the textured 2DGS/hash/MLP code.
			// ============================================================
			if (!tex_bw && ewa_conic != nullptr) {
				const float4 con_o = collected_ewa_conic_bw[j];
				if (con_o.w <= 0.0f) continue;             // culled in preprocess
				const int gid_e = collected_id[j];
				const float2 xy_e = collected_xy[j];
				const float2 de = { xy_e.x - pixf.x, xy_e.y - pixf.y };
				const float me = con_o.x * de.x * de.x
				               + 2.0f * con_o.y * de.x * de.y
				               + con_o.z * de.y * de.y;
				const float opa_e = con_o.w;

				// Forward-replay alpha/G + dG/dm (must match the forward EWA branch).
				// Untextured kernel = `--kernel2` (render_mode bits[16..19] =
				// kernel_type2+1; 0 ⇒ fall back to kernel_type). MUST match the
				// forward EWA decode exactly.
				const int _utk_n = (render_mode >> 16) & 0xF;
				const int ut_kt = _utk_n ? (_utk_n - 1) : kernel_type;
				float G_e, dG_dm, base_e = 0.0f, beta_e = 0.0f;
				bool is_beta_e = (ut_kt == 1 || ut_kt == 4);
				if (is_beta_e) {
					const float k_sq = (ut_kt == 4) ? 9.0f : 1.0f;
					if (me >= k_sq + 1e-6f) continue;      // compact support
					beta_e = collected_shapes[j].x;
					base_e = fmaxf(0.0f, 1.0f - me / k_sq);
					G_e = powf(base_e, beta_e);
					dG_dm = (base_e > 1e-9f) ? (-(beta_e / k_sq) * powf(base_e, beta_e - 1.0f)) : 0.0f;
				} else {
					const float power = -0.5f * me;
					if (power > 0.0f) continue;            // me < 0 (degenerate)
					G_e = expf(power);
					dG_dm = -0.5f * G_e;
				}
				float alpha_e = fminf(0.99f, opa_e * G_e);
				if (alpha_e < 1.0f / 255.0f) continue;

				// ---- shared per-pixel recurrence (mirror textured path) ----
				T = T / (1.f - alpha_e);
				// `--method res_3d` dual cascade: untextured Gauss decay the
				// SV-aux cascade only. T_sv_back recovers T_sv_pre_at_this_Gauss.
				T_sv_back = T_sv_back / (1.f - alpha_e);
				const float dchannel_dcolor = alpha_e * T_sv_back;
				const float w_e = alpha_e * T;
				// dL_dalpha split: color contributions scaled by T_sv_back
				// (per-cascade), mask/dist/normal contributions scaled by T
				// (joint cascade — they're joint outputs). Combined at end.
				float dL_dalpha_color = 0.0f;
				float dL_dalpha = 0.0f;

				for (int ch = 0; ch < C; ch++) {
					// SV-cascade recurrence (independent of textured cascade).
					accum_rec_sv[ch] = last_alpha_sv * last_color_sv[ch] + (1.f - last_alpha_sv) * accum_rec_sv[ch];
					const float cval = RGB_TO_FLOAT(colors[gid_e * 3 + ch]);
					last_color_sv[ch] = cval;
					// dL_dpixel == dL_dout_color which routes equally to both
					// cascades (out_color = C_sv_aux + C_tex_aux, linear sum).
					dL_dalpha_color += (cval - accum_rec_sv[ch]) * dL_dpixel[ch];
					atomicAdd(&(dL_dcolors[gid_e * 3 + ch]), dchannel_dcolor * dL_dpixel[ch]);
				}

#if RENDER_AXUTILITY
				const float c_d = depths[gid_e];          // camera-space z (forward used this)
				// `--method mixed_3d`: EWA Gaussians are DETACHED from the
				// normal-consistency loss (--lambda_normal / --w_normal).
				// Forward keeps them in D (so surf_normal = depth_to_normal(D)
				// sees the full depth) and in the recurrence state (so textured
				// contributors get correct dL_dalpha via their own
				// (c_d_tex - accum_depth_rec) terms), but EWA's own dL_dalpha
				// does NOT pick up the depth/normal grads:
				//   • skip dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth
				//   • skip dL_dalpha += (0 - accum_normal_rec) * dL_dnormal2D
				// Distortion is already skipped (dL_dweight = 0). Color and
				// mask (dL_dpixel / dL_daccum) grads still flow into EWA so
				// it keeps being optimized for photometric + opacity tasks.
				float dL_dweight = 0.0f;
				dL_dalpha += dL_dweight - last_dL_dT;
				last_dL_dT = dL_dweight * alpha_e + (1 - alpha_e) * last_dL_dT;

				accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
				last_depth = c_d;
				// EWA detached from depth loss (see comment above).

				accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
				dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

				// untextured surfel normal ≡ 0 (forward set N += 0). Advance the
				// recurrence state so textured neighbours stay consistent; EWA
				// detached from normal loss (no dL_dalpha contribution here).
				for (int ch = 0; ch < 3; ch++) {
					accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
					last_normal[ch] = 0.0f;
				}
#endif

				// `--method res_3d` dual cascade: scale per-cascade. Color uses
				// T_sv_back; mask/dist/normal use joint T.
				dL_dalpha = dL_dalpha_color * T_sv_back + dL_dalpha * T;

				// regularization (weight-squared only) — overdraw is textured-only
				// in mixed/mixed_3d; this is the untextured EWA branch, so it
				// neither receives an overdraw grad nor advances `overdraw_accum`
				// (forward also skipped its sig contribution).
				float dL_dalpha_reg = 0.0f;
				if (dL_dwr != 0.0f) {
					float safe_denom_wr = fmaxf(1.0f - alpha_e, 1e-7f);
					dL_dalpha_reg += dL_dwr * (2.0f * w_e * T - 2.0f * wr_accum / safe_denom_wr);
					wr_accum += w_e * w_e;
				}

				last_alpha = alpha_e;
				// Advance SV-cascade per-iteration state for next reverse step.
				last_alpha_sv = alpha_e;
				dL_dalpha += dL_dalpha_reg;

				// Opacity clamp: alpha = min(.99, opa·G). When clamped, dα/dopa =
				// dα/dG = 0 → skip opacity/conic/mean2D/shape grads (FastGS).
				if (opa_e * G_e > 0.99f) continue;

				atomicAdd(&(dL_dopacity[gid_e]), G_e * dL_dalpha);

				const float dL_dG = opa_e * dL_dalpha;
				const float dL_dm = dL_dG * dG_dm;
				// dL/dconic (a,b,c) carried via dL_dtransMat[gid*9+0..2].
				// FastGS's verbatim computeCov2DCUDA (in ewa_backward_vjp)
				// expects the off-diagonal in its half convention
				// (dL_dconic.y = dG/dB with ∂m/∂B treated as dx·dy, NOT
				// 2·dx·dy) — see FastGS render: -0.5·gdx·d.y·dL_dG. The
				// dL_dmean2D path below DOES use the full 2·dx·dy (verified
				// exact via the means3D gradcheck), so only this carrier
				// drops the factor 2 on the b channel.
				atomicAdd(&dL_dtransMat[gid_e * 9 + 0], dL_dm * de.x * de.x);
				atomicAdd(&dL_dtransMat[gid_e * 9 + 1], dL_dm * de.x * de.y);
				atomicAdd(&dL_dtransMat[gid_e * 9 + 2], dL_dm * de.y * de.y);
				// dL/d(screen mean) via m, chained NDC→pixel (×0.5·W/H)
				const float dL_dddx = dL_dm * (2.0f * con_o.x * de.x + 2.0f * con_o.y * de.y);
				const float dL_dddy = dL_dm * (2.0f * con_o.y * de.x + 2.0f * con_o.z * de.y);
				const float gx = dL_dddx * 0.5f * (float)W;
				const float gy = dL_dddy * 0.5f * (float)H;
				atomicAdd(&dL_dmean2D[gid_e].x, gx);
				atomicAdd(&dL_dmean2D[gid_e].y, gy);
				atomicAdd(&dL_dmean2D[gid_e].z, fabsf(gx));   // AbsGS densify proxy
				atomicAdd(&dL_dmean2D[gid_e].w, fabsf(gy));

				// Restricted-beta shape gradient (kernel_type 1/4): dL/dβ =
				// dL_dG · G · ln(base), base ∈ (0,1].
				if (is_beta_e && dL_dshapes != nullptr && base_e > 1e-7f) {
					atomicAdd(&dL_dshapes[gid_e], dL_dG * G_e * logf(base_e));
				}
				continue;
			}

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			const float splat_size = collected_size[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y);
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

		// compute intersection and depth (full 2DGS, both halves)
		float rho = min(rho3d, rho2d);
		float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
		if (c_d < near_n) continue;
		float4 nor_o = collected_normal_opacity[j];
		float normal[3] = {nor_o.x, nor_o.y, nor_o.z};  // Already normalized in preprocessing
		float opa = nor_o.w;

		// accumulations

		float alpha, G = 0.0f, demon = 1.0f;
		float shape_val = 0.0f;  // For beta kernel gradient
		float base = 0.0f;       // For beta kernel gradient
		float per_gaussian_beta = 0.0f;  // For flex kernel gradient
		float G_raw = 0.0f;  // For flex kernel gradient (raw Gaussian before beta transform)
		float general_beta = 0.0f;  // For general kernel gradient
		float general_pow_term = 0.0f;  // For general kernel gradient: (r²)^(β/2)
		float general_rho_safe = 0.0f;  // For general kernel gradient: max(rho, 1e-8)

		// For beta kernel max-pool gradient routing
		bool beta_wins = false;
		float alpha_beta = 0.0f;
		float alpha_lp = 0.0f;

		// For beta_scaled kernel (type 4), we need k_sq for gradient computation
		float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;

		// AA-2DGS mip-filter state (populated when d_aa_kernel_size > 0 and standard Gaussian kernel)
		bool is_aa = false;
		float aa_coef = 1.0f;

		// `--method mixed`: untextured uses the run's kernel (same as textured);
		// only the color path differs (skip hash/MLP).
		if (kernel_type == 1 || kernel_type == 4) {
			// Beta kernel with separate G_obj (Beta) and G_screen (Gaussian low-pass)
			// kernel_type 1: k²=1 (unit circle cutoff)
			// kernel_type 4: k²=9 (3σ scaled, matches Gaussian extent)

			// 1. Hard support check on object-space distance
			if (rho3d >= k_sq + 1e-6f)
				continue;  // Outside compact support - skip entirely

			shape_val = collected_shapes[j].x;

			// 2. Object-space Beta kernel
			base = fmaxf(0.0f, 1.0f - rho3d / k_sq);
			alpha_beta = powf(base, shape_val);

			// 3. Screen-space Gaussian low-pass
			alpha_lp = expf(-rho2d / 2.0f);

			// 4. Max-pool handoff: track which branch won for gradient routing
			beta_wins = (alpha_beta >= alpha_lp);
			G = beta_wins ? alpha_beta : alpha_lp;

			// 5. Final alpha
			alpha = fminf(0.99f, opa * G);
		} else if (kernel_type == 2) {
			// Flex kernel: Standard Gaussian with per-Gaussian learnable beta
			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			G_raw = exp(power);
			per_gaussian_beta = collected_shapes[j].x;  // shapes array holds per-Gaussian beta
			if (per_gaussian_beta > 0.0f) {
				demon = 1.0f + per_gaussian_beta * G_raw;
				G = (1.0f + per_gaussian_beta) * G_raw / demon;
			} else {
				G = G_raw;
			}
			alpha = min(0.99f, opa * G);
		} else if (kernel_type == 3) {
			// General kernel: Isotropic Generalized Gaussian
			// Formula: G = exp(-0.5 * (r²)^(β/2))
			general_beta = collected_shapes[j].x;  // beta in range [2.0, 8.0]
			float exponent = 0.5f * general_beta;  // β/2

			general_rho_safe = fmaxf(rho, 1e-8f);
			general_pow_term = powf(general_rho_safe, exponent);  // (r²)^(β/2)
			float power = -0.5f * general_pow_term;

			if (power > 0.0f)
				continue;

			G = expf(power);
			alpha = min(0.99f, opa * G);
		} else if (d_aa_kernel_size > 0.0f) {
			// AA-2DGS Jacobian-based mip filter (forward replay).
			is_aa = true;
			const float ks = d_aa_kernel_size;
			const float k_sq_aa = ks * ks;
			const float pz_inv_aa = 1.0f / p.z;
			const float pz_sq_inv_aa = pz_inv_aa * pz_inv_aa;
			const float3 dp_dx_aa = cross(Tv, Tw);
			const float3 dp_dy_aa = cross(Tw, Tu);
			const float J_a = (dp_dx_aa.x * p.z - p.x * dp_dx_aa.z) * pz_sq_inv_aa;
			const float J_b = (dp_dx_aa.y * p.z - p.y * dp_dx_aa.z) * pz_sq_inv_aa;
			const float J_c = (dp_dy_aa.x * p.z - p.x * dp_dy_aa.z) * pz_sq_inv_aa;
			const float J_d = (dp_dy_aa.y * p.z - p.y * dp_dy_aa.z) * pz_sq_inv_aa;
			const float det_J = J_a * J_d - J_b * J_c;
			const float trace_JJT = J_a*J_a + J_b*J_b + J_c*J_c + J_d*J_d;
			const float det_V = k_sq_aa * det_J * det_J + ks * trace_JJT + 1.0f;
			if (fabsf(det_V) < 1e-8f) continue;
			const float det_V_inv = 1.0f / det_V;
			aa_coef = sqrtf(det_V_inv + 1e-8f);
			const float term1 = J_d * s.x - J_c * s.y;
			const float term2 = J_a * s.y - J_b * s.x;
			const float rho_aa_num = (s.x*s.x + s.y*s.y) + ks * (term1*term1 + term2*term2);
			const float rho_aa = rho_aa_num * det_V_inv;
			const float power_aa = -0.5f * rho_aa;
			if (power_aa > 0.0f) continue;
			G = expf(power_aa);
			alpha = fminf(0.99f, aa_coef * opa * G);
		} else {
			// Standard Gaussian kernel
			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			G = exp(power);
			if(beta > 0.0){
				demon = 1.0 + beta * G;
				G = (1.0 + beta) * G / demon;
			}

			alpha = min(0.99f, opa * G);
		}

		if (alpha < 1.0f / 255.0f)
			continue;

			T = T / (1.f - alpha);
			// `--method res_3d` dual cascade: textured Gauss decay tex-aux
			// cascade. The `!tex_bw && ewa_conic` branch above already short-
			// circuits untex-EWA Gauss for res_3d, so tex_bw == True everywhere
			// here in practice; the routing handles `--method mixed` fallback.
			const bool _route_tex_std = tex_bw;
			if (_route_tex_std) {
				T_tex_back = T_tex_back / (1.f - alpha);
			} else {
				T_sv_back = T_sv_back / (1.f - alpha);
			}
			const float T_color_std = _route_tex_std ? T_tex_back : T_sv_back;
			const float dchannel_dcolor = alpha * T_color_std;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair). dL_dalpha split per-cascade: color via T_color_std,
			// other outputs (mask/dist/normal/depth) via joint T.
			float dL_dalpha_color = 0.0f;
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];

			float dL_dxyz[3] = {0};
			
			if(level == 0){
				for (int ch = 0; ch < C; ch++)
				{
					const float c = collected_colors[ch * BLOCK_SIZE + j];

					// Update last color (to be used in the next iteration)
					// `--method res_3d`: per-cascade accum_rec / last_alpha / last_color.
					float* const _ar = _route_tex_std ? accum_rec_tex  : accum_rec_sv;
					float* const _lc = _route_tex_std ? last_color_tex : last_color_sv;
					const float _la  = _route_tex_std ? last_alpha_tex : last_alpha_sv;
					_ar[ch] = _la * _lc[ch] + (1.f - _la) * _ar[ch];
					_lc[ch] = c;

					const float dL_dchannel = dL_dpixel[ch];
					dL_dalpha_color += (c - _ar[ch]) * dL_dchannel;
					// Update the gradients w.r.t. color of the Gaussian.
					// Atomic, since this pixel is just one of potentially
					// many that were affected by this Gaussian.
					atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				}
			}
			else {
				
				// Calculate and get features & dy_dx.
				// `--method mixed`: world-xyz reconstruction is needed ONLY for the
				// hash query (textured). Skip it + the two shared loads for untex.
				float3 xyz = {0.0f, 0.0f, 0.0f};
				if (tex_bw) {
					const float3 pk = collected_pk[j];
					// intersection pts xyz
					if(rho3d <= rho2d){
						const float3 sutu = collected_SuTu[j];
						const float3 svtv = collected_SvTv[j];
						xyz = {s.x * sutu.x + s.y * svtv.x + pk.x,
							s.x * sutu.y + s.y * svtv.y + pk.y,
							s.x * sutu.z + s.y * svtv.z + pk.z};
					}
					else xyz = pk;
				}

				float dL_dchannels[C], grad_feat[C], feat[C];
				float sum_grad = 0.0;
				for(int ch = 0; ch < C; ch++){
					// const float dL_dchannel = dL_dpixel[ch];
					dL_dchannels[ch] = dL_dpixel[ch];
					grad_feat[ch] =  dchannel_dcolor * dL_dchannels[ch];
					sum_grad += fabs(grad_feat[ch]);
				}

				atomicAdd(&(dL_gradsum[global_id]), sum_grad);
				
				bool debug = false;
				uint32_t appearance_level = collected_ap_level[j];

				bool contract = if_contract;

			// hashgrid feature interpolation
			// in BW, query_feature will update dL_dfeatures & dL_dxyz
			// Note: render_mode may have flags in upper bits (e.g., inference flag), so mask to get base mode
			switch (render_mode & 0xFF){
			case 5: {
				// 3D_SH_res backward: SH base color + hash MLP residual
				// feat = ReLU(SH) + ReLU(MLP(hash(xyz))), identity MLP activation

				// `--method mixed` untextured fast-path backward: identical 2DGS
				// geometry (computed above), simple-splatting color — no hash query,
				// no MLP recompute, no dL_dxyz pull-back. Forward used
				// `feat[ch] = rgb[gid*3+ch]` (the SV/SH baseline = `colors` here),
				// so:
				//   • dL_dcolors[gid] += w · dL_dpixel  (BACKWARD::preprocess → dL_dsh)
				//   • feat[ch] := baseline, so the post-switch accum-chain
				//     `dL_dalpha += (feat - accum_rec)·dL_dchannel` is CORRECT
				//     (this is what lets untextured surfels learn scale/rot/opacity
				//     to match the image — exactly like 2DGS).
				if (!tex_bw) {
					for (int ch = 0; ch < C; ch++) {
						const float dL_dchannel = dL_dpixel[ch];
						atomicAdd(&(dL_dcolors[global_id * 3 + ch]),
						          dchannel_dcolor * dL_dchannel);
						feat[ch] = RGB_TO_FLOAT(colors[global_id * 3 + ch]);
					}
					break;
				}

				// 0. Compute xyz intersection point (same as forward pass)
				const float3 pk = collected_pk[j];
				float3 xyz;
				if (rho3d <= rho2d) {
					const float3 sutu = collected_SuTu[j];
					const float3 svtv = collected_SvTv[j];
					xyz = {s.x * sutu.x + s.y * svtv.x + pk.x,
					       s.x * sutu.y + s.y * svtv.y + pk.y,
					       s.x * sutu.z + s.y * svtv.z + pk.z};
				} else {
					xyz = pk;
				}

				const int active_hashgrid_levels = (level >> 8) & 0xFF;

				// 1. Query hash features — skip for tail pixels (matches forward)
				bool skip_hash = (d_contrib_thresh_bw > 0.0f && w < d_contrib_thresh_bw)
				                 || (d_count_thresh_bw > 0 && contributor >= (uint32_t)d_count_thresh_bw);
				const int hash_dim_px = active_hashgrid_levels * l_dim;
				float hash_feat[16] = {0};
				if (!skip_hash && active_hashgrid_levels > 0 && l_dim == 4) {
					if (hash_dim_px == 4)
						query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 8)
						query_feature<false, 8, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 12)
						query_feature<false, 12, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 16)
						query_feature<false, 16, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
				} else if (!skip_hash && active_hashgrid_levels > 0 && l_dim == 2) {
					// 2D per level — supports 1..8 hash levels (hash_dim ∈ {2,4,6,8,10,12,14,16}).
					if (hash_dim_px == 2)
						query_feature<false, 2, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 4)
						query_feature<false, 4, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 6)
						query_feature<false, 6, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 8)
						query_feature<false, 8, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 10)
						query_feature<false, 10, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 12)
						query_feature<false, 12, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 14)
						query_feature<false, 14, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_px == 16)
						query_feature<false, 16, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false);
				}

				// 2. Build MLP input: [hash(hash_dim) | pad(16-hash_dim)] = 16D
				float mlp_input[TC_INPUT_DIM];
				for (int i = 0; i < TC_INPUT_DIM; i++) mlp_input[i] = 0.0f;
				for (int i = 0; i < hash_dim_px && i < TC_INPUT_DIM; i++) mlp_input[i] = hash_feat[i];

				// 3. Recompute MLP forward → residual (identity, no sigmoid)
				float h1_post[TC_HIDDEN_DIM], h2_post[TC_HIDDEN_DIM];
				float residual[ORIG_OUTPUT_DIM];
				MlpWeights smem_mlp2 = {smem_mlp_W1, smem_mlp_W2, smem_mlp_W3};
				mlp_forward_inline(mlp_input, residual, h1_post, h2_post, false, smem_mlp2);

				// d_residual_mode = 0: feat = ReLU( ReLU(SH+sh_bias) + residual + res_bias )
				// d_residual_mode = 1: feat = ReLU(SH+sh_bias) + ReLU(residual + res_bias)
				// d_residual_mode = 2: feat = ReLU(SH+sh_bias) + (residual + res_bias)    [mixed; per-pixel ReLU in Python]
				// SH's own inner ReLU is handled upstream in preprocessCUDA (clamped[]).
				// `--method res_3d` per-Gauss bias gate: tex carriers force
				// sh_color = 0 (mirror of forward). Affects the gate_sh/gate_res
				// expressions below where they read `sh_color_bw[c] + residual + bias`.
				float sh_color_bw[3];
				for (int c = 0; c < 3; c++)
					sh_color_bw[c] = (d_textured_bias_gate && tex_bw) ? 0.0f : RGB_TO_FLOAT(colors[global_id * 3 + c]);

				float dL_drgb[3];     // grad flowing into residual (MLP) path
				float dL_drgb_sh[3];  // grad flowing into SH path
				for (int c = 0; c < 3; c++) {
					float gate_res, gate_sh;
					if (d_residual_mode == 1) {
						gate_res = (residual[c] + d_res_bias > 0.0f) ? 1.0f : 0.0f;
						gate_sh = 1.0f;
					} else if (d_residual_mode == 2) {
						// mixed: signed residual, no per-Gauss ReLU; both gates open.
						gate_res = 1.0f;
						gate_sh = 1.0f;
					} else {
						// `--ste` sign-aware: STE pass only when loss wants this
						// channel HIGHER at the clamped pixel (dL_dchannels < 0
						// → release-clamp direction).
						float g = ((sh_color_bw[c] + residual[c] + d_res_bias > 0.0f) ||
						           (d_ste_relu && dL_dchannels[c] < 0.0f))
						          ? 1.0f : d_lru_slope;  // `--lru` α (0 = std ReLU)
						gate_res = g;
						gate_sh = g;
					}
					dL_drgb[c] = dL_dchannels[c] * w * gate_res;
					dL_drgb_sh[c] = dL_dchannels[c] * w * gate_sh;
				}

				// SH gradient: gated separately under add mode.
				for (int ch = 0; ch < 3; ch++)
					atomicAdd(&(dL_dcolors[global_id * 3 + ch]), dL_drgb_sh[ch]);

				// 6. MLP backward
				float dL_dinput_full[TC_INPUT_DIM];
				if (dL_dmlp_W1 != nullptr) {
					// Full backward: weight grads + input grads
					mlp_backward<TC_INPUT_DIM, TC_HIDDEN_DIM, ORIG_OUTPUT_DIM>(
						mlp_input, residual, dL_drgb,
						h1_post, h2_post,
						dL_dinput_full,
						tile_dL_dW1,
						tile_dL_dW2,
						tile_dL_dW3,
						smem_mlp2,
						false  // identity activation, no sigmoid
					);
				} else {
					// freeze_mlp: input grads only (for hash backward), skip weight grads
					mlp_backward_input_only<TC_INPUT_DIM, TC_HIDDEN_DIM, ORIG_OUTPUT_DIM>(
						mlp_input, residual, dL_drgb,
						h1_post, h2_post,
						dL_dinput_full,
						smem_mlp2,
						false
					);
				}

				// 7. Backprop to hash features (first hash_dim elements of dL_dinput)
				const int hash_dim = active_hashgrid_levels * l_dim;
				if (!skip_hash && active_hashgrid_levels > 0 && l_dim == 4) {
					float dL_dhash[16];
					for (int i = 0; i < hash_dim; i++)
						dL_dhash[i] = dL_dinput_full[i];

					float hash_feat_dummy[16];
					if (hash_dim == 4) {
						query_feature<true, 4, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 8) {
						query_feature<true, 8, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 12) {
						query_feature<true, 12, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 16) {
						query_feature<true, 16, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					}
					if (detach_hash_grad) { dL_dxyz[0] = 0; dL_dxyz[1] = 0; dL_dxyz[2] = 0; }
				} else if (!skip_hash && active_hashgrid_levels > 0 && l_dim == 2) {
					// 2D per level — supports 1..8 hash levels (hash_dim ∈ {2,4,6,8,10,12,14,16}).
					float dL_dhash[16];
					for (int i = 0; i < hash_dim; i++)
						dL_dhash[i] = dL_dinput_full[i];

					float hash_feat_dummy[16];
					if (hash_dim == 2) {
						query_feature<true, 2, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 4) {
						query_feature<true, 4, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 6) {
						query_feature<true, 6, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 8) {
						query_feature<true, 8, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 10) {
						query_feature<true, 10, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 12) {
						query_feature<true, 12, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 14) {
						query_feature<true, 14, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					} else if (hash_dim == 16) {
						query_feature<true, 16, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, contract, false,
						                           dL_dhash, dL_dfeatures, dL_dxyz);
					}
					if (detach_hash_grad) { dL_dxyz[0] = 0; dL_dxyz[1] = 0; dL_dxyz[2] = 0; }
				}

				// 8. Set feat for alpha gradient (matches forward activation).
				// `--method res_3d` per-Gauss bias gate (mirror of forward).
				for (int ch = 0; ch < C; ch++) {
					float sh_c = (d_textured_bias_gate && tex_bw) ? 0.0f : RGB_TO_FLOAT(colors[global_id * 3 + ch]);
					if (d_residual_mode == 1)
						feat[ch] = sh_c + fmaxf(0.0f, residual[ch] + d_res_bias);
					else if (d_residual_mode == 2)
						feat[ch] = sh_c + residual[ch] + d_res_bias;       // mixed: signed
					else
						feat[ch] = fmaxf(0.0f, sh_c + residual[ch] + d_res_bias);
				}

				break;
			}
			case 6: {
				// 3D_SH_cat backward: same as case 5 but MLP input includes DC SH
				// colors[] = full SH eval (via color_ptr), dc_features[] = DC SH (3D per Gaussian)

				// 0. Compute xyz intersection point
				const float3 pk6 = collected_pk[j];
				float3 xyz6;
				if (rho3d <= rho2d) {
					const float3 sutu6 = collected_SuTu[j];
					const float3 svtv6 = collected_SvTv[j];
					xyz6 = {s.x * sutu6.x + s.y * svtv6.x + pk6.x,
					        s.x * sutu6.y + s.y * svtv6.y + pk6.y,
					        s.x * sutu6.z + s.y * svtv6.z + pk6.z};
				} else {
					xyz6 = pk6;
				}

				const int active_hashgrid_levels_6 = (level >> 8) & 0xFF;
				const int hash_dim_6 = active_hashgrid_levels_6 * l_dim;

				// 1. Query hash features
				bool skip_hash_6 = (d_contrib_thresh_bw > 0.0f && w < d_contrib_thresh_bw)
				                   || (d_count_thresh_bw > 0 && contributor >= (uint32_t)d_count_thresh_bw);
				float hash_feat_6[12] = {0};
				if (!skip_hash_6 && active_hashgrid_levels_6 > 0 && l_dim == 4) {
					if (hash_dim_6 == 4)
						query_feature<false, 4, 4>(hash_feat_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_6 == 8)
						query_feature<false, 8, 4>(hash_feat_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false);
					else if (hash_dim_6 == 12)
						query_feature<false, 12, 4>(hash_feat_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false);
				}

				// 2. Load DC SH from dc_features
				float dc_sh_bw[3] = {0};
				if (dc_features != nullptr) {
					for (int ch = 0; ch < 3; ch++)
						dc_sh_bw[ch] = dc_features[global_id * 3 + ch];
				}

				// 3. Build MLP input: [hash | dc_sh(3) | bias(1) | pad]
				float mlp_input_6[TC_INPUT_DIM];
				for (int i = 0; i < TC_INPUT_DIM; i++) mlp_input_6[i] = 0.0f;
				int pos6 = 0;
				for (int i = 0; i < hash_dim_6 && pos6 < TC_INPUT_DIM; i++) mlp_input_6[pos6++] = hash_feat_6[i];
				for (int i = 0; i < 3 && pos6 < TC_INPUT_DIM; i++) mlp_input_6[pos6++] = dc_sh_bw[i];
				if (pos6 < TC_INPUT_DIM) mlp_input_6[pos6] = 1.0f;

				// 4. Recompute MLP forward
				float h1_post_6[TC_HIDDEN_DIM], h2_post_6[TC_HIDDEN_DIM];
				float residual_6[ORIG_OUTPUT_DIM];
				MlpWeights smem_mlp_6 = {smem_mlp_W1, smem_mlp_W2, smem_mlp_W3};
				mlp_forward_inline(mlp_input_6, residual_6, h1_post_6, h2_post_6, false, smem_mlp_6);

				// d_residual_mode = 0: outer ReLU gates both paths together.
				// d_residual_mode = 1: separate ReLUs for SH and residual.
				// NOTE: for case 6, colors[] holds DC_SH (MLP input), not the SH-
				// evaluated sh_color. We use DC_SH as a stand-in in the gate test —
				// this matches the pre-decoupling approximation in this kernel.
				float sh_color_bw_6[3];
				for (int c = 0; c < 3; c++)
					sh_color_bw_6[c] = RGB_TO_FLOAT(colors[global_id * 3 + c]);

				float dL_drgb_6[3];      // grad flowing into residual (MLP) path
				float dL_drgb_sh_6[3];   // grad flowing into SH path
				for (int c = 0; c < 3; c++) {
					float gate_res, gate_sh;
					if (d_residual_mode == 1) {
						gate_res = (residual_6[c] + d_res_bias > 0.0f) ? 1.0f : 0.0f;
						gate_sh = 1.0f;
					} else {
						// `--ste` sign-aware: see case-5 site above.
						float g = ((sh_color_bw_6[c] + residual_6[c] + d_res_bias > 0.0f) ||
						           (d_ste_relu && dL_dchannels[c] < 0.0f))
						          ? 1.0f : d_lru_slope;  // `--lru` α (0 = std ReLU)
						gate_res = g;
						gate_sh = g;
					}
					dL_drgb_6[c] = dL_dchannels[c] * w * gate_res;
					dL_drgb_sh_6[c] = dL_dchannels[c] * w * gate_sh;
				}

				// SH gradient: gated separately under add mode.
				for (int ch = 0; ch < 3; ch++)
					atomicAdd(&(dL_dcolors[global_id * 3 + ch]), dL_drgb_sh_6[ch]);

				// 6. MLP backward
				float dL_dinput_full_6[TC_INPUT_DIM];
				if (dL_dmlp_W1 != nullptr) {
					mlp_backward<TC_INPUT_DIM, TC_HIDDEN_DIM, ORIG_OUTPUT_DIM>(
						mlp_input_6, residual_6, dL_drgb_6,
						h1_post_6, h2_post_6,
						dL_dinput_full_6,
						tile_dL_dW1, tile_dL_dW2, tile_dL_dW3,
						smem_mlp_6, false);
				} else {
					mlp_backward_input_only<TC_INPUT_DIM, TC_HIDDEN_DIM, ORIG_OUTPUT_DIM>(
						mlp_input_6, residual_6, dL_drgb_6,
						h1_post_6, h2_post_6,
						dL_dinput_full_6,
						smem_mlp_6, false);
				}

				// 6b. Route MLP's dc_sh-slot gradients into dL_dcolors so they flow back
				//     to _features_dc through the SH backward chain (dL_dshs[0,:] gets
				//     SH_C0 * dL_dcolors, which PyTorch chains through `_effective_shs()`
				//     → `_features_dc`). The Python-side `colors_precomp` expression is
				//     `.detach()`'d w.r.t. `_features_dc` to prevent double-counting via
				//     `grad_colors_precomp`. MLP input layout: [hash | dc_sh(3) | bias(1) | pad].
				for (int ch = 0; ch < 3; ch++) {
					atomicAdd(&(dL_dcolors[global_id * 3 + ch]),
					          dL_dinput_full_6[hash_dim_6 + ch]);
				}

				// 7. Backprop to hash features
				if (!skip_hash_6 && active_hashgrid_levels_6 > 0 && l_dim == 4) {
					// 4D per level — cat caps hash_dim at 12 (1..3 hash levels of 4D each).
					float dL_dhash_6[16];
					for (int i = 0; i < hash_dim_6; i++)
						dL_dhash_6[i] = dL_dinput_full_6[i];

					float hash_feat_dummy_6[16];
					if (hash_dim_6 == 4)
						query_feature<true, 4, 4>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					else if (hash_dim_6 == 8)
						query_feature<true, 8, 4>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					else if (hash_dim_6 == 12)
						query_feature<true, 12, 4>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					else if (hash_dim_6 == 16)
						query_feature<true, 16, 4>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					if (detach_hash_grad) { dL_dxyz[0] = 0; dL_dxyz[1] = 0; dL_dxyz[2] = 0; }
				} else if (!skip_hash_6 && active_hashgrid_levels_6 > 0 && l_dim == 2) {
					// 2D per level — cat caps hash_dim at 12 (1..6 hash levels of 2D each).
					float dL_dhash_6[16];
					for (int i = 0; i < hash_dim_6; i++)
						dL_dhash_6[i] = dL_dinput_full_6[i];

					float hash_feat_dummy_6[16];
					if (hash_dim_6 == 2)
						query_feature<true, 2, 2>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					else if (hash_dim_6 == 4)
						query_feature<true, 4, 2>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					else if (hash_dim_6 == 6)
						query_feature<true, 6, 2>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					else if (hash_dim_6 == 8)
						query_feature<true, 8, 2>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					else if (hash_dim_6 == 10)
						query_feature<true, 10, 2>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					else if (hash_dim_6 == 12)
						query_feature<true, 12, 2>(hash_feat_dummy_6, xyz6, voxel_min, voxel_max, collec_offsets,
						    appearance_level, hash_features, active_hashgrid_levels_6,
						    l_scale, Base, align_corners, interp, contract, false,
						    dL_dhash_6, dL_dfeatures, dL_dxyz);
					if (detach_hash_grad) { dL_dxyz[0] = 0; dL_dxyz[1] = 0; dL_dxyz[2] = 0; }
				}

				// 8. Set feat for alpha gradient (matches forward activation).
				// NOTE: for case 6, colors[] holds DC_SH (MLP input), not the SH-evaluated
				// sh_color. We use DC_SH as a stand-in — matches prior approximation.
				for (int ch = 0; ch < C; ch++) {
					float sh_c = RGB_TO_FLOAT(colors[global_id * 3 + ch]);
					if (d_residual_mode == 1)
						feat[ch] = sh_c + fmaxf(0.0f, residual_6[ch] + d_res_bias);
					else
						feat[ch] = fmaxf(0.0f, sh_c + residual_6[ch] + d_res_bias);
				}

				break;
			}
			default: printf("BW unsupported render_mode: %d\n", render_mode & 0xFF);
				break;
			}

				// Update dL_dalpha_color (per-cascade) and get grad_feat
				for (int ch = 0; ch < C; ch++)
				{
					const float c = feat[ch];
					// `--method res_3d`: per-cascade accum_rec / last_alpha / last_color
					// for the COLOR recurrence (the joint accum_rec/last_color
					// arrays are no longer needed — they only fed dL_dalpha for
					// color, which is now per-cascade).
					float* const _ar = _route_tex_std ? accum_rec_tex  : accum_rec_sv;
					float* const _lc = _route_tex_std ? last_color_tex : last_color_sv;
					const float _la  = _route_tex_std ? last_alpha_tex : last_alpha_sv;
					_ar[ch] = _la * _lc[ch] + (1.f - _la) * _ar[ch];
					_lc[ch] = c;

					dL_dalpha_color += (c - _ar[ch]) * dL_dchannels[ch];
				}

			}
			
			float dL_dz = 0.0f;
			float dL_dweight = 0;

#if RENDER_AXUTILITY
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			// `--method mixed_3d`: final_A_tex excludes EWA occlusion (T_tex), making
// the dist gradient strictly textured-only. final_A_tex == final_A for
// non-mixed_3d so this is a no-op there.
dL_dweight += (final_D2 + m_d * m_d * final_A_tex - 2 * m_d * final_D) * dL_dreg;
#endif

			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			// `--method mixed_3d`: final_A_tex for textured-only dist gradient.
const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A_tex - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;

			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			// `--method res_3d` dual cascade: scale color by per-cascade T,
			// other (depth/dist/normal/mask) contributions by joint T.
			dL_dalpha = dL_dalpha_color * T_color_std + dL_dalpha * T;

			// Regularization gradients: only affect opacity, NOT geometry
			float dL_dalpha_reg = 0.0f;

			// Overdraw regularization gradient (non-GEMM path) — gated on
			// `tex_bw` so --method mixed/mixed_3d's untextured 2DGS rows are
			// excluded. `is_textured == nullptr` ⇒ tex_bw defaults true ⇒
			// 3D_SH_res byte-identical.
			if (dL_doverdraw > 0.0f && tex_bw) {
				float sig_od = 1.0f / (1.0f + expf(-OD_K * (w - OD_THRESH)));
				float dsig_od = OD_K * sig_od * (1.0f - sig_od);
				float safe_denom = fmaxf(1.0f - alpha, 1e-7f);
				dL_dalpha_reg += dL_doverdraw * (dsig_od * T - overdraw_accum / safe_denom);
				overdraw_accum += dsig_od * w;
			}

			// Weight-squared regularization gradient (direct + indirect)
			if (dL_dwr != 0.0f) {
				float safe_denom_wr = fmaxf(1.0f - alpha, 1e-7f);
				dL_dalpha_reg += dL_dwr * (2.0f * w * T - 2.0f * wr_accum / safe_denom_wr);
				wr_accum += w * w;
			}

			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;
			// `--method res_3d`: advance per-cascade last_alpha for COLOR
			// recurrence; the joint last_alpha advance above is kept for
			// depth/normal/dist recurrences.
			if (_route_tex_std) last_alpha_tex = alpha;
			else                last_alpha_sv  = alpha;

			// Geometry gradient: only from RGB loss (no reg)
			// AA: alpha = coef * opa * G  →  dL/dG = coef * opa * dL_dalpha
			float dL_dG = (is_aa ? aa_coef : 1.0f) * nor_o.w * dL_dalpha;
			// AA: dL/dcoef = opa * G * dL_dalpha (used only in AA backward block below)
			float dL_dcoef_aa = is_aa ? (nor_o.w * G * dL_dalpha) : 0.0f;
			// Opacity gradient: RGB + reg
			dL_dalpha += dL_dalpha_reg;

			// `--method mixed`: untextured uses the run's kernel, so its shape
			// gradients flow exactly like textured.
			if (kernel_type == 1 || kernel_type == 4) {
				// Beta kernel with max-pool: gradient only flows through winning branch
				// Shape gradient only applies when beta branch won
				if (beta_wins && dL_dshapes != nullptr && base > 1e-7f) {
					// dL/dshape = dL/dalpha * dalpha/dG * dG/dshape
					//           = dL/dalpha * opacity * alpha_beta * ln(base)
					float dL_dshape = dL_dalpha * opa * alpha_beta * logf(base);
					atomicAdd(&dL_dshapes[global_id], dL_dshape);
				}
				// dL_dG stays as nor_o.w * dL_dalpha for position gradient propagation
			} else if (kernel_type == 2 && per_gaussian_beta > 0.0f) {
				// Flex kernel: G = (1+beta)*G_raw / (1+beta*G_raw)
				// dG/dG_raw = (1+beta) / demon^2
				// dG/dbeta = G_raw * (1 - G_raw) / demon^2
				const float dG_dg_raw = (1.0f + per_gaussian_beta) / (demon * demon);
				dL_dG *= dG_dg_raw;  // dL_dG now refers to G_raw for position gradient

				// Gradient w.r.t. per-Gaussian beta (stored in shapes array)
				if (dL_dshapes != nullptr) {
					// dL/dbeta = dL/dalpha * dalpha/dG * dG/dbeta
					// dG/dbeta = G_raw * (1 - G_raw) / demon^2
					float dG_dbeta = G_raw * (1.0f - G_raw) / (demon * demon);
					float dL_dbeta = dL_dalpha * opa * dG_dbeta;
					atomicAdd(&dL_dshapes[global_id], dL_dbeta);
				}
			} else if (kernel_type == 3) {
				// General kernel: G = exp(-0.5 * (r²)^(β/2))
				// Let pow_term = (r²)^(β/2), then G = exp(-0.5 * pow_term)
				// dG/d(pow_term) = -0.5 * G
				// d(pow_term)/d(r²) = (β/2) * (r²)^(β/2 - 1) = (β/2) * pow_term / r²
				// d(pow_term)/dβ = 0.5 * pow_term * ln(r²)

				// Position gradients will be computed later using dG_factor
				// dL_dG stays as nor_o.w * dL_dalpha

				// Gradient w.r.t. β (stored in shapes array)
				if (dL_dshapes != nullptr) {
					// dG/dβ = dG/d(pow_term) * d(pow_term)/dβ
					//       = -0.5 * G * 0.5 * pow_term * ln(r²)
					//       = -0.25 * G * pow_term * ln(r²)
					float log_rho = logf(general_rho_safe);
					float dG_dbeta = -0.25f * G * general_pow_term * log_rho;
					float dL_dbeta = dL_dalpha * opa * dG_dbeta;
					atomicAdd(&dL_dshapes[global_id], dL_dbeta);
				}
			} else if(beta > 0.0){
				// with beta activation (Gaussian kernel only)
				const float dG_dg = (1.0 + beta) / (demon * demon);
				dL_dG *=  dG_dg; // dL_dg now infact
			}


#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif

			// homotrans grad
			float2 dL_duv = {0.0, 0.0};
			if(homotrans != nullptr && level > 0){

				// // uv -> xyz, grad from dL_dxyzs
				const float dL_dpx = dL_dxyz[0];
				const float dL_dpy = dL_dxyz[1];
				const float dL_dpz = dL_dxyz[2];
				
				if(rho3d <= rho2d || is_aa){
					const float3 sutu = collected_SuTu[j];
					const float3 svtv = collected_SvTv[j];
					const float3 pk = collected_pk[j];

					dL_duv = {
						dL_dpx * sutu.x + dL_dpy * sutu.y + dL_dpz * sutu.z,
						dL_dpx * svtv.x + dL_dpy * svtv.y + dL_dpz * svtv.z
					};

					// atmoicAdd to dL_dhomoMat, glm::3x4
					atomicAdd(&dL_dhomoMat[global_id * 9 + 0],  dL_dpx * s.x);
					atomicAdd(&dL_dhomoMat[global_id * 9 + 1],  dL_dpy * s.x);
					atomicAdd(&dL_dhomoMat[global_id * 9 + 2],  dL_dpz * s.x);
					atomicAdd(&dL_dhomoMat[global_id * 9 + 3],  dL_dpx * s.y);
					atomicAdd(&dL_dhomoMat[global_id * 9 + 4],  dL_dpy * s.y);
					atomicAdd(&dL_dhomoMat[global_id * 9 + 5],  dL_dpz * s.y);
				}
				// for both rho3d and rho2d
				atomicAdd(&dL_dhomoMat[global_id * 9 + 6],  dL_dpx);
				atomicAdd(&dL_dhomoMat[global_id * 9 + 7],  dL_dpy);
				atomicAdd(&dL_dhomoMat[global_id * 9 + 8],  dL_dpz);
				
			}


			if (is_aa) {
				// AA-2DGS Jacobian-based mip filter backward.
				// Recompute forward-replay variables for the gradient chain.
				const float ks = d_aa_kernel_size;
				const float k_sq_aa = ks * ks;
				const float pz_inv_aa = 1.0f / p.z;
				const float pz_sq_inv_aa = pz_inv_aa * pz_inv_aa;
				const float3 dp_dx_aa = cross(Tv, Tw);
				const float3 dp_dy_aa = cross(Tw, Tu);
				const float J_a = (dp_dx_aa.x * p.z - p.x * dp_dx_aa.z) * pz_sq_inv_aa;
				const float J_b = (dp_dx_aa.y * p.z - p.y * dp_dx_aa.z) * pz_sq_inv_aa;
				const float J_c = (dp_dy_aa.x * p.z - p.x * dp_dy_aa.z) * pz_sq_inv_aa;
				const float J_d = (dp_dy_aa.y * p.z - p.y * dp_dy_aa.z) * pz_sq_inv_aa;
				const float det_J = J_a * J_d - J_b * J_c;
				const float trace_JJT = J_a*J_a + J_b*J_b + J_c*J_c + J_d*J_d;
				const float det_V = k_sq_aa * det_J * det_J + ks * trace_JJT + 1.0f;
				const float det_V_inv = 1.0f / det_V;
				const float coef = aa_coef;  // = sqrtf(det_V_inv + 1e-8f) from forward replay
				const float term1 = J_d * s.x - J_c * s.y;
				const float term2 = J_a * s.y - J_b * s.x;
				const float rho_numerator = (s.x*s.x + s.y*s.y) + ks * (term1*term1 + term2*term2);

				// 1. Through G = exp(power), power = -0.5 * rho
				float dL_drho = dL_dG * G * (-0.5f);
				// 2. Through coef = sqrt(det_V_inv)
				float dL_ddet_V_inv = 0.0f;
				if (coef > 1e-9f) dL_ddet_V_inv = dL_dcoef_aa * (0.5f / coef);
				// 3. Through rho = rho_numerator * det_V_inv
				float dL_drho_numerator = dL_drho * det_V_inv;
				dL_ddet_V_inv += dL_drho * rho_numerator;
				// 4. Through det_V_inv = 1 / det_V
				float dL_ddet_V = dL_ddet_V_inv * (-det_V_inv * det_V_inv);
				// 5. dL/ds from s² part + hashgrid feature gradient via xyz = s·SuTu + s·SvTv + pk
				float2 dL_ds = { dL_drho_numerator * 2.0f * s.x + dL_duv.x,
				                 dL_drho_numerator * 2.0f * s.y + dL_duv.y };
				// 6. dL/dterm1, dL/dterm2
				float dL_dterm1 = dL_drho_numerator * ks * 2.0f * term1;
				float dL_dterm2 = dL_drho_numerator * ks * 2.0f * term2;
				// term1 = J_d * s.x - J_c * s.y
				float dL_dJ_d = dL_dterm1 * s.x;
				float dL_dJ_c = dL_dterm1 * (-s.y);
				dL_ds.x += dL_dterm1 * J_d;
				dL_ds.y += dL_dterm1 * (-J_c);
				// term2 = J_a * s.y - J_b * s.x
				float dL_dJ_a = dL_dterm2 * s.y;
				float dL_dJ_b = dL_dterm2 * (-s.x);
				dL_ds.y += dL_dterm2 * J_a;
				dL_ds.x += dL_dterm2 * (-J_b);
				// 7. Through det_V: k²·det_J² + k·trace + 1
				float dL_ddet_J = dL_ddet_V * k_sq_aa * 2.0f * det_J;
				float dL_dtrace = dL_ddet_V * ks;
				// trace_JJT = J_a² + J_b² + J_c² + J_d²
				float trace_coef = dL_dtrace * 2.0f;
				dL_dJ_a += trace_coef * J_a;
				dL_dJ_b += trace_coef * J_b;
				dL_dJ_c += trace_coef * J_c;
				dL_dJ_d += trace_coef * J_d;
				// det_J = J_a*J_d - J_b*J_c
				dL_dJ_a += dL_ddet_J * J_d;
				dL_dJ_d += dL_ddet_J * J_a;
				dL_dJ_b += dL_ddet_J * (-J_c);
				dL_dJ_c += dL_ddet_J * (-J_b);
				// 8. Depth gradient contribution to dL/ds and dL/dTw
				dL_ds.x += dL_dz * Tw.x;
				dL_ds.y += dL_dz * Tw.y;
				float3 dL_dTu = {0.f, 0.f, 0.f};
				float3 dL_dTv = {0.f, 0.f, 0.f};
				float3 dL_dTw = {dL_dz * s.x, dL_dz * s.y, dL_dz};
				// 9. Backprop J quotient rule → dp_dx, dp_dy, p
				float3 dL_ddp_dx = {0.f, 0.f, 0.f};
				float3 dL_ddp_dy = {0.f, 0.f, 0.f};
				float3 dL_dp_aa = {0.f, 0.f, 0.f};
				// val = (vec.comp * p.z - p.comp * vec.z) * pz_squared_inv
				#define ACCUM_QG(dL_dval, val, vec_comp, vec_z, p_comp, dL_dvec_comp, dL_dvec_z, dL_dp_comp) \
				    { \
				        float term_qg = (dL_dval) * pz_sq_inv_aa; \
				        dL_dvec_comp += term_qg * p.z; \
				        dL_dvec_z    -= term_qg * p_comp; \
				        dL_dp_comp   -= term_qg * vec_z; \
				        dL_dp_aa.z   += term_qg * vec_comp - 2.0f * (dL_dval) * (val) * pz_inv_aa; \
				    }
				ACCUM_QG(dL_dJ_a, J_a, dp_dx_aa.x, dp_dx_aa.z, p.x, dL_ddp_dx.x, dL_ddp_dx.z, dL_dp_aa.x);
				ACCUM_QG(dL_dJ_b, J_b, dp_dx_aa.y, dp_dx_aa.z, p.y, dL_ddp_dx.y, dL_ddp_dx.z, dL_dp_aa.y);
				ACCUM_QG(dL_dJ_c, J_c, dp_dy_aa.x, dp_dy_aa.z, p.x, dL_ddp_dy.x, dL_ddp_dy.z, dL_dp_aa.x);
				ACCUM_QG(dL_dJ_d, J_d, dp_dy_aa.y, dp_dy_aa.z, p.y, dL_ddp_dy.y, dL_ddp_dy.z, dL_dp_aa.y);
				#undef ACCUM_QG
				// 10. Perspective division s = p/p.z
				float dL_dsx_pz = dL_ds.x * pz_inv_aa;
				float dL_dsy_pz = dL_ds.y * pz_inv_aa;
				dL_dp_aa.x += dL_dsx_pz;
				dL_dp_aa.y += dL_dsy_pz;
				dL_dp_aa.z -= (dL_dsx_pz * s.x + dL_dsy_pz * s.y);
				// 11. Through cross products
				// dp_dx = cross(Tv, Tw) → dL/dTv += cross(Tw, dL_ddp_dx); dL/dTw += cross(dL_ddp_dx, Tv)
				{
					float3 a1 = cross(Tw, dL_ddp_dx);
					float3 a2 = cross(dL_ddp_dx, Tv);
					dL_dTv.x += a1.x; dL_dTv.y += a1.y; dL_dTv.z += a1.z;
					dL_dTw.x += a2.x; dL_dTw.y += a2.y; dL_dTw.z += a2.z;
				}
				// dp_dy = cross(Tw, Tu) → dL/dTw += cross(Tu, dL_ddp_dy); dL/dTu += cross(dL_ddp_dy, Tw)
				{
					float3 b1 = cross(Tu, dL_ddp_dy);
					float3 b2 = cross(dL_ddp_dy, Tw);
					dL_dTw.x += b1.x; dL_dTw.y += b1.y; dL_dTw.z += b1.z;
					dL_dTu.x += b2.x; dL_dTu.y += b2.y; dL_dTu.z += b2.z;
				}
				// p = cross(k, l) → dL/dk = cross(l, dL_dp); dL/dl = cross(dL_dp, k)
				const float3 dL_dk_aa = cross(l, dL_dp_aa);
				const float3 dL_dl_aa = cross(dL_dp_aa, k);
				// k = pixf.x * Tw - Tu ; l = pixf.y * Tw - Tv
				dL_dTu.x -= dL_dk_aa.x; dL_dTu.y -= dL_dk_aa.y; dL_dTu.z -= dL_dk_aa.z;
				dL_dTv.x -= dL_dl_aa.x; dL_dTv.y -= dL_dl_aa.y; dL_dTv.z -= dL_dl_aa.z;
				dL_dTw.x += pixf.x * dL_dk_aa.x + pixf.y * dL_dl_aa.x;
				dL_dTw.y += pixf.x * dL_dk_aa.y + pixf.y * dL_dl_aa.y;
				dL_dTw.z += pixf.x * dL_dk_aa.z + pixf.y * dL_dl_aa.z;
				// Atomic updates
				atomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dTw.z);
				// AbsGS: abs of Tu.z/Tv.z for densification signal
				atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dTu.z));
				atomicAdd(&dL_dmean2D[global_id].w, fabsf(dL_dTv.z));
			} else if (rho3d <= rho2d) {
				float2 dL_ds;
				if (kernel_type == 5) {
					// Nexel kernel: anisotropic gamma exponents
					float gamma_x = collected_shapes[j].x;
					float gamma_y = collected_shapes[j].y;
					const float GAMMA_EPS = 1e-6f;
					float comp_x = fminf(s.x * s.x + GAMMA_EPS, powf(1000.0f, 1.0f / gamma_x));
					float comp_y = fminf(s.y * s.y + GAMMA_EPS, powf(1000.0f, 1.0f / gamma_y));
					dL_ds = {
						dL_dG * (-G) * gamma_x * powf(comp_x, gamma_x - 1.0f) * s.x + dL_dz * Tw.x,
						dL_dG * (-G) * gamma_y * powf(comp_y, gamma_y - 1.0f) * s.y + dL_dz * Tw.y
					};
					// dL_dgamma
					float dL_dgamma_x = -0.5f * dL_dG * G * logf(comp_x) * powf(comp_x, gamma_x);
					float dL_dgamma_y = -0.5f * dL_dG * G * logf(comp_y) * powf(comp_y, gamma_y);
					atomicAdd(&dL_dshapes[global_id * 2 + 0], dL_dgamma_x);
					atomicAdd(&dL_dshapes[global_id * 2 + 1], dL_dgamma_y);
				} else {
					float dG_factor;
					if (kernel_type == 1 || kernel_type == 4) {
						if (beta_wins && base > 1e-7f) {
							dG_factor = -shape_val * alpha_beta / (base * k_sq);
						} else {
							dG_factor = 0.0f;
						}
					} else if (kernel_type == 2) {
						dG_factor = -G_raw;
					} else if (kernel_type == 3) {
						dG_factor = -0.5f * general_beta * G * general_pow_term / general_rho_safe;
					} else {
						dG_factor = -G;
					}
					dL_ds = {
						dL_dG * dG_factor * s.x + dL_dz * Tw.x,
						dL_dG * dG_factor * s.y + dL_dz * Tw.y
					};
				}

				dL_ds.x += dL_duv.x;
				dL_ds.y += dL_duv.y;

				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
				// AbsGS: accumulate absolute Tu.z / Tv.z for cancellation-free densification
				atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dTu.z));
				atomicAdd(&dL_dmean2D[global_id].w, fabsf(dL_dTv.z));
			} else {
				// Update gradients w.r.t. center of Gaussian 2D mean position
				float dG_factor_2d;
				if (kernel_type == 1 || kernel_type == 4) {
					// Beta kernel with max-pool: gradient depends on which branch won
					if (!beta_wins) {
						// Gaussian low-pass won: dG/drho2d = -0.5 * alpha_lp
						dG_factor_2d = -0.5f * alpha_lp * FilterInvSquare;
					} else {
						// Beta branch won, but we're in rho2d branch
						// alpha_beta depends on rho3d, not rho2d, so gradient is 0
						dG_factor_2d = 0.0f;
					}
				} else if (kernel_type == 2) {
					// Flex kernel: same as Gaussian, use G_raw
					dG_factor_2d = -G_raw * FilterInvSquare;
				} else if (kernel_type == 3) {
					// General kernel: dG/dd.x = dG/drho2d * drho2d/dd.x
					// dG/drho = -0.25 * β * G * pow_term / rho
					// rho2d = FilterInvSquare * (d.x² + d.y²), so drho2d/dd.x = 2 * FilterInvSquare * d.x
					// dG_factor_2d should satisfy: dG_factor_2d * d.x = dG/dd.x
					// Therefore: dG_factor_2d = dG/drho * 2 * FilterInvSquare = -0.5 * β * G * pow_term / rho * FilterInvSquare
					dG_factor_2d = -0.5f * general_beta * G * general_pow_term / general_rho_safe * FilterInvSquare;
				} else {
					// Gaussian kernel
					dG_factor_2d = -G * FilterInvSquare;
				}
				const float dG_ddelx = dG_factor_2d * d.x;
				const float dG_ddely = dG_factor_2d * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				// AbsGS: abs gradient from low-pass filter path
				atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dG * dG_ddelx));
				atomicAdd(&dL_dmean2D[global_id].w, fabsf(dL_dG * dG_ddely));
				if (render_mode & 0x400) {
					// --lowpass: propagate low-pass filter + depth gradient to transMat
					const float dL_dxy_x = dL_dG * dG_ddelx;
					const float dL_dxy_y = dL_dG * dG_ddely;
					const float inv_Tw_z = 1.0f / (Tw.z + 1e-7f);
					const float2 dL_ds_lp = {dL_dz * Tw.x, dL_dz * Tw.y};
					const float3 dz_dTw_lp = {s.x, s.y, 1.0f};
					const float dsx_pz_lp = dL_ds_lp.x / p.z;
					const float dsy_pz_lp = dL_ds_lp.y / p.z;
					const float3 dL_dp_lp = {dsx_pz_lp, dsy_pz_lp, -(dsx_pz_lp * s.x + dsy_pz_lp * s.y)};
					const float3 dL_dk_lp = cross(l, dL_dp_lp);
					const float3 dL_dl_lp = cross(dL_dp_lp, k);
					const float3 dL_dTu_lp = {-dL_dk_lp.x, -dL_dk_lp.y, -dL_dk_lp.z};
					const float3 dL_dTv_lp = {-dL_dl_lp.x, -dL_dl_lp.y, -dL_dl_lp.z};
					const float3 dL_dTw_lp = {
						pixf.x * dL_dk_lp.x + pixf.y * dL_dl_lp.x + dL_dz * dz_dTw_lp.x + dL_dxy_x * inv_Tw_z,
						pixf.x * dL_dk_lp.y + pixf.y * dL_dl_lp.y + dL_dz * dz_dTw_lp.y + dL_dxy_y * inv_Tw_z,
						pixf.x * dL_dk_lp.z + pixf.y * dL_dl_lp.z + dL_dz * dz_dTw_lp.z - (dL_dxy_x * xy.x + dL_dxy_y * xy.y) * inv_Tw_z};
					atomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dTu_lp.x);
					atomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dTu_lp.y);
					atomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dTu_lp.z);
					atomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dTv_lp.x);
					atomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dTv_lp.y);
					atomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dTv_lp.z);
					atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dTw_lp.x);
					atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dTw_lp.y);
					atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dTw_lp.z);
				} else {
					atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dz);
				}
			}

			// Update gradients w.r.t. opacity of the Gaussian
			// AA: alpha = coef * opa * G → dL/dopa = coef * G * dL_dalpha
			atomicAdd(&(dL_dopacity[global_id]), (is_aa ? aa_coef : 1.0f) * G * dL_dalpha);
		}
	}

	// Flush tile-local MLP gradients to global memory (once per tile, bias-free).
	// When d_skip_mlp_grad is set, the three wmma_gemm_layer* calls above were
	// skipped so tile_dL_dW* buffers remain zero — flushing would just do 48
	// zero atomicAdds per tile, wasted bandwidth. Skip cleanly.
	if (((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) && dL_dmlp_W1 != nullptr
	        && !d_skip_mlp_grad) {
		block.sync();
		unsigned long long _prof_flush_t0 = clock64();
		MODES::flush_tile_mlp_grads(
			tile_dL_dW1, tile_dL_dW2, tile_dL_dW3,
			dL_dmlp_W1, dL_dmlp_W2, dL_dmlp_W3,
			block.thread_rank());
		block.sync();
		unsigned long long _prof_flush_t1 = clock64();
		if (block.thread_rank() == 0) {
			atomicAdd(&d_bw_profile[3], _prof_flush_t1 - _prof_flush_t0);
			atomicAdd(&d_bw_profile_counts[2], 1);
		}
	}

}



__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float4* dL_dmean2Ds,
	float* dL_dTs, 
	float* dL_dhomoMat,
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	 glm::vec4* dL_drots)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
		);

		glm::mat4 world2ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
			projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
			projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
			projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
		);

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
			glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		);

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float4 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	glm::mat3x4 dL_dhomo = glm::mat3x4(
		glm::vec4(dL_dhomoMat[idx * 9 + 0], dL_dhomoMat[idx * 9 + 1], dL_dhomoMat[idx * 9 + 2], 0.0),
		glm::vec4(dL_dhomoMat[idx * 9 + 3], dL_dhomoMat[idx * 9 + 4], dL_dhomoMat[idx * 9 + 5], 0.0),
		glm::vec4(dL_dhomoMat[idx * 9 + 6], dL_dhomoMat[idx * 9 + 7], dL_dhomoMat[idx * 9 + 8], 0.0)
	);

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);

	dL_dM = dL_dM + dL_dhomo;

	// TOGGLE NORMAL NORMALIZATION GRADIENT: Must match forward pass
	// Backprop through normalization if NORMALIZE_SURFACE_NORMALS is defined
	#define NORMALIZE_SURFACE_NORMALS
	#ifdef NORMALIZE_SURFACE_NORMALS
	float3 dL_dnormal_normalized = dL_dnormals[idx];
	float normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
	float3 dL_dnormal_unnorm;
	if(normal_len > 1e-7f) {
		float dot_grad_norm = dL_dnormal_normalized.x * normal.x + dL_dnormal_normalized.y * normal.y + dL_dnormal_normalized.z * normal.z;
		dL_dnormal_unnorm.x = (dL_dnormal_normalized.x - dot_grad_norm * normal.x / normal_len) / normal_len;
		dL_dnormal_unnorm.y = (dL_dnormal_normalized.y - dot_grad_norm * normal.y / normal_len) / normal_len;
		dL_dnormal_unnorm.z = (dL_dnormal_normalized.z - dot_grad_norm * normal.z / normal_len) / normal_len;
	} else {
		dL_dnormal_unnorm = make_float3(0.0f, 0.0f, 0.0f);
	}
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormal_unnorm, viewmatrix);
	#else
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
	#endif
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

// ============================================================================
// `--method mixed_3d` — UNTEXTURED EWA 3D-ellipsoid backward VJP.
//
// Fused VERBATIM port of FastGS backward.cu: computeCov2DCUDA + computeCov3D +
// the projection-mean part of preprocessCUDA, restricted to one Gaussian and
// recomputing cov3D internally from (scale3, rot). Mirrors the forward
// `compute_ewa_conic` exactly so the analytic grad matches the forward to
// numerical precision (gradcheck-verified). `scale3=(sx,sy,sz)` activated;
// `q=(r,x,y,z)` (no normalization, same as forward). Writes:
//   dL_dscales[idx*2+0/1] = dL/dsx, dL/dsy   (FastGS computeCov3D, '=')
//   dL_dscaling_z[idx]    = dL/dsz
//   dL_drots[idx*4+0..3]  = dL/dquat         (FastGS, '=')
//   dL_dmean3D[idx]      += cov2D-part + proj-part   (computeColorFromSH adds SH part)
// dL_dconic = (dL_da, dL_db, dL_dc) is the inverse-cov grad (a=conic.x,
// b=conic.y, c=conic.z). dL_dmean2D is the NDC-scaled screen grad (.x,.y).
// ============================================================================
__device__ void ewa_backward_vjp(
	int idx,
	const float3& p_orig,
	const glm::vec3 scale3,
	const glm::vec4 q,
	const float* viewmatrix,
	const float* projmatrix,
	const float h_x, const float h_y,
	const float tan_fovx, const float tan_fovy,
	const float dL_da, const float dL_db, const float dL_dc,
	const float2 dL_dmean2D,
	float* dL_dscales,
	float* dL_dscaling_z,
	float* dL_drots,
	glm::vec3* dL_dmean3D)
{
	// ---- recompute cov3D (FastGS computeCov3D forward) ----
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale3.x; S[1][1] = scale3.y; S[2][2] = scale3.z;
	float r = q.x, x = q.y, y = q.z, z = q.w;
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	glm::mat3 Mm = S * R;
	glm::mat3 Sigma = glm::transpose(Mm) * Mm;
	float cov3D[6] = { Sigma[0][0], Sigma[0][1], Sigma[0][2],
	                   Sigma[1][1], Sigma[1][2], Sigma[2][2] };

	// ---- computeCov2DCUDA (FastGS, verbatim) ----
	float3 dL_dconic = { dL_da, dL_db, dL_dc };
	float3 t = transformPoint4x3(p_orig, viewmatrix);
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	const float x_grad_mul = (txtz < -limx || txtz > limx) ? 0 : 1;
	const float y_grad_mul = (tytz < -limy || tytz > limy) ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	glm::mat3 T = W * J;
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da_c = 0, dL_db_c = 0, dL_dc_c = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
	float dL_dcov[6];
	if (denom2inv != 0)
	{
		dL_da_c = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc_c = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db_c = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		dL_dcov[0] = (T[0][0] * T[0][0] * dL_da_c + T[0][0] * T[1][0] * dL_db_c + T[1][0] * T[1][0] * dL_dc_c);
		dL_dcov[3] = (T[0][1] * T[0][1] * dL_da_c + T[0][1] * T[1][1] * dL_db_c + T[1][1] * T[1][1] * dL_dc_c);
		dL_dcov[5] = (T[0][2] * T[0][2] * dL_da_c + T[0][2] * T[1][2] * dL_db_c + T[1][2] * T[1][2] * dL_dc_c);
		dL_dcov[1] = 2 * T[0][0] * T[0][1] * dL_da_c + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db_c + 2 * T[1][0] * T[1][1] * dL_dc_c;
		dL_dcov[2] = 2 * T[0][0] * T[0][2] * dL_da_c + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db_c + 2 * T[1][0] * T[1][2] * dL_dc_c;
		dL_dcov[4] = 2 * T[0][2] * T[0][1] * dL_da_c + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db_c + 2 * T[1][1] * T[1][2] * dL_dc_c;
	}
	else
	{
		for (int i = 0; i < 6; i++) dL_dcov[i] = 0;
	}

	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da_c +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db_c;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da_c +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db_c;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da_c +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db_c;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_c +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db_c;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_c +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db_c;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_c +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db_c;

	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
	float3 dL_dmean_cov = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, viewmatrix);

	// ---- computeCov3D backward (FastGS, verbatim) ----
	glm::vec3 dunc(dL_dcov[0], dL_dcov[3], dL_dcov[5]);
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov[0], 0.5f * dL_dcov[1], 0.5f * dL_dcov[2],
		0.5f * dL_dcov[1], dL_dcov[3], 0.5f * dL_dcov[4],
		0.5f * dL_dcov[2], 0.5f * dL_dcov[4], dL_dcov[5]);
	glm::mat3 dL_dM = 2.0f * Mm * dL_dSigma;
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	dL_dscales[idx * 2 + 0] = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscales[idx * 2 + 1] = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscaling_z[idx]      = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= scale3.x;
	dL_dMt[1] *= scale3.y;
	dL_dMt[2] *= scale3.z;

	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
	dL_drots[idx * 4 + 0] = dL_dq.x;
	dL_drots[idx * 4 + 1] = dL_dq.y;
	dL_drots[idx * 4 + 2] = dL_dq.z;
	dL_drots[idx * 4 + 3] = dL_dq.w;

	// ---- projection-mean part (FastGS preprocessCUDA, verbatim) ----
	float4 m_hom = transformPoint4x4(p_orig, projmatrix);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);
	float mul1 = (projmatrix[0] * p_orig.x + projmatrix[4] * p_orig.y + projmatrix[8] * p_orig.z + projmatrix[12]) * m_w * m_w;
	float mul2 = (projmatrix[1] * p_orig.x + projmatrix[5] * p_orig.y + projmatrix[9] * p_orig.z + projmatrix[13]) * m_w * m_w;
	glm::vec3 dL_dmean_proj;
	dL_dmean_proj.x = (projmatrix[0] * m_w - projmatrix[3] * mul1) * dL_dmean2D.x + (projmatrix[1] * m_w - projmatrix[3] * mul2) * dL_dmean2D.y;
	dL_dmean_proj.y = (projmatrix[4] * m_w - projmatrix[7] * mul1) * dL_dmean2D.x + (projmatrix[5] * m_w - projmatrix[7] * mul2) * dL_dmean2D.y;
	dL_dmean_proj.z = (projmatrix[8] * m_w - projmatrix[11] * mul1) * dL_dmean2D.x + (projmatrix[9] * m_w - projmatrix[11] * mul2) * dL_dmean2D.y;

	dL_dmean3D[idx] += glm::vec3(dL_dmean_cov.x, dL_dmean_cov.y, dL_dmean_cov.z) + dL_dmean_proj;
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	float* dL_dhomoMat,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float4* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	const bool pixel_center = false,
	// `--method mixed_3d`: untextured EWA rows take the FastGS VJP instead of
	// the 2DGS transMat VJP. nullptr scaling_z → pure mixed/2DGS path.
	const bool* is_textured = nullptr,
	const float* scaling_z = nullptr,
	float* dL_dscaling_z = nullptr)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);

	// `--method mixed_3d`: UNTEXTURED EWA rows — geometry VJP is the FastGS
	// EWA path (conic→cov2D→cov3D→scale/scaling_z/rot + proj→mean3D), NOT the
	// 2DGS transMat VJP. dL_dconic was carried in dL_dtransMats[idx*9+0..2] and
	// the NDC screen grad in dL_dmean2Ds[idx].x/.y by the render backward.
	// Color VJP (computeColorFromSH) is identical → still run it.
	if (scaling_z != nullptr && is_textured != nullptr && !is_textured[idx]) {
		const float3 p_orig = make_float3(means3D[idx].x, means3D[idx].y, means3D[idx].z);
		const glm::vec3 scale3 = glm::vec3(scales[idx].x, scales[idx].y, scaling_z[idx]);
		const float2 dL_dm2D = make_float2(dL_dmean2Ds[idx].x, dL_dmean2Ds[idx].y);
		ewa_backward_vjp(
			idx, p_orig, scale3, rotations[idx],
			viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy,
			dL_dtransMats[idx * 9 + 0], dL_dtransMats[idx * 9 + 1], dL_dtransMats[idx * 9 + 2],
			dL_dm2D,
			(float*)dL_dscales, dL_dscaling_z, (float*)dL_drots, dL_dmean3Ds);
		if (shs)
			computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped,
			                   (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
		// dL_dmean2Ds[idx].x/.y already hold the screen-space densification
		// proxy from the render backward — leave them (and .z/.w) untouched.
		return;
	}

	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dhomoMat,
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots
	);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc

	// AbsGS: scale the abs signal accumulated during the render backward
	// (per-pixel fabs(dL_dTu.z) / fabs(dL_dG * dG_ddelx)) to match the
	// densification coordinate system (same depth * 0.5 * W/H factor as x/y).
	dL_dmean2Ds[idx].z *= depth * 0.5f * float(W);
	dL_dmean2Ds[idx].w *= depth * 0.5f * float(H);
}


// Setter kernels for backward's own threshold copies
__global__ void setContribThreshBwKernel(float val) { d_contrib_thresh_bw = val; }
__global__ void setCountThreshBwKernel(int val) { d_count_thresh_bw = val; }

void BACKWARD::setContribThresh(float val) {
	setContribThreshBwKernel<<<1, 1>>>(val);
}

void BACKWARD::setCountThresh(int val) {
	setCountThreshBwKernel<<<1, 1>>>(val);
}

__global__ void setOverdrawLambdaBwKernel(float val) { d_overdraw_lambda_bw = val; }
void BACKWARD::setOverdrawLambda(float val) {
	setOverdrawLambdaBwKernel<<<1, 1>>>(val);
}

__global__ void setWeightRegLambdaBwKernel(float val) { d_weight_reg_lambda_bw = val; }
void BACKWARD::setWeightRegLambda(float val) {
	setWeightRegLambdaBwKernel<<<1, 1>>>(val);
}

__global__ void setResBiasBwKernel(float val) { d_res_bias = val; }
void BACKWARD::setResBias(float val) {
	setResBiasBwKernel<<<1, 1>>>(val);
}

__global__ void setResidualModeBwKernel(int v) { d_residual_mode = v; }
void BACKWARD::setResidualMode(int mode) {
	setResidualModeBwKernel<<<1, 1>>>(mode);
}

// `--ste`: straight-through estimator on the per-Gauss outer ReLU (mode 0 only).
__global__ void setSteReluBwKernel(int v) { d_ste_relu = v; }
void BACKWARD::setSteRelu(int v) {
	setSteReluBwKernel<<<1, 1>>>(v);
}

// `--method res_3d_double`: backward mirror.
__global__ void setTexturedBiasGateBwKernel(int v) { d_textured_bias_gate = v; }
void BACKWARD::setTexturedBiasGate(int v) {
	setTexturedBiasGateBwKernel<<<1, 1>>>(v);
}

// `--lru`: leaky-ReLU slope α for the outer per-Gauss ReLU (mode 0 only).
__global__ void setLruSlopeBwKernel(float v) { d_lru_slope = v; }
void BACKWARD::setLruSlope(float v) {
	setLruSlopeBwKernel<<<1, 1>>>(v);
}

__global__ void setAaKernelSizeBwKernel(float val) { d_aa_kernel_size = val; }
void BACKWARD::setAaKernelSize(float val) {
	setAaKernelSizeBwKernel<<<1, 1>>>(val);
}

__global__ void setSkipMlpGradKernel(bool val) { d_skip_mlp_grad = val; }
void BACKWARD::setSkipMlpGrad(bool val) {
	setSkipMlpGradKernel<<<1, 1>>>(val);
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos,
	float4* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dhomoMat,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	const bool pixel_center,
	const bool* is_textured,
	const float* scaling_z,
	float* dL_dscaling_z)
{
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dhomoMat,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots,
		pixel_center,
		is_textured,
		scaling_z,
		dL_dscaling_z
	);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const float beta,
	int W, int H,
	uint32_t C, uint32_t level, uint32_t l_dim, float l_scale, uint32_t Base,
	bool align_corners, uint32_t interp,
	const bool if_contract,
	float focal_x, float focal_y,
	const glm::vec2* scales,
	const float* other_maps,
	const int* out_index,
	const float* bg_color,
	const float2* means2D,
	const float4* normal_opacity,
	const rgb_t* colors,  // FP16 SH baseline (geomState.rgb)
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
	const float* dL_dpixels_untex,
	const float* dL_depths,
	float* dL_dfeatures,
	float * dL_dtransMat,
	float * dL_dhomoMat,
	float4* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_gradsum,
	const glm::vec3* cam_pos,
	const uint32_t D_diffuse,
	const float* hash_features_diffuse,
	const int* level_offsets_diffuse,
	const float* gridrange_diffuse,
	float* dL_dfeatures_diffuse,
	const int render_mode,
	const float* shapes,
	const int kernel_type,
	float* dL_dshapes,
	const bool detach_hash_grad,
	float* dL_dmlp_W1,
	float* dL_dmlp_W2,
	float* dL_dmlp_W3,
	const float* dc_features,
	const bool* is_textured,
	const float4* ewa_conic,
	const float* scaling_z)
{
	// Get MLP weight pointers for passing to the kernel (for fused MLP modes, FP16)
	__half *mlp_W1_ptr = nullptr;
	__half *mlp_W2_ptr = nullptr;
	__half *mlp_W3_ptr = nullptr;
	if ((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) {
		FORWARD::getMlpWeightPointers(
			&mlp_W1_ptr,
			&mlp_W2_ptr,
			&mlp_W3_ptr);
	}

	// Determine D_DIFFUSE template parameter for kernel dispatch
	const uint32_t D_DIFFUSE_TEMPLATE = D_diffuse;

	// Dynamic shared memory for mode 5 (3D_SH_res) collaborative GEMM (FP16 buffers)
	// All layers use: 256*16*2 + 256*16*2 = 8KB + 8KB = 16KB (FP16)
	size_t smem_size = 0;
	bool use_collaborative_gemm = false;

	if (((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) && dL_dmlp_W1 != nullptr) {
		// Check for debug override to disable collaborative GEMM
		static int force_disable = -1;
		if (force_disable == -1) {
			const char* env = getenv("DISABLE_COLLABORATIVE_GEMM");
			force_disable = (env && atoi(env) != 0) ? 1 : 0;
		}

		// `--method mixed_3d`: collaborative GEMM is ENABLED (same as
		// `--method mixed` / `3D_SH_res`). The MODE-5 path has a block-uniform
		// untextured-EWA branch that handles those Gaussians and skips the
		// GEMM, so textured surfels keep the tensor-core MLP backward.
		if (!force_disable) {
			// Try to enable collaborative GEMM with ~16KB FP16 shared memory (single batch of 256)
			// All layers need 256*(16+16)*2 = 16384 bytes
			// Will fall back to atomics if GPU doesn't support it
			const size_t required_smem = COLLABORATIVE_SMEM_SIZE;
			smem_size = required_smem;
			use_collaborative_gemm = true;
		}
	}

	// Set max dynamic shared memory attribute if needed (must be done before launch)
	// If this fails, the GPU doesn't support enough shared memory - fall back to atomics
	if (use_collaborative_gemm && smem_size > 0) {
		// Get GPU's max shared memory per block
		int device;
		cudaGetDevice(&device);
		int max_smem_per_block;
		cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

		// Get kernel's static shared memory usage
		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, renderCUDAsurfelBackward<3, 0>);

		// Check if we have enough headroom
		size_t total_needed = attr.sharedSizeBytes + smem_size;

		if (total_needed > (size_t)max_smem_per_block) {
			// GPU doesn't support enough shared memory, fall back to atomics
			smem_size = 0;
			use_collaborative_gemm = false;
		} else {
			// Request the larger shared memory allocation
			cudaError_t err = cudaFuncSetAttribute(renderCUDAsurfelBackward<3, 0>,
			                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
			if (err != cudaSuccess) {
				// Fall back to atomics-based path
				smem_size = 0;
				use_collaborative_gemm = false;
				cudaGetLastError();  // Clear the error
			}
		}
	}

	// Adjust render_mode for kernel: bit 8 = use collaborative GEMM
	int adjusted_render_mode = render_mode;
	if (((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) && use_collaborative_gemm) {
		adjusted_render_mode = render_mode | 0x100;  // Set bit 8 to indicate collaborative GEMM
	}

	// FP16 lean library: only C=3 (RGB) is needed for mode 5
	if (C != 3) {
		printf("diff_surfel_3D_16: Unsupported channel count %d (only C=3 supported)\n", C);
		return;
	}
	renderCUDAsurfelBackward<3, 0> <<<grid, block, smem_size>>>(
			ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
			means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
			dL_dpixels, dL_dpixels_untex, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
			hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, adjusted_render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad,
			dL_dmlp_W1, dL_dmlp_W2, dL_dmlp_W3,
			mlp_W1_ptr, mlp_W2_ptr, mlp_W3_ptr,
			dc_features,
			is_textured,
			ewa_conic,
			scaling_z);

}

// Compute opacity gradients for 3D mode with full transmittance chain
// This kernel processes intersections per pixel back-to-front to correctly
// compute dL/dalpha including the transmittance chain effect
// Outputs dL_dalpha per intersection for use by geometry gradient kernel
__global__ void compute_opacity_gradient_3D_kernel(
    int M,                                      // Total intersections
    int N,                                      // Total Gaussians
    int num_pixels,                             // H * W
    const float* __restrict__ dL_dweight,       // [M] from PyTorch
    const float* __restrict__ T_values,         // [M] transmittance
    const float* __restrict__ G_values,         // [M] kernel value
    const float* __restrict__ alpha_values,     // [M] alpha
    const int* __restrict__ gaussian_ids,       // [M] Gaussian indices
    const int* __restrict__ pixel_starts,       // [num_pixels+1] boundaries
    float* __restrict__ dL_dopacity,            // [N] opacity gradient output
    float* __restrict__ dL_dalpha_out)          // [M] per-intersection dL_dalpha output
{
    // One thread per pixel
    int pix_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix_id >= num_pixels) return;

    int start = pixel_starts[pix_id];
    int end = pixel_starts[pix_id + 1];

    float last_dL_dT = 0.0f;

    // Process back-to-front (matches backward.cu logic)
    for (int i = end - 1; i >= start; i--) {
        float dL_dw = dL_dweight[i];
        float alpha = alpha_values[i];
        float T = T_values[i];
        float G = G_values[i];
        int gid = gaussian_ids[i];

        // Full transmittance chain gradient (same as backward.cu)
        // dL_dalpha = (dL_dweight - last_dL_dT) * T
        float dL_dalpha = (dL_dw - last_dL_dT) * T;

        // Propagate transmittance chain
        // last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT
        last_dL_dT = dL_dw * alpha + (1.0f - alpha) * last_dL_dT;

        // dL_dopacity = G * dL_dalpha (opacity gradient)
        atomicAdd(&dL_dopacity[gid], G * dL_dalpha);

        // Output dL_dalpha for use by geometry gradient kernel
        dL_dalpha_out[i] = dL_dalpha;
    }
}

// Compute geometry gradients for 3D mode using geomBuffer's transMat
// This kernel takes dL_dalpha per intersection and computes dL_dtransMat
// which is then used by preprocess backward to get exact scale/rotation/position gradients
__global__ void compute_geometry_gradient_3D_kernel(
    int M,                                      // Total intersections
    int N,                                      // Total Gaussians
    int W, int H,                               // Image dimensions
    const float* __restrict__ dL_dalpha,        // [M] from opacity kernel
    const float* __restrict__ opacity_values,   // [M] per-intersection opacity
    const float* __restrict__ G_values,         // [M] kernel value
    const float* __restrict__ s_x_values,       // [M] intersection s.x
    const float* __restrict__ s_y_values,       // [M] intersection s.y
    const float* __restrict__ rho_flag,         // [M] 1.0=disk, 0.0=center
    const int* __restrict__ gaussian_ids,       // [M] Gaussian indices
    const int* __restrict__ pixel_ids,          // [M] pixel indices
    const float* __restrict__ transMat,         // [N*9] from geomBuffer
    float* __restrict__ dL_dtransMat)           // [N*9] output
{
    // One thread per intersection
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    // Skip center intersections (rho2d < rho3d) - they don't contribute to transMat grads
    // Only disk intersections (rho3d <= rho2d) have gradients through transMat
    if (rho_flag[idx] < 0.5f) return;

    int gid = gaussian_ids[idx];
    int pix_id = pixel_ids[idx];

    // Pixel coordinates (center of pixel)
    float pixf_x = (float)(pix_id % W) + 0.5f;
    float pixf_y = (float)(pix_id / W) + 0.5f;

    // Get transMat for this Gaussian (Tu, Tv, Tw as 3 rows of 3 elements each)
    float Tu_x = transMat[gid * 9 + 0], Tu_y = transMat[gid * 9 + 1], Tu_z = transMat[gid * 9 + 2];
    float Tv_x = transMat[gid * 9 + 3], Tv_y = transMat[gid * 9 + 4], Tv_z = transMat[gid * 9 + 5];
    float Tw_x = transMat[gid * 9 + 6], Tw_y = transMat[gid * 9 + 7], Tw_z = transMat[gid * 9 + 8];

    // Compute k, l vectors (same as forward pass)
    // k = -Tu + pixf.x * Tw
    // l = -Tv + pixf.y * Tw
    float3 k = {-Tu_x + pixf_x * Tw_x, -Tu_y + pixf_x * Tw_y, -Tu_z + pixf_x * Tw_z};
    float3 l = {-Tv_x + pixf_y * Tw_x, -Tv_y + pixf_y * Tw_y, -Tv_z + pixf_y * Tw_z};

    // p = cross(k, l)
    float3 p = {k.y * l.z - k.z * l.y, k.z * l.x - k.x * l.z, k.x * l.y - k.y * l.x};

    // Avoid division by zero
    if (fabsf(p.z) < 1e-7f) return;

    // Get intersection values
    float s_x = s_x_values[idx];
    float s_y = s_y_values[idx];
    float G = G_values[idx];
    float opa = opacity_values[idx];
    float dL_da = dL_dalpha[idx];

    // dL_dG = opacity * dL_dalpha
    float dL_dG = opa * dL_da;

    // For Gaussian kernel: dG_factor = -G (dG/drho = -0.5*G, drho/ds = 2*s, so dG/ds = -G*s)
    float dG_factor = -G;

    // Compute dL_ds
    float dL_ds_x = dL_dG * dG_factor * s_x;
    float dL_ds_y = dL_dG * dG_factor * s_y;

    // dL_dp from dL_ds (matches backward.cu lines 1244-1246)
    float dsx_pz = dL_ds_x / p.z;
    float dsy_pz = dL_ds_y / p.z;
    float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s_x + dsy_pz * s_y)};

    // dL_dk = cross(l, dL_dp), dL_dl = cross(dL_dp, k) (matches backward.cu lines 1247-1248)
    float3 dL_dk = {l.y * dL_dp.z - l.z * dL_dp.y, l.z * dL_dp.x - l.x * dL_dp.z, l.x * dL_dp.y - l.y * dL_dp.x};
    float3 dL_dl = {dL_dp.y * k.z - dL_dp.z * k.y, dL_dp.z * k.x - dL_dp.x * k.z, dL_dp.x * k.y - dL_dp.y * k.x};

    // dL_dTu, dL_dTv, dL_dTw (matches backward.cu lines 1250-1255)
    float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
    float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
    float3 dL_dTw = {
        pixf_x * dL_dk.x + pixf_y * dL_dl.x,
        pixf_x * dL_dk.y + pixf_y * dL_dl.y,
        pixf_x * dL_dk.z + pixf_y * dL_dl.z
    };

    // Accumulate into dL_dtransMat (matches backward.cu lines 1259-1267)
    atomicAdd(&dL_dtransMat[gid * 9 + 0], dL_dTu.x);
    atomicAdd(&dL_dtransMat[gid * 9 + 1], dL_dTu.y);
    atomicAdd(&dL_dtransMat[gid * 9 + 2], dL_dTu.z);
    atomicAdd(&dL_dtransMat[gid * 9 + 3], dL_dTv.x);
    atomicAdd(&dL_dtransMat[gid * 9 + 4], dL_dTv.y);
    atomicAdd(&dL_dtransMat[gid * 9 + 5], dL_dTv.z);
    atomicAdd(&dL_dtransMat[gid * 9 + 6], dL_dTw.x);
    atomicAdd(&dL_dtransMat[gid * 9 + 7], dL_dTw.y);
    atomicAdd(&dL_dtransMat[gid * 9 + 8], dL_dTw.z);
}

// Unified backward kernel for 3D mode that reads transMat from geomBuffer
// Computes both dL_dopacity and dL_dtransMat in one pass
// This avoids needing to expose transMat to Python
// NEW: Also accepts dL_duv from hash/xyz gradient path (like cat mode)
__global__ void backward_from_weight_grad_kernel(
    int num_pixels,                             // H * W (for pixel iteration)
    int N,                                      // Total Gaussians
    int W, int H,                               // Image dimensions
    const float* __restrict__ dL_dweight,       // [M] from PyTorch
    const int* __restrict__ gaussian_ids,       // [M] Gaussian indices
    const int* __restrict__ pixel_ids,          // [M] pixel indices
    const int* __restrict__ pixel_starts,       // [num_pixels+1] boundaries
    const float* __restrict__ T_values,         // [M] transmittance
    const float* __restrict__ G_values,         // [M] kernel value
    const float* __restrict__ alpha_values,     // [M] alpha
    const float* __restrict__ opacity_values,   // [M] per-intersection opacity
    const float* __restrict__ s_x_values,       // [M] intersection s.x
    const float* __restrict__ s_y_values,       // [M] intersection s.y
    const float* __restrict__ rho_flag,         // [M] 1.0=disk, 0.0=center
    const float* __restrict__ dL_duv_x,         // [M] xyz gradient contribution to s.x (from hash backward)
    const float* __restrict__ dL_duv_y,         // [M] xyz gradient contribution to s.y (from hash backward)
    const float* __restrict__ transMat,         // [N*9] from geomBuffer (accessed directly)
    const float* __restrict__ mean2D_precomp,   // [N*2] pre-computed mean2D (x,y) from forward pass
    float* __restrict__ dL_dopacity,            // [N] opacity gradient output
    float* __restrict__ dL_dtransMat,           // [N*9] transMat gradient output
    float* __restrict__ dL_dmean2D)             // [N*2] mean2D gradient output (for densification/position)
{
    // One thread per pixel
    int pix_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix_id >= num_pixels) return;

    int start = pixel_starts[pix_id];
    int end = pixel_starts[pix_id + 1];

    if (start >= end) return;  // No intersections for this pixel

    // Get actual pixel ID from the intersection buffer (all intersections in range share same pixel)
    // IMPORTANT: pix_id is just an index into unique pixels, NOT the actual pixel coordinate!
    int actual_pixel_id = pixel_ids[start];
    // NOTE: Native CUDA kernel uses integer pixel coordinates WITHOUT +0.5 offset!
    float pixf_x = (float)(actual_pixel_id % W);
    float pixf_y = (float)(actual_pixel_id / W);

    float last_dL_dT = 0.0f;

    // Process back-to-front (matches backward.cu logic for transmittance chain)
    for (int i = end - 1; i >= start; i--) {
        float dL_dw = dL_dweight[i];
        float alpha = alpha_values[i];
        float T = T_values[i];
        float G = G_values[i];
        float opa = opacity_values[i];
        float s_x = s_x_values[i];
        float s_y = s_y_values[i];
        float is_disk = rho_flag[i];  // 1.0 = disk intersection, 0.0 = center
        int gid = gaussian_ids[i];

        // Full transmittance chain gradient (same as backward.cu)
        float dL_dalpha = (dL_dw - last_dL_dT) * T;
        last_dL_dT = dL_dw * alpha + (1.0f - alpha) * last_dL_dT;

        // dL_dopacity = G * dL_dalpha
        atomicAdd(&dL_dopacity[gid], G * dL_dalpha);

        // Get transMat for this Gaussian (Tu, Tv, Tw as 3 rows)
        float Tu_x = transMat[gid * 9 + 0], Tu_y = transMat[gid * 9 + 1], Tu_z = transMat[gid * 9 + 2];
        float Tv_x = transMat[gid * 9 + 3], Tv_y = transMat[gid * 9 + 4], Tv_z = transMat[gid * 9 + 5];
        float Tw_x = transMat[gid * 9 + 6], Tw_y = transMat[gid * 9 + 7], Tw_z = transMat[gid * 9 + 8];

        // Handle rho2d case (center intersection) - gradient through mean2D directly
        if (is_disk < 0.5f) {
            // Use pre-computed mean2D from forward pass (matches native backward which reads from points_xy_image)
            // This avoids any mismatch due to different cutoff values or numerical precision
            float mean2D_x, mean2D_y;
            if (mean2D_precomp != nullptr) {
                mean2D_x = mean2D_precomp[gid * 2 + 0];
                mean2D_y = mean2D_precomp[gid * 2 + 1];
            } else {
                // Fallback: compute mean2D using t_vec formula (MUST match forward compute_aabb!)
                // Forward: t = (cutoff², cutoff², -1), d = dot(t, T[2]*T[2]), f = t/d
                // mean2D = (dot(f, T[0]*T[2]), dot(f, T[1]*T[2]))
                float3 Tu = {Tu_x, Tu_y, Tu_z};
                float3 Tv = {Tv_x, Tv_y, Tv_z};
                float3 Tw = {Tw_x, Tw_y, Tw_z};

                // t_vec = (9, 9, -1) corresponds to cutoff=3 (used with ADAPTIVE_CUTOFF)
                // For default cutoff=4, use (16, 16, -1)
                float3 t_vec = {9.0f, 9.0f, -1.0f};
                float3 Tw_sq = {Tw.x * Tw.x, Tw.y * Tw.y, Tw.z * Tw.z};
                float d_denom = t_vec.x * Tw_sq.x + t_vec.y * Tw_sq.y + t_vec.z * Tw_sq.z;

                // Avoid division by zero
                if (fabsf(d_denom) < 1e-7f) continue;

                float3 f = {t_vec.x / d_denom, t_vec.y / d_denom, t_vec.z / d_denom};

                // Tu_dot_Tw = Tu * Tw elementwise (for dot(f, Tu*Tw))
                float Tu_dot_Tw = f.x * (Tu.x * Tw.x) + f.y * (Tu.y * Tw.y) + f.z * (Tu.z * Tw.z);
                float Tv_dot_Tw = f.x * (Tv.x * Tw.x) + f.y * (Tv.y * Tw.y) + f.z * (Tv.z * Tw.z);

                mean2D_x = Tu_dot_Tw;
                mean2D_y = Tv_dot_Tw;
            }

            // d = mean2D - pixel (matches native kernel convention)
            float d_x = mean2D_x - pixf_x;
            float d_y = mean2D_y - pixf_y;

            // dL_dG = opacity * dL_dalpha
            float dL_dG = opa * dL_dalpha;

            // FilterInvSquare = 2.0 for anti-aliasing (matches forward.cu)
            const float FilterInvSquare = 2.0f;

            // For Gaussian kernel: dG/drho2d = -G, rho2d = FilterInvSquare * (d.x² + d.y²)
            // dG/dd.x = dG/drho2d * drho2d/dd.x = -G * 2*FilterInvSquare*d.x
            // dG_factor_2d * d.x = dG/dd.x, so dG_factor_2d = -G * FilterInvSquare * 2 / 2 = -G * FilterInvSquare
            float dG_factor_2d = -G * FilterInvSquare;

            float dG_ddelx = dG_factor_2d * d_x;
            float dG_ddely = dG_factor_2d * d_y;

            atomicAdd(&dL_dmean2D[gid * 2 + 0], dL_dG * dG_ddelx);
            atomicAdd(&dL_dmean2D[gid * 2 + 1], dL_dG * dG_ddely);

            continue;
        }

        // Compute k, l vectors (same as forward pass)
        float3 k = {-Tu_x + pixf_x * Tw_x, -Tu_y + pixf_x * Tw_y, -Tu_z + pixf_x * Tw_z};
        float3 l = {-Tv_x + pixf_y * Tw_x, -Tv_y + pixf_y * Tw_y, -Tv_z + pixf_y * Tw_z};

        // p = cross(k, l)
        float3 p = {k.y * l.z - k.z * l.y, k.z * l.x - k.x * l.z, k.x * l.y - k.y * l.x};

        // Avoid division by zero
        if (fabsf(p.z) < 1e-7f) continue;

        // dL_dG = opacity * dL_dalpha
        float dL_dG = opa * dL_dalpha;

        // For Gaussian kernel: dG/ds = -G * s
        float dG_factor = -G;

        // Compute dL_ds from kernel shape gradient
        float dL_ds_x = dL_dG * dG_factor * s_x;
        float dL_ds_y = dL_dG * dG_factor * s_y;

        // ADD dL_duv contribution from hash/xyz gradient path (matches cat mode backward.cu lines 1240-1241)
        // This is the crucial term that connects hash feature gradients to geometry
        if (dL_duv_x != nullptr && dL_duv_y != nullptr) {
            dL_ds_x += dL_duv_x[i];
            dL_ds_y += dL_duv_y[i];
        }

        // dL_dp from dL_ds (matches backward.cu)
        float dsx_pz = dL_ds_x / p.z;
        float dsy_pz = dL_ds_y / p.z;
        float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s_x + dsy_pz * s_y)};

        // dL_dk = cross(l, dL_dp), dL_dl = cross(dL_dp, k)
        float3 dL_dk = {l.y * dL_dp.z - l.z * dL_dp.y, l.z * dL_dp.x - l.x * dL_dp.z, l.x * dL_dp.y - l.y * dL_dp.x};
        float3 dL_dl = {dL_dp.y * k.z - dL_dp.z * k.y, dL_dp.z * k.x - dL_dp.x * k.z, dL_dp.x * k.y - dL_dp.y * k.x};

        // dL_dTu, dL_dTv, dL_dTw (matches backward.cu)
        float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
        float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
        float3 dL_dTw = {
            pixf_x * dL_dk.x + pixf_y * dL_dl.x,
            pixf_x * dL_dk.y + pixf_y * dL_dl.y,
            pixf_x * dL_dk.z + pixf_y * dL_dl.z
        };

        // Accumulate into dL_dtransMat
        atomicAdd(&dL_dtransMat[gid * 9 + 0], dL_dTu.x);
        atomicAdd(&dL_dtransMat[gid * 9 + 1], dL_dTu.y);
        atomicAdd(&dL_dtransMat[gid * 9 + 2], dL_dTu.z);
        atomicAdd(&dL_dtransMat[gid * 9 + 3], dL_dTv.x);
        atomicAdd(&dL_dtransMat[gid * 9 + 4], dL_dTv.y);
        atomicAdd(&dL_dtransMat[gid * 9 + 5], dL_dTv.z);
        atomicAdd(&dL_dtransMat[gid * 9 + 6], dL_dTw.x);
        atomicAdd(&dL_dtransMat[gid * 9 + 7], dL_dTw.y);
        atomicAdd(&dL_dtransMat[gid * 9 + 8], dL_dTw.z);

        // NOTE: For disk path (rho3d <= rho2d), dL_dmean2D is NOT computed in render backward!
        // It will be derived from dL_dT in transMat_to_scale_rot_grad_kernel (matching native backward).
        // Only the 2D fallback path (above) computes dL_dmean2D directly from Gaussian kernel gradient.
    }
}

// Host wrapper to launch the unified backward kernel
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
    float* dL_dmean2D)
{
    const int block_size = 256;
    const int grid_size = (num_pixels + block_size - 1) / block_size;

    backward_from_weight_grad_kernel<<<grid_size, block_size>>>(
        num_pixels, N, W, H,
        dL_dweight, gaussian_ids, pixel_ids, pixel_starts,
        T_values, G_values, alpha_values, opacity_values,
        s_x_values, s_y_values, rho_flag,
        dL_duv_x, dL_duv_y,
        transMat,
        mean2D_precomp,
        dL_dopacity, dL_dtransMat, dL_dmean2D
    );
}

// =============================================================================
// New kernel: Convert dL_dtransMat to dL_dscale and dL_drotation
// This performs the proper coordinate space conversion that the native backward does:
//   dL_dM = P * transpose(dL_dT)
//   dL_dscale = [dot(dL_dM[0], R[0]), dot(dL_dM[1], R[1])]
//   dL_dR = [dL_dM[0] * scale.x, dL_dM[1] * scale.y, 0]
//   dL_drot = quat_to_rotmat_vjp(rot, dL_dR)
// =============================================================================

__global__ void transMat_to_scale_rot_grad_kernel(
    int N,
    int W, int H,  // Image dimensions for ndc2pix transformation
    const float* __restrict__ dL_dtransMat,   // [N, 9] - screen-space transMat gradient
    const float* __restrict__ dL_dhomoMat,    // [N, 9] - xyz gradient contribution (can be nullptr)
                                               // Layout: [col0.xyz, col1.xyz, col2.xyz] where each is sum(dL_dxyz * s)
    const float* __restrict__ dL_dmean2D,     // [N, 2] - 2D mean gradient (can be nullptr)
    const float* __restrict__ dL_dnormal3D,   // [N, 3] - normal gradient from depth/normal loss (can be nullptr)
    const float* __restrict__ means3D,         // [N, 3] - world-space positions (needed for dL_dmean2D)
    const float* __restrict__ transMat_precomp, // [N, 9] - forward pass transMat (can be nullptr, will reconstruct if needed)
    const float* __restrict__ scales,          // [N, 2]
    const float* __restrict__ rotations,       // [N, 4] - quaternions (w,x,y,z stored as x,y,z,w in glm)
    const float* __restrict__ projmatrix,      // [16] - 4x4 projection matrix (column-major)
    const float* __restrict__ viewmatrix,      // [16] - 4x4 view matrix (for normal gradient transform)
    float* __restrict__ dL_dscales,            // [N, 2] output
    float* __restrict__ dL_drots,              // [N, 4] output
    float* __restrict__ dL_dmeans,             // [N, 3] output
    const bool pixel_center = false)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Read scale and rotation for this Gaussian
    glm::vec2 scale = glm::vec2(scales[idx * 2 + 0], scales[idx * 2 + 1]);
    glm::vec4 rot = glm::vec4(
        rotations[idx * 4 + 0],  // w component (stored in x)
        rotations[idx * 4 + 1],  // x component (stored in y)
        rotations[idx * 4 + 2],  // y component (stored in z)
        rotations[idx * 4 + 3]   // z component (stored in w)
    );

    // Compute rotation matrix R from quaternion
    glm::mat3 R = quat_to_rotmat(rot);

    // Build world2ndc matrix from projmatrix (first 3 rows of the 4x4 projection)
    // projmatrix is column-major: [col0, col1, col2, col3]
    glm::mat4 world2ndc = glm::mat4(
        projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
        projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
        projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
        projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
    );

    // Build ndc2pix transformation (matches forward.cu and native backward)
    const float ndc_off_x = pixel_center ? float(W) / 2.0f : float(W-1) / 2.0f;
    const float ndc_off_y = pixel_center ? float(H) / 2.0f : float(H-1) / 2.0f;
    glm::mat3x4 ndc2pix = glm::mat3x4(
        glm::vec4(float(W) / 2.0f, 0.0f, 0.0f, ndc_off_x),
        glm::vec4(0.0f, float(H) / 2.0f, 0.0f, ndc_off_y),
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );

    // Compute P = world2ndc * ndc2pix (this matches the native backward exactly)
    glm::mat3x4 P = world2ndc * ndc2pix;

    // Read dL_dT (3x3 in row-major layout: Tu, Tv, Tw as rows)
    // Stored as [Tu.x, Tu.y, Tu.z, Tv.x, Tv.y, Tv.z, Tw.x, Tw.y, Tw.z]
    glm::mat3 dL_dT = glm::mat3(
        dL_dtransMat[idx * 9 + 0], dL_dtransMat[idx * 9 + 1], dL_dtransMat[idx * 9 + 2],
        dL_dtransMat[idx * 9 + 3], dL_dtransMat[idx * 9 + 4], dL_dtransMat[idx * 9 + 5],
        dL_dtransMat[idx * 9 + 6], dL_dtransMat[idx * 9 + 7], dL_dtransMat[idx * 9 + 8]
    );

    // Handle dL_dmean2D contribution using t_vec formula (matches compute_transmat_aabb)
    // NOTE: The mean2D used in render is from compute_aabb which uses the same t_vec formula.
    // Both compute_aabb (forward) and compute_transmat_aabb (backward) use:
    //   t = (cutoff², cutoff², -1) with cutoff=3 → t_vec = (9, 9, -1)
    //   mean2D = dot(f, T[0]*T[2]), dot(f, T[1]*T[2]) where f = t / dot(t, T[2]*T[2])
    if (dL_dmean2D != nullptr && transMat_precomp != nullptr) {
        float dL_dm2D_x = dL_dmean2D[idx * 2 + 0];
        float dL_dm2D_y = dL_dmean2D[idx * 2 + 1];

        if (dL_dm2D_x != 0.0f || dL_dm2D_y != 0.0f) {
            // Get transMat T from forward pass (columns as T[0], T[1], T[2])
            glm::mat3 T = glm::mat3(
                transMat_precomp[idx * 9 + 0], transMat_precomp[idx * 9 + 1], transMat_precomp[idx * 9 + 2],
                transMat_precomp[idx * 9 + 3], transMat_precomp[idx * 9 + 4], transMat_precomp[idx * 9 + 5],
                transMat_precomp[idx * 9 + 6], transMat_precomp[idx * 9 + 7], transMat_precomp[idx * 9 + 8]
            );

            // Same t_vec formula as native backward (matches forward compute_aabb with cutoff=3)
            glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
            float d = glm::dot(t_vec, T[2] * T[2]);

            // Avoid division by zero
            if (fabsf(d) < 1e-7f) return;

            glm::vec3 f_vec = t_vec * (1.0f / d);
            glm::vec3 dL_dT0 = dL_dm2D_x * f_vec * T[2];
            glm::vec3 dL_dT1 = dL_dm2D_y * f_vec * T[2];
            glm::vec3 dL_dT3 = dL_dm2D_x * f_vec * T[0] + dL_dm2D_y * f_vec * T[1];
            glm::vec3 dL_df = dL_dm2D_x * T[0] * T[2] + dL_dm2D_y * T[1] * T[2];
            float dL_dd = glm::dot(dL_df, f_vec) * (-1.0f / d);
            glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
            dL_dT3 += dL_dd * dd_dT3;

            // Add to dL_dT
            dL_dT[0] += dL_dT0;
            dL_dT[1] += dL_dT1;
            dL_dT[2] += dL_dT3;
        }
    }

    // Convert screen-space gradient to world-space: dL_dM = P * transpose(dL_dT)
    // dL_dM is 3x4 (columns are the gradients for splat2world matrix columns)
    glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);

    // Add xyz gradient contribution (dL_dhomoMat) if provided
    // This is pre-computed in Python as sum over intersections of (dL_dxyz * s)
    // Layout: [col0.xyz, col1.xyz, col2.xyz]
    // - col0: sum(dL_dxyz * s_x) for scale_x direction
    // - col1: sum(dL_dxyz * s_y) for scale_y direction
    // - col2: sum(dL_dxyz) for mean position
    if (dL_dhomoMat != nullptr) {
        // Column 0 contribution (indices 0-2)
        dL_dM[0].x += dL_dhomoMat[idx * 9 + 0];
        dL_dM[0].y += dL_dhomoMat[idx * 9 + 1];
        dL_dM[0].z += dL_dhomoMat[idx * 9 + 2];
        // Column 1 contribution (indices 3-5)
        dL_dM[1].x += dL_dhomoMat[idx * 9 + 3];
        dL_dM[1].y += dL_dhomoMat[idx * 9 + 4];
        dL_dM[1].z += dL_dhomoMat[idx * 9 + 5];
        // Column 2 contribution (indices 6-8) - for mean gradient
        dL_dM[2].x += dL_dhomoMat[idx * 9 + 6];
        dL_dM[2].y += dL_dhomoMat[idx * 9 + 7];
        dL_dM[2].z += dL_dhomoMat[idx * 9 + 8];
    }

    // Compute normal gradient contribution (matches compute_transmat_aabb in native backward)
    // Normal in view space is computed in forward as: normal = transformVec4x3(R[:,2], viewmatrix)
    // where R[:,2] is the z-column of the rotation matrix (the normal direction in world space).
    // The backward transforms dL_dnormal3D through viewmatrix transpose back to world space.
    glm::vec3 dL_dtn_vec(0.0f);
    if (dL_dnormal3D != nullptr && viewmatrix != nullptr) {
        float3 dL_dn = {dL_dnormal3D[idx * 3 + 0], dL_dnormal3D[idx * 3 + 1], dL_dnormal3D[idx * 3 + 2]};

        // Check if there's any normal gradient
        if (dL_dn.x != 0.0f || dL_dn.y != 0.0f || dL_dn.z != 0.0f) {
            // Compute the normal in view space (same as forward: transformVec4x3(R[:,2], viewmatrix))
            // For 2D surfels with scale_z = 1, normal = R[:,2]
            glm::vec3 normal_world = R[2];  // Third column of rotation matrix
            float3 normal_view = transformVec4x3({normal_world.x, normal_world.y, normal_world.z}, viewmatrix);

            // Handle normalization gradient (NORMALIZE_SURFACE_NORMALS is defined in forward)
            // Forward normalizes the view-space normal before outputting
            float normal_len = sqrtf(normal_view.x * normal_view.x + normal_view.y * normal_view.y + normal_view.z * normal_view.z);
            float3 dL_dnormal_unnorm;
            if (normal_len > 1e-7f) {
                float inv_len = 1.0f / normal_len;
                float3 normal_normalized = {normal_view.x * inv_len, normal_view.y * inv_len, normal_view.z * inv_len};
                float dot_grad_norm = dL_dn.x * normal_normalized.x + dL_dn.y * normal_normalized.y + dL_dn.z * normal_normalized.z;
                dL_dnormal_unnorm.x = (dL_dn.x - dot_grad_norm * normal_normalized.x) * inv_len;
                dL_dnormal_unnorm.y = (dL_dn.y - dot_grad_norm * normal_normalized.y) * inv_len;
                dL_dnormal_unnorm.z = (dL_dn.z - dot_grad_norm * normal_normalized.z) * inv_len;
            } else {
                dL_dnormal_unnorm = make_float3(0.0f, 0.0f, 0.0f);
            }

            // Transform back to world space: dL_dtn = viewmatrix^T * dL_dnormal_unnorm
            float3 dL_dtn = transformVec4x3Transpose(dL_dnormal_unnorm, viewmatrix);
            dL_dtn_vec = glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z);
        }
    }

    // Extract dL_dRS from dL_dM (first two columns plus normal gradient in third)
    // This matches compute_transmat_aabb in native backward which uses dL_dtn for the normal
    glm::mat3 dL_dRS = glm::mat3(
        glm::vec3(dL_dM[0]),  // dL_dM column 0 -> vec3
        glm::vec3(dL_dM[1]),  // dL_dM column 1 -> vec3
        dL_dtn_vec            // Normal gradient contribution (was zeros before)
    );

    // Compute scale gradients: dL_dscale.x = dot(dL_dRS[0], R[0])
    // R[0] is first column of rotation matrix (the u-axis direction in world space)
    dL_dscales[idx * 2 + 0] = glm::dot(dL_dRS[0], R[0]);
    dL_dscales[idx * 2 + 1] = glm::dot(dL_dRS[1], R[1]);

    // Compute rotation gradients: dL_dR = [dL_dRS[0] * scale.x, dL_dRS[1] * scale.y, 0]
    glm::mat3 dL_dR = glm::mat3(
        dL_dRS[0] * glm::vec3(scale.x),
        dL_dRS[1] * glm::vec3(scale.y),
        dL_dRS[2]  // zeros
    );

    // Convert rotation matrix gradient to quaternion gradient
    glm::vec4 dL_drot = quat_to_rotmat_vjp(rot, dL_dR);

    // Write output
    dL_drots[idx * 4 + 0] = dL_drot.x;
    dL_drots[idx * 4 + 1] = dL_drot.y;
    dL_drots[idx * 4 + 2] = dL_drot.z;
    dL_drots[idx * 4 + 3] = dL_drot.w;

    // Write mean gradient from column 2
    if (dL_dmeans != nullptr) {
        dL_dmeans[idx * 3 + 0] = dL_dM[2].x;
        dL_dmeans[idx * 3 + 1] = dL_dM[2].y;
        dL_dmeans[idx * 3 + 2] = dL_dM[2].z;
    }
}

// Host wrapper for transMat to scale/rotation gradient conversion
void transMat_to_scale_rot_grad(
    int N,
    int W, int H,  // Image dimensions for ndc2pix transformation
    const float* dL_dtransMat,   // [N, 9] screen-space transMat gradient
    const float* dL_dhomoMat,    // [N, 9] xyz gradient contribution (can be nullptr)
    const float* dL_dmean2D,     // [N, 2] 2D mean gradient (can be nullptr)
    const float* dL_dnormal3D,   // [N, 3] normal gradient (can be nullptr)
    const float* means3D,        // [N, 3] world-space positions (needed for dL_dmean2D)
    const float* transMat_precomp, // [N, 9] forward pass transMat (can be nullptr)
    const float* scales,
    const float* rotations,
    const float* projmatrix,
    const float* viewmatrix,     // [16] 4x4 view matrix (for normal gradient transform)
    float* dL_dscales,
    float* dL_drots,
    float* dL_dmeans)
{
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    transMat_to_scale_rot_grad_kernel<<<grid_size, block_size>>>(
        N,
        W, H,
        dL_dtransMat,
        dL_dhomoMat,
        dL_dmean2D,
        dL_dnormal3D,
        means3D,
        transMat_precomp,
        scales,
        rotations,
        projmatrix,
        viewmatrix,
        dL_dscales,
        dL_drots,
        dL_dmeans
    );
}

// ============================================================================
// BACKWARD KERNEL PROFILING - Host functions
// These must be in the same compilation unit as the __device__ symbols
// ============================================================================

void resetBackwardProfile() {
    unsigned long long zeros_ull[6] = {0};
    unsigned int zeros_uint[4] = {0};
    cudaMemcpyToSymbol(d_bw_profile, zeros_ull, sizeof(zeros_ull));
    cudaMemcpyToSymbol(d_bw_profile_counts, zeros_uint, sizeof(zeros_uint));
}

void readBackwardProfile(unsigned long long* cycles, unsigned int* counts) {
    cudaMemcpyFromSymbol(cycles, d_bw_profile, 6 * sizeof(unsigned long long));
    cudaMemcpyFromSymbol(counts, d_bw_profile_counts, 4 * sizeof(unsigned int));
}

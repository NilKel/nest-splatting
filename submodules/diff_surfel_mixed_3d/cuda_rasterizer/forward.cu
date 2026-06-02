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

#include "forward.h"
#include "auxiliary.h"
#include "hashgrid.h"
#include "mma_utils.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "modes/mode_3d_direct_fused.cu"
namespace cg = cooperative_groups;

// ============================================================================
// MLP WEIGHTS IN GLOBAL DEVICE MEMORY (3D_SH_res: tiny residual MLP)
// All weight matrices are [16×16] FP16 (single WMMA tile per layer)
// ============================================================================
// MLP weights (no bias): [hash(hash_dim) | pad(16-hash_dim)] = 16D input
// L1: W1[16×16], L2: W2[16×16], L3: W3[16×16] (only first 3 rows = RGB residual)
__device__ __half* d_mlp_W1 = nullptr;      // 16D input → 16D hidden1 (256 halfs = 512B)
__device__ __half* d_mlp_W2 = nullptr;      // 16D hidden1 → 16D hidden2 (512B)
__device__ __half* d_mlp_W3 = nullptr;      // 16D hidden2 → 16D output (only 3 rows = RGB residual)

// Contribution threshold: skip hash query when w = T*alpha < this value (0 = disabled)
__device__ float d_contrib_thresh = 0.0f;

// Count threshold: skip hash query after this many contributing Gaussians per pixel (0 = disabled)
__device__ int d_count_thresh = 0;

// Overdraw regularization: lambda weight for sigmoid-based soft contributor count loss
// k (steepness) is hardcoded to 10. When lambda > 0, forward outputs overdraw map,
// backward adds dL_dalpha contribution to reduce per-pixel contributor count.
__device__ float d_overdraw_lambda = 0.0f;

// Weight-squared regularization: penalize (1 - sum(w_i^2)) per pixel
// When lambda > 0, backward adds dL_dalpha = -lambda * 2 * w * T per Gaussian
__device__ float d_weight_reg_lambda = 0.0f;

// Activation biases for SH and MLP residual.
// d_residual_mode selects the OUTER activation:
//   0 = 3D_SH_res (default):   color = ReLU(ReLU(SH + sh_bias) + residual + res_bias)
//   1 = 3D_SH_add:             color = ReLU(SH + sh_bias) + ReLU(residual + res_bias)
// (In both modes the SH branch's inner ReLU is applied by computeColorFromSH.)
// Default: sh_bias=0.5, res_bias=0.5 (standard 3DGS gray init + residual offset)
// For decomposition: sh_only sets res_bias=-999 (ReLU clamps to 0), tex_only sets sh_bias=-999
__device__ int d_residual_mode = 0;
// `--ste`: straight-through estimator on the outer per-Gauss ReLU. Only the
// backward consumes this; mirrored here for setter symmetry.
__device__ int d_ste_relu_fwd = 0;
__device__ float d_sh_bias = 0.5f;
// Nexels-style anti-aliasing d_aa_factor / d_aa_focal are now declared in hashgrid.h
// (per-TU static __device__). Setters below update this TU's copy.
// AA-2DGS mip filter kernel size σ. 0 disables (use standard rho3d/rho2d).
__device__ float d_aa_kernel_size = 0.0f;
__device__ float d_res_bias = 0.5f;
// FastGS Compact Box multiplier. Scales the Mahalanobis² threshold used by the
// AdR cutoff in preprocessCUDA:  cutoff = sqrt(2·log(opacity·255)·mult).
// Default 1.0 = unchanged (matches our existing AdR cutoff). FastGS paper uses 0.5.
__device__ float d_compact_mult = 1.0f;

// Host-side pointers for memory management
static __half* h_mlp_W1 = nullptr;
static __half* h_mlp_W2 = nullptr;
static __half* h_mlp_W3 = nullptr;
static bool mlp_weights_allocated = false;

// Convenience macros to access MLP weights
#define mlp_W1 d_mlp_W1
#define mlp_W2 d_mlp_W2
#define mlp_W3 d_mlp_W3

// MLP gradient buffers (device global memory, allocated once)
// These accumulate gradients across all intersections, then retrieved by Python
float* d_dL_dW1 = nullptr;   // [16, 16] = W1_SIZE = 256 floats
float* d_dL_dW2 = nullptr;   // [16, 16] = W2_SIZE = 256 floats
float* d_dL_dW3 = nullptr;   // [16, 16] = W3_SIZE = 256 floats

// Flag to track if gradient buffers are allocated
bool mlp_grad_buffers_allocated = false;

// ============================================================================
// PERFORMANCE TEST MACRO
// Uncomment to test if powf is the performance bottleneck in the general kernel.
// When enabled, bypasses the expensive powf(rho, beta/2) with a simple rho (beta=2).
// If FPS increases significantly (>20%), powf is the bottleneck.
// If FPS stays the same, memory bandwidth is the bottleneck.
// ============================================================================
// #define FAST_POW_TEST 1

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += d_sh_bias;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Inline SH evaluation for hybrid_SH mode
// SH coefficients format: [sh0_r, sh0_g, sh0_b, sh1_r, sh1_g, sh1_b, ...]
__device__ void eval_sh_inline(int deg, const float* sh, const float3& dir, float* result) {
	// DC component (degree 0)
	result[0] = SH_C0 * sh[0] + d_sh_bias;  // R
	result[1] = SH_C0 * sh[1] + d_sh_bias;  // G
	result[2] = SH_C0 * sh[2] + d_sh_bias;  // B

	if (deg > 0) {
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		// Degree 1 (coefficients 1-3, indices 3-11)
		result[0] = result[0] - SH_C1 * y * sh[3]  + SH_C1 * z * sh[6]  - SH_C1 * x * sh[9];   // R
		result[1] = result[1] - SH_C1 * y * sh[4]  + SH_C1 * z * sh[7]  - SH_C1 * x * sh[10];  // G
		result[2] = result[2] - SH_C1 * y * sh[5]  + SH_C1 * z * sh[8]  - SH_C1 * x * sh[11];  // B

		if (deg > 1) {
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			// Degree 2 (coefficients 4-8, indices 12-26)
			result[0] = result[0] +
				SH_C2[0] * xy * sh[12] +
				SH_C2[1] * yz * sh[15] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[18] +
				SH_C2[3] * xz * sh[21] +
				SH_C2[4] * (xx - yy) * sh[24];

			result[1] = result[1] +
				SH_C2[0] * xy * sh[13] +
				SH_C2[1] * yz * sh[16] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[19] +
				SH_C2[3] * xz * sh[22] +
				SH_C2[4] * (xx - yy) * sh[25];

			result[2] = result[2] +
				SH_C2[0] * xy * sh[14] +
				SH_C2[1] * yz * sh[17] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[20] +
				SH_C2[3] * xz * sh[23] +
				SH_C2[4] * (xx - yy) * sh[26];

			if (deg > 2) {
				// Degree 3 (coefficients 9-15, indices 27-47)
				result[0] = result[0] +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[27] +
					SH_C3[1] * xy * z * sh[30] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[33] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[36] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[39] +
					SH_C3[5] * z * (xx - yy) * sh[42] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[45];

				result[1] = result[1] +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[28] +
					SH_C3[1] * xy * z * sh[31] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[34] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[37] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[40] +
					SH_C3[5] * z * (xx - yy) * sh[43] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[46];

				result[2] = result[2] +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[29] +
					SH_C3[1] * xy * z * sh[32] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[35] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[38] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[41] +
					SH_C3[5] * z * (xx - yy) * sh[44] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[47];
			}
		}
	}

	// Clamp to [0, inf)
	result[0] = fmaxf(result[0], 0.0f);
	result[1] = fmaxf(result[1], 0.0f);
	result[2] = fmaxf(result[2], 0.0f);
}


// Inline SH evaluation WITHOUT activation (for hybrid_SH mode)
// Returns raw SH RGB values before activation
__device__ void eval_sh_raw(int deg, const float* sh, const float3& dir, float* result) {
	// DC component (degree 0) - NO +0.5 offset
	result[0] = SH_C0 * sh[0];  // R
	result[1] = SH_C0 * sh[1];  // G
	result[2] = SH_C0 * sh[2];  // B

	if (deg > 0) {
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		// Degree 1 (coefficients 1-3, indices 3-11)
		result[0] = result[0] - SH_C1 * y * sh[3]  + SH_C1 * z * sh[6]  - SH_C1 * x * sh[9];   // R
		result[1] = result[1] - SH_C1 * y * sh[4]  + SH_C1 * z * sh[7]  - SH_C1 * x * sh[10];  // G
		result[2] = result[2] - SH_C1 * y * sh[5]  + SH_C1 * z * sh[8]  - SH_C1 * x * sh[11];  // B

		if (deg > 1) {
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			// Degree 2 (coefficients 4-8, indices 12-26)
			result[0] = result[0] +
				SH_C2[0] * xy * sh[12] +
				SH_C2[1] * yz * sh[15] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[18] +
				SH_C2[3] * xz * sh[21] +
				SH_C2[4] * (xx - yy) * sh[24];

			result[1] = result[1] +
				SH_C2[0] * xy * sh[13] +
				SH_C2[1] * yz * sh[16] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[19] +
				SH_C2[3] * xz * sh[22] +
				SH_C2[4] * (xx - yy) * sh[25];

			result[2] = result[2] +
				SH_C2[0] * xy * sh[14] +
				SH_C2[1] * yz * sh[17] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[20] +
				SH_C2[3] * xz * sh[23] +
				SH_C2[4] * (xx - yy) * sh[26];

			if (deg > 2) {
				// Degree 3 (coefficients 9-15, indices 27-47)
				result[0] = result[0] +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[27] +
					SH_C3[1] * xy * z * sh[30] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[33] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[36] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[39] +
					SH_C3[5] * z * (xx - yy) * sh[42] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[45];

				result[1] = result[1] +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[28] +
					SH_C3[1] * xy * z * sh[31] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[34] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[37] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[40] +
					SH_C3[5] * z * (xx - yy) * sh[43] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[46];

				result[2] = result[2] +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[29] +
					SH_C3[1] * xy * z * sh[32] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[35] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[38] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[41] +
					SH_C3[5] * z * (xx - yy) * sh[44] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[47];
			}
		}
	}
	// NO activation here - returns raw values
}

// ============================================================================
// MLP FORWARD PASS FOR FUSED MODES (3D_fused, 3D_direct_fused)
// Unrolled scalar MLP evaluation per-thread
// Used in both forward and backward (backward recomputes forward to get h1, h2)
// ============================================================================

// Spherical Harmonics encoding degree 4 (16D output)
// Matches Python's tcnn.Encoding with otype="SphericalHarmonics" degree=4
// Python normalizes input: d = (d+1)/2 before encoding
// tcnn then converts back: d = input * 2 - 1 and computes SH
// Net effect: view_dir passes through unchanged, but tcnn uses (-x, -y, z) convention
__device__ void encode_view_direction(const float3& view_dir, float* view_enc) {
	// tcnn's convention: negate x and y components
	// This was determined empirically by comparing with tcnn output
	float x = -view_dir.x;
	float y = -view_dir.y;
	float z = view_dir.z;

	// Spherical harmonics basis functions (real, degree 4)
	// Constants are sqrt((2*l+1)/(4*pi) * (l-m)!/(l+m)!)
	// l=0 (1 coefficient)
	view_enc[0] = 0.28209479177387814f;  // Y_0^0 = 0.5 * sqrt(1/pi)

	// l=1 (3 coefficients)
	view_enc[1] = 0.4886025119029199f * y;   // Y_1^{-1}
	view_enc[2] = 0.4886025119029199f * z;   // Y_1^0
	view_enc[3] = 0.4886025119029199f * x;   // Y_1^1

	// l=2 (5 coefficients)
	view_enc[4] = 1.0925484305920792f * x * y;  // Y_2^{-2}
	view_enc[5] = 1.0925484305920792f * y * z;  // Y_2^{-1}
	view_enc[6] = 0.31539156525252005f * (3.0f * z * z - 1.0f);  // Y_2^0
	view_enc[7] = 1.0925484305920792f * x * z;  // Y_2^1
	view_enc[8] = 0.5462742152960396f * (x * x - y * y);  // Y_2^2

	// l=3 (7 coefficients)
	view_enc[9]  = 0.5900435899266435f * y * (3.0f * x * x - y * y);  // Y_3^{-3}
	view_enc[10] = 2.890611442640554f * x * y * z;  // Y_3^{-2}
	view_enc[11] = 0.4570457994644658f * y * (5.0f * z * z - 1.0f);  // Y_3^{-1}
	view_enc[12] = 0.3731763325901154f * z * (5.0f * z * z - 3.0f);  // Y_3^0
	view_enc[13] = 0.4570457994644658f * x * (5.0f * z * z - 1.0f);  // Y_3^1
	view_enc[14] = 1.4453057213202769f * (x * x - y * y) * z;  // Y_3^2
	view_enc[15] = 0.5900435899266435f * x * (x * x - 3.0f * y * y);  // Y_3^3
}

// MLP forward pass (no bias): IN_DIM input → HIDDEN_DIM hidden1 → HIDDEN_DIM hidden2 → OUT_DIM output
// 3D_SH_res: All dimensions = 16 (single WMMA tile per layer)
// Input: [hash(hash_dim) | pad(16-hash_dim)] = 16D. No implicit bias.
// Output: 3D RGB residual (identity activation, no sigmoid)
template <int IN_DIM, int HIDDEN_DIM, int OUT_DIM>
__device__ void mlp_forward_fused(
	const float* input,       // [IN_DIM] = 16D (hash + pad)
	float* output,            // [OUT_DIM] = 3 (RGB residual)
	float* hidden1,           // [HIDDEN_DIM] = 16 (caller-provided buffer)
	float* hidden2,           // [HIDDEN_DIM] = 16 (caller-provided buffer)
	bool apply_sigmoid,       // false for residual (identity), true for RGB (sigmoid)
	const __half* W1 = nullptr,   // Shared memory W1 (falls back to global if nullptr)
	const __half* W2 = nullptr,   // Shared memory W2
	const __half* W3 = nullptr    // Shared memory W3
) {
	// Use shared memory weights if provided, otherwise fall back to global memory
	const __half* w1 = W1 ? W1 : mlp_W1;
	const __half* w2 = W2 ? W2 : mlp_W2;
	const __half* w3 = W3 ? W3 : mlp_W3;

	// Convert input to FP16 once. For 3D_SH_res the input is [hash(hash_dim) |
	// pad(16-hash_dim) all zeros] — NO implicit bias slot. MLP(0) ≡ 0.
	// (3D_SH_cat / 3D_direct_fused use a different layout with input[4]=1.0
	// as an implicit L1 bias; that path is in mode_3d_direct_fused.cu, not here.)
	__half input_h[IN_DIM];
	#pragma unroll
	for (int i = 0; i < IN_DIM; i++)
		input_h[i] = __float2half(input[i]);

	// Layer 1: IN_DIM → HIDDEN_DIM (ReLU) — FP16 __half2 dot products
	#pragma unroll
	for (int h = 0; h < HIDDEN_DIM; h++) {
		__half2 acc2 = __float2half2_rn(0.0f);
		const __half* w1_row = &w1[h * IN_DIM];
		#pragma unroll
		for (int i = 0; i < IN_DIM; i += 2) {
			__half2 in2 = *reinterpret_cast<const __half2*>(&input_h[i]);
			__half2 wt2 = *reinterpret_cast<const __half2*>(&w1_row[i]);
			acc2 = __hfma2(in2, wt2, acc2);
		}
		float acc = __half2float(acc2.x) + __half2float(acc2.y);
		hidden1[h] = fmaxf(0.0f, acc);  // ReLU (store FP32 for backward)
	}

	// Convert hidden1 to FP16 for layer 2
	__half h1_h[HIDDEN_DIM];
	#pragma unroll
	for (int i = 0; i < HIDDEN_DIM; i++)
		h1_h[i] = __float2half(hidden1[i]);

	// Layer 2: HIDDEN_DIM → HIDDEN_DIM (ReLU) — FP16 __half2 dot products
	#pragma unroll
	for (int h = 0; h < HIDDEN_DIM; h++) {
		__half2 acc2 = __float2half2_rn(0.0f);
		const __half* w2_row = &w2[h * HIDDEN_DIM];
		#pragma unroll
		for (int i = 0; i < HIDDEN_DIM; i += 2) {
			__half2 in2 = *reinterpret_cast<const __half2*>(&h1_h[i]);
			__half2 wt2 = *reinterpret_cast<const __half2*>(&w2_row[i]);
			acc2 = __hfma2(in2, wt2, acc2);
		}
		float acc = __half2float(acc2.x) + __half2float(acc2.y);
		hidden2[h] = fmaxf(0.0f, acc);  // ReLU (store FP32 for backward)
	}

	// Convert hidden2 to FP16 for layer 3
	__half h2_h[HIDDEN_DIM];
	#pragma unroll
	for (int i = 0; i < HIDDEN_DIM; i++)
		h2_h[i] = __float2half(hidden2[i]);

	// Layer 3: HIDDEN_DIM → OUT_DIM (identity or sigmoid) — FP16 __half2 dot products
	#pragma unroll
	for (int o = 0; o < OUT_DIM; o++) {
		__half2 acc2 = __float2half2_rn(0.0f);
		const __half* w3_row = &w3[o * HIDDEN_DIM];
		#pragma unroll
		for (int h = 0; h < HIDDEN_DIM; h += 2) {
			__half2 in2 = *reinterpret_cast<const __half2*>(&h2_h[h]);
			__half2 wt2 = *reinterpret_cast<const __half2*>(&w3_row[h]);
			acc2 = __hfma2(in2, wt2, acc2);
		}
		float acc = __half2float(acc2.x) + __half2float(acc2.y);
		output[o] = apply_sigmoid ? (1.0f / (1.0f + expf(-acc))) : acc;
	}
}

// ============================================================================

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
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

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	
	// TOGGLE NORMAL NORMALIZATION: Comment/uncomment the line below
	// Uncomment to normalize normals for surface mode (recommended for consistent dot product magnitudes)
	// Comment out to use unnormalized normals (scaled by Gaussian dimensions)
	#define NORMALIZE_SURFACE_NORMALS
	#ifdef NORMALIZE_SURFACE_NORMALS
	float normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
	if(normal_len > 1e-7f) {
		normal.x /= normal_len;
		normal.y /= normal_len;
		normal.z /= normal_len;
	}
	#endif

}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// ============================================================================
// `--method mixed_3d` — EWA 3D-ellipsoid path for UNTEXTURED surfels.
//
// computeCov3D / computeCov2D are ported VERBATIM from FastGS
// (diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu, the stock
// Inria EWA splatting math) so that, fed identical Gaussians, this submodule's
// untextured output matches FastGS bit-for-bit. Quaternion layout (r,x,y,z),
// no quaternion normalization, eps2d=0.3 low-pass — all kept identical to
// FastGS on purpose. The textured half is untouched (compute_transmat).
// ============================================================================
__device__ void ewa_computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ float3 ewa_computeCov2D(const float3& mean, float focal_x, float focal_y,
	float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Returns false if the Gaussian is culled (degenerate conic).
// On success: conic = inverse 2D covariance (a,b,c), point_image = pixel-space
// center, my_radius = FastGS ceil(3·sqrt(max λ)).
__device__ bool compute_ewa_conic(
	const float3& p_orig,
	const glm::vec3 scale3,
	float mod,
	const glm::vec4 rot,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const int W, const int H,
	float3& conic,
	float2& point_image,
	float& my_radius)
{
	float cov3D[6];
	ewa_computeCov3D(scale3, mod, rot, cov3D);

	float3 cov = ewa_computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return false;
	float det_inv = 1.f / det;
	conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	int* radii_x,  // Separate X radius for rectangular AABB
	int* radii_y,  // Separate Y radius for rectangular AABB
	float2* points_xy_image,
	float* depths,
	float* transMats,
	rgb_t* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	float4* conic_t,            // SnugBox conic cache (aabb_mode==5)
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode,
	const int render_mode,
	const bool* is_textured,
	const float* scaling_z,
	float4* ewa_conic)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	radii_x[idx] = 0;
	radii_y[idx] = 0;
	tiles_touched[idx] = 0;
	conic_t[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // default: rect AABB fallback
	if (ewa_conic != nullptr)
		ewa_conic[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// `--method mixed_3d`: UNTEXTURED surfels render as EWA 3D ellipsoids
	// (FastGS-parity geometry). Engaged only post-texsplit (scaling_z != nullptr)
	// for rows flagged untextured; textured rows fall through to compute_transmat
	// (bit-identical to `--method mixed`). conic_t stays 0 → binning uses the
	// rect-AABB path with rx==ry==FastGS radius.
	const bool ewa_path = (scaling_z != nullptr) && (is_textured != nullptr)
	                      && (ewa_conic != nullptr) && (!is_textured[idx]);
	if (ewa_path)
	{
		const float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
		glm::vec3 scale3 = glm::vec3(scales[idx].x, scales[idx].y, scaling_z[idx]);
		float3 conic; float2 pimg; float my_radius;
		if (!compute_ewa_conic(p_orig, scale3, scale_modifier, rotations[idx],
				viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy,
				W, H, conic, pimg, my_radius))
			return;

		uint2 rect_min, rect_max;
		getRectXY(pimg, (int)my_radius, (int)my_radius, rect_min, rect_max, grid);
		uint32_t n_tiles = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
		if (n_tiles == 0)
			return;

		// SH/SV baseline color (untextured: no hash, no MLP). Same fill rule
		// as the textured/standard path below.
		if (colors_precomp == nullptr || (render_mode & 0xFF) == 6) {
			glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
			rgb[idx * C + 0] = FLOAT_TO_RGB(result.x);
			rgb[idx * C + 1] = FLOAT_TO_RGB(result.y);
			rgb[idx * C + 2] = FLOAT_TO_RGB(result.z);
		} else {
			for (int i = 0; i < C; i++)
				rgb[idx * C + i] = FLOAT_TO_RGB(colors_precomp[idx * C + i]);
		}

		depths[idx] = p_view.z;
		radii[idx] = (int)my_radius;
		radii_x[idx] = (int)my_radius;
		radii_y[idx] = (int)my_radius;
		points_xy_image[idx] = pimg;
		normal_opacity[idx] = { 0.0f, 0.0f, 1.0f, opacities[idx] };
		ewa_conic[idx] = make_float4(conic.x, conic.y, conic.z, opacities[idx]);

		// `--aabb accutile` (aabb_mode==5) on the untextured EWA half: write a
		// per-Gauss SnugBox conic so duplicateWithKeys routes this row through
		// the ellipse-tight tile walk (auxiliary.h:processTiles) instead of the
		// rect AABB enum. The EWA conic IS the inverse 2D covariance, so the
		// quadratic form `m = a·dx² + 2b·dx·dy + c·dy²` is exactly what AccuTile
		// expects — the only work is deriving the Mahalanobis² cutoff `t_m` per
		// (effective) kernel and capping at m=9 (FastGS 3σ rect bound) so the
		// AccuTile count never exceeds the rect AABB count.
		if (aabb_mode == 5) {
			const float opa = opacities[idx];
			if (opa >= (1.0f / 255.0f)) {
				// Decode the --kernel2 override the same way the EWA render
				// kernel does (render_mode bits [16..19] = ut_kt+1; nibble 0 →
				// unset, fall back to kernel_type). See forward.cu render loop
				// (~line 1769).
				const int _utk_n_pp = (render_mode >> 16) & 0xF;
				const int ut_kt_pp  = _utk_n_pp ? (_utk_n_pp - 1) : kernel_type;
				float t_m = 0.0f;
				const float ratio = 1.0f / (255.0f * opa);
				if ((ut_kt_pp == 1 || ut_kt_pp == 4) && shapes != nullptr) {
					// Restricted-beta: α = opa·max(0, 1 − m/k²)^β.
					// α ≥ 1/255 ⇒ m ≤ k²·(1 − (1/(opa·255))^(1/β)).
					const float k_sq = (ut_kt_pp == 4) ? 9.0f : 1.0f;
					const float beta_val = fmaxf(shapes[idx], 1e-3f);
					const float thr = powf(ratio, 1.0f / beta_val);
					if (thr < 1.0f) t_m = k_sq * (1.0f - thr);
				} else {
					// FastGS Gaussian: α = opa·exp(-0.5·m).
					// α ≥ 1/255 ⇒ m ≤ 2·ln(opa·255).
					t_m = 2.0f * logf(255.0f * opa);
				}
				// FastGS radius is ceil(3·√maxλ) → bounds m to 9. Cap so AccuTile
				// stays inside the rect AABB (the count safety check below would
				// reject anyway, but capping avoids the wasted scan).
				t_m = fminf(t_m, 9.0f);
				const float det_c = conic.x * conic.z - conic.y * conic.y;
				if (t_m > 0.0f && conic.x > 0.0f && conic.z > 0.0f && det_c > 0.0f) {
					uint32_t n_tiles_sb = duplicateToTilesTouched(
						conic.x, conic.y, conic.z, t_m, pimg, grid,
						0, 0, 0.0f, nullptr, nullptr);
					if (n_tiles_sb > 0 && n_tiles_sb <= n_tiles) {
						conic_t[idx] = make_float4(conic.x, conic.y, conic.z, t_m);
						tiles_touched[idx] = n_tiles_sb;
						return;
					}
				}
			}
			// Fallthrough: degenerate / culled / inflated → rect AABB below.
		}

		tiles_touched[idx] = n_tiles;
		return;  // skip 2DGS transmat/AABB entirely
	}

	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

	// Compute cutoff for bounding box
	// aabb_mode: 0 = square (default), 1 = AdR cutoff, 2 = rectangular, 3 = AdR + rectangular,
	//            4 = beta (fixed r=1), 5 = AdR + rectangular + AccuTile ellipse cull
	float cutoff;
	bool use_adr_cutoff = (aabb_mode == 1 || aabb_mode == 3 || aabb_mode == 5);  // modes 1, 3, 5 use AdR
	bool use_beta_cutoff = (aabb_mode == 4);  // mode 4: fixed r=1 for beta kernels
	if (use_adr_cutoff && kernel_type == 3 && shapes != nullptr) {
		// AdR: Adaptive bounding box based on opacity and beta
		// For general kernel (kernel_type=3), the kernel is exp(-0.5 * (r²)^(β/2))
		// We want: opacity * exp(-0.5 * k^β) = 1/255
		// Solving: k = (2 * ln(opacity * 255))^(1/β)
		float opacity_val = opacities[idx];
		float beta = shapes[idx];  // Beta parameter for this Gaussian, range [2.0, 8.0]

		// Early cull: if opacity < 1/255, Gaussian is invisible everywhere
		if (opacity_val < (1.0f / 255.0f)) {
			radii[idx] = 0;
			tiles_touched[idx] = 0;
			return;
		}

		// k = (2 * ln(opacity * 255))^(1/beta)
		float log_term = 2.0f * logf(opacity_val * 255.0f);
		if (log_term > 0.0f) {
			float k = powf(log_term, 1.0f / beta);
			// Safety clamp: don't let soft Gaussians (low beta) grow beyond 4σ
			cutoff = fminf(k, 4.0f);
		} else {
			// log_term <= 0 means opacity <= 1/255, shouldn't happen after early cull
			cutoff = 0.1f;  // Minimal bounding box
		}
	} else if (use_adr_cutoff && (kernel_type == 1 || kernel_type == 4) && shapes != nullptr) {
		// AdR for beta kernel with max-pool: need to consider both Beta and Gaussian radii
		// kernel_type 1: k²=1 (unit disk), kernel_type 4: k²=9 (3σ scaled)
		float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
		float k = (kernel_type == 4) ? 3.0f : 1.0f;

		float opacity_val = opacities[idx];
		float shape = shapes[idx];  // Shape parameter

		// Early cull: if opacity < 1/255, splat is invisible everywhere
		if (opacity_val < (1.0f / 255.0f)) {
			radii[idx] = 0;
			tiles_touched[idx] = 0;
			return;
		}

		// Visibility threshold (alpha_min = 1/255)
		float ratio = 1.0f / (255.0f * opacity_val);

		// Beta kernel radius: solve opacity * (1 - r²/k²)^shape >= 1/255
		// => r <= k * sqrt(1 - ratio^(1/shape))
		float r_beta = 0.0f;
		float threshold = powf(ratio, 1.0f / shape);
		if (threshold < 1.0f) {
			r_beta = k * sqrtf(1.0f - threshold);
		}

		// Gaussian low-pass radius: solve opacity * exp(-r²/2) >= 1/255
		// => r² <= -2 * ln(ratio) = 2 * ln(255 * opacity)
		// => r <= sqrt(2 * ln(255 * opacity))
		float r_lp = 0.0f;
		float log_term = logf(255.0f * opacity_val);
		if (log_term > 0.0f) {
			r_lp = sqrtf(2.0f * log_term);
		}

		// Combined cutoff: max of Beta and Gaussian radii (lossless)
		cutoff = fmaxf(r_beta, r_lp);

		// Safety clamp to prevent excessively large bounding boxes
		cutoff = fminf(cutoff, k + 2.0f);
	} else if (use_adr_cutoff) {
		// AdR for standard Gaussian kernel (kernel_type == 0 or 2, no shapes)
		// Solve: opacity * exp(-0.5 * r²) >= 1/255
		// => r <= sqrt(2 * ln(255 * opacity))
		float opacity_val = opacities[idx];

		// Early cull: if opacity < 1/255, Gaussian is invisible everywhere
		if (opacity_val < (1.0f / 255.0f)) {
			radii[idx] = 0;
			tiles_touched[idx] = 0;
			return;
		}

		float log_term = logf(255.0f * opacity_val);
		if (log_term > 0.0f) {
			// FastGS Compact Box: scale Mahalanobis² by d_compact_mult (paper: 0.5).
			// d_compact_mult=1.0 reproduces our existing AdR cutoff.
			cutoff = sqrtf(2.0f * log_term * d_compact_mult);
		} else {
			cutoff = 0.1f;
		}
		// Don't exceed original 4σ
		cutoff = fminf(cutoff, 4.0f);
	} else if (use_beta_cutoff) {
		// Beta kernel: fixed cutoff for compact support with low-pass consideration
		float k = (kernel_type == 4) ? 3.0f : 1.0f;
		// Low-pass radius for typical opacity (~0.5): sqrt(2 * ln(127.5)) ≈ 3.1
		float r_lp_typical = sqrtf(2.0f * logf(127.5f));
		cutoff = fmaxf(k * 1.1f, r_lp_typical);
	} else {
#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
		// the effective extent is now depended on the opacity of gaussian.
		cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
		// 2DGS default: fixed 4σ cutoff
		cutoff = 4.0f;
#endif
	}

	// Compute center and AABB bounds
	// aabb_mode: 0 = square AABB (2DGS default), 1 = AdR cutoff only, 2 = rectangular AABB only, 3 = AdR + rectangular
	// Compute center and AABB bounds
	// aabb_mode: 0 = square AABB (2DGS default), 1 = AdR cutoff only, 2 = rectangular AABB only, 3 = AdR + rectangular
	// Rectangular AABB uses separate X/Y radii instead of max(x,y)
	float2 point_image;
	float2 extent;
	bool ok = compute_aabb(T, cutoff, point_image, extent);
	if (!ok) return;

	uint2 rect_min, rect_max;
	float radius;  // For output to radii array (max of x,y for compatibility)
	int rx, ry;    // Separate X and Y radii
	bool use_rect_aabb = (aabb_mode >= 2);  // modes 2, 3, 4, 5 use rectangular AABB

	if (use_rect_aabb) {
		// Rectangular AABB: keep separate X and Y radii
		// This allows elongated Gaussians to have smaller tile coverage
		rx = (int)ceilf(fmaxf(extent.x, cutoff * FilterSize));
		ry = (int)ceilf(fmaxf(extent.y, cutoff * FilterSize));
		getRectXY(point_image, rx, ry, rect_min, rect_max, grid);
		radius = fmaxf((float)rx, (float)ry);  // For radii output (compatibility)
	} else {
		// Square AABB: use max(x, y) as scalar radius (original 2DGS behavior)
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
		rx = (int)radius;
		ry = (int)radius;
		getRect(point_image, (int)radius, rect_min, rect_max, grid);
	}

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Store separate X/Y radii for tile binning
	radii_x[idx] = rx;
	radii_y[idx] = ry;

	// Compute colors
	// NOTE: In baseline hashgrid mode (render_mode=0, level>0), colors_precomp is empty
	// and we skip this entirely - colors come from hashgrid query in render kernel
	// render_mode 6 (3D_SH_cat): colors_precomp has DC SH, but we still need full SH eval
	if (colors_precomp == nullptr || (render_mode & 0xFF) == 6) {
		// SH mode: evaluate spherical harmonics to RGB
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = FLOAT_TO_RGB(result.x);
		rgb[idx * C + 1] = FLOAT_TO_RGB(result.y);
		rgb[idx * C + 2] = FLOAT_TO_RGB(result.z);
	}
	else {
		// Per-Gaussian features mode (cat, adaptive, etc.): copy precomputed features
		// For empty colors_precomp (baseline hashgrid), skip this - rgb buffer won't be used
		// Only copy if C matches expected dimension (otherwise it's a size mismatch)
		for(int i = 0; i < C; i++){
			rgb[idx * C + i] = FLOAT_TO_RGB(colors_precomp[idx * C + i]);
		}
	}
	
	// if(idx == 0 ){
	// 	printf("depth %.4f\n", p_view.z);
	// }
	depths[idx] = p_view.z;
	// Store max radius for the sorter (needs non-zero to be valid)
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};

	uint32_t rect_tile_count = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	// SnugBox+AccuTile (aabb_mode==5): per-row ellipse intersection emits keys
	// only for tiles the ellipse actually crosses. Counts here MUST equal the
	// emit-phase count in duplicateWithKeys (see auxiliary.h — uses no-fusion
	// FP intrinsics for bit-identicality across the two specializations).
	if (aabb_mode == 5) {
		float A_c, B_c, E_c, t_c;
		float2 p_c;
		if (compute_conic_from_transmat(T, cutoff, A_c, B_c, E_c, t_c, p_c)) {
			uint32_t n_tiles_sb = duplicateToTilesTouched(
				A_c, B_c, E_c, t_c, p_c, grid,
				0, 0, 0.0f, nullptr, nullptr);
			// AccuTile must be ≤ rect AABB. If not, the conic is near-degenerate
			// (disc → 0⁻) and the scan inflated outside the rect — fall back.
			if (n_tiles_sb > 0 && n_tiles_sb <= rect_tile_count) {
				conic_t[idx] = make_float4(A_c, B_c, E_c, t_c);
				// Use conic center so the emit-phase walk matches exactly.
				points_xy_image[idx] = p_c;
				tiles_touched[idx] = n_tiles_sb;
				return;
			}
		}
	}

	tiles_touched[idx] = rect_tile_count;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	uint32_t render_number = 0;


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

	// Max contributor tracking for depth reinit
	float max_w = 0.0f;
	float max_depth = 0.0f;
#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

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
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (depth < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;


#if RENDER_AXUTILITY
			// Track max contributor per pixel (for depth reinit)
			if (w > max_w) {
				max_w = w;
				max_depth = depth;
			}

			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		out_others[pix_id + NUM_OFFSET * H * W] = render_number;
		out_others[pix_id + MAXDEPTH_OFFSET * H * W] = max_depth;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}


// Nest Gaussian
// Compute Zip-NeRF style anti-aliasing weight for a hash level
// Returns weight in [0, 1]: 0 = fully attenuated, 1 = full contribution
__device__ __forceinline__ float compute_aa_weight(
	float focal,      // Focal length in pixels
	float depth,      // Intersection depth
	float resolution, // Hash level resolution (s_l)
	float aa_scale    // User-specified scale factor
) {
	if (aa_scale <= 0.0f || depth <= 0.0f) return 1.0f;

	// Zip-NeRF formula: w_l = 1 - exp(-f^2 / (2*pi * s_l^2 * z^2))
	// With user scale: w_l = 1 - exp(-aa_scale * f^2 / (2*pi * s_l^2 * z^2))
	float f_sq = focal * focal;
	float s_sq = resolution * resolution;
	float z_sq = depth * depth;
	float denom = 2.0f * 3.14159265f * s_sq * z_sq;

	float exponent = -aa_scale * f_sq / denom;
	return 1.0f - expf(exponent);
}

template <uint32_t CHANNELS, uint32_t D_DIFFUSE = 0>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDAsurfelForward(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float beta,
	int W, int H,
	uint32_t level, const uint32_t l_dim, float l_scale, uint32_t Base,
	bool align_corners, uint32_t interp,
	const bool if_contract, const bool record_transmittance,
	const glm::vec2* scales,
	float focal_x, float focal_y,
	const float* __restrict__ means3D,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ homotrans,
	const float* __restrict__ ap_level,
	const __half* __restrict__ hash_features,
	const int* __restrict__ level_offsets,
	const float* __restrict__ gridrange,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	int* __restrict__ out_index,
	float* __restrict__ cover_pixel,
	float* __restrict__ trans_avg,
	const glm::vec3* __restrict__ cam_pos,
	const float* __restrict__ hash_features_diffuse = nullptr,
	const int* __restrict__ level_offsets_diffuse = nullptr,
	const float* __restrict__ gridrange_diffuse = nullptr,
	const int render_mode = 0,
	const rgb_t* __restrict__ rgb = nullptr,
	const uint32_t max_intersections = 0,
	const float* __restrict__ shapes = nullptr,
	const int kernel_type = 0,
	const float aa = 0.0f,
	const float aa_threshold = 0.01f,
	// 3D mode intersection buffer outputs
	float* __restrict__ intersection_buffer = nullptr,
	uint32_t* __restrict__ intersection_count = nullptr,
	const uint32_t max_intersections_per_pixel = 0,
	// Pre-encoded view directions for 3D_direct_fused (H*W, 16) - one 16D vector per pixel
	const float* __restrict__ viewdirs_enc = nullptr,
	// FastGS VCD/VCP counters. When metric_map[pix_id]==1, every Gaussian that
	// passes the alpha > 1/255 gate at this pixel atomic-adds to metric_counts[id].
	// No effect when either pointer is null.
	const int* __restrict__ metric_map = nullptr,
	int* __restrict__ metric_counts = nullptr,
	// `--method mixed` per-Gauss bool flag [P]. nullptr → all-textured behavior.
	const bool* __restrict__ is_textured = nullptr,
	// `--method mixed_3d` per-Gauss EWA conic [P] (a,b,c,opacity). nullptr →
	// untextured surfels keep the 2DGS ray-splat geometry below.
	const float4* __restrict__ ewa_conic = nullptr)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	const float pix_off = (render_mode & 0x800) ? 0.5f : 0.0f;
	float2 pixf = { (float)pix.x + pix_off, (float)pix.y + pix_off};
	const bool fastgs_count = (metric_counts != nullptr && metric_map != nullptr);

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float2 collected_size[BLOCK_SIZE];

	__shared__ float3 collected_SuTu[BLOCK_SIZE];
	__shared__ float3 collected_SvTv[BLOCK_SIZE];
	__shared__ float3 collected_pk[BLOCK_SIZE];
	__shared__ uint32_t collected_ap_level[BLOCK_SIZE];
	__shared__ float2 collected_shapes[BLOCK_SIZE];  // Kernel shape: .x = primary (beta/general/flex), .y = nexel gamma_y
	__shared__ bool collected_is_textured[BLOCK_SIZE];  // `--method mixed` per-Gauss flag
	__shared__ float4 collected_ewa_conic[BLOCK_SIZE];  // `--method mixed_3d` EWA conic

	// Shared memory for per-Gaussian baseline features (dual hashgrid mode)
	// NOTE: Disabled for baseline_double/baseline_blend_double due to shared memory limits
	// We query on-demand instead (less efficient but fits in shared memory)
	// __shared__ float collected_feat_pk[BLOCK_SIZE][6 * 4];  // 6 levels × 4 features per Gaussian

	// Shared memory cache for MLP weights (FP16, all [16×16], 1.5KB total)
	// Loaded once per tile, eliminates global memory reads in mlp_forward_fused
	__shared__ __half smem_mlp_W1[W1_SIZE];   // 16*16 = 512 bytes
	__shared__ __half smem_mlp_W2[W2_SIZE];   // 16*16 = 512 bytes
	__shared__ __half smem_mlp_W3[W3_SIZE];   // 16*16 = 512 bytes

	// WMMA forward buffers (case 5 batched MLP) — dynamic shared memory
	extern __shared__ char fw_dynamic_smem[];
	__half* smem_fw_half  = reinterpret_cast<__half*>(fw_dynamic_smem);                                    // 256*16*2 = 8,192 bytes
	float*  smem_fw_float = reinterpret_cast<float*>(fw_dynamic_smem + TC_BATCH * TC_INPUT_DIM * sizeof(__half));  // 256*16*4 = 16,384 bytes

	// Cooperatively load MLP weights into shared memory (mode 5 and 6)
	if (((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) && mlp_W1 != nullptr) {
		const int tid = block.thread_rank();
		for (int idx = tid; idx < W1_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W1[idx] = mlp_W1[idx];
		for (int idx = tid; idx < W2_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W2[idx] = mlp_W2[idx];
		for (int idx = tid; idx < W3_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W3[idx] = mlp_W3[idx];
		block.sync();
	}

	// Initialize helper variables
	float T = 1.0f;
	// `--method mixed_3d`: A_tex is the explicit running sum of textured
	// weights w_j = alpha_j * T_full_j (so the 2DGS distortion identity
	// L_dist = Σ w_i (m_i² A − 2 m_i M1 + M2) holds with A, M1, M2 all
	// summed over the same textured subset). Initialised to 0 here; the
	// EWA branch never touches it; the textured branch accumulates inside
	// the distortion block alongside M1/M2. Persisted to final_T slot 3
	// for the backward to read directly (no 1 − x flip).
	float A_tex = 0.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float SH_RGB[3] = { 0 };  // Separate accumulator for residual_hybrid SH RGB (render_mode==11)
	uint32_t render_number = 0;
	float vis_appearance[3] = {0};
	float overdraw_sum = 0.0f;  // Soft contributor count: sum of sigmoid(k*(w-t))
	float w_square_sum = 0.0f;  // Sum of squared weights: sum(w_i^2) for weight_reg
	float beta_sum    = 0.0f;   // sum_i w_i * beta_i — per-pixel shape reg target (--w_lambda_perpix) loss

#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

	// Max contributor tracking for depth reinit
	float max_w = 0.0f;
	float max_depth = 0.0f;
	int max_idx = -1;  // global Gaussian id of the max-weight contributor (or -1 if none)

	int collec_offsets[16] = {0};
	// float feat[CHANNELS] = { 0 };
	float voxel_min = 0.0f;
	float voxel_max = 0.0f;
	float pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
	if(level > 0){
		// For cat mode (render_mode==1), 'level' is encoded as:
		// (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
		int hashgrid_levels = level;  // Default: use level as-is (baseline mode)

		if(render_mode == 3){
			// 3D mode: level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			// Hashgrid query happens in PyTorch, not CUDA - so hashgrid_levels = 0
			hashgrid_levels = 0;
		} else if((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6){
			// 3D_SH_res / 3D_SH_cat mode: level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			// Hash query happens in CUDA kernel (like cat mode), so use active_hashgrid_levels
			int active_hashgrid_levels = (level >> 8) & 0xFF;
			hashgrid_levels = active_hashgrid_levels;
		} else if(level > 16){
			printf("Error: level %d > 16.", level);
			return;
		}

		// Copy offsets for hashgrid query (up to max 16 levels)
		for(int l = 0; l <= hashgrid_levels && l < 17; l++) collec_offsets[l] = level_offsets[l];
		voxel_min = gridrange[0];
		voxel_max = gridrange[1];
	}


#endif

	// Cache per-pixel view direction encoding ONCE before Gaussian loop (mode 5)
	// viewdirs_enc is per-pixel (same for all Gaussians at this pixel), so avoid
	// re-reading 16 floats from global memory per intersection
	float cached_view_enc[16] = {0};
	bool has_cached_view_enc = false;
	if (inside && viewdirs_enc != nullptr && render_mode == 5) {
		has_cached_view_enc = true;
		for (int vi = 0; vi < 16; vi++)
			cached_view_enc[vi] = viewdirs_enc[pix_id * 16 + vi];
	}

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};

			collected_size[block.thread_rank()].x = scales[coll_id].x;
			collected_size[block.thread_rank()].y = scales[coll_id].y;

			// from 2dgs eq.(5)
			if(homotrans != nullptr){
				collected_SuTu[block.thread_rank()] = {homotrans[16 * coll_id+0], homotrans[16 * coll_id+4], homotrans[16 * coll_id+8]};
				collected_SvTv[block.thread_rank()] = {homotrans[16 * coll_id+1], homotrans[16 * coll_id+5], homotrans[16 * coll_id+9]};
				collected_pk[block.thread_rank()] = {homotrans[16 * coll_id+3], homotrans[16 * coll_id+7], homotrans[16 * coll_id+11]};
			}
		if(ap_level != nullptr){
			collected_ap_level[block.thread_rank()] = floorf(ap_level[coll_id]);
		}
		// Collect shape for kernel (beta/general/flex: 1 float; nexel: 2 floats)
		if(shapes != nullptr){
			if (kernel_type == 5) {
				// Nexel: load [gamma_x, gamma_y] from [N, 2] tensor
				collected_shapes[block.thread_rank()] = {shapes[coll_id * 2], shapes[coll_id * 2 + 1]};
			} else {
				// Other kernels: single float per Gaussian
				collected_shapes[block.thread_rank()] = {shapes[coll_id], 0.0f};
			}
		}
		// `--method mixed`: pull per-Gauss bool flag. Default to true (textured) when null.
		collected_is_textured[block.thread_rank()] = (is_textured == nullptr) ? true : is_textured[coll_id];
		collected_ewa_conic[block.thread_rank()] = (ewa_conic == nullptr) ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : ewa_conic[coll_id];

		// NOTE: Per-Gaussian feature caching disabled due to shared memory limits
		// Features are now queried on-demand in the per-pixel loop (cases 4, 5, 12)

	}
		block.sync();

		// ================================================================
		// Case 5 WMMA forward: Tensor core batched MLP for 3D_SH_res
		// All 256 threads process each Gaussian together:
		//   Phase 1: per-thread alpha/weight + xyz + hash query + SH load
		//   Phase 2: write MLP input [hash(hash_dim)|pad] to smem (zeros if inactive)
		//   Phase 3: wmma_forward_all() — tensor core batched MLP
		//   Phase 4: read result, feat = SH + ReLU(residual), accumulate
		// ================================================================
		if (false && (render_mode & 0xFF) == 5) {
		  const int tid = block.thread_rank();

		  for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		  {
			bool active = !done;
			float my_w = 0.0f;
			float my_test_T = T;
			float2 my_s = {0, 0};
			float my_rho3d = 0, my_rho2d = 0, my_depth = 0, my_alpha = 0;
			float my_normal[3] = {0, 0, 0};
			float my_sh_color[3] = {0, 0, 0};

			// Phase 1: Geometric + alpha + xyz + hash query + SH load
			do {
				if (!active) break;

				contributor++;
				if (max_intersections > 0 && contributor >= max_intersections) {
					done = true; active = false; break;
				}

				const float2 xy = collected_xy[j];
				const float3 Tu = collected_Tu[j];
				const float3 Tv = collected_Tv[j];
				const float3 Tw = collected_Tw[j];
				float3 k = pix.x * Tw - Tu;
				float3 l = pix.y * Tw - Tv;
				float3 p = cross(k, l);
				if (p.z == 0.0) { active = false; break; }

				my_s = {p.x / p.z, p.y / p.z};
				my_rho3d = my_s.x * my_s.x + my_s.y * my_s.y;
				float2 d = {xy.x - pixf.x, xy.y - pixf.y};
				my_rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
				float rho = min(my_rho3d, my_rho2d);

				my_depth = (my_rho3d <= my_rho2d) ?
					(my_s.x * Tw.x + my_s.y * Tw.y) + Tw.z : Tw.z;
				if (my_depth < near_n) { active = false; break; }

				float4 nor_o = collected_normal_opacity[j];
				my_normal[0] = nor_o.x; my_normal[1] = nor_o.y; my_normal[2] = nor_o.z;
				float opa = nor_o.w;

				if (kernel_type == 1 || kernel_type == 4) {
					float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
					float shape = collected_shapes[j].x;
					if (my_rho3d >= k_sq + 1e-6f) { active = false; break; }
					float base = fmaxf(0.0f, 1.0f - my_rho3d / k_sq);
					float alpha_beta = powf(base, shape);
					float alpha_lp = expf(-my_rho2d / 2.0f);
					my_alpha = fminf(0.99f, opa * fmaxf(alpha_beta, alpha_lp));
				} else if (kernel_type == 2) {
					float power = -0.5f * rho;
					if (power > 0.0f) { active = false; break; }
					float G = exp(power);
					float per_gaussian_beta = collected_shapes[j].x;
					if (per_gaussian_beta > 0.0f)
						G = (1.0f + per_gaussian_beta) * G / (1.0f + per_gaussian_beta * G);
					my_alpha = min(0.99f, opa * G);
				} else if (kernel_type == 3) {
					float beta_param = collected_shapes[j].x;
					float rho_safe = fmaxf(rho, 1e-8f);
					float pow_term = powf(rho_safe, 0.5f * beta_param);
					float power = -0.5f * pow_term;
					if (power > 0.0f) { active = false; break; }
					my_alpha = min(0.99f, opa * expf(power));
				} else if (kernel_type == 5) {
					// Nexel kernel (WMMA path)
					float gamma_x = collected_shapes[j].x;
					float gamma_y = collected_shapes[j].y;
					const float GAMMA_EPS = 1e-6f;
					float comp_x = fminf(my_s.x * my_s.x + GAMMA_EPS, powf(1000.0f, 1.0f / gamma_x));
					float comp_y = fminf(my_s.y * my_s.y + GAMMA_EPS, powf(1000.0f, 1.0f / gamma_y));
					float power = -0.5f * (powf(comp_x, gamma_x) + powf(comp_y, gamma_y));
					if (power > 0.0f) { active = false; break; }
					my_alpha = min(0.99f, opa * expf(power));
				} else {
					float power = -0.5f * rho;
					if (power > 0.0f) { active = false; break; }
					float G = exp(power);
					if (beta > 0.0)
						G = (1.0 + beta) * G / (1.0 + beta * G);
					my_alpha = min(0.99f, opa * G);
				}

				if (my_alpha < 1.0f / 255.0f) { active = false; break; }

				// FastGS VCD/VCP: count high-error-pixel hits per Gaussian.
				if (fastgs_count && inside && metric_map[pix_id] == 1) {
					atomicAdd(&metric_counts[collected_id[j]], 1);
				}

				float test_T = T * (1 - my_alpha);
				if (test_T < 0.0001f) { done = true; active = false; break; }

				my_w = my_alpha * T;
				my_test_T = test_T;
				render_number++;

#if RENDER_AXUTILITY
				// Track max contributor per pixel (id used by mini depth-reinit SH transfer)
				if (my_w > max_w) {
					max_w = my_w;
					max_depth = my_depth;
					max_idx = collected_id[j];
				}

				float A = 1-T;
				float m = far_n / (far_n - near_n) * (1 - near_n / my_depth);
				distortion += (m * m * A + M2 - 2 * m * M1) * my_w;
				D += my_depth * my_w;
				M1 += m * my_w;
				M2 += m * m * my_w;
				if (T > 0.5) {
					median_depth = my_depth;
					median_contributor = contributor;
				}
				for (int ch = 0; ch < 3; ch++) N[ch] += my_normal[ch] * my_w;
#endif
			} while(0);

			// Phase 2: Compute hash features + SH, write MLP input to shared memory
			if (active) {
				// Compute xyz intersection
				const float3 pk = collected_pk[j];
				float3 xyz;
				if (my_rho3d <= my_rho2d) {
					const float3 sutu = collected_SuTu[j];
					const float3 svtv = collected_SvTv[j];
					xyz = {my_s.x * sutu.x + my_s.y * svtv.x + pk.x,
					       my_s.x * sutu.y + my_s.y * svtv.y + pk.y,
					       my_s.x * sutu.z + my_s.y * svtv.z + pk.z};
				} else {
					xyz = pk;
				}

				// Load SH base color
				int gauss_id = collected_id[j];
				for (int ch = 0; ch < 3; ch++)
					my_sh_color[ch] = RGB_TO_FLOAT(rgb[gauss_id * 3 + ch]);

				// Query hashgrid (up to 16D for 4 levels × 4D)
				const int active_hashgrid_levels = (level >> 8) & 0xFF;
				const int hash_dim_batched = active_hashgrid_levels * l_dim;
				uint32_t my_ap_level = collected_ap_level[j];
				float hash_feat[16] = {0};
				if (active_hashgrid_levels > 0 && l_dim == 4) {
					if (hash_dim_batched == 4)
						query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 8)
						query_feature<false, 8, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 12)
						query_feature<false, 12, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 16)
						query_feature<false, 16, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
				} else if (active_hashgrid_levels > 0 && l_dim == 2) {
					// 2D per level — supports 1..8 hash levels (hash_dim ∈ {2,4,6,8,10,12,14,16}).
					if (hash_dim_batched == 2)
						query_feature<false, 2, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 4)
						query_feature<false, 4, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 6)
						query_feature<false, 6, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 8)
						query_feature<false, 8, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 10)
						query_feature<false, 10, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 12)
						query_feature<false, 12, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 14)
						query_feature<false, 14, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
					else if (hash_dim_batched == 16)
						query_feature<false, 16, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							my_ap_level, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false,
							nullptr, nullptr, nullptr, my_depth);
				}

				// Write MLP input: [hash(hash_dim) | pad(16-hash_dim)] = 16D
				for (int i = 0; i < TC_INPUT_DIM; i++)
					smem_fw_half[tid * TC_INPUT_DIM + i] = __half(0);
				for (int i = 0; i < hash_dim_batched && i < TC_INPUT_DIM; i++)
					smem_fw_half[tid * TC_INPUT_DIM + i] = __float2half(hash_feat[i]);
			} else {
				// Inactive: write zeros (bias-free MLP: W@0=0, contributes nothing)
				for (int i = 0; i < TC_INPUT_DIM; i++)
					smem_fw_half[tid * TC_INPUT_DIM + i] = __half(0);
			}
			__syncthreads();

			// Phase 3: WMMA batched MLP forward (all 8 warps, tensor cores)
			wmma_forward_all(smem_fw_half, smem_fw_float,
				smem_mlp_W1, smem_mlp_W2, smem_mlp_W3);

			// Phase 4: Read MLP result, combine with SH, accumulate.
			// Activation is selected by d_residual_mode (see top of file):
			//   0 (3D_SH_res): feat = ReLU(ReLU(SH+sh_bias) + residual + res_bias)
			//   1 (3D_SH_add): feat = ReLU(SH+sh_bias) + ReLU(residual + res_bias)
			if (active) {
				for (int ch = 0; ch < ORIG_OUTPUT_DIM; ch++) {
					float residual = smem_fw_float[tid * TC_OUTPUT_DIM + ch];
					float feat_ch;
					if (d_residual_mode == 1) {
						// my_sh_color is already ReLU(SH+sh_bias) from computeColorFromSH.
						feat_ch = my_sh_color[ch] + fmaxf(0.0f, residual + d_res_bias);
					} else if (d_residual_mode == 2) {
						// mixed: signed residual, no per-Gauss ReLU.
						feat_ch = my_sh_color[ch] + residual + d_res_bias;
					} else {
						feat_ch = fmaxf(0.0f, my_sh_color[ch] + residual + d_res_bias);
					}
					C[ch] += feat_ch * my_w;
				}

				if (record_transmittance) {
					atomicAdd(&(cover_pixel[collected_id[j]]), 1.0f);
					atomicAdd(&(trans_avg[collected_id[j]]), my_w);
				}

				T = my_test_T;
				last_contributor = contributor;
			}
		  }  // end WMMA inner loop
		} else
		// Iterate over current batch (scalar path for non-mode-5)
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Early termination based on total evaluations (for benchmarking kernel overhead)
			// This caps BEFORE kernel computation to fix number of powf/exp calls
			if (max_intersections > 0 && contributor >= max_intersections)
			{
				done = true;
				continue;
			}

			// `--method mixed`: untextured Gaussians use the SAME 2DGS ray-splat
			// geometry as textured/2DGS (the splat shape IS the rho3d ray-plane
			// intersection — rho2d alone is only a fixed ~1px anti-alias floor
			// and carries no size/orientation). They differ ONLY in the color
			// path (skip hash query + MLP residual; use the SV/SH baseline).
			const bool tex_fwd = collected_is_textured[j];

			// ============================================================
			// `--method mixed_3d` — UNTEXTURED EWA 3D-ellipsoid fast path.
			// Shape comes from the precomputed FastGS conic (NOT the 2DGS
			// ray-splat). Fully self-contained: no Tu/Tv/Tw, no hash, no
			// MLP. Color = SH/SV baseline from `rgb`. The alpha math below
			// is byte-identical to FastGS renderCUDA so untextured output
			// matches FastGS on identical inputs. `continue` past all the
			// textured 2DGS code.
			// ============================================================
			if (!tex_fwd && ewa_conic != nullptr) {
				const float4 con_o = collected_ewa_conic[j];
				if (con_o.w <= 0.0f) continue;  // culled in preprocess
				const float2 xy_e = collected_xy[j];
				const float2 dd = { xy_e.x - pixf.x, xy_e.y - pixf.y };
				// Squared Mahalanobis radius in pixel space (EWA "r²"); the
				// conic already carries the eps2d=0.3 low-pass.
				const float m = con_o.x * dd.x * dd.x
				              + 2.0f * con_o.y * dd.x * dd.y
				              + con_o.z * dd.y * dd.y;
				float alpha_e;
				// Untextured surfels honour `--kernel2` if given, else `--kernel`
				// (the 2DGS textured half always uses kernel_type). `--kernel2`
				// rides render_mode bits [16..19] = (kernel_type2+1); a zero
				// nibble means unset → untextured fall back to kernel_type.
				// beta / beta_scaled → compact restricted-beta falloff (β =
				// activated get_shape ∈ [0,5]); anything else → FastGS Gaussian.
				const int _utk_n = (render_mode >> 16) & 0xF;
				const int ut_kt = _utk_n ? (_utk_n - 1) : kernel_type;
				if (ut_kt == 1 || ut_kt == 4) {
					const float k_sq = (ut_kt == 4) ? 9.0f : 1.0f;
					if (m >= k_sq + 1e-6f) continue;          // compact support
					const float beta_s = collected_shapes[j].x;
					const float base = fmaxf(0.0f, 1.0f - m / k_sq);
					alpha_e = fminf(0.99f, con_o.w * powf(base, beta_s));
				} else {
					const float power = -0.5f * m;
					if (power > 0.0f) continue;               // m < 0 (degenerate)
					alpha_e = min(0.99f, con_o.w * exp(power));
				}
				if (alpha_e < 1.0f / 255.0f) continue;

				if (fastgs_count && inside && metric_map[pix_id] == 1)
					atomicAdd(&metric_counts[collected_id[j]], 1);

				const float test_T_e = T * (1 - alpha_e);
				if (test_T_e < 0.0001f) { done = true; continue; }
				const float w_e = alpha_e * T;
				render_number++;

				// --method mixed/mixed_3d: overdraw metric is textured-only.
				// Untextured EWA / specular surfels are by design a layered
				// set; penalizing their pile-up is counterproductive. The
				// metric we expose to Python (render_overdraw) and the
				// CUDA-side --overdraw_reg gradient both follow this gate.
				// (w_square_sum / --weight_reg stays unmasked.)
				w_square_sum += w_e * w_e;
				// untextured: no kernel shape → no beta_sum term, no overdraw contribution

				const int gid_e = collected_id[j];
				const float depth_e = depths[gid_e];
#if RENDER_AXUTILITY
				if (w_e > max_w) { max_w = w_e; max_depth = depth_e; max_idx = gid_e; }
				{
					// `--method mixed_3d`: untextured EWA Gaussians do NOT contribute to
					// the per-pixel depth-distortion accumulator (distortion, M1, M2,
					// A_tex). Distortion stays tex-only at the gradient level — the
					// backward EWA branches mirror by leaving dL_dweight=0.
					// Depth accumulator D and median_depth DO get the EWA contribution
					// because the depth-derived normal (used by --lambda_normal) needs
					// the full rendered depth, and downstream consumers (TSDF mesh
					// extraction) also expect the full-visibility depth map.
					D  += depth_e * w_e;
					if (T > 0.5f) { median_depth = depth_e; median_contributor = contributor; }
					// untextured 3D ellipsoid: no 2DGS surfel normal → 0 normal contribution
				}
#endif
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += RGB_TO_FLOAT(rgb[gid_e * CHANNELS + ch]) * w_e;

				if (record_transmittance) {
					atomicAdd(&(cover_pixel[gid_e]), 1.0f);
					atomicAdd(&(trans_avg[gid_e]), w_e);
				}

				T = test_T_e;
				last_contributor = contributor;
				continue;
			}

			// Fisrt compute two homogeneous planes, See Eq. (8) — 2DGS, both halves.
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

		float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
		if (depth < near_n) continue;
		float4 nor_o = collected_normal_opacity[j];
		float normal[3] = {nor_o.x, nor_o.y, nor_o.z};  // Already normalized in preprocessing
		float opa = nor_o.w;

		float alpha;
		// `--method mixed`: untextured surfels use the run's chosen kernel
		// (beta/general/flex/nexel/Gaussian) — identical kernel + geometry to
		// textured; they differ ONLY in the color path (skip hash/MLP).
		if (kernel_type == 1 || kernel_type == 4) {
			// Beta kernel with separate G_obj (Beta) and G_screen (Gaussian low-pass)
			// kernel_type 1: k²=1 (unit circle cutoff)
			// kernel_type 4: k²=9 (3σ scaled, matches Gaussian extent)
			float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
			float shape = collected_shapes[j].x;

			// 1. Hard support check on object-space distance (with epsilon for numerical safety)
			if (rho3d >= k_sq + 1e-6f)
				continue;  // Outside compact support - skip entirely

			// 2. Object-space Beta kernel (geometry)
			float base = fmaxf(0.0f, 1.0f - rho3d / k_sq);
			float alpha_beta = powf(base, shape);

			// 3. Screen-space Gaussian low-pass (anti-aliasing)
			// Gaussian: exp(-rho2d / 2) where rho2d = FilterInvSquare * d^2
			float alpha_lp = expf(-rho2d / 2.0f);

			// 4. Max-pool handoff: smooth transition between Beta (close) and Gaussian (far)
			float kernel_val = fmaxf(alpha_beta, alpha_lp);

			// 5. Final alpha with opacity
			alpha = fminf(0.99f, opa * kernel_val);
		} else if (kernel_type == 2) {
			// Flex kernel: Standard Gaussian with per-Gaussian learnable beta
			// Same formula as Gaussian but beta comes from shapes array instead of global config
			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			float G = exp(power);
			float per_gaussian_beta = collected_shapes[j].x;  // shapes array holds per-Gaussian beta
			if (per_gaussian_beta > 0.0f)
				G = (1.0f + per_gaussian_beta) * G / (1.0f + per_gaussian_beta * G);

			alpha = min(0.99f, opa * G);
		} else if (kernel_type == 3) {
			// General kernel: Isotropic Generalized Gaussian
			// Formula: G = exp(-0.5 * (r²)^(β/2))
			// β = 2.0: standard Gaussian, β = 8.0: super-Gaussian (box-like)
			float beta_param = collected_shapes[j].x;  // shapes array holds beta in range [2.0, 8.0]

#ifdef FAST_POW_TEST
			// PERFORMANCE TEST: Bypass expensive powf
			// Forces beta=2 behavior (standard Gaussian) to measure powf overhead
			// If FPS increases significantly, powf is the bottleneck
			(void)beta_param;  // Suppress unused variable warning
			float pow_term = rho;  // Equivalent to rho^1 (beta=2 gives exponent=1)
#else
			float exponent = 0.5f * beta_param;  // β/2
			// Avoid numerical issues with rho=0
			float rho_safe = fmaxf(rho, 1e-8f);
			float pow_term = powf(rho_safe, exponent);  // (r²)^(β/2)
#endif

			float power = -0.5f * pow_term;

			if (power > 0.0f)
				continue;

			float G = expf(power);
			alpha = min(0.99f, opa * G);
		} else if (kernel_type == 5) {
			// Nexel kernel: per-axis gamma exponents (anisotropic generalized Gaussian)
			// G = exp(-0.5 * (pow(s_x² + eps, gamma_x) + pow(s_y² + eps, gamma_y)))
			// gamma=1: standard Gaussian. gamma>1: softer. gamma<1: sharper (but activation prevents <1).
			float gamma_x = collected_shapes[j].x;
			float gamma_y = collected_shapes[j].y;
			float sx2 = s.x * s.x;
			float sy2 = s.y * s.y;
			const float GAMMA_EPS = 1e-6f;
			float comp_x = fminf(sx2 + GAMMA_EPS, powf(1000.0f, 1.0f / gamma_x));
			float comp_y = fminf(sy2 + GAMMA_EPS, powf(1000.0f, 1.0f / gamma_y));
			float power = -0.5f * (powf(comp_x, gamma_x) + powf(comp_y, gamma_y));
			if (power > 0.0f)
				continue;
			float G = expf(power);
			alpha = min(0.99f, opa * G);
		} else if (d_aa_kernel_size > 0.0f) {
			// AA-2DGS Jacobian-based mip filter (replaces rho3d/rho2d heuristic).
			// Σ'_local = I + ks·J·Jᵀ, alpha = coef·opa·exp(-0.5·rho_new)
			const float ks = d_aa_kernel_size;
			const float k_sq_aa = ks * ks;
			const float pz_inv = 1.0f / p.z;
			const float pz_sq_inv = pz_inv * pz_inv;
			const float3 dp_dx = cross(Tv, Tw);
			const float3 dp_dy = cross(Tw, Tu);
			const float J_a = (dp_dx.x * p.z - p.x * dp_dx.z) * pz_sq_inv;
			const float J_b = (dp_dx.y * p.z - p.y * dp_dx.z) * pz_sq_inv;
			const float J_c = (dp_dy.x * p.z - p.x * dp_dy.z) * pz_sq_inv;
			const float J_d = (dp_dy.y * p.z - p.y * dp_dy.z) * pz_sq_inv;
			const float det_J = J_a * J_d - J_b * J_c;
			const float trace_JJT = J_a * J_a + J_b * J_b + J_c * J_c + J_d * J_d;
			const float det_V = k_sq_aa * det_J * det_J + ks * trace_JJT + 1.0f;
			if (fabsf(det_V) < 1e-8f) continue;
			const float det_V_inv = 1.0f / det_V;
			const float coef = sqrtf(det_V_inv + 1e-8f);
			const float term1 = J_d * s.x - J_c * s.y;
			const float term2 = J_a * s.y - J_b * s.x;
			const float rho_aa_num = (s.x * s.x + s.y * s.y) + ks * (term1 * term1 + term2 * term2);
			const float rho_aa = rho_aa_num * det_V_inv;
			const float power_aa = -0.5f * rho_aa;
			if (power_aa > 0.0f) continue;
			alpha = fminf(0.99f, coef * opa * expf(power_aa));
		} else {
			// Standard Gaussian kernel
			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float G = exp(power);

			if(beta > 0.0)
				G = (1.0 + beta) * G / (1.0 + beta * G);

			alpha = min(0.99f, opa * G);
		}

		if (alpha < 1.0f / 255.0f)
			continue;

		// FastGS VCD/VCP: count high-error-pixel hits per Gaussian.
		if (fastgs_count && inside && metric_map[pix_id] == 1) {
			atomicAdd(&metric_counts[collected_id[j]], 1);
		}

		float test_T = T * (1 - alpha);
		if (test_T < 0.0001f)
		{
			done = true;
			continue;
		}

		float w = alpha * T;

		render_number++;

		// Soft contributor count: sigmoid(k*(w - 1/255)) per Gaussian.
		// Always computed for monitoring; gradient only when overdraw_lambda > 0.
		// --method mixed/mixed_3d: gate on `tex_fwd` so the overdraw metric +
		// gradient are textured-only (untextured = specular layer is expected
		// to overlap by design). When `is_textured == nullptr` (3D_SH_res
		// etc.), `collected_is_textured` defaults to true → byte-identical.
		if (tex_fwd) {
			const float OD_K = 10.0f;
			const float OD_THRESH = 1.0f / 255.0f;
			float sig = 1.0f / (1.0f + expf(-OD_K * (w - OD_THRESH)));
			overdraw_sum += sig;
		}
		// Weight squared accumulation for weight_reg loss: sum(w_i^2)
		// Ideal surface: one Gaussian with w=1 → sum=1. Overdraw: multiple w<1 → sum<1.
		w_square_sum += w * w;

		// Per-pixel beta-mass accumulator: sum_i w_i * beta_i.
		// Drives the --w_lambda_perpix shape reg (Python loss: (w_r * beta_sum).mean()).
		// Note: we use collected_shapes[j].x (the primary shape — beta for beta/beta_scaled/
		// general/flex kernels). Always computed (cheap); only gates gradient via the
		// Python autograd path. Beta kernels only — for scalar Gaussian kernels shape=0.
		beta_sum += w * collected_shapes[j].x;

		// NOTE: max_intersections check moved earlier (before kernel computation)
		// to cap total evaluations for benchmarking, not just valid intersections

#if RENDER_AXUTILITY
			// Track max contributor per pixel (id used by mini depth-reinit SH transfer)
			if (w > max_w) {
				max_w = w;
				max_depth = depth;
				max_idx = collected_id[j];
			}

			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			// `--method mixed_3d`: A_tex is the explicit textured-only running
			// sum of w (NOT 1 - T_tex). With interleaved EWA absorbers, the
			// identity 1 - Π(1-α) = Σ α·T only holds when every absorber in the
			// chain contributes to A; since EWA Gaussians DON'T contribute, we
			// must track A_tex by explicit accumulation to keep the distortion
			// identity self-consistent across A, M1, M2 (all over textured subset).
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A_tex + M2 - 2 * m * M1) * w;
			D  += depth * w;
			A_tex += w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}

			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif
			
			//// now color part is in ingp model
			// Eq. (3) from 3D Gaussian splatting paper.
			// MyGs, now color calculation is in ngp part.

			// Special handling for adaptive_cat_fast (mode 13): check weight BEFORE 3D intersection
			// This allows skipping expensive intersection computation for Gaussian-only primitives.
			// NOTE: This path overloads `rgb` as a variable-width [N, total_dim+1] feature buffer
			// (not [N,3] color). Incompatible with FP16_RGB (which assumes [N,3] colors). Gated out
			// when FP16_RGB=1 — only modes 1/5/6 are supported in FP16-rgb mode.
			const int base_mode = render_mode & 0xFF;
#if !FP16_RGB
			if (base_mode == 13) {
				// adaptive_cat_fast: Skip 3D intersection for Gaussian-only primitives
				const bool use_inference = (render_mode >> 8) & 0x1;
				const int total_levels = (level >> 16) & 0xFF;
				const int hashgrid_levels = (level >> 8) & 0xFF;
				const int hybrid_levels = level & 0xFF;
				const int per_level_dim = l_dim;
				const int total_dim = total_levels * per_level_dim;

				int gauss_id = collected_id[j];
				const float* gauss_feat = &rgb[gauss_id * (total_dim + 1)];
				const float weight = gauss_feat[total_dim];

				float feat[CHANNELS];
				for(int i = 0; i < CHANNELS; i++) feat[i] = 0.0f;

				if (use_inference && weight > 0.5f) {
					// FAST PATH: Gaussian-only, skip 3D intersection entirely
					// Just copy per-Gaussian features, no hash query needed
					for(int i = 0; i < total_dim && i < CHANNELS; i++) {
						feat[i] = gauss_feat[i];
					}
				} else {
					// SLOW PATH: Need 3D intersection for hash query
					const float3 pk = collected_pk[j];
					float3 xyz;
					if(rho3d <= rho2d){
						const float3 sutu = collected_SuTu[j];
						const float3 svtv = collected_SvTv[j];
						xyz = {s.x * sutu.x + s.y * svtv.x + pk.x,
							s.x * sutu.y + s.y * svtv.y + pk.y,
							s.x * sutu.z + s.y * svtv.z + pk.z};
					}
					else xyz = pk;

					// Accumulate weighted position (for visualization)
					pos_x += w * xyz.x;
					pos_y += w * xyz.y;
					pos_z += w * xyz.z;

					uint32_t appearance_level = collected_ap_level[j];
					bool contract = if_contract;
					bool debug = false;

					if (use_inference) {
						// Inference: Use hashgrid for fine levels, Gaussian for coarse
						for(int i = 0; i < hybrid_levels * per_level_dim && i < CHANNELS; i++) {
							feat[i] = gauss_feat[i];
						}

						if (hashgrid_levels > 0) {
							float hash_feat[16 * 4];
							float voxel_min = gridrange[0];
							float voxel_max = gridrange[1];
							int collec_offsets[17];
							for(int lv = 0; lv <= hashgrid_levels; lv++){
								collec_offsets[lv] = level_offsets[lv];
							}

							if(l_dim == 4) {
								query_feature<false, 16*4, 4>(hash_feat, xyz, voxel_min, voxel_max,
								                               collec_offsets, appearance_level, hash_features,
								                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
							} else if(l_dim == 2) {
								query_feature<false, 16*4, 2>(hash_feat, xyz, voxel_min, voxel_max,
								                               collec_offsets, appearance_level, hash_features,
								                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
							} else if(l_dim == 8) {
								query_feature<false, 16*4, 8>(hash_feat, xyz, voxel_min, voxel_max,
								                               collec_offsets, appearance_level, hash_features,
								                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
							}

							const int fine_start = hybrid_levels * per_level_dim;
							for(int i = 0; i < hashgrid_levels * per_level_dim && (fine_start + i) < CHANNELS; i++) {
								feat[fine_start + i] = hash_feat[i];
							}
						}
					} else {
						// Training: Smooth blending
						for(int i = 0; i < hybrid_levels * per_level_dim && i < CHANNELS; i++) {
							feat[i] = gauss_feat[i] * weight;
						}

						if (hashgrid_levels > 0) {
							float hash_feat[16 * 4];
							float voxel_min = gridrange[0];
							float voxel_max = gridrange[1];
							int collec_offsets[17];
							for(int lv = 0; lv <= hashgrid_levels; lv++){
								collec_offsets[lv] = level_offsets[lv];
							}

							if(l_dim == 4) {
								query_feature<false, 16*4, 4>(hash_feat, xyz, voxel_min, voxel_max,
								                               collec_offsets, appearance_level, hash_features,
								                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
							} else if(l_dim == 2) {
								query_feature<false, 16*4, 2>(hash_feat, xyz, voxel_min, voxel_max,
								                               collec_offsets, appearance_level, hash_features,
								                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
							} else if(l_dim == 8) {
								query_feature<false, 16*4, 8>(hash_feat, xyz, voxel_min, voxel_max,
								                               collec_offsets, appearance_level, hash_features,
								                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
							}

							const int fine_start = hybrid_levels * per_level_dim;
							for(int i = 0; i < hashgrid_levels * per_level_dim && (fine_start + i) < CHANNELS; i++) {
								feat[fine_start + i] = weight * gauss_feat[fine_start + i]
								                     + (1.0f - weight) * hash_feat[i];
							}
						}
					}
				}

				// Accumulate features
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += feat[ch] * w;

				// Skip the rest of the normal processing
				T = T * (1 - alpha);
				continue;
			}
#endif  // !FP16_RGB (mode 13 adaptive_cat_fast)

			if(level == 0){
				// SH-only fallback. Read from FP16 rgb (preprocessed) for the
				// hashgrid-mode paths; features (colors_precomp) is FP32 for cat.
				if (rgb != nullptr) {
					for (int ch = 0; ch < CHANNELS; ch++)
						C[ch] += RGB_TO_FLOAT(rgb[collected_id[j] * CHANNELS + ch]) * w;
				} else {
					for (int ch = 0; ch < CHANNELS; ch++)
						C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
				}
			}
			else{

				// `--method mixed` untextured surfels do NOT query the hashgrid, so
				// the world-space ray-splat intersection point (xyz) — needed ONLY
				// for the hash lookup — is pure waste for them. Reference 2DGS never
				// computes it either. Skip the reconstruction + the two shared loads
				// (collected_SuTu / collected_SvTv) + pos accumulation for untextured.
				float3 xyz = {0.0f, 0.0f, 0.0f};  // only read on the textured hash path
				if (tex_fwd) {
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

					pos_x += w * xyz.x;
					pos_y += w * xyz.y;
					pos_z += w * xyz.z;
				}

				bool debug = false;

				float feat[CHANNELS];
				uint32_t appearance_level = collected_ap_level[j];

				// bool contract = false;
				// bool contract = true;
				bool contract = if_contract;

				// hashgrid feature interpolation
				// Note: render_mode may have flags encoded in upper bits (e.g., inference flag for adaptive_cat)
				// Extract base mode for switch, keep full value for mode-specific flag extraction
			switch (render_mode & 0xFF){
		case 5: {
			/* 3D_SH_res mode: SH base color + hash MLP residual
			 *
			 * Pipeline:
			 *   1. Compute xyz intersection point
			 *   2. Load SH base color (from preprocessing, per-Gaussian)
			 *   3. Query hashgrid for fine features at xyz
			 *   4. Build MLP input: [hash(hash_dim) | pad(16-hash_dim)] = 16D
			 *   5. Run MLP → 3D RGB residual (identity activation, no sigmoid)
			 *   6. feat = SH_color + residual
			 */

			// `--method mixed` untextured fast path: simple 2DGS — per-Gauss SH
			// color only, no hash query, no MLP, no SV. The rgb buffer is the
			// FP16 SH-evaluated baseline (computed in preprocessCUDA). Backward
			// writes dL_dcolors[gid * 3 + ch], which BACKWARD::preprocess maps
			// back to dL_dsh.
			if (!tex_fwd) {
				int gid = collected_id[j];
				for (int ch = 0; ch < CHANNELS; ch++) feat[ch] = 0.0f;
				for (int ch = 0; ch < 3 && ch < CHANNELS; ch++)
					feat[ch] = RGB_TO_FLOAT(rgb[gid * 3 + ch]);
				break;
			}

			// 1. Load SH base color from preprocessing (rgb stores SH-evaluated 3D colors, FP16)
			int gauss_id = collected_id[j];
			float sh_color[3];
			for (int ch = 0; ch < 3; ch++)
				sh_color[ch] = RGB_TO_FLOAT(rgb[gauss_id * 3 + ch]);

			// Skip hash query when:
			// - contribution w = T*alpha is too small (tail pixels), or
			// - pixel has already processed count_thresh Gaussians (depth cutoff)
			bool skip_hash = (d_contrib_thresh > 0.0f && w < d_contrib_thresh)
			                 || (d_count_thresh > 0 && contributor > (uint32_t)d_count_thresh);

			// 0. Compute xyz intersection point
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

			// Decode level parameter: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			const int active_hashgrid_levels = (level >> 8) & 0xFF;

			// 2. Query hashgrid for fine features at xyz
			// hash_dim = active_hashgrid_levels * l_dim (up to 12D for 3 levels × 4D)
			const int hash_dim = active_hashgrid_levels * l_dim;
			float hash_feat[16];  // Max 4 levels × 4D = 16D (= TC_INPUT_DIM)
			for (int i = 0; i < 16; i++) hash_feat[i] = 0.0f;
			if (!skip_hash && active_hashgrid_levels > 0 && l_dim == 4) {
				if (hash_dim == 4) {
					query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 8) {
					query_feature<false, 8, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 12) {
					query_feature<false, 12, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 16) {
					query_feature<false, 16, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				}
			} else if (!skip_hash && active_hashgrid_levels > 0 && l_dim == 2) {
				// 2D per level — supports 1..8 hash levels (hash_dim ∈ {2,4,6,8,10,12,14,16}).
				if (hash_dim == 2) {
					query_feature<false, 2, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 4) {
					query_feature<false, 4, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 6) {
					query_feature<false, 6, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 8) {
					query_feature<false, 8, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 10) {
					query_feature<false, 10, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 12) {
					query_feature<false, 12, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 14) {
					query_feature<false, 14, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				} else if (hash_dim == 16) {
					query_feature<false, 16, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug,
					                           nullptr, nullptr, nullptr, depth);
				}
			}

			// 3. Build MLP input: [hash(hash_dim) | pad(16-hash_dim)] = 16D
			float mlp_input[TC_INPUT_DIM];  // TC_INPUT_DIM = 16
			for (int i = 0; i < TC_INPUT_DIM; i++) mlp_input[i] = 0.0f;
			for (int i = 0; i < hash_dim && i < TC_INPUT_DIM; i++) mlp_input[i] = hash_feat[i];

			// 4. Run MLP → RGB residual (identity activation, NO sigmoid)
			float h1[TC_HIDDEN_DIM], h2[TC_HIDDEN_DIM];
			float residual[ORIG_OUTPUT_DIM];
			mlp_forward_fused<TC_INPUT_DIM, TC_HIDDEN_DIM, ORIG_OUTPUT_DIM>(
				mlp_input, residual, h1, h2, false,  // false = identity (no sigmoid)
				smem_mlp_W1, smem_mlp_W2, smem_mlp_W3);

			// Activation: d_residual_mode selects:
			//   0 (3D_SH_res): feat = ReLU(ReLU(SH+sh_bias) + residual + res_bias)  — outer ReLU
			//   1 (3D_SH_add): feat = ReLU(SH+sh_bias) + ReLU(residual + res_bias)  — separate ReLUs
			//   2 (mixed):     feat = ReLU(SH+sh_bias) + (residual + res_bias)      — signed residual, no per-Gauss ReLU
			//                  Per-pixel ReLU on the final blended color is applied in Python (autograd).
			// sh_color is already ReLU(SH+sh_bias) from computeColorFromSH (inner ReLU).
			for (int ch = 0; ch < 3; ch++) {
				if (d_residual_mode == 1) {
					feat[ch] = sh_color[ch] + fmaxf(0.0f, residual[ch] + d_res_bias);
				} else if (d_residual_mode == 2) {
					feat[ch] = sh_color[ch] + residual[ch] + d_res_bias;
				} else {
					feat[ch] = fmaxf(0.0f, sh_color[ch] + residual[ch] + d_res_bias);
				}
			}

			break;
		}
		case 6: {
			/* 3D_SH_cat mode: SH base color + hash+DC MLP residual
			 * Same as case 5 but MLP input includes per-Gaussian DC SH for identity.
			 * features[] = DC SH (3D per Gaussian), rgb[] = full SH eval
			 */

			// 1. Load full SH base color from rgb (preprocessed in geomState)
			int gauss_id_6 = collected_id[j];
			float sh_color_6[3];
			for (int ch = 0; ch < 3; ch++)
				sh_color_6[ch] = RGB_TO_FLOAT(rgb[gauss_id_6 * 3 + ch]);

			// 2. Query hashgrid (same as case 5)
			const int active_hashgrid_levels_6 = (level >> 8) & 0xFF;
			const int hash_dim_6 = active_hashgrid_levels_6 * l_dim;
			bool skip_hash_6 = (d_contrib_thresh > 0.0f && w < d_contrib_thresh)
			                   || (d_count_thresh > 0 && contributor > (uint32_t)d_count_thresh);
			float hash_feat_6[12];
			for (int i = 0; i < 12; i++) hash_feat_6[i] = 0.0f;
			if (!skip_hash_6 && active_hashgrid_levels_6 > 0 && l_dim == 4) {
				// 4D per level — cat caps hash_dim at 12 (1..3 hash levels of 4D each).
				if (hash_dim_6 == 4) {
					query_feature<false, 4, 4>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else if (hash_dim_6 == 8) {
					query_feature<false, 8, 4>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else if (hash_dim_6 == 12) {
					query_feature<false, 12, 4>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				}
			} else if (!skip_hash_6 && active_hashgrid_levels_6 > 0 && l_dim == 2) {
				// 2D per level — cat caps hash_dim at 12 (1..6 hash levels of 2D each).
				if (hash_dim_6 == 2) {
					query_feature<false, 2, 2>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else if (hash_dim_6 == 4) {
					query_feature<false, 4, 2>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else if (hash_dim_6 == 6) {
					query_feature<false, 6, 2>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else if (hash_dim_6 == 8) {
					query_feature<false, 8, 2>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else if (hash_dim_6 == 10) {
					query_feature<false, 10, 2>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else if (hash_dim_6 == 12) {
					query_feature<false, 12, 2>(hash_feat_6, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels_6,
					                           l_scale, Base, align_corners, interp, contract, debug);
				}
			}

			// 3. Load DC SH from features[] (per-Gaussian identity, 3D)
			float dc_sh[3];
			for (int ch = 0; ch < 3; ch++)
				dc_sh[ch] = features[gauss_id_6 * 3 + ch];

			// 4. Build MLP input: [hash(hash_dim) | DC_SH(3) | bias(1) | pad] = 16D
			float mlp_input_6[TC_INPUT_DIM];
			for (int i = 0; i < TC_INPUT_DIM; i++) mlp_input_6[i] = 0.0f;
			int pos = 0;
			for (int i = 0; i < hash_dim_6 && pos < TC_INPUT_DIM; i++) mlp_input_6[pos++] = hash_feat_6[i];
			for (int i = 0; i < 3 && pos < TC_INPUT_DIM; i++) mlp_input_6[pos++] = dc_sh[i];
			if (pos < TC_INPUT_DIM) mlp_input_6[pos] = 1.0f;  // bias

			// 5. Run MLP
			float h1_6[TC_HIDDEN_DIM], h2_6[TC_HIDDEN_DIM];
			float residual_6[ORIG_OUTPUT_DIM];
			mlp_forward_fused<TC_INPUT_DIM, TC_HIDDEN_DIM, ORIG_OUTPUT_DIM>(
				mlp_input_6, residual_6, h1_6, h2_6, false,
				smem_mlp_W1, smem_mlp_W2, smem_mlp_W3);

			// Combo #6 (same as case 5): feat = ReLU( ReLU(SH+sh_bias) + residual + res_bias )
			for (int ch = 0; ch < 3; ch++)
				feat[ch] = fmaxf(0.0f, sh_color_6[ch] + residual_6[ch] + d_res_bias);

			break;
		}
		default:
			printf("FW unsupported render_mode: %d\n", render_mode & 0xFF);
			for(int i = 0; i < CHANNELS; i++) feat[i] = 0.0f;
			break;
		}  // End of switch (render_mode & 0xFF)

		// Accumulate hashgrid features
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += feat[ch] * w;
				
				// For residual_hybrid (render_mode=11): also accumulate SH RGB separately
				// This evaluates per-Gaussian SH and accumulates into SH_RGB[]
				// SH coefficients format: [sh0_r, sh0_g, sh0_b, sh1_r, sh1_g, sh1_b, ...] (interleaved)
				// For degree 3: 16 coefficients per channel = 48 total per Gaussian
				
				// max level is 6
				float ap_color[3] = {0};
				if(appearance_level <= 4){
					ap_color[0] = (appearance_level - 2) * 0.5;
					ap_color[1] = 1.0f;
				}
				else {
					ap_color[0] = 1.0f;
					ap_color[1] = 1.0 - (appearance_level - 4) * 0.5;
				}

				for (int ch = 0; ch < 3; ch++)
					vis_appearance[ch] += ap_color[ch] * w;

			}
			
			if(record_transmittance){
				atomicAdd(&(cover_pixel[collected_id[j]]), 1.0f);
				atomicAdd(&(trans_avg[collected_id[j]]), w);  // accum_weights: sum of alpha*T per Gaussian
			}

			T = test_T;
			// A_tex (textured-only weight sum) was already incremented inside the
			// distortion block above — no per-alpha update needed here.

			// Keep track of last range entry to update this pixel.
			last_contributor = contributor;

		}
	}

	// NOTE: baseline_blend_double post-processing was removed during render mode cleanup
	// (it was old mode 3, now deleted)

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;

		for (int ch = 0; ch < CHANNELS; ch++)
		{
			out_color[ch * H * W + pix_id] = C[ch];
			// if(CHANNELS == 3)out_color[ch * H * W + pix_id] += T * bg_color[ch];
		}


#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		// `--method mixed_3d`: slot 3 = A_tex (textured-only running weight sum).
		// Backward reads it directly as final_A_tex (no 1 − x flip) for the
		// textured dist-gradient sites. See A_tex comment at top of kernel for
		// why this differs from 1 - T_tex when EWA absorbers are interleaved.
		final_T[pix_id + 3 * H * W] = A_tex;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		out_others[pix_id + NUM_OFFSET * H * W] = render_number;

		out_others[pix_id + (POS_OFFSET + 0) * H * W] = pos_x;
		out_others[pix_id + (POS_OFFSET + 1) * H * W] = pos_y;
		out_others[pix_id + (POS_OFFSET + 2) * H * W] = pos_z;

		out_others[pix_id + (VIS_OFFSET + 0) * H * W] = vis_appearance[0];
		out_others[pix_id + (VIS_OFFSET + 1) * H * W] = vis_appearance[1];
		out_others[pix_id + (VIS_OFFSET + 2) * H * W] = vis_appearance[2];
		out_others[pix_id + OVERDRAW_OFFSET * H * W] = overdraw_sum;
		out_others[pix_id + MAXDEPTH_OFFSET * H * W] = max_depth;
		out_others[pix_id + WSQUARE_OFFSET * H * W] = w_square_sum;
		out_others[pix_id + BETA_SUM_OFFSET * H * W] = beta_sum;
		// Per-pixel id of the max-weight Gaussian (for mini depth-reinit SH transfer).
		// out_index is a separate int32 [H, W] buffer plumbed all the way to Python.
		if (out_index != nullptr) out_index[pix_id] = max_idx;

		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}


// Kernel to set device pointers (needed because __device__ vars can't be set from host directly)
__global__ void setMlpPointersKernel(
	__half* W1, __half* W2, __half* W3)
{
	d_mlp_W1 = W1;
	d_mlp_W2 = W2;
	d_mlp_W3 = W3;
}

// Conversion kernel: float -> __half (launched on device)
__global__ void float2half_kernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		dst[idx] = __float2half(src[idx]);
	}
}

// Copy MLP weights to global device memory (bias-free, stored as FP16)
// Accepts FP32 from Python, converts to FP16 on device
// 3D_SH_res: All [16×16] = 256 elements each
void FORWARD::setMlpWeights(
	const float* W1,
	const float* W2,
	const float* W3)
{
	// Allocate FP16 device memory if not already done
	if (!mlp_weights_allocated) {
		cudaMalloc(&h_mlp_W1, W1_SIZE * sizeof(__half));    // W1[16×16] = 256 halfs
		cudaMalloc(&h_mlp_W2, W2_SIZE * sizeof(__half));    // W2[16×16] = 256 halfs
		cudaMalloc(&h_mlp_W3, W3_SIZE * sizeof(__half));    // W3[16×16] = 256 halfs

		// Set the device pointers
		setMlpPointersKernel<<<1, 1>>>(h_mlp_W1, h_mlp_W2, h_mlp_W3);
		cudaDeviceSynchronize();

		mlp_weights_allocated = true;
	}

	// Convert FP32 weights to FP16 on device
	const int block = 256;
	float2half_kernel<<<(W1_SIZE + block-1)/block, block>>>(W1, h_mlp_W1, W1_SIZE);
	float2half_kernel<<<(W2_SIZE + block-1)/block, block>>>(W2, h_mlp_W2, W2_SIZE);
	float2half_kernel<<<(W3_SIZE + block-1)/block, block>>>(W3, h_mlp_W3, W3_SIZE);
}

// Set contribution threshold for hash query skip (w = T*alpha)
__global__ void setContribThreshKernel(float val) { d_contrib_thresh = val; }
void FORWARD::setContribThresh(float val) {
	setContribThreshKernel<<<1, 1>>>(val);
}

// Set count threshold for hash query skip (per-pixel Gaussian count)
__global__ void setCountThreshKernel(int val) { d_count_thresh = val; }
void FORWARD::setCountThresh(int val) {
	setCountThreshKernel<<<1, 1>>>(val);
}

// Set overdraw regularization lambda
__global__ void setOverdrawLambdaKernel(float val) { d_overdraw_lambda = val; }
void FORWARD::setOverdrawLambda(float val) {
	setOverdrawLambdaKernel<<<1, 1>>>(val);
}

// Set weight-squared regularization lambda
__global__ void setWeightRegLambdaKernel(float val) { d_weight_reg_lambda = val; }
void FORWARD::setWeightRegLambda(float val) {
	setWeightRegLambdaKernel<<<1, 1>>>(val);
}

// Set activation biases for SH and MLP residual
__global__ void setActivationBiasKernel(float sh, float res) { d_sh_bias = sh; d_res_bias = res; }
void FORWARD::setActivationBias(float sh_bias, float res_bias) {
	setActivationBiasKernel<<<1, 1>>>(sh_bias, res_bias);
}

// Set residual activation mode (0 = 3D_SH_res stacked outer ReLU, 1 = 3D_SH_add separate ReLUs)
__global__ void setResidualModeFwdKernel(int v) { d_residual_mode = v; }
void FORWARD::setResidualMode(int mode) {
	setResidualModeFwdKernel<<<1, 1>>>(mode);
}

// `--ste`: straight-through estimator on the per-Gauss outer ReLU.
__global__ void setSteReluFwdKernel(int v) { d_ste_relu_fwd = v; }
void FORWARD::setSteRelu(int v) {
	setSteReluFwdKernel<<<1, 1>>>(v);
}

// Set anti-aliasing params (Nexels-style hash-grid down-weighting)
__global__ void setAntiAliasKernel(float factor, float focal) { d_aa_factor = factor; d_aa_focal = focal; }
void FORWARD::setAntiAlias(float factor, float focal) {
	setAntiAliasKernel<<<1, 1>>>(factor, focal);
}

// Set FastGS Compact Box Mahalanobis² multiplier (scales AdR cutoff).
__global__ void setCompactMultKernel(float val) { d_compact_mult = val; }
void FORWARD::setCompactMult(float val) {
	setCompactMultKernel<<<1, 1>>>(val);
}

// Set AA-2DGS mip filter kernel size σ (0 disables; matches AA-2DGS's kernel_size, default 0.1)
__global__ void setAaKernelSizeKernel(float val) { d_aa_kernel_size = val; }
void FORWARD::setAaKernelSize(float val) {
	setAaKernelSizeKernel<<<1, 1>>>(val);
}

// Get MLP weight device pointers for passing to kernels (bias-free, FP16)
void FORWARD::getMlpWeightPointers(
	__half** W1,
	__half** W2,
	__half** W3)
{
	*W1 = h_mlp_W1;
	*W2 = h_mlp_W2;
	*W3 = h_mlp_W3;
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const float beta,
	int W, int H,
	uint32_t C, uint32_t level, uint32_t l_dim, float l_scale, uint32_t Base,
	bool align_corners, uint32_t interp,
	const bool if_contract, const bool record_transmittance,
	float focal_x, float focal_y,
	const glm::vec2* scales,
	const float* means3D,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* homotrans,
	const float* ap_level,
	const __half* hash_features,
	const int* level_offsets,
	const float* gridrange,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others,
	int* out_index,
	float* cover_pixels,
	float* trans_avg,
	const glm::vec3* cam_pos,
	const uint32_t D_diffuse,
	const float* hash_features_diffuse,
	const int* level_offsets_diffuse,
	const float* gridrange_diffuse,
	const int render_mode,
	const uint32_t max_intersections,
	const float* shapes,
	const int kernel_type,
	const float aa,
	const float aa_threshold,
	float* intersection_buffer,
	uint32_t* intersection_count,
	uint32_t max_intersections_per_pixel,
	const float* viewdirs_enc,
	const rgb_t* rgb_override,
	const int* metric_map,
	int* metric_counts,
	const bool* is_textured,
	const float4* ewa_conic)
{
	const uint32_t D_DIFFUSE_TEMPLATE = D_diffuse;

	// FP16 lean library: only C=3 (RGB) is needed for mode 5
	if (C != 3) {
		printf("diff_surfel_3D_16: Unsupported channel count %d (only C=3 supported)\n", C);
		return;
	}
	// FP16 SH baseline (geomState.rgb) — read by case 5 / case 6 inside the kernel.
	// `colors` (FP32) carries colors_precomp / DC SH separately.
	const rgb_t* rgb_ptr = rgb_override;

	// WMMA forward disabled: 55KB shared memory (31KB static + 24KB dynamic) kills occupancy
	// (1 block/SM vs 3 blocks/SM), and 7 __syncthreads per Gaussian adds overhead.
	// Scalar FP16 (__half2) path is faster for 16×16×16 MLP.
	renderCUDAsurfelForward<3, 0> <<<grid, block>>>(
		ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
		depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg, cam_pos,
		hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, rgb_ptr, max_intersections, shapes, kernel_type, aa, aa_threshold,
		intersection_buffer, intersection_count, max_intersections_per_pixel, viewdirs_enc,
		metric_map, metric_counts, is_textured, ewa_conic);

}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	int* radii_x,  // Separate X radius for rectangular AABB
	int* radii_y,  // Separate Y radius for rectangular AABB
	float2* means2D,
	float* depths,
	float* transMats,
	rgb_t* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	float4* conic_t,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode,
	const int render_mode,
	const bool* is_textured,
	const float* scaling_z,
	float4* ewa_conic)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		radii_x,
		radii_y,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		conic_t,
		prefiltered,
		shapes,
		kernel_type,
		aabb_mode,
		render_mode,
		is_textured,
		scaling_z,
		ewa_conic
		);
}

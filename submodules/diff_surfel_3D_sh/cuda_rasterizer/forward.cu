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
// MLP WEIGHTS IN GLOBAL DEVICE MEMORY (for 3D_fused and 3D_direct_fused modes)
// Using device pointers instead of constant memory to allow sharing across
// compilation units (constant memory requires -rdc=true which breaks PyTorch linking)
// ============================================================================
// Bias-free MLP weights in global device memory (WMMA-padded dimensions)
// L1: W1[32×32] (input[24]=1.0 for implicit bias, positions 25-31 = 0 padding)
// L2: W2[32×32] (no bias)
// L3: W3_sh[48×32] (48 SH coefficients = 16 per channel × 3 RGB)
__device__ __half* d_mlp_W1 = nullptr;      // 32D input → 32D hidden1 (W1_SIZE×2 = 2KB)
__device__ __half* d_mlp_W2 = nullptr;      // 32D hidden1 → 32D hidden2 (2KB)
__device__ __half* d_mlp_W3_sh = nullptr;   // 32D → 48 SH coefficients (W3_SIZE×2 = 3KB)
__device__ __half* d_mlp_W3_rgb = nullptr;  // Unused in SH mode (kept for API compatibility)

// Host-side pointers for memory management
static __half* h_mlp_W1 = nullptr;
static __half* h_mlp_W2 = nullptr;
static __half* h_mlp_W3_sh = nullptr;
static __half* h_mlp_W3_rgb = nullptr;
static bool mlp_weights_allocated = false;

// Convenience macros to access MLP weights
#define mlp_W1 d_mlp_W1
#define mlp_W2 d_mlp_W2
#define mlp_W3_sh d_mlp_W3_sh
#define mlp_W3_rgb d_mlp_W3_rgb

// MLP gradient buffers (device global memory, allocated once)
// These accumulate gradients across all intersections, then retrieved by Python
float* d_dL_dW1 = nullptr;   // [32, 32] = W1_SIZE = 1024 floats
float* d_dL_dW2 = nullptr;   // [32, 32] = W2_SIZE = 1024 floats
float* d_dL_dW3 = nullptr;   // [48, 32] = W3_SIZE = 1536 floats

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
	result += 0.5f;

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
	result[0] = SH_C0 * sh[0] + 0.5f;  // R
	result[1] = SH_C0 * sh[1] + 0.5f;  // G
	result[2] = SH_C0 * sh[2] + 0.5f;  // B

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

// MLP forward pass (bias-free): IN_DIM input → HIDDEN_DIM hidden1 → HIDDEN_DIM hidden2 → OUT_DIM output
// WMMA-padded: IN_DIM=32 (positions 25-31 = 0 padding), OUT_DIM=48 (SH coefficients)
// Input must have input[24] = 1.0f for implicit L1 bias (last column before padding)
// L2/L3 have no bias. Template parameters allow compile-time loop unrolling.
template <int IN_DIM, int HIDDEN_DIM, int OUT_DIM>
__device__ void mlp_forward_fused(
	const float* input,       // [IN_DIM] = 48D (24 features + 16 view + 1 bias + 7 zero-pad)
	float* output,            // [OUT_DIM] = 48 (SH) or 3 (RGB)
	float* hidden1,           // [HIDDEN_DIM] = 32 (caller-provided buffer)
	float* hidden2,           // [HIDDEN_DIM] = 32 (caller-provided buffer)
	bool apply_sigmoid,       // true for RGB output, false for SH output
	const __half* W1 = nullptr,   // Shared memory W1 (falls back to global if nullptr)
	const __half* W2 = nullptr,   // Shared memory W2
	const __half* W3 = nullptr    // Shared memory W3
) {
	// Use shared memory weights if provided, otherwise fall back to global memory
	const __half* w1 = W1 ? W1 : mlp_W1;
	const __half* w2 = W2 ? W2 : mlp_W2;
	const __half* w3 = W3 ? W3 : (OUT_DIM == 48 ? mlp_W3_sh : mlp_W3_rgb);

	// Layer 1: IN_DIM → HIDDEN_DIM (ReLU, no explicit bias, FP16 weights)
	#pragma unroll
	for (int h = 0; h < HIDDEN_DIM; h++) {
		float acc = 0;
		#pragma unroll
		for (int i = 0; i < IN_DIM; i++) {
			acc += input[i] * __half2float(w1[h * IN_DIM + i]);
		}
		hidden1[h] = fmaxf(0.0f, acc);  // ReLU
	}

	// Layer 2: HIDDEN_DIM → HIDDEN_DIM (ReLU, no bias, FP16 weights)
	#pragma unroll
	for (int h = 0; h < HIDDEN_DIM; h++) {
		float acc = 0;
		#pragma unroll
		for (int i = 0; i < HIDDEN_DIM; i++) {
			acc += hidden1[i] * __half2float(w2[h * HIDDEN_DIM + i]);
		}
		hidden2[h] = fmaxf(0.0f, acc);  // ReLU
	}

	// Layer 3: HIDDEN_DIM → OUT_DIM (Sigmoid optional, no bias, FP16 weights)
	#pragma unroll
	for (int o = 0; o < OUT_DIM; o++) {
		float acc = 0;
		#pragma unroll
		for (int h = 0; h < HIDDEN_DIM; h++) {
			acc += hidden2[h] * __half2float(w3[o * HIDDEN_DIM + h]);
		}
		output[o] = apply_sigmoid ? (1.0f / (1.0f + expf(-acc))) : acc;
	}
}

// SH evaluation for MLP output (48 coefficients → 3 RGB)
// Format: [sh0_r, sh0_g, sh0_b, sh1_r, ...] = 16 coefficients × 3 RGB
__device__ void eval_sh_from_mlp(const float* sh_coeffs, const float3& view_dir, float* rgb) {
	// Use existing eval_sh_raw for degree 3
	eval_sh_raw(3, sh_coeffs, view_dir, rgb);
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
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode)
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

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
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
	// aabb_mode: 0 = square (default), 1 = AdR cutoff, 2 = rectangular, 3 = AdR + rectangular, 4 = beta (fixed r=1)
	float cutoff;
	bool use_adr_cutoff = (aabb_mode == 1 || aabb_mode == 3);  // modes 1 and 3 use AdR
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
	bool use_rect_aabb = (aabb_mode >= 2);  // modes 2 and 3 use rectangular AABB

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
	if (colors_precomp == nullptr) {
		// SH mode: evaluate spherical harmonics to RGB
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}
	else {
		// Per-Gaussian features mode (cat, adaptive, etc.): copy precomputed features
		// For empty colors_precomp (baseline hashgrid), skip this - rgb buffer won't be used
		// Only copy if C matches expected dimension (otherwise it's a size mismatch)
		for(int i = 0; i < C; i++){
			rgb[idx * C + i] = colors_precomp[idx * C + i];
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
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
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
	float2 pixf = { (float)pix.x, (float)pix.y};

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
	const float* __restrict__ rgb = nullptr,
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
	const float* __restrict__ viewdirs_enc = nullptr)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

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
	__shared__ float collected_shapes[BLOCK_SIZE];  // Beta kernel shape parameter

	// Shared memory for per-Gaussian baseline features (dual hashgrid mode)
	// NOTE: Disabled for baseline_double/baseline_blend_double due to shared memory limits
	// We query on-demand instead (less efficient but fits in shared memory)
	// __shared__ float collected_feat_pk[BLOCK_SIZE][6 * 4];  // 6 levels × 4 features per Gaussian

	// Shared memory cache for MLP weights (FP16, WMMA-padded, ~7.1KB total)
	// Loaded once per tile, eliminates global memory reads in mlp_forward_fused
	__shared__ __half smem_mlp_W1[W1_SIZE];   // 32*32 = 2,048 bytes
	__shared__ __half smem_mlp_W2[W2_SIZE];   // 32*32 = 2,048 bytes
	__shared__ __half smem_mlp_W3[W3_SIZE];   // 48*32 = 3,072 bytes

	// WMMA forward buffers (case 5 batched MLP) — dynamic shared memory
	extern __shared__ char fw_dynamic_smem[];
	__half* smem_fw_half  = reinterpret_cast<__half*>(fw_dynamic_smem);                                    // 256*48*2 = 24,576 bytes
	float*  smem_fw_float = reinterpret_cast<float*>(fw_dynamic_smem + TC_BATCH * TC_INPUT_DIM * sizeof(__half));  // 256*32*4 = 32,768 bytes

	// Cooperatively load MLP weights into shared memory (mode 5 only)
	if (render_mode == 5 && mlp_W1 != nullptr) {
		const int tid = block.thread_rank();
		for (int idx = tid; idx < W1_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W1[idx] = mlp_W1[idx];
		for (int idx = tid; idx < W2_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W2[idx] = mlp_W2[idx];
		for (int idx = tid; idx < W3_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W3[idx] = mlp_W3_sh[idx];
		block.sync();
	}

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float SH_RGB[3] = { 0 };  // Separate accumulator for residual_hybrid SH RGB (render_mode==11)
	uint32_t render_number = 0;
	float vis_appearance[3] = {0};

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
		} else if(render_mode == 5){
			// 3D_direct_fused mode: level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
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

	// SH mode: view direction is computed per-intersection from xyz and cam_pos
	// No need to cache encoded view directions (SH evaluation uses raw viewdir)

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
		// Collect shape for beta kernel (only when using beta kernel)
		if(shapes != nullptr){
			collected_shapes[block.thread_rank()] = shapes[coll_id];
		}

		// NOTE: Per-Gaussian feature caching disabled due to shared memory limits
		// Features are now queried on-demand in the per-pixel loop (cases 4, 5, 12)

	}
		block.sync();

		// ================================================================
		// Case 5 WMMA forward: DISABLED — scalar per-thread MLP is faster
		// because early-exit and zero-sync beats batched WMMA for small MLPs.
		// Benchmarked: WMMA 25ms vs scalar 11ms (2.3x slower).
		// Keeping code for reference / future use with larger hidden dims.
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

			// Phase 1: Geometric + alpha (do-while(0) for structured break)
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
					float shape = collected_shapes[j];
					if (my_rho3d >= k_sq + 1e-6f) { active = false; break; }
					float base = fmaxf(0.0f, 1.0f - my_rho3d / k_sq);
					float alpha_beta = powf(base, shape);
					float alpha_lp = expf(-my_rho2d / 2.0f);
					my_alpha = fminf(0.99f, opa * fmaxf(alpha_beta, alpha_lp));
				} else if (kernel_type == 2) {
					float power = -0.5f * rho;
					if (power > 0.0f) { active = false; break; }
					float G = exp(power);
					float per_gaussian_beta = collected_shapes[j];
					if (per_gaussian_beta > 0.0f)
						G = (1.0f + per_gaussian_beta) * G / (1.0f + per_gaussian_beta * G);
					my_alpha = min(0.99f, opa * G);
				} else if (kernel_type == 3) {
					float beta_param = collected_shapes[j];
					float rho_safe = fmaxf(rho, 1e-8f);
					float pow_term = powf(rho_safe, 0.5f * beta_param);
					float power = -0.5f * pow_term;
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

				float test_T = T * (1 - my_alpha);
				if (test_T < 0.0001f) { done = true; active = false; break; }

				my_w = my_alpha * T;
				my_test_T = test_T;
				render_number++;

#if RENDER_AXUTILITY
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

			// Phase 2: Build MLP input → shared memory (half)
			if (active) {
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

				pos_x += my_w * xyz.x;
				pos_y += my_w * xyz.y;
				pos_z += my_w * xyz.z;

				const int hybrid_levels = level & 0xFF;
				const int per_gaussian_dim = hybrid_levels * l_dim;
				const int active_hashgrid_levels = (level >> 8) & 0xFF;

				float gauss_feat[20];
				if (hybrid_levels > 0 && rgb != nullptr) {
					int gauss_id = collected_id[j];
					const float* pgf = &rgb[gauss_id * per_gaussian_dim];
					for (int i = 0; i < per_gaussian_dim && i < 20; i++) gauss_feat[i] = pgf[i];
					for (int i = per_gaussian_dim; i < 20; i++) gauss_feat[i] = 0.0f;
				} else {
					for (int i = 0; i < 20; i++) gauss_feat[i] = 0.0f;
				}

				float hash_feat[4] = {0};
				if (active_hashgrid_levels > 0) {
					uint32_t ap_lvl = collected_ap_level[j];
					float voxel_min = gridrange[0];
					float voxel_max = gridrange[1];
					int collec_offsets[2];
					for (int lv = 0; lv <= active_hashgrid_levels; lv++)
						collec_offsets[lv] = level_offsets[lv];
					if (l_dim == 4)
						query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							ap_lvl, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false);
					else if (l_dim == 2)
						query_feature<false, 4, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
							ap_lvl, hash_features, active_hashgrid_levels,
							l_scale, Base, align_corners, interp, if_contract, false);
				}

				// Build 32D input: [gauss(20) | hash(4) | 1.0 bias | zeros(7)]
				for (int i = 0; i < 20; i++)
					smem_fw_half[tid * TC_INPUT_DIM + i] = __float2half(gauss_feat[i]);
				for (int i = 0; i < 4; i++)
					smem_fw_half[tid * TC_INPUT_DIM + 20 + i] = __float2half(hash_feat[i]);
				smem_fw_half[tid * TC_INPUT_DIM + 24] = __float2half(1.0f);  // Implicit L1 bias
				for (int i = ORIG_INPUT_DIM; i < TC_INPUT_DIM; i++)
					smem_fw_half[tid * TC_INPUT_DIM + i] = __half(0);
			} else {
				for (int i = 0; i < TC_INPUT_DIM; i++)
					smem_fw_half[tid * TC_INPUT_DIM + i] = __half(0);
			}
			__syncthreads();

			// Phase 3: WMMA batched MLP forward (all warps)
			wmma_forward_all(smem_fw_half, smem_fw_float,
				smem_mlp_W1, smem_mlp_W2, smem_mlp_W3);

			// Phase 4: Read 48D SH coefficients, evaluate at viewdir, accumulate RGB
			if (active) {
				// Read SH coefficients from WMMA output
				float sh_out[TC_OUTPUT_DIM];
				for (int i = 0; i < TC_OUTPUT_DIM; i++)
					sh_out[i] = smem_fw_float[tid * TC_OUTPUT_DIM + i];

				// Compute per-intersection view direction
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
				glm::vec3 cp = *cam_pos;
				float3 view_dir = {xyz.x - cp.x, xyz.y - cp.y, xyz.z - cp.z};
				float inv_len = rsqrtf(view_dir.x*view_dir.x + view_dir.y*view_dir.y + view_dir.z*view_dir.z + 1e-7f);
				view_dir.x *= inv_len; view_dir.y *= inv_len; view_dir.z *= inv_len;

				// Evaluate SH basis and compute RGB
				float sh_basis[16];
				eval_sh_basis_degree3(view_dir.x, view_dir.y, view_dir.z, sh_basis);
				for (int ch = 0; ch < 3; ch++) {
					float val = eval_sh_channel(&sh_out[ch * 16], sh_basis) + 0.5f;
					C[ch] += fmaxf(0.0f, val) * my_w;
				}

				uint32_t appearance_level = collected_ap_level[j];
				float ap_color[3] = {0};
				if (appearance_level <= 4) {
					ap_color[0] = (appearance_level - 2) * 0.5;
					ap_color[1] = 1.0f;
				} else {
					ap_color[0] = 1.0f;
					ap_color[1] = 1.0 - (appearance_level - 4) * 0.5;
				}
				for (int ch = 0; ch < 3; ch++)
					vis_appearance[ch] += ap_color[ch] * my_w;

				if (record_transmittance) {
					atomicAdd(&(cover_pixel[collected_id[j]]), 1.0f);
					atomicAdd(&(trans_avg[collected_id[j]]), T);
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
		float normal[3] = {nor_o.x, nor_o.y, nor_o.z};  // Already normalized in preprocessing
		float opa = nor_o.w;

		float alpha;
		if (kernel_type == 1 || kernel_type == 4) {
			// Beta kernel with separate G_obj (Beta) and G_screen (Gaussian low-pass)
			// kernel_type 1: k²=1 (unit circle cutoff)
			// kernel_type 4: k²=9 (3σ scaled, matches Gaussian extent)
			float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
			float shape = collected_shapes[j];

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
			float per_gaussian_beta = collected_shapes[j];  // shapes array holds per-Gaussian beta
			if (per_gaussian_beta > 0.0f)
				G = (1.0f + per_gaussian_beta) * G / (1.0f + per_gaussian_beta * G);

			alpha = min(0.99f, opa * G);
		} else if (kernel_type == 3) {
			// General kernel: Isotropic Generalized Gaussian
			// Formula: G = exp(-0.5 * (r²)^(β/2))
			// β = 2.0: standard Gaussian, β = 8.0: super-Gaussian (box-like)
			float beta_param = collected_shapes[j];  // shapes array holds beta in range [2.0, 8.0]

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

		float test_T = T * (1 - alpha);
		if (test_T < 0.0001f)
		{
			done = true;
			continue;
		}

		float w = alpha * T;

		render_number++;

		// NOTE: max_intersections check moved earlier (before kernel computation)
		// to cap total evaluations for benchmarking, not just valid intersections

#if RENDER_AXUTILITY
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
			
			//// now color part is in ingp model
			// Eq. (3) from 3D Gaussian splatting paper.
			// MyGs, now color calculation is in ngp part.

			// Special handling for adaptive_cat_fast (mode 13): check weight BEFORE 3D intersection
			// This allows skipping expensive intersection computation for Gaussian-only primitives
			const int base_mode = render_mode & 0xFF;
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

			if(level == 0){
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			}
			else{

				const float3 pk = collected_pk[j];
				float3 xyz;
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
				case 0:
					// Baseline mode: use l_dim directly (includes surface_blend with 12D features)
					if(l_dim == 2) {
						query_feature<false, CHANNELS, 2>(feat, xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
					} else if(l_dim == 4) {
						query_feature<false, CHANNELS, 4>(feat, xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
					} else if(l_dim == 8) {
						query_feature<false, CHANNELS, 8>(feat, xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
					} else if(l_dim == 12) {
						query_feature<false, CHANNELS, 12>(feat, xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
					} else {
						printf("FW unsupported level dim : %d\n", l_dim);
					}
					break;
		case 3: {
			/* 3D mode: Output intersection buffer for PyTorch processing
			 * Instead of blending features in CUDA, we output raw intersection data (12 floats per intersection):
			 *   0. gaussian_id: which Gaussian this intersection belongs to
			 *   1. weight: alpha * T (blending weight)
			 *   2. pixel_id: which pixel this intersection belongs to
			 *   3-5. xyz.x, xyz.y, xyz.z: 3D intersection point in world space (for hash query)
			 *   6-7. s_x, s_y: 2D disk coordinates (for backward gradient computation)
			 *   8. rho_flag: 1.0 if disk intersection, 0.0 if Gaussian center (for backward)
			 *   9. alpha: opacity * G (for gradient computation)
			 *   10. T: transmittance before this intersection (for gradient computation)
			 *   11. G: kernel value (for gradient computation: dL/dopacity = G * dL/dalpha)
			 *
			 * The PyTorch pipeline then:
			 *   1. Uses xyz directly for hash encoding (exact match with CAT mode)
			 *   2. Gathers per-Gaussian features by ID
			 *   3. Concatenates and passes through MLP to get SH coefficients
			 *   4. Blends RGB per pixel using alpha compositing
			 *   5. Backward uses T, alpha, G, s_x, s_y, rho_flag for correct gradients
			 */

			if (intersection_buffer != nullptr && intersection_count != nullptr && max_intersections_per_pixel > 0) {
				int gauss_id = collected_id[j];

				// Atomically get write index for this pixel
				uint32_t write_idx = atomicAdd(&intersection_count[pix_id], 1);

				// Only write if within per-pixel cap
				if (write_idx < max_intersections_per_pixel) {
					// Padded layout: each pixel gets max_intersections_per_pixel slots
					uint32_t global_idx = pix_id * max_intersections_per_pixel + write_idx;

					// Compute G from alpha and opacity (alpha = min(0.99, opa * G))
					// If alpha was clamped, G_recovered will be slightly off, but this is rare
					float G_recovered = alpha / (opa + 1e-7f);

					// Compute xyz intersection point (same formula as CAT mode for hash query)
					// xyz = pk + s.x * SuTu + s.y * SvTv  (when rho3d <= rho2d, i.e., disk intersection)
					// xyz = pk                            (when rho2d < rho3d, i.e., center fallback)
					const float3 pk = collected_pk[j];
					float3 xyz;
					float rho_flag = (rho3d <= rho2d) ? 1.0f : 0.0f;
					if (rho3d <= rho2d) {
						const float3 sutu = collected_SuTu[j];
						const float3 svtv = collected_SvTv[j];
						xyz = {s.x * sutu.x + s.y * svtv.x + pk.x,
						       s.x * sutu.y + s.y * svtv.y + pk.y,
						       s.x * sutu.z + s.y * svtv.z + pk.z};
					} else {
						xyz = pk;
					}

					// Write intersection data: 12 floats per intersection
					intersection_buffer[global_idx * 12 + 0] = __int_as_float(gauss_id);
					intersection_buffer[global_idx * 12 + 1] = w;              // weight = alpha * T
					intersection_buffer[global_idx * 12 + 2] = __int_as_float(pix_id);
					intersection_buffer[global_idx * 12 + 3] = xyz.x;          // world-space xyz for hash query
					intersection_buffer[global_idx * 12 + 4] = xyz.y;
					intersection_buffer[global_idx * 12 + 5] = xyz.z;
					intersection_buffer[global_idx * 12 + 6] = s.x;            // disk coordinate for backward
					intersection_buffer[global_idx * 12 + 7] = s.y;
					intersection_buffer[global_idx * 12 + 8] = rho_flag;       // disk vs center flag for backward
					intersection_buffer[global_idx * 12 + 9] = alpha;          // alpha = opacity * G
					intersection_buffer[global_idx * 12 + 10] = T;              // transmittance BEFORE this intersection
					intersection_buffer[global_idx * 12 + 11] = G_recovered;    // kernel value for dL/dopacity = G * dL/dalpha
				}
			}

			// For 3D mode, we don't accumulate features here - just output intersection data
			// Set feat to zeros so the accumulation loop below does nothing
			for(int i = 0; i < CHANNELS; i++) feat[i] = 0.0f;

			break;
		}
		case 5: {
			/* 3D_SH_TC mode: In-kernel MLP → SH coefficients → eval at viewdir → RGB
			 * MLP outputs 48D SH coefficients (16 per channel × 3 RGB), no view input.
			 * View direction is used AFTER MLP for SH evaluation, enabling baking.
			 *
			 * Pipeline:
			 *   1. Compute xyz intersection point
			 *   2. Get per-Gaussian features from colors_precomp (coarse levels)
			 *   3. Query hashgrid for fine levels at xyz
			 *   4. Build MLP input: [gauss(20) | hash(4) | 1.0 bias | zeros(7)] = 32D
			 *   5. Run MLP → 48D SH coefficients (no activation)
			 *   6. Compute per-intersection view direction from xyz and cam_pos
			 *   7. Evaluate SH basis at viewdir, dot with coefficients → RGB
			 */

			// 1. Compute xyz intersection point (same as cat mode)
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
			const int hybrid_levels = level & 0xFF;
			const int per_gaussian_dim = hybrid_levels * l_dim;  // Coarse feature dimension

			// Determine hashgrid levels from active_hashgrid_levels
			const int active_hashgrid_levels = (level >> 8) & 0xFF;

			// 2. Get per-Gaussian features (coarse levels)
			float gauss_feat[20];  // Max 5 hybrid_levels * 4D = 20D
			if (hybrid_levels > 0 && rgb != nullptr) {
				int gauss_id = collected_id[j];
				const float* per_gaussian_feat = &rgb[gauss_id * per_gaussian_dim];
				for (int i = 0; i < per_gaussian_dim && i < 20; i++) {
					gauss_feat[i] = per_gaussian_feat[i];
				}
			} else {
				for (int i = 0; i < 20; i++) gauss_feat[i] = 0.0f;
			}

			// 3. Query hashgrid for fine levels at xyz
			float hash_feat[4];  // Fine levels: typically 1 level * 4D = 4D
			if (active_hashgrid_levels > 0) {
				if (l_dim == 4) {
					query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else if (l_dim == 2) {
					query_feature<false, 4, 2>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, debug);
				} else {
					for (int i = 0; i < 4; i++) hash_feat[i] = 0.0f;
				}
			} else {
				for (int i = 0; i < 4; i++) hash_feat[i] = 0.0f;
			}

			// 4. Build MLP input: [gauss(20) | hash(4) | 1.0 bias | zeros(7)] = 32D
			float mlp_input[TC_INPUT_DIM];
			for (int i = 0; i < TC_INPUT_DIM; i++) mlp_input[i] = 0.0f;
			for (int i = 0; i < 20; i++) mlp_input[i] = gauss_feat[i];
			for (int i = 0; i < 4; i++)  mlp_input[20 + i] = hash_feat[i];
			mlp_input[24] = 1.0f;  // Implicit L1 bias (positions 25-31 = 0 padding)

			// 5. Run MLP → 48D SH coefficients (no sigmoid, identity activation)
			float sh_out[TC_OUTPUT_DIM];  // 48 SH coefficients
			float h1[TC_HIDDEN_DIM], h2[TC_HIDDEN_DIM];
			mlp_forward_fused<TC_INPUT_DIM, TC_HIDDEN_DIM, TC_OUTPUT_DIM>(
				mlp_input, sh_out, h1, h2, false,
				smem_mlp_W1, smem_mlp_W2, smem_mlp_W3);

			// 6. Compute per-intersection view direction
			glm::vec3 cp = *cam_pos;
			float3 view_dir = {xyz.x - cp.x, xyz.y - cp.y, xyz.z - cp.z};
			float inv_len = rsqrtf(view_dir.x*view_dir.x + view_dir.y*view_dir.y + view_dir.z*view_dir.z + 1e-7f);
			view_dir.x *= inv_len; view_dir.y *= inv_len; view_dir.z *= inv_len;

			// 7. Evaluate SH: basis(viewdir) · coefficients + 0.5, clamped to [0, inf)
			// SH layout: [R_coef0..R_coef15, G_coef0..G_coef15, B_coef0..B_coef15]
			float sh_basis[16];
			eval_sh_basis_degree3(view_dir.x, view_dir.y, view_dir.z, sh_basis);

			for (int ch = 0; ch < 3; ch++) {
				float val = eval_sh_channel(&sh_out[ch * 16], sh_basis) + 0.5f;
				feat[ch] = fmaxf(0.0f, val);
			}

			break;
		}
		default:
			// Unsupported render_mode - zero features
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
				atomicAdd(&(trans_avg[collected_id[j]]), T);
			}
			
			T = test_T;

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
		
		// For residual_hybrid mode (render_mode=11): store SH RGB in out_others[14:16]
		// This is written separately from the main C[] output which contains hashgrid features
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}


// Kernel to set device pointers (needed because __device__ vars can't be set from host directly)
__global__ void setMlpPointersKernel(
	__half* W1, __half* W2, __half* W3_sh, __half* W3_rgb)
{
	d_mlp_W1 = W1;
	d_mlp_W2 = W2;
	d_mlp_W3_sh = W3_sh;
	d_mlp_W3_rgb = W3_rgb;
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
void FORWARD::setMlpWeights(
	const float* W1,
	const float* W2,
	const float* W3,
	bool is_sh_mode)
{
	// Allocate FP16 device memory if not already done
	if (!mlp_weights_allocated) {
		cudaMalloc(&h_mlp_W1, W1_SIZE * sizeof(__half));    // W1[32×32] WMMA-padded
		cudaMalloc(&h_mlp_W2, W2_SIZE * sizeof(__half));    // W2[32×32]
		cudaMalloc(&h_mlp_W3_sh, W3_SIZE * sizeof(__half)); // W3_sh[48×32] SH coefficients
		cudaMalloc(&h_mlp_W3_rgb, W3_SIZE * sizeof(__half)); // W3_rgb (unused in SH mode)

		// Set the device pointers
		setMlpPointersKernel<<<1, 1>>>(
			h_mlp_W1, h_mlp_W2, h_mlp_W3_sh, h_mlp_W3_rgb);
		cudaDeviceSynchronize();

		mlp_weights_allocated = true;
	}

	// Convert FP32 weights to FP16 on device
	const int block = 256;
	float2half_kernel<<<(W1_SIZE + block-1)/block, block>>>(W1, h_mlp_W1, W1_SIZE);
	float2half_kernel<<<(W2_SIZE + block-1)/block, block>>>(W2, h_mlp_W2, W2_SIZE);

	if (is_sh_mode) {
		float2half_kernel<<<(W3_SIZE + block-1)/block, block>>>(W3, h_mlp_W3_sh, W3_SIZE);
	} else {
		float2half_kernel<<<(W3_SIZE + block-1)/block, block>>>(W3, h_mlp_W3_rgb, W3_SIZE);
	}
}

// Get MLP weight device pointers for passing to kernels (bias-free, FP16)
void FORWARD::getMlpWeightPointers(
	__half** W1,
	__half** W2,
	__half** W3_sh,
	__half** W3_rgb)
{
	*W1 = h_mlp_W1;
	*W2 = h_mlp_W2;
	*W3_sh = h_mlp_W3_sh;
	*W3_rgb = h_mlp_W3_rgb;
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
	const float* viewdirs_enc)
{
	// Determine D_DIFFUSE template parameter for kernel dispatch
	// For dual hashgrid modes (baseline_double, baseline_blend_double, surface_rgb), use D_diffuse
	// Otherwise default to 0
	const uint32_t D_DIFFUSE_TEMPLATE = D_diffuse;
	
	// FP16 lean library: only C=3 (RGB) is needed for mode 5
	if (C != 3) {
		printf("diff_surfel_3D_16: Unsupported channel count %d (only C=3 supported)\n", C);
		return;
	}
	// WMMA forward disabled (scalar is faster for small MLPs).
	// Dynamic shared memory not needed for forward — only backward uses WMMA.
	renderCUDAsurfelForward<3, 0> <<<grid, block>>>(
		ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
		depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg, cam_pos,
		hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors, max_intersections, shapes, kernel_type, aa, aa_threshold,
		intersection_buffer, intersection_count, max_intersections_per_pixel, viewdirs_enc);

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
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode)
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
		prefiltered,
		shapes,
		kernel_type,
		aabb_mode
		);
}

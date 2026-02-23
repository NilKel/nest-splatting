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
	float3* __restrict__ dL_dmean2D,
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
	float dL_dpixel[C];

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
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
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
			if (contributor >= last_contributor)
				continue;

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
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
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
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dfeatures,
	float * __restrict__ dL_dtransMat,
	float * __restrict__ dL_dhomoMat,
	float3* __restrict__ dL_dmean2D,
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
	const __half* __restrict__ mlp_W3_ptr = nullptr)
{
	// Create MLP weights struct from parameters (bias-free, all [16×16])
	MlpWeights mlp_weights = {
		mlp_W1_ptr,
		mlp_W2_ptr,
		mlp_W3_ptr
	};

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

	// __shared__ float collected_colors[C * BLOCK_SIZE];

	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float collected_size[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	__shared__ float3 collected_SuTu[BLOCK_SIZE];
	__shared__ float3 collected_SvTv[BLOCK_SIZE];
	__shared__ float3 collected_pk[BLOCK_SIZE];
	__shared__ uint32_t collected_ap_level[BLOCK_SIZE];
	__shared__ float collected_shapes[BLOCK_SIZE];  // Beta kernel shape parameter

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
	float dL_dpixel[C];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		// here dL_ddepth is dL_dD (blended depth value), so no change here. 
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];

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
		} else if((render_mode & 0xFF) == 5){
			// 3D_direct_fused mode: level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			// Hash query happens in CUDA kernel (like cat mode), so use active_hashgrid_levels
			// Note: render_mode may have bit 8 set (0x100) for collaborative GEMM, so mask to get base mode
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
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Note: 3D_SH_res does not use view direction encoding (MLP is view-independent)
	// View dependence comes from SH evaluation in preprocessing

	// Initialize tile-local MLP gradient buffers for render_mode=5 (3D_SH_res)
	// Each thread zeroes a subset of the buffer collaboratively
	// Note: (render_mode & 0xFF) masks out the bit 8 flag to get base mode
	if ((render_mode & 0xFF) == 5 && dL_dmlp_W1 != nullptr) {
		const int tid = block.thread_rank();
		for (int idx = tid; idx < W1_SIZE; idx += BLOCK_SIZE)
			tile_dL_dW1[idx] = 0.0f;
		for (int idx = tid; idx < W2_SIZE; idx += BLOCK_SIZE)
			tile_dL_dW2[idx] = 0.0f;
		for (int idx = tid; idx < W3_SIZE; idx += BLOCK_SIZE)
			tile_dL_dW3[idx] = 0.0f;
		// Load MLP weights into shared memory (eliminates global reads in backward)
		for (int idx = tid; idx < W1_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W1[idx] = mlp_weights.W1[idx];
		for (int idx = tid; idx < W2_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W2[idx] = mlp_weights.W2[idx];
		for (int idx = tid; idx < W3_SIZE; idx += BLOCK_SIZE)
			smem_mlp_W3[idx] = mlp_weights.W3[idx];
		block.sync();
	}

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
			
			collected_size[block.thread_rank()] =  PI * scales[coll_id].x * scales[coll_id].y;
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

				// Per-pixel intersection data
				float2 s = {0, 0};
				float rho3d = 0, rho2d = 0, rho = 0;
				float c_d = 0, alpha = 0, G = 0, w = 0;
				float3 xyz = {0, 0, 0};
				// Store k, l, p for geometry gradients later
				float3 k_stored = {0, 0, 0};
				float3 l_stored = {0, 0, 0};
				float3 p_stored = {0, 0, 0};

				// Beta/flex/general kernel variables (needed across alpha + geometry gradient phases)
				float shape_val = 0.0f, base = 0.0f;
				float alpha_beta = 0.0f, alpha_lp = 0.0f;
				bool beta_wins = false;
				float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
				float per_gaussian_beta_val = 0.0f, G_raw = 0.0f, demon_val = 1.0f;
				float general_beta_val = 0.0f, general_pow_term_val = 0.0f, general_rho_safe_val = 0.0f;

				// Compute intersection (only for participating pixels)
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
							if (kernel_type == 1 || kernel_type == 4) {
								if (rho3d < k_sq + 1e-6f) {
									shape_val = collected_shapes[j];
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
									per_gaussian_beta_val = collected_shapes[j];
									G = G_raw; demon_val = 1.0f;
									if (per_gaussian_beta_val > 0.0f) {
										demon_val = 1.0f + per_gaussian_beta_val * G_raw;
										G = (1.0f + per_gaussian_beta_val) * G_raw / demon_val;
									}
									alpha = fminf(0.99f, opa * G);
									valid_alpha = true;
								}
							} else if (kernel_type == 3) {
								general_beta_val = collected_shapes[j];
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
								w = alpha * T;
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
							} else participates = false;
						} else participates = false;
					} else participates = false;
				}

				// ======== INTERLEAVED MLP BACKWARD WITH COLLABORATIVE GEMM ========
				// 3D_SH_res: SH base color + hash MLP residual
				// MLP input: [hash(4) | bias(1) | pad(11)] = 16D
				// MLP output: 3D RGB residual (identity activation, NO sigmoid)
				// feat = SH_color + MLP_residual
				const int active_hashgrid_levels = (level >> 8) & 0xFF;

				// --- Phase 1: Forward recomputation + identity backward → dL_dz3 ---
				float my_input[TC_INPUT_DIM] = {0};
				float my_h1_post[TC_HIDDEN_DIM] = {0};
				float my_h2_post[TC_HIDDEN_DIM] = {0};
				float my_residual[ORIG_OUTPUT_DIM] = {0};
				float my_dL_dz3[TC_OUTPUT_DIM] = {0};

				if (participates) {
					// 1. Query hash features (4D)
					if (active_hashgrid_levels > 0 && l_dim == 4) {
						float hash_feat[4] = {0};
						uint32_t appearance_level = collected_ap_level[j];
						query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
						                           appearance_level, hash_features, active_hashgrid_levels,
						                           l_scale, Base, align_corners, interp, if_contract, false);
						for (int i = 0; i < 4; i++) my_input[i] = hash_feat[i];
					}
					my_input[4] = 1.0f;  // Implicit bias
					// Positions 5-15 are zero (WMMA padding)

					// 2. Recompute MLP forward → residual (identity activation, no sigmoid)
					MlpWeights smem_mlp = {smem_mlp_W1, smem_mlp_W2, smem_mlp_W3};
					mlp_forward_inline(my_input, my_residual, my_h1_post, my_h2_post, false, smem_mlp);

					// 3. Load SH base color and compute feat = SH + residual
					float sh_color[3];
					for (int ch = 0; ch < 3; ch++)
						sh_color[ch] = colors[global_id * 3 + ch];

					// 4. Identity backward → dL_dz3 (no sigmoid derivative)
					// dL/d(feat) = dL/d(pixel) * w, and feat = SH + residual
					// So dL/d(residual) = dL/d(feat) = dL/d(pixel) * w
					#pragma unroll
					for (int o = 0; o < ORIG_OUTPUT_DIM; o++) {
						my_dL_dz3[o] = dL_dpixel[o] * w;
					}
					// Positions 3-15 of dL_dz3 stay zero (WMMA padding)

					// 5. Accumulate dL to SH colors (same gradient flows to both)
					for (int ch = 0; ch < 3; ch++) {
						atomicAdd(&(dL_dcolors[global_id * 3 + ch]), dL_dpixel[ch] * w);
					}
				}

				// Note: no __syncthreads needed here - ballot check above provides sync
				extern __shared__ __half dynamic_smem[];
				wmma_gemm_layer3(my_dL_dz3, my_h2_post, tile_dL_dW3, dynamic_smem);

				// --- Phase 2: Layer 3 backward → dL_dz2 ---
				float my_dL_dz2[TC_HIDDEN_DIM] = {0};
				if (participates) {
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
				wmma_gemm_layer2(my_dL_dz2, my_h1_post, tile_dL_dW2, dynamic_smem);

				// --- Phase 3: Layer 2 backward → dL_dz1 ---
				float my_dL_dz1[TC_HIDDEN_DIM] = {0};
				if (participates) {
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
				wmma_gemm_layer1(my_dL_dz1, my_input, tile_dL_dW1, dynamic_smem);

				// ======== Phase 4: dL_dinput → hash & geometry gradients ========
				if (participates) {
					// dL_dinput = W1^T @ dL_dz1 (only first 4 elements = hash features)
					float my_dL_dinput[4];
					#pragma unroll
					for (int i = 0; i < 4; i++) {
						float sum = 0;
						#pragma unroll
						for (int h = 0; h < TC_HIDDEN_DIM; h++) {
							sum += my_dL_dz1[h] * __half2float(smem_mlp_W1[h * TC_INPUT_DIM + i]);
						}
						my_dL_dinput[i] = sum;
					}

					// Backprop to hash features - dL_dxyz flows to geometry
					float dL_dxyz[3] = {0, 0, 0};
					if (active_hashgrid_levels > 0 && l_dim == 4) {
						float dL_dhash[4];
						for (int i = 0; i < 4; i++) dL_dhash[i] = my_dL_dinput[i];
						float hash_feat_dummy[4];
						uint32_t appearance_level = collected_ap_level[j];
						query_feature<true, 4, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
						                          appearance_level, hash_features, active_hashgrid_levels,
						                          l_scale, Base, align_corners, interp, if_contract, false,
						                          dL_dhash, dL_dfeatures, dL_dxyz);
					}

					// ======== GEOMETRY GRADIENTS ========
					float dL_dalpha = 0.0f;
					// feat = SH_color + residual (recompute for alpha gradient)
					float feat[C];
					float sh_color_recomp[3];
					for (int ch = 0; ch < 3; ch++) {
						sh_color_recomp[ch] = colors[global_id * 3 + ch];
						feat[ch] = sh_color_recomp[ch] + my_residual[ch];
					}

					// Update accumulators
					for (int ch = 0; ch < C; ch++) {
						accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
						last_color[ch] = feat[ch];
						dL_dalpha += (feat[ch] - accum_rec[ch]) * dL_dpixel[ch];
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
					dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
					dL_dalpha += dL_dweight - last_dL_dT;
					last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
					const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
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
						atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
					}
#endif

					dL_dalpha *= T;
					last_alpha = alpha;

					float dL_dG = opa * dL_dalpha;

					// Kernel-specific shape gradients and dL_dG adjustments
					if (kernel_type == 1 || kernel_type == 4) {
						if (beta_wins && dL_dshapes != nullptr && base > 1e-7f) {
							float dL_dshape = dL_dalpha * opa * alpha_beta * logf(base);
							atomicAdd(&dL_dshapes[global_id], dL_dshape);
						}
					} else if (kernel_type == 2 && per_gaussian_beta_val > 0.0f) {
						const float dG_dg_raw = (1.0f + per_gaussian_beta_val) / (demon_val * demon_val);
						dL_dG *= dG_dg_raw;
						if (dL_dshapes != nullptr) {
							float dG_dbeta = G_raw * (1.0f - G_raw) / (demon_val * demon_val);
							float dL_dbeta = dL_dalpha * opa * dG_dbeta;
							atomicAdd(&dL_dshapes[global_id], dL_dbeta);
						}
					} else if (kernel_type == 3 && dL_dshapes != nullptr) {
						float log_rho = logf(general_rho_safe_val);
						float dG_dbeta = -0.25f * G * general_pow_term_val * log_rho;
						float dL_dbeta = dL_dalpha * opa * dG_dbeta;
						atomicAdd(&dL_dshapes[global_id], dL_dbeta);
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
							atomicAdd(&dL_dhomoMat[global_id * 9 + 0],  dL_dpx * s.x);
							atomicAdd(&dL_dhomoMat[global_id * 9 + 1],  dL_dpy * s.x);
							atomicAdd(&dL_dhomoMat[global_id * 9 + 2],  dL_dpz * s.x);
							atomicAdd(&dL_dhomoMat[global_id * 9 + 3],  dL_dpx * s.y);
							atomicAdd(&dL_dhomoMat[global_id * 9 + 4],  dL_dpy * s.y);
							atomicAdd(&dL_dhomoMat[global_id * 9 + 5],  dL_dpz * s.y);
						}
						// for both rho3d and rho2d
						atomicAdd(&dL_dhomoMat[global_id * 9 + 6],  dL_dxyz[0]);
						atomicAdd(&dL_dhomoMat[global_id * 9 + 7],  dL_dxyz[1]);
						atomicAdd(&dL_dhomoMat[global_id * 9 + 8],  dL_dxyz[2]);
					}

					// Geometry gradients based on whether rho3d or rho2d was used
					if (rho3d <= rho2d) {
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
						float2 dL_ds = {
							dL_dG * dG_factor * s.x + dL_dz * Tw.x,
							dL_dG * dG_factor * s.y + dL_dz * Tw.y
						};
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
						atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx);
						atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely);
						atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz);
					}

					atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

					// NOTE: T was already updated at the start of this iteration
					// (T = T / (1-alpha) to recover T_before, matching 2DGS backward)
					// No T-based termination needed - loop runs through all contributors
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
			if (contributor >= last_contributor)
				continue;

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

		// compute intersection and depth
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

		if (kernel_type == 1 || kernel_type == 4) {
			// Beta kernel with separate G_obj (Beta) and G_screen (Gaussian low-pass)
			// kernel_type 1: k²=1 (unit circle cutoff)
			// kernel_type 4: k²=9 (3σ scaled, matches Gaussian extent)

			// 1. Hard support check on object-space distance
			if (rho3d >= k_sq + 1e-6f)
				continue;  // Outside compact support - skip entirely

			shape_val = collected_shapes[j];

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
			per_gaussian_beta = collected_shapes[j];  // shapes array holds per-Gaussian beta
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
			general_beta = collected_shapes[j];  // beta in range [2.0, 8.0]
			float exponent = 0.5f * general_beta;  // β/2

			general_rho_safe = fmaxf(rho, 1e-8f);
			general_pow_term = powf(general_rho_safe, exponent);  // (r²)^(β/2)
			float power = -0.5f * general_pow_term;

			if (power > 0.0f)
				continue;

			G = expf(power);
			alpha = min(0.99f, opa * G);
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
			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];

			float dL_dxyz[3] = {0};
			
			if(level == 0){
				for (int ch = 0; ch < C; ch++)
				{
					const float c = colors[global_id * C + ch];

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
			}
			else {
				
				// Calculate and get features & dy_dx
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
				case 0:
					// Baseline mode: use l_dim directly (includes surface_blend with 12D features)
					if(l_dim == 2) {
						query_feature<true, C, 2>(feat, xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, grad_feat, dL_dfeatures, dL_dxyz);
					} else if(l_dim == 4) {
						query_feature<true, C, 4>(feat, xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, grad_feat, dL_dfeatures, dL_dxyz);
					} else if(l_dim == 8) {
						query_feature<true, C, 8>(feat, xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, grad_feat, dL_dfeatures, dL_dxyz);
					} else if(l_dim == 12) {
						query_feature<true, C, 12>(feat, xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, grad_feat, dL_dfeatures, dL_dxyz);
					} else {
						printf("BW unsupported level dim : %d\n", l_dim);
					}
					break;
			case 3:
				// 3D mode: All gradients handled in PyTorch post-processing
				// No CUDA hashgrid backward needed - just skip
				break;
			case 5: {
				// 3D_SH_res backward: SH base color + hash MLP residual
				// feat = SH_color + MLP(hash(xyz)), identity activation
				if (dL_dmlp_W1 == nullptr) break;

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

				// 1. Query hash features (4D)
				float hash_feat[4] = {0};
				if (active_hashgrid_levels > 0 && l_dim == 4) {
					query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, false);
				}

				// 2. Build MLP input: [hash(4) | bias(1) | pad(11)] = 16D
				float mlp_input[TC_INPUT_DIM];
				for (int i = 0; i < TC_INPUT_DIM; i++) mlp_input[i] = 0.0f;
				for (int i = 0; i < 4; i++) mlp_input[i] = hash_feat[i];
				mlp_input[4] = 1.0f;

				// 3. Recompute MLP forward → residual (identity, no sigmoid)
				float h1_post[TC_HIDDEN_DIM], h2_post[TC_HIDDEN_DIM];
				float residual[ORIG_OUTPUT_DIM];
				MlpWeights smem_mlp2 = {smem_mlp_W1, smem_mlp_W2, smem_mlp_W3};
				mlp_forward_inline(mlp_input, residual, h1_post, h2_post, false, smem_mlp2);

				// 4. dL/d(feat) = dL/d(pixel) * w
				// feat = SH_color + residual → dL/d(residual) = dL/d(feat) = dL/d(pixel) * w
				float dL_drgb[3];
				for (int c = 0; c < 3; c++)
					dL_drgb[c] = dL_dchannels[c] * w;

				// 5. Accumulate dL to SH colors
				for (int ch = 0; ch < 3; ch++)
					atomicAdd(&(dL_dcolors[global_id * 3 + ch]), dL_drgb[ch]);

				// 6. MLP backward (legacy per-pixel atomic path, identity activation)
				float dL_dinput_full[TC_INPUT_DIM];
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

				// 7. Backprop to hash features (first 4D of dL_dinput)
				if (active_hashgrid_levels > 0 && l_dim == 4) {
					float dL_dhash[4];
					for (int i = 0; i < 4; i++)
						dL_dhash[i] = dL_dinput_full[i];

					float hash_feat_dummy[4];
					query_feature<true, 4, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
					                           appearance_level, hash_features, active_hashgrid_levels,
					                           l_scale, Base, align_corners, interp, contract, false,
					                           dL_dhash, dL_dfeatures, dL_dxyz);
				}

				// 8. Set feat = SH + residual for alpha gradient computation below
				for (int ch = 0; ch < C; ch++) {
					feat[ch] = colors[global_id * 3 + ch] + residual[ch];
				}

				break;
			}
			default: printf("BW unsupported level dim : %d\n", l_dim);
				break;
			}

				// Update dL_dalpha and get grad_feat
				for (int ch = 0; ch < C; ch++)
				{
					const float c = feat[ch];
					// Update last color (to be used in the next iteration)
					accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_color[ch] = c;

					dL_dalpha += (c - accum_rec[ch]) * dL_dchannels[ch];
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
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif

			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
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

			// Helpful reusable temporary variables
			float dL_dG = nor_o.w * dL_dalpha;

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
				
				if(rho3d <= rho2d){
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


			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				// For Gaussian kernel: dG/ds.x = -G * s.x (from exp(-0.5*(s.x²+s.y²)))
				// For Beta kernel: dG/ds.x = -shape * G / base * s.x (from pow(1-rho, shape))
				// For Flex kernel: same as Gaussian but use G_raw (already accounted for in dL_dG)
				// For General kernel: dG/drho = -0.25 * β * G * pow_term / rho
				float dG_factor;
				if (kernel_type == 1 || kernel_type == 4) {
					// Beta kernel with max-pool: gradient depends on which branch won
					if (beta_wins && base > 1e-7f) {
						// Beta branch: dG/drho3d = -shape * alpha_beta / (base * k_sq)
						dG_factor = -shape_val * alpha_beta / (base * k_sq);
					} else {
						// Gaussian low-pass won, but we're in rho3d branch
						// alpha_lp depends on rho2d, not rho3d, so gradient is 0
						dG_factor = 0.0f;
					}
				} else if (kernel_type == 2) {
					// Flex kernel: same as Gaussian, use G_raw for position gradient
					// dL_dG has already been adjusted in the gradient section above
					dG_factor = -G_raw;
				} else if (kernel_type == 3) {
					// General kernel: dG/ds.x = dG/drho * drho/ds.x
					// dG/drho = -0.5 * G * (β/2) * pow_term / rho = -0.25 * β * G * pow_term / rho
					// drho/ds.x = 2 * s.x, so dG/ds.x = dG/drho * 2 * s.x
					// dG_factor should satisfy: dG_factor * s.x = dG/ds.x
					// Therefore: dG_factor = dG/drho * 2 = -0.5 * β * G * pow_term / rho
					dG_factor = -0.5f * general_beta * G * general_pow_term / general_rho_safe;
				} else {
					// Gaussian kernel: dG/drho = -0.5 * G, drho/ds.x = 2*s.x
					// Combined: dG/ds.x = -G * s.x
					dG_factor = -G;
				}
				float2 dL_ds = {
					dL_dG * dG_factor * s.x + dL_dz * Tw.x,
					dL_dG * dG_factor * s.y + dL_dz * Tw.y
				};

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
			} else {
				// Update gradients w.r.t. center of Gaussian 2D mean position
				// For rho2d case: rho2d = FilterInvSquare * (d.x² + d.y²)
				// For Gaussian kernel: dG/dd.x = -G * FilterInvSquare * d.x
				// For Beta kernel: dG/dd.x = -shape * G / base * FilterInvSquare * d.x
				// For Flex kernel: same as Gaussian but use G_raw
				// For General kernel: dG/dd.x = dG/drho * FilterInvSquare * d.x
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
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz); // propagate depth loss
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}

	// Flush tile-local MLP gradients to global memory (once per tile, bias-free)
	if ((render_mode & 0xFF) == 5 && dL_dmlp_W1 != nullptr) {
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
	const float3* dL_dmean2Ds, 
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
	float3 dL_dmean2D = dL_dmean2Ds[idx];
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
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
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
	float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dhomoMat,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
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
		dL_drots
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
	const float* colors,
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
	float * dL_dtransMat,
	float * dL_dhomoMat,
	float3* dL_dmean2D,
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
	float* dL_dmlp_W3)
{
	// Get MLP weight pointers for passing to the kernel (for 3D_direct_fused mode, FP16)
	__half *mlp_W1_ptr = nullptr;
	__half *mlp_W2_ptr = nullptr;
	__half *mlp_W3_ptr = nullptr;
	if (render_mode == 5) {
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

	if (render_mode == 5 && dL_dmlp_W1 != nullptr) {
		// Check for debug override to disable collaborative GEMM
		static int force_disable = -1;
		if (force_disable == -1) {
			const char* env = getenv("DISABLE_COLLABORATIVE_GEMM");
			force_disable = (env && atoi(env) != 0) ? 1 : 0;
		}

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
	if (render_mode == 5 && use_collaborative_gemm) {
		adjusted_render_mode = 5 | 0x100;  // Set bit 8 to indicate collaborative GEMM
	}

	// FP16 lean library: only C=3 (RGB) is needed for mode 5
	if (C != 3) {
		printf("diff_surfel_3D_16: Unsupported channel count %d (only C=3 supported)\n", C);
		return;
	}
	renderCUDAsurfelBackward<3, 0> <<<grid, block, smem_size>>>(
			ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
			means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
			dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
			hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, adjusted_render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad,
			dL_dmlp_W1, dL_dmlp_W2, dL_dmlp_W3,
			mlp_W1_ptr, mlp_W2_ptr, mlp_W3_ptr);

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
    // This must match forward.cu line 589: float2 pixf = { (float)pix.x, (float)pix.y};
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
    float* __restrict__ dL_dmeans)             // [N, 3] output
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
    // This converts from NDC [-1,1] to pixel coordinates [0,W-1] x [0,H-1]
    glm::mat3x4 ndc2pix = glm::mat3x4(
        glm::vec4(float(W) / 2.0f, 0.0f, 0.0f, float(W-1) / 2.0f),
        glm::vec4(0.0f, float(H) / 2.0f, 0.0f, float(H-1) / 2.0f),
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

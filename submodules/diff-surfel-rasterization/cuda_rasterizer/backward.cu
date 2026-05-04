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
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Periodic-freeze flag (mirrors diff_surfel_3D_sh_res). When true, the hash
// query in the backward kernel runs forward-only so feat[] stays correct for
// the alpha reconstruction, but no gradient is propagated into hash features
// or xyz from the hash path. Toggled from Python via BACKWARD::setSkipMlpGrad().
__device__ bool d_skip_mlp_grad = false;

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
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
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
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled (overwritten by preprocess)
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled (overwritten by preprocess)
				// AbsGS: per-pixel absolute screen-space grad magnitude. Survives the
				// preprocess overwrite (only .x/.y are overwritten there); .z/.w are
				// scaled by the same depth*W/H factor in preprocess to land in NDC units.
				atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dG * dG_ddelx));
				atomicAdd(&dL_dmean2D[global_id].w, fabsf(dL_dG * dG_ddely));
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
	const float* __restrict__ hash_features,
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
	const bool detach_hash_grad = false)
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
		// For cat mode (render_mode==1), level is encoded as:
		// (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
		// Decode to get actual hashgrid levels for offset copying
		int actual_levels = level;
		if(render_mode == 1){
			// cat mode: Extract active_hashgrid_levels from encoded value
			int active_hashgrid_levels = (level >> 8) & 0xFF;
			actual_levels = active_hashgrid_levels;  // Use ACTIVE hashgrid levels for coarse-to-fine
		} else if(level > 16){
			printf("Error: level %d  > 16.", level);
			return;
		}
		for(int l = 0; l <= actual_levels; l++) collec_offsets[l] = level_offsets[l];
		voxel_min = gridrange[0];
		voxel_max = gridrange[1];
	}
	
	// Dual hashgrid offsets (unused by current modes, kept for interface compatibility)
	int collec_offsets_diffuse[16] = {0};
	float voxel_min_diffuse = 0.0f;
	float voxel_max_diffuse = 0.0f;

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
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
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
		// Features are now queried on-demand in the per-pixel loop
		}
		block.sync();

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
					// Baseline mode: use l_dim directly (includes surface_blend with 12D features).
					// d_skip_mlp_grad: forward-only query, no gradient propagation into hash/xyz.
					if (!d_skip_mlp_grad) {
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
					} else {
						if(l_dim == 2) {
							query_feature<false, C, 2>(feat, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, nullptr, nullptr, nullptr);
						} else if(l_dim == 4) {
							query_feature<false, C, 4>(feat, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, nullptr, nullptr, nullptr);
						} else if(l_dim == 8) {
							query_feature<false, C, 8>(feat, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, nullptr, nullptr, nullptr);
						} else if(l_dim == 12) {
							query_feature<false, C, 12>(feat, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, nullptr, nullptr, nullptr);
						} else {
							printf("BW unsupported level dim : %d\n", l_dim);
						}
					}
					break;
			case 1: {
				// hybrid_features mode backward: Split gradients between per-Gaussian and hashgrid
				// Decode level parameter: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
				const int total_levels = level >> 16;  // Extract bits 16-31
				const int hybrid_levels = level & 0xFF;  // Extract bits 0-7
				const int per_gaussian_dim = hybrid_levels * l_dim;

				// Determine hashgrid levels from offsets array (hashgrid contains only the coarse levels)
				int hashgrid_levels = 0;
				for(int i = 1; i < 17; i++){
					if(i < 17 && collec_offsets[i] > collec_offsets[i-1]){
						hashgrid_levels = i;
					}
				}

				// Reconstruct per-Gaussian features into feat array (if present)
				if (hybrid_levels > 0) {
					for(int i = 0; i < per_gaussian_dim; i++){
						feat[i] = colors[global_id * per_gaussian_dim + i];
					}
				}

				// Backprop through hashgrid at xyz
				// Use separate buffers to avoid stack overflow: query_feature<true, 16*4, LD>
				// zeros C=64 elements, but feat/grad_feat are only C=24 (kernel template).
				if (hashgrid_levels > 0) {
					float feat_hashgrid[16 * 4] = {0};
					float grad_feat_hashgrid[16 * 4] = {0};
					const int hashgrid_dim = hashgrid_levels * l_dim;

					// Copy hash portion of grad_feat into separate buffer
					for(int i = 0; i < hashgrid_dim; i++){
						grad_feat_hashgrid[i] = grad_feat[per_gaussian_dim + i];
					}

					// d_skip_mlp_grad: forward-only query (populates feat_hashgrid for the
					// alpha reconstruction below); no backprop into hash features or xyz.
					if (!d_skip_mlp_grad) {
						if(l_dim == 2) {
							query_feature<true, 16 * 4, 2>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
								grad_feat_hashgrid, dL_dfeatures, dL_dxyz);
						} else if(l_dim == 4) {
							query_feature<true, 16 * 4, 4>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
								grad_feat_hashgrid, dL_dfeatures, dL_dxyz);
						} else if(l_dim == 8) {
							query_feature<true, 16 * 4, 8>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
								grad_feat_hashgrid, dL_dfeatures, dL_dxyz);
						} else if(l_dim == 12) {
							query_feature<true, 16 * 4, 12>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
								grad_feat_hashgrid, dL_dfeatures, dL_dxyz);
						} else {
							printf("BW unsupported level dim : %d\n", l_dim);
						}
					} else {
						if(l_dim == 2) {
							query_feature<false, 16 * 4, 2>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
								nullptr, nullptr, nullptr);
						} else if(l_dim == 4) {
							query_feature<false, 16 * 4, 4>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
								nullptr, nullptr, nullptr);
						} else if(l_dim == 8) {
							query_feature<false, 16 * 4, 8>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
								nullptr, nullptr, nullptr);
						} else if(l_dim == 12) {
							query_feature<false, 16 * 4, 12>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
								appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
								nullptr, nullptr, nullptr);
						} else {
							printf("BW unsupported level dim : %d\n", l_dim);
						}
					}

					// Copy hash features back into feat array
					for(int i = 0; i < hashgrid_dim; i++){
						feat[per_gaussian_dim + i] = feat_hashgrid[i];
					}
				}

				// Backprop to per-Gaussian features (first hybrid_levels×D of gradient)
				if (hybrid_levels > 0) {
					for(int i = 0; i < per_gaussian_dim; i++){
						atomicAdd(&(dL_dcolors[global_id * per_gaussian_dim + i]), grad_feat[i]);
					}
				}

				break;
			}
			default: printf("BW unsupported render_mode : %d\n", render_mode & 0xFF);
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
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled (overwritten by preprocess)
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled (overwritten by preprocess)
				// AbsGS: per-pixel absolute screen-space grad magnitude. Survives the
				// preprocess overwrite (only .x/.y are overwritten there); .z/.w are
				// scaled by the same depth*W/H factor in preprocess to land in NDC units.
				atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dG * dG_ddelx));
				atomicAdd(&dL_dmean2D[global_id].w, fabsf(dL_dG * dG_ddely));
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz); // propagate depth loss
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
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

	// AbsGS: scale the abs signal accumulated during the render backward
	// (per-pixel fabs(dL_dG * dG_ddelx)) to match the densification coordinate
	// system (same depth * 0.5 * W/H factor as x/y).
	dL_dmean2Ds[idx].z *= depth * 0.5f * float(W);
	dL_dmean2Ds[idx].w *= depth * 0.5f * float(H);
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
	const float* hash_features,
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
	const bool detach_hash_grad)
{
	switch (C) {
		case 3:
			renderCUDAsurfelBackward<3, 0> <<<grid, block>>>(
					ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
					means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
					dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
					hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
			break;
		case 8:
			renderCUDAsurfelBackward<8, 0> <<<grid, block>>>(
					ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
					means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
					dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
					hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
			break;
		case 16:
			renderCUDAsurfelBackward<16, 0> <<<grid, block>>>(
					ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
					means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
					dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
					hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
			break;
	case 24:
		// Always use D_DIFFUSE=0 template and handle dual hashgrids at runtime
		// This avoids shared memory issues from instantiating multiple templates
		renderCUDAsurfelBackward<24, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
		break;
	case 32:
		renderCUDAsurfelBackward<32, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
		break;
	case 42:
		renderCUDAsurfelBackward<42, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
		break;
	case 48:
		renderCUDAsurfelBackward<48, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
		break;
	case 72:
		renderCUDAsurfelBackward<72, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
		break;
	case 90:
		renderCUDAsurfelBackward<90, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum, cam_pos,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode, shapes, kernel_type, dL_dshapes, detach_hash_grad);
		break;
	default:
		printf("Unsupported channel count: %d\n", C);
	}

}

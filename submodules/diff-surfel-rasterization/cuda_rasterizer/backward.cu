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
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_gradsum,
	const float* __restrict__ hash_features_diffuse = nullptr,
	const int* __restrict__ level_offsets_diffuse = nullptr,
	const float* __restrict__ gridrange_diffuse = nullptr,
	float* __restrict__ dL_dfeatures_diffuse = nullptr,
	const int render_mode = 0)
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
	
	// Shared memory for per-Gaussian baseline features (dual hashgrid mode)
	// NOTE: Disabled for baseline_double/baseline_blend_double due to shared memory limits  
	// We query on-demand instead (less efficient but fits in shared memory)
	// Only used for surface_rgb mode (render_mode == 1)
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
		// For hybrid_features (render_mode==4), adaptive_add (render_mode==7), and adaptive_cat (render_mode==12), level is encoded as:
		// (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
		// Decode to get actual hashgrid levels for offset copying
		int actual_levels = level;
		if(render_mode == 4){
			// Extract active_hashgrid_levels from encoded value
			int total_levels = level >> 16;
			int active_hashgrid_levels = (level >> 8) & 0xFF;
			int hybrid_levels = level & 0xFF;
			actual_levels = active_hashgrid_levels;  // Use ACTIVE hashgrid levels for coarse-to-fine
		} else if(render_mode == 7){
			// adaptive_add mode: level = (total_levels << 16) | (active_hashgrid_levels << 8)
			int active_hashgrid_levels = (level >> 8) & 0xFF;
			actual_levels = active_hashgrid_levels;
		} else if((render_mode & 0xFF) == 12){
			// adaptive_cat mode: level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			// Note: render_mode may have inference flag in upper bits, so mask to get base mode
			int active_hashgrid_levels = (level >> 8) & 0xFF;
			actual_levels = active_hashgrid_levels;
		} else if(render_mode == 8 || render_mode == 9 || render_mode == 10){
			// hybrid_SH, hybrid_SH_post, and hybrid_SH_raw modes: level encodes (decompose_flag << 24) | (sh_degree << 16) | (1 << 8)
			// Don't check level > 16 since we use higher bits for encoding
			actual_levels = 1;  // Always 1 hashgrid level
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
	   (render_mode == 2 || render_mode == 3 || render_mode == 1)){
		for(int l = 0; l <= level; l++) collec_offsets_diffuse[l] = level_offsets_diffuse[l];
		voxel_min_diffuse = gridrange_diffuse[0];
		voxel_max_diffuse = gridrange_diffuse[1];
	}

	// baseline_blend_double: Precompute gradient for spatial hashgrid at blended position
	// This is done once per pixel BEFORE the Gaussian loop
	float dL_dblended_pos[3] = {0, 0, 0};
	// NOTE: Changed from compile-time D_DIFFUSE check to runtime check
	if(inside && render_mode == 3 && level > 0){
		// Get blended 3D position from other_maps
		float3 blended_pos = {
			other_maps[pix_id + (POS_OFFSET + 0) * H * W],
			other_maps[pix_id + (POS_OFFSET + 1) * H * W],
			other_maps[pix_id + (POS_OFFSET + 2) * H * W]
		};
		
		// Get appearance level (use first contributor's level)
		uint32_t pixel_ap_level = 0;
		if(ap_level != nullptr && last_contributor > 0){
			int first_gaussian_id = point_list[ranges[block.group_index().y * horizontal_blocks + block.group_index().x].x];
			pixel_ap_level = floorf(ap_level[first_gaussian_id]);
		}
		
		// Gradient from output: dL/dC flows through spatial features
		float grad_spatial[16 * 4];
		for(int ch = 0; ch < C; ch++){
			grad_spatial[ch] = dL_dpixel[ch];  // Gradient w.r.t. output color
		}
		
		// Backprop through spatial hashgrid query
		float feat_dummy[16 * 4];
		query_feature<true, 16 * 4, 4>(feat_dummy, blended_pos, voxel_min, voxel_max, collec_offsets, 
			pixel_ap_level, hash_features, level, l_scale, Base, align_corners, interp, if_contract, false,
			grad_spatial, dL_dfeatures, dL_dblended_pos);
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
			
		// NOTE: Per-Gaussian feature caching disabled due to shared memory limits
		// Features are now queried on-demand in the per-pixel loop (cases 4, 5, 12)
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

		float power = -0.5f * rho;
		if (power > 0.0f)
			continue;

		float G = exp(power), demon = 1.0;
			// const float beta = 0.0f;
			if(beta > 0.0){
				demon = 1.0 + beta * G;
				G = (1.0 + beta) * G / demon;
			}
			
			float alpha = min(0.99f, opa * G);
			
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
			switch (render_mode){
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
		case 2: {
			// baseline_double mode backward: dual 4D hashgrids
			// Forward: feat = [feat_xyz | feat_pk] (concatenation)
			// Gradients must be split: first part to hashgrid 1, second part to hashgrid 2

			float feat_xyz[16 * 4] = {0};   // Dummy for forward recompute (MUST initialize!)
			float feat_pk[16 * 4] = {0};    // Dummy for forward recompute (MUST initialize!)
			float dL_dpk_local[3] = {0};  // Not used (pk is fixed Gaussian center)

				// Determine number of levels for second hashgrid
				int level_diffuse = level;  // Default to same level count
				
				// Split gradients for concatenated features
				float grad_feat_xyz[16 * 4];
				float grad_feat_pk[16 * 4];
				for(int i = 0; i < level * 4; i++){
					grad_feat_xyz[i] = grad_feat[i];  // First part: hashgrid 1
				}
				for(int i = 0; i < level_diffuse * 4; i++){
					grad_feat_pk[i] = grad_feat[level * 4 + i];  // Second part: hashgrid 2
				}

				// Backprop through hashgrid 1 at xyz
				query_feature<true, 16 * 4, 4>(feat_xyz, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug,
					grad_feat_xyz, dL_dfeatures, dL_dxyz);

				// Backprop through hashgrid 2 at pk
				if(hash_features_diffuse != nullptr && dL_dfeatures_diffuse != nullptr && level_offsets_diffuse != nullptr) {
					int collec_offsets_pk[17];
					for(int lv = 0; lv <= level_diffuse; lv++){
						collec_offsets_pk[lv] = level_offsets_diffuse[lv];
					}
					float voxel_min_pk = gridrange_diffuse[0];
					float voxel_max_pk = gridrange_diffuse[1];

					query_feature<true, 16 * 4, 4>(feat_pk, pk, voxel_min_pk, voxel_max_pk,
						collec_offsets_pk, appearance_level, hash_features_diffuse, level_diffuse, l_scale, Base,
						align_corners, interp, contract, debug, grad_feat_pk, dL_dfeatures_diffuse, dL_dpk_local);
				}

				// Note: dL_dxyz is already updated by the first query_feature call
				// dL_dpk_local is not used (pk is a fixed Gaussian center, no gradient needed)

				break;
			}
				case 3: {
					// baseline_blend_double mode backward: dual 4D hashgrids
					// Forward: feat = feat_pk (per-Gaussian, blended), then feat_spatial added after loop
					// Gradients: per-Gaussian features get alpha-weighted gradients
					// Spatial features: gradient dL/dblended_pos distributed to xyz weighted by (alpha * T)
					
					// Backprop through per-Gaussian features (hashgrid 2)
					if(hash_features_diffuse != nullptr && dL_dfeatures_diffuse != nullptr && level_offsets_diffuse != nullptr) {
						// Gradient for per-Gaussian features: just alpha-weighted
						float grad_pk[16 * 4];
						for(int i = 0; i < level * 4; i++){
							grad_pk[i] = w * grad_feat[i];
						}
						
						float feat_dummy[16 * 4];
						float dL_dpk_local[3] = {0};
						
						int collec_offsets_pk[17];
						for(int lv = 0; lv <= level; lv++){
							collec_offsets_pk[lv] = level_offsets_diffuse[lv];
						}
						float voxel_min_pk = gridrange_diffuse[0];
						float voxel_max_pk = gridrange_diffuse[1];
						
						query_feature<true, 16 * 4, 4>(feat_dummy, pk, voxel_min_pk, voxel_max_pk, 
							collec_offsets_pk, appearance_level, hash_features_diffuse, level, l_scale, Base, 
							align_corners, interp, contract, debug, grad_pk, dL_dfeatures_diffuse, dL_dpk_local);
					}
					
					// Distribute spatial gradient to xyz (blended position gradient)
					// Forward: blended_pos = Σ (alpha * T * xyz)
					// Backward: dL/dxyz += (alpha * T) * dL/dblended_pos
					dL_dxyz[0] = w * dL_dblended_pos[0];  // w = alpha * T
					dL_dxyz[1] = w * dL_dblended_pos[1];
					dL_dxyz[2] = w * dL_dblended_pos[2];
					
					break;
				}
			case 4: {
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

				// Backprop through hashgrid at xyz - pass grad_feat slice directly like case 0
				if (hashgrid_levels > 0) {
					if(l_dim == 2) {
						query_feature<true, 16 * 4, 2>(&feat[per_gaussian_dim], xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
							&grad_feat[per_gaussian_dim], dL_dfeatures, dL_dxyz);
					} else if(l_dim == 4) {
						query_feature<true, 16 * 4, 4>(&feat[per_gaussian_dim], xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
							&grad_feat[per_gaussian_dim], dL_dfeatures, dL_dxyz);
					} else if(l_dim == 8) {
						query_feature<true, 16 * 4, 8>(&feat[per_gaussian_dim], xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
							&grad_feat[per_gaussian_dim], dL_dfeatures, dL_dxyz);
					} else if(l_dim == 12) {
						query_feature<true, 16 * 4, 12>(&feat[per_gaussian_dim], xyz, voxel_min, voxel_max, collec_offsets,
							appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
							&grad_feat[per_gaussian_dim], dL_dfeatures, dL_dxyz);
					} else {
						printf("BW unsupported level dim : %d\n", l_dim);
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
		case 6: {
			// adaptive mode backward: soft blend gradients
			// Forward: feat = mask * adaptive_features + (1-mask) * hashgrid_features
			// Backward: 
			//   d_adaptive_features = mask * grad_feat
			//   d_hashgrid_features = (1-mask) * grad_feat  
			//   d_mask = grad_feat * (adaptive_features - hashgrid_features) [handled in Python]
			
			const int num_levels = level;
			const int feat_dim = num_levels * l_dim;
			
			// Read per-Gaussian adaptive features and mask from colors buffer
			const float* adaptive_features = &colors[global_id * (2 * feat_dim)];
			const float* mask = &colors[global_id * (2 * feat_dim) + feat_dim];
			
			// Query full hashgrid at intersection point (needed for mask gradient)
			float feat_hashgrid[16 * 4] = {0};
			if(l_dim == 2) {
				query_feature<false, 16 * 4, 2>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, num_levels, l_scale, Base, align_corners, interp, contract, debug);
			} else if(l_dim == 4) {
				query_feature<false, 16 * 4, 4>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, num_levels, l_scale, Base, align_corners, interp, contract, debug);
			} else if(l_dim == 8) {
				query_feature<false, 16 * 4, 8>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, num_levels, l_scale, Base, align_corners, interp, contract, debug);
			} else if(l_dim == 12) {
				query_feature<false, 16 * 4, 12>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, num_levels, l_scale, Base, align_corners, interp, contract, debug);
			}
			
			// Compute scaled gradients for hashgrid backprop: grad_hash = (1-mask) * grad_feat
			float grad_hashgrid[16 * 4];
			for(int i = 0; i < feat_dim && i < 64; i++) {
				grad_hashgrid[i] = (1.0f - mask[i]) * grad_feat[i];
			}
			
			// Backprop to hashgrid with scaled gradients
			if(l_dim == 2) {
				query_feature<true, 16 * 4, 2>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, num_levels, l_scale, Base, align_corners, interp, contract, debug,
					grad_hashgrid, dL_dfeatures, dL_dxyz);
			} else if(l_dim == 4) {
				query_feature<true, 16 * 4, 4>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, num_levels, l_scale, Base, align_corners, interp, contract, debug,
					grad_hashgrid, dL_dfeatures, dL_dxyz);
			} else if(l_dim == 8) {
				query_feature<true, 16 * 4, 8>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, num_levels, l_scale, Base, align_corners, interp, contract, debug,
					grad_hashgrid, dL_dfeatures, dL_dxyz);
			} else if(l_dim == 12) {
				query_feature<true, 16 * 4, 12>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, num_levels, l_scale, Base, align_corners, interp, contract, debug,
					grad_hashgrid, dL_dfeatures, dL_dxyz);
			}
			
			// Backprop to adaptive features and mask
			// dL_dcolors layout: [d_adaptive | d_mask]
			for(int i = 0; i < feat_dim && i < 64; i++) {
				float d_adaptive = mask[i] * grad_feat[i];
				float d_mask = grad_feat[i] * (adaptive_features[i] - feat_hashgrid[i]);
				atomicAdd(&(dL_dcolors[global_id * (2 * feat_dim) + i]), d_adaptive);
				atomicAdd(&(dL_dcolors[global_id * (2 * feat_dim) + feat_dim + i]), d_mask);
			}
			
			break;
		}
		case 7: {
			// adaptive_add mode backward: weighted sum gradients
			// Forward: feat = weight * per_gaussian + (1 - weight) * hashgrid
			// Backward:
			//   d_per_gaussian = weight * grad_feat
			//   d_hashgrid = (1 - weight) * grad_feat
			//   d_weight = sum(grad_feat * (per_gaussian - hashgrid))
			
			// Decode level parameter: (total_levels << 16) | (active_hashgrid_levels << 8)
			const int total_levels = (level >> 16) & 0xFF;
			const int active_hashgrid_levels = (level >> 8) & 0xFF;
			const int feat_dim = total_levels * l_dim;
			
			// Read per-Gaussian features and weight from colors buffer
			// Layout: [feat_dim features | 1 weight] = feat_dim + 1 total
			const int colors_precomp_stride = feat_dim + 1;
			const float* per_gaussian_data = &colors[global_id * colors_precomp_stride];
			const float weight = per_gaussian_data[feat_dim];  // Already sigmoid'd
			const float inv_weight = 1.0f - weight;
			
			// Query hashgrid at intersection point (needed for weight gradient)
			float feat_hashgrid[16 * 4] = {0};
			if (active_hashgrid_levels > 0) {
				if(l_dim == 2) {
					query_feature<false, 16 * 4, 2>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, active_hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else if(l_dim == 4) {
					query_feature<false, 16 * 4, 4>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, active_hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else if(l_dim == 8) {
					query_feature<false, 16 * 4, 8>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, active_hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else if(l_dim == 12) {
					query_feature<false, 16 * 4, 12>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, active_hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				}
			}
			
			// Compute scaled gradients for hashgrid backprop: grad_hash = (1 - weight) * grad_feat
			float grad_hashgrid[16 * 4];
			for(int i = 0; i < feat_dim && i < 64; i++) {
				grad_hashgrid[i] = inv_weight * grad_feat[i];
			}
			
			// Backprop to hashgrid with scaled gradients
			if (active_hashgrid_levels > 0) {
				if(l_dim == 2) {
					query_feature<true, 16 * 4, 2>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, active_hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
						grad_hashgrid, dL_dfeatures, dL_dxyz);
				} else if(l_dim == 4) {
					query_feature<true, 16 * 4, 4>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, active_hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
						grad_hashgrid, dL_dfeatures, dL_dxyz);
				} else if(l_dim == 8) {
					query_feature<true, 16 * 4, 8>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, active_hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
						grad_hashgrid, dL_dfeatures, dL_dxyz);
				} else if(l_dim == 12) {
					query_feature<true, 16 * 4, 12>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, active_hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
						grad_hashgrid, dL_dfeatures, dL_dxyz);
				}
			}
			
			// Backprop to per-Gaussian features: d_per_gaussian = weight * grad_feat
			// and compute d_weight = sum(grad_feat * (per_gaussian - hashgrid))
			float d_weight = 0.0f;
			for(int i = 0; i < feat_dim && i < 64; i++) {
				float d_per_gaussian = weight * grad_feat[i];
				atomicAdd(&(dL_dcolors[global_id * colors_precomp_stride + i]), d_per_gaussian);
				
				// Accumulate weight gradient
				d_weight += grad_feat[i] * (per_gaussian_data[i] - feat_hashgrid[i]);
			}
			
			// Backprop through sigmoid: d_gamma = d_weight * weight * (1 - weight)
			// (sigmoid derivative: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)))
			float d_gamma = d_weight * weight * inv_weight;
			atomicAdd(&(dL_dcolors[global_id * colors_precomp_stride + feat_dim]), d_gamma);

			break;
		}
		case 8: {
			// hybrid_SH mode backward pass
			// Forward: gaussian_rgb = clamp(gaussian_raw + 0.5), hashgrid_rgb = sigmoid(hashgrid_raw)
			//          feat = clamp(gaussian_rgb + hashgrid_rgb)
			// Need to backprop through: clamp → addition → [clamp+0.5, sigmoid]
			
			// Decode level parameter
			const int decompose_flag = (level >> 24) & 0xFF;   // 0=normal, 1=gaussian_only, 2=ngp_only
			const int hashgrid_levels = (level >> 8) & 0xFF;   // Should be 1
			
			int gauss_id = collected_id[j];
			
			// Get forward pass values to compute derivatives
			float gaussian_raw[3];
			gaussian_raw[0] = colors[gauss_id * 3 + 0];
			gaussian_raw[1] = colors[gauss_id * 3 + 1];
			gaussian_raw[2] = colors[gauss_id * 3 + 2];
			
			// Recompute activated values
			float gaussian_rgb[3];
			gaussian_rgb[0] = fmaxf(0.0f, fminf(1.0f, gaussian_raw[0] + 0.5f));
			gaussian_rgb[1] = fmaxf(0.0f, fminf(1.0f, gaussian_raw[1] + 0.5f));
			gaussian_rgb[2] = fmaxf(0.0f, fminf(1.0f, gaussian_raw[2] + 0.5f));
			
			// Query hashgrid to get raw values for sigmoid derivative
			float hashgrid_raw[3] = {0.0f, 0.0f, 0.0f};
			if (hashgrid_levels > 0 && decompose_flag != 1) {
				query_feature<false, 3, 3>(hashgrid_raw, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, 1, l_scale, Base, align_corners, interp, contract, debug);
			}
			
			float hashgrid_rgb[3];
			hashgrid_rgb[0] = 1.0f / (1.0f + expf(-hashgrid_raw[0]));
			hashgrid_rgb[1] = 1.0f / (1.0f + expf(-hashgrid_raw[1]));
			hashgrid_rgb[2] = 1.0f / (1.0f + expf(-hashgrid_raw[2]));
			
			// Backprop through final clamp
			float grad_pre_clamp[3];
			for(int c = 0; c < 3; c++) {
				float sum_val = gaussian_rgb[c] + hashgrid_rgb[c];
				// Clamp derivative: 1 if in range [0,1], else 0
				grad_pre_clamp[c] = (sum_val >= 0.0f && sum_val <= 1.0f) ? grad_feat[c] : 0.0f;
			}
			
			// Backprop to gaussian_rgb and hashgrid_rgb
			if (decompose_flag != 2) {  // Not ngp_only
				for(int c = 0; c < 3; c++) {
					// Backprop through gaussian clamp (+0.5)
					float val_after_add = gaussian_raw[c] + 0.5f;
					float grad_gaussian_raw = (val_after_add >= 0.0f && val_after_add <= 1.0f) ? grad_pre_clamp[c] : 0.0f;
					atomicAdd(&dL_dcolors[gauss_id * 3 + c], alpha * T * grad_gaussian_raw);
				}
			}
			
			// Backprop to hashgrid through sigmoid
			if (hashgrid_levels > 0 && decompose_flag != 1) {  // Not gaussian_only
				float grad_hashgrid_raw[3];
				for(int c = 0; c < 3; c++) {
					// Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
					grad_hashgrid_raw[c] = alpha * T * grad_pre_clamp[c] * hashgrid_rgb[c] * (1.0f - hashgrid_rgb[c]);
				}
				
				float feat_dummy[3];
				query_feature<true, 3, 3>(feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, 1, l_scale, Base, align_corners, interp, contract, debug,
					grad_hashgrid_raw, dL_dfeatures, dL_dxyz);
			}

			break;
		}
		case 9: {
			// hybrid_SH_post mode backward pass
			// Forward: sh_coeffs[DC indices] += dc_residual from hashgrid, output all 48 coefficients
			// DC indices: 0 (R), 16 (G), 32 (B) due to PyTorch memory layout
			// Backward: grad flows to both SH coefficients and hashgrid

			// Decode level parameter
			const int decompose_flag = (level >> 24) & 0xFF;   // 0=normal, 1=gaussian_only, 2=ngp_only
			const int sh_degree = (level >> 16) & 0xFF;        // Active SH degree (0-3)
			const int hashgrid_levels = (level >> 8) & 0xFF;   // Should be 1

			int gauss_id = collected_id[j];

			// Backprop to SH coefficients
			// grad_feat contains gradients for all 48 coefficients
			if (decompose_flag != 2) {  // Not ngp_only - SH coefficients receive gradients
				for(int i = 0; i < 48; i++) {
					atomicAdd(&dL_dcolors[gauss_id * 48 + i], alpha * T * grad_feat[i]);
				}
			}

			// Backprop to hashgrid through DC residual
			// DC indices: 0 (R), 16 (G), 32 (B) due to PyTorch memory layout
			if (hashgrid_levels > 0 && decompose_flag != 1) {  // Not gaussian_only
				float grad_dc_residual[3];
				grad_dc_residual[0] = alpha * T * grad_feat[0];   // grad_DC_R
				grad_dc_residual[1] = alpha * T * grad_feat[16];  // grad_DC_G
				grad_dc_residual[2] = alpha * T * grad_feat[32];  // grad_DC_B

				float feat_dummy[3];
				query_feature<true, 3, 3>(feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, 1, l_scale, Base, align_corners, interp, contract, debug,
					grad_dc_residual, dL_dfeatures, dL_dxyz);
			}

			break;
		}
		case 10: {
			// hybrid_SH_raw mode backward pass
			// Forward: combined_raw = gaussian_raw + hashgrid_raw, feat = sigmoid(combined_raw)
			// Need to backprop through: sigmoid → addition → [gaussian_raw, hashgrid_raw]
			
			// Decode level parameter
			const int decompose_flag = (level >> 24) & 0xFF;   // 0=normal, 1=gaussian_only, 2=ngp_only
			const int hashgrid_levels = (level >> 8) & 0xFF;   // Should be 1
			
			int gauss_id = collected_id[j];
			
			// Get forward pass values
			float gaussian_raw[3];
			gaussian_raw[0] = colors[gauss_id * 3 + 0];
			gaussian_raw[1] = colors[gauss_id * 3 + 1];
			gaussian_raw[2] = colors[gauss_id * 3 + 2];
			
			// Query hashgrid to get raw values
			float hashgrid_raw[3] = {0.0f, 0.0f, 0.0f};
			if (hashgrid_levels > 0 && decompose_flag != 1) {
				query_feature<false, 3, 3>(hashgrid_raw, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, 1, l_scale, Base, align_corners, interp, contract, debug);
			}
			
			// Recompute combined and activated values
			float combined_raw[3];
			if (decompose_flag == 2) {
				combined_raw[0] = hashgrid_raw[0];
				combined_raw[1] = hashgrid_raw[1];
				combined_raw[2] = hashgrid_raw[2];
			} else if (decompose_flag == 1) {
				combined_raw[0] = gaussian_raw[0];
				combined_raw[1] = gaussian_raw[1];
				combined_raw[2] = gaussian_raw[2];
			} else {
				combined_raw[0] = gaussian_raw[0] + hashgrid_raw[0];
				combined_raw[1] = gaussian_raw[1] + hashgrid_raw[1];
				combined_raw[2] = gaussian_raw[2] + hashgrid_raw[2];
			}
			
			// Compute sigmoid values for derivative
			float sigmoid_val[3];
			sigmoid_val[0] = 1.0f / (1.0f + expf(-combined_raw[0]));
			sigmoid_val[1] = 1.0f / (1.0f + expf(-combined_raw[1]));
			sigmoid_val[2] = 1.0f / (1.0f + expf(-combined_raw[2]));
			
			// Backprop through sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
			float grad_combined_raw[3];
			for(int c = 0; c < 3; c++) {
				grad_combined_raw[c] = grad_feat[c] * sigmoid_val[c] * (1.0f - sigmoid_val[c]);
			}
			
			// Backprop to gaussian_raw
			if (decompose_flag != 2) {  // Not ngp_only
				for(int c = 0; c < 3; c++) {
					atomicAdd(&dL_dcolors[gauss_id * 3 + c], alpha * T * grad_combined_raw[c]);
				}
			}
			
			// Backprop to hashgrid_raw
			if (hashgrid_levels > 0 && decompose_flag != 1) {  // Not gaussian_only
				float grad_hashgrid_raw[3];
				for(int c = 0; c < 3; c++) {
					grad_hashgrid_raw[c] = alpha * T * grad_combined_raw[c];
				}
				
				float feat_dummy[3];
				query_feature<true, 3, 3>(feat_dummy, xyz, voxel_min, voxel_max, collec_offsets,
					appearance_level, hash_features, 1, l_scale, Base, align_corners, interp, contract, debug,
					grad_hashgrid_raw, dL_dfeatures, dL_dxyz);
			}
			
			break;
		}
		case 11: {
			// residual_hybrid mode backward: Gradients for dual outputs
			// Per-Gaussian SH→RGB (precomputed) + MLP(alpha-blended hashgrid features)
			// grad_feat[C] contains gradients w.r.t. hashgrid features from MLP backward pass

			// Decode level parameter
			const int hybrid_levels = level & 0xFF;
			const int hashgrid_levels = (level >> 8) & 0xFF;
			const int hashgrid_dim = hashgrid_levels * l_dim;

			// Backprop through hashgrid query at 3D intersection point
			// grad_feat[C] should contain gradients for first hashgrid_dim channels
			if (hashgrid_levels > 0 && hashgrid_dim <= C) {
				float feat_hashgrid[16 * 4];
				float grad_hashgrid[16 * 4] = {0.0f};
				float dL_dxyz_local[3] = {0.0f};
				
				// Copy gradients for hashgrid features (first hashgrid_dim elements)
				for(int i = 0; i < hashgrid_dim && i < C; i++) {
					grad_hashgrid[i] = grad_feat[i];
				}

				if(l_dim == 2) {
					query_feature<true, 16 * 4, 2>(feat_hashgrid, xyz, voxel_min, voxel_max,
						collec_offsets, appearance_level, hash_features, hashgrid_levels,
						l_scale, Base, align_corners, interp, contract, debug,
						grad_hashgrid, dL_dfeatures, dL_dxyz_local);
				} else if(l_dim == 4) {
					query_feature<true, 16 * 4, 4>(feat_hashgrid, xyz, voxel_min, voxel_max,
						collec_offsets, appearance_level, hash_features, hashgrid_levels,
						l_scale, Base, align_corners, interp, contract, debug,
						grad_hashgrid, dL_dfeatures, dL_dxyz_local);
				} else if(l_dim == 8) {
					query_feature<true, 16 * 4, 8>(feat_hashgrid, xyz, voxel_min, voxel_max,
						collec_offsets, appearance_level, hash_features, hashgrid_levels,
						l_scale, Base, align_corners, interp, contract, debug,
						grad_hashgrid, dL_dfeatures, dL_dxyz_local);
				} else if(l_dim == 12) {
					query_feature<true, 16 * 4, 12>(feat_hashgrid, xyz, voxel_min, voxel_max,
						collec_offsets, appearance_level, hash_features, hashgrid_levels,
						l_scale, Base, align_corners, interp, contract, debug,
						grad_hashgrid, dL_dfeatures, dL_dxyz_local);
				}
				
				// Note: dL_dxyz_local gradients could be accumulated if needed for position gradients
				// For now, we don't use them since xyz is computed from Gaussian parameters
			}

			// Note: SH RGB gradients handled separately in accumulation loop
			// (SH coefficients are precomputed in Python, gradients flow via autograd)

			break;
		}
		case 12: {
			/* adaptive_cat mode backward: Additive blending gradients
			 * Forward: coarse *= w, fine = w * gauss + (1-w) * hash
			 * Gradients flow to: gaussian features (scaled by w), weight, and hashgrid
			 */

			const int total_levels = (level >> 16) & 0xFF;
			const int hashgrid_levels = (level >> 8) & 0xFF;
			const int hybrid_levels = level & 0xFF;
			const int per_level_dim = l_dim;
			const int total_dim = total_levels * per_level_dim;

			// Validate decoded values are reasonable
			if (total_levels == 0 || total_levels > 32 || hashgrid_levels > 32 || hybrid_levels > 32) {
				if (debug) {
					printf("BW Error: Invalid decoded level values. level=%d, total=%d, hash=%d, hybrid=%d\n",
					       level, total_levels, hashgrid_levels, hybrid_levels);
				}
				break;
			}

			int gauss_id = collected_id[j];
			const float weight = colors[gauss_id * (total_dim + 1) + total_dim];
			const float* gauss_feat = &colors[gauss_id * (total_dim + 1)];

			// Coarse levels: dL/dgauss = w * dL/dfeat
			for(int i = 0; i < hybrid_levels * per_level_dim; i++) {
				atomicAdd(&dL_dcolors[gauss_id * (total_dim + 1) + i], weight * grad_feat[i]);
			}

			// Gradient for weight from coarse levels: dL/dw = sum(gauss * dL/dfeat)
			float dL_dweight = 0.0f;
			for(int i = 0; i < hybrid_levels * per_level_dim; i++) {
				dL_dweight += gauss_feat[i] * grad_feat[i];
			}

			// Fine levels: blend gradients
			if (hashgrid_levels > 0) {
				// Re-query hashgrid for gradient computation
				float hash_feat[16 * 4];

				if(l_dim == 2) {
					query_feature<false, 16*4, 2>(hash_feat, xyz, voxel_min, voxel_max,
					                               collec_offsets, appearance_level, hash_features,
					                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else if(l_dim == 4) {
					query_feature<false, 16*4, 4>(hash_feat, xyz, voxel_min, voxel_max,
					                               collec_offsets, appearance_level, hash_features,
					                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else if(l_dim == 8) {
					query_feature<false, 16*4, 8>(hash_feat, xyz, voxel_min, voxel_max,
					                               collec_offsets, appearance_level, hash_features,
					                               hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				}

				const int fine_start = hybrid_levels * per_level_dim;

				// Gradient for per-Gaussian fine features: dL/dgauss_fine = w * dL/dfeat_fine
				for(int i = 0; i < hashgrid_levels * per_level_dim; i++) {
					atomicAdd(&dL_dcolors[gauss_id * (total_dim + 1) + fine_start + i],
					          weight * grad_feat[fine_start + i]);
				}

				// Gradient for weight from fine levels: dL/dw += sum((gauss - hash) * dL/dfeat)
				for(int i = 0; i < hashgrid_levels * per_level_dim; i++) {
					dL_dweight += (gauss_feat[fine_start + i] - hash_feat[i]) * grad_feat[fine_start + i];
				}

				// Gradient for hashgrid: dL/dhash = (1-w) * dL/dfeat
				float grad_hash[16 * 4];
				for(int i = 0; i < hashgrid_levels * per_level_dim; i++) {
					grad_hash[i] = (1.0f - weight) * grad_feat[fine_start + i];
				}

				// Backprop through hashgrid
				float hash_feat_dummy[16 * 4];
				if(l_dim == 2) {
					query_feature<true, 16*4, 2>(hash_feat_dummy, xyz, voxel_min, voxel_max,
					                              collec_offsets, appearance_level, hash_features,
					                              hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
					                              grad_hash, dL_dfeatures, dL_dxyz);
				} else if(l_dim == 4) {
					query_feature<true, 16*4, 4>(hash_feat_dummy, xyz, voxel_min, voxel_max,
					                              collec_offsets, appearance_level, hash_features,
					                              hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
					                              grad_hash, dL_dfeatures, dL_dxyz);
				} else if(l_dim == 8) {
					query_feature<true, 16*4, 8>(hash_feat_dummy, xyz, voxel_min, voxel_max,
					                              collec_offsets, appearance_level, hash_features,
					                              hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug,
					                              grad_hash, dL_dfeatures, dL_dxyz);
				}
			}

			// Write weight gradient
			atomicAdd(&dL_dcolors[gauss_id * (total_dim + 1) + total_dim], dL_dweight);

			break;
		}
		case 1: {
		// Surface mode backward with optional baseline features
		// Forward: feat = ReLU(-dot(vec, normal) + baseline)
		// Gradients flow to both surface hashgrid and baseline hashgrid (if present)
		
		float grad_feat_vec[16 * 12];  // Gradient w.r.t. surface hashgrid
		for(int i = 0; i < 16 * 12; i++) grad_feat_vec[i] = 0.0f;
		
		// Gradient w.r.t. normal (accumulate across all levels)
		float dL_dnormal[3] = {0.0f, 0.0f, 0.0f};
		
		// Query surface features from forward pass to compute normal gradient
		float feat_vec[16 * 12];
		query_feature<false, 16 * 12, 12>(feat_vec, xyz, voxel_min, voxel_max, collec_offsets, 
			appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
		
	// Query per-Gaussian baseline features on-demand (if dual hashgrid mode)
	// NOTE: Changed to on-demand querying instead of caching due to shared memory limits
	float feat_baseline[16 * 4] = {0};
	if(render_mode == 1 && hash_features_diffuse != nullptr){
		const float3 pk_center = collected_pk[j];
		int collec_offsets_pk[17];
		for(int lv = 0; lv <= level; lv++){
			collec_offsets_pk[lv] = level_offsets_diffuse[lv];
		}
		float voxel_min_pk = gridrange_diffuse[0];
		float voxel_max_pk = gridrange_diffuse[1];
		
		query_feature<false, 16 * 4, 4>(feat_baseline, pk_center, voxel_min_pk, voxel_max_pk, collec_offsets_pk, 
			appearance_level, hash_features_diffuse, level, l_scale, Base, align_corners, interp, contract, debug);
	}
		
		// Expand gradients through dot product with ReLU
		for(int lv = 0; lv < level; lv++){
			int scalar_start = lv * 4;  // Gradient position
			int vec_start = lv * 12;     // Surface feature position
			
			for(int i = 0; i < 4; i++){
				// Recompute forward value to check ReLU activation
				float dot_prod = 0.0f;
				for(int j = 0; j < 3; j++){
					dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
				}
				// feat_baseline is now a local array (initialized to zero if not queried)
				float baseline_val = feat_baseline[lv * 4 + i];
				float forward_val = -dot_prod + baseline_val;
				
				// ReLU gradient: zero if forward_val <= 0
				float dL_dscalar = (forward_val > 0.0f) ? grad_feat[scalar_start + i] : 0.0f;
				
				// dL/dvec = -dL/dscalar * normal (chain rule for dot product)
				for(int j = 0; j < 3; j++){
					grad_feat_vec[vec_start + i * 3 + j] = -dL_dscalar * normal[j];
				}
				// dL/dnormal += -dL/dscalar * vec
				for(int j = 0; j < 3; j++){
					dL_dnormal[j] += -dL_dscalar * feat_vec[vec_start + i * 3 + j];
				}
		}
	}
		
		// Accumulate normal gradient (normalization backprop handled in compute_transmat_aabb)
		atomicAdd(&dL_dnormal3D[global_id * 3 + 0], alpha * T * dL_dnormal[0]);
		atomicAdd(&dL_dnormal3D[global_id * 3 + 1], alpha * T * dL_dnormal[1]);
		atomicAdd(&dL_dnormal3D[global_id * 3 + 2], alpha * T * dL_dnormal[2]);
			
			// Backprop through surface hashgrid
			float feat_dummy[16 * 12];
			query_feature<true, 16 * 12, 12>(feat_dummy, xyz, voxel_min, voxel_max, collec_offsets, 
				appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, 
				grad_feat_vec, dL_dfeatures, dL_dxyz);
			
			// Backprop through baseline hashgrid (if dual hashgrid mode)
			if(hash_features_diffuse != nullptr && dL_dfeatures_diffuse != nullptr && level_offsets_diffuse != nullptr) {
				// Gradient w.r.t. baseline: dL/dbaseline = dL/dfeat * ReLU_mask
				float grad_baseline[16 * 4];
				for(int lv = 0; lv < level; lv++){
					for(int c = 0; c < 4; c++){
						// Recompute forward value for ReLU mask
						int vec_start = lv * 12;
						float dot_prod = 0.0f;
						for(int j = 0; j < 3; j++){
							dot_prod += feat_vec[vec_start + c * 3 + j] * normal[j];
						}
						float baseline_val = feat_baseline ? feat_baseline[lv * 4 + c] : 0.0f;
						float forward_val = -dot_prod + baseline_val;
						float relu_mask = (forward_val > 0.0f) ? 1.0f : 0.0f;
						
						grad_baseline[lv * 4 + c] = w * grad_feat[lv * 4 + c] * relu_mask;
					}
				}
				
				// Backprop through baseline hashgrid at pk (Gaussian center)
				float feat_dummy_baseline[16 * 4];
				float dL_dpk[3] = {0};
				
				int collec_offsets_baseline[17];
				for(int l = 0; l <= level; l++) collec_offsets_baseline[l] = level_offsets_diffuse[l];
				float voxel_min_baseline = gridrange_diffuse[0];
				float voxel_max_baseline = gridrange_diffuse[1];
				
				query_feature<true, 16 * 4, 4>(feat_dummy_baseline, pk, voxel_min_baseline, voxel_max_baseline, 
					collec_offsets_baseline, appearance_level, hash_features_diffuse, level, l_scale, Base, 
					align_corners, interp, contract, debug, grad_baseline, dL_dfeatures_diffuse, dL_dpk);
			}
			
		break;
	}
		case 5: {
			// Surface RGB mode backward: backprop through ReLU(-dot product) + RGB
			// Output has 7 features per level: 4 scalars (from ReLU(-dot)) + 3 RGB
			
			// Buffer sized for actual number of levels: level * 15 features
			const int vec_buffer_size = level * 15;  // e.g., 6 levels × 15 = 90
			float grad_feat_vec[16 * 15];  // Max supported: 16 levels × 15 = 240
			for(int i = 0; i < 16 * 15; i++) grad_feat_vec[i] = 0.0f;
			
			float grad_feat_rgb[16 * 15];  // Gradients for RGB query at pk
			for(int i = 0; i < 16 * 15; i++) grad_feat_rgb[i] = 0.0f;
			
			// Gradient w.r.t. normal (accumulate across all levels and features)
			float dL_dnormal[3] = {0.0f, 0.0f, 0.0f};
			
			// Query the features from forward pass to compute normal gradient
			float feat_vec[16 * 15];
			query_feature<false, 16 * 15, 15>(feat_vec, xyz, voxel_min, voxel_max, collec_offsets, 
				appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
			
			// Process each level
			for(int lv = 0; lv < level; lv++){
				int out_start = lv * 7;     // Position in grad_feat (4 scalars + 3 RGB)
				int vec_start = lv * 15;    // Position in hashgrid output
				
				// Backprop through ReLU and dot product for first 4 features with normal
				for(int i = 0; i < 4; i++){
					// Recompute forward value to check ReLU activation
					float dot_prod = 0.0f;
					for(int j = 0; j < 3; j++){
						dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
					}
					float forward_val = -dot_prod;  // No baseline in case 5
					
					// ReLU gradient: zero if forward_val <= 0
					float dL_dscalar = (forward_val > 0.0f) ? grad_feat[out_start + i] : 0.0f;
					
					// dL/dvec = -dL/dscalar * normal
					for(int j = 0; j < 3; j++){
						grad_feat_vec[vec_start + i * 3 + j] = -dL_dscalar * normal[j];
					}
					// dL/dnormal = -dL/dscalar * vec
					for(int j = 0; j < 3; j++){
						dL_dnormal[j] += -dL_dscalar * feat_vec[vec_start + i * 3 + j];
					}
				}
				
			// Copy RGB gradients (queried at pk)
			for(int c = 0; c < 3; c++){
				grad_feat_rgb[vec_start + 12 + c] = grad_feat[out_start + 4 + c];
		}
	}
		
		// Accumulate normal gradient (normalization backprop handled in compute_transmat_aabb)
		atomicAdd(&dL_dnormal3D[global_id * 3 + 0], alpha * T * dL_dnormal[0]);
		atomicAdd(&dL_dnormal3D[global_id * 3 + 1], alpha * T * dL_dnormal[1]);
		atomicAdd(&dL_dnormal3D[global_id * 3 + 2], alpha * T * dL_dnormal[2]);
				
				// Dummy arrays for feat (not used in backward)
				float feat_dummy_vec[16 * 15];
				float feat_dummy_rgb[16 * 15];
				
				// Backprop through hashgrid at xyz for vector potentials
				float dL_dxyz_vec[3] = {0};
				query_feature<true, 16 * 15, 15>(feat_dummy_vec, xyz, voxel_min, voxel_max, collec_offsets, 
					appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, grad_feat_vec, dL_dfeatures, dL_dxyz_vec);
				
				// Add to total position gradient
				for(int i = 0; i < 3; i++) dL_dxyz[i] += dL_dxyz_vec[i];
				
				// Backprop through hashgrid at pk for RGB
				float dL_dxyz_rgb[3] = {0};
				query_feature<true, 16 * 15, 15>(feat_dummy_rgb, pk, voxel_min, voxel_max, collec_offsets, 
					appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug, grad_feat_rgb, dL_dfeatures, dL_dxyz_rgb);
				// Note: pk gradients not used since Gaussian center has other gradient sources
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

			if(beta > 0.0){
				// with beta activation
				const float dG_dg = (1.0 + beta) / (demon * demon); 
				dL_dG *=  dG_dg; // dL_dg now infact
			}
			// dL_dG *= 0.8;


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
				float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
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
	float3* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_gradsum,
	const uint32_t D_diffuse,
	const float* hash_features_diffuse,
	const int* level_offsets_diffuse,
	const float* gridrange_diffuse,
	float* dL_dfeatures_diffuse,
	const int render_mode)
{
	// Determine D_DIFFUSE template parameter for kernel dispatch
	const uint32_t D_DIFFUSE_TEMPLATE = D_diffuse;
	
	switch (C) {
		case 3:
			renderCUDAsurfelBackward<3, 0> <<<grid, block>>>(
					ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
					means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
					dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
					hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
			break;
		case 8:
			renderCUDAsurfelBackward<8, 0> <<<grid, block>>>(
					ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
					means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
					dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
					hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
			break;
		case 16:
			renderCUDAsurfelBackward<16, 0> <<<grid, block>>>(
					ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
					means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
					dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
					hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
			break;
	case 24:
		// Always use D_DIFFUSE=0 template and handle dual hashgrids at runtime
		// This avoids shared memory issues from instantiating multiple templates
		renderCUDAsurfelBackward<24, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
		break;
	case 32:
		renderCUDAsurfelBackward<32, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
		break;
	case 42:
		renderCUDAsurfelBackward<42, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
		break;
	case 48:
		renderCUDAsurfelBackward<48, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
		break;
	case 72:
		renderCUDAsurfelBackward<72, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
		break;
	case 90:
		renderCUDAsurfelBackward<90, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, scales, focal_x, focal_y, other_maps, out_index, bg_color,
				means2D, normal_opacity, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange, colors, depths, final_Ts, n_contrib,
				dL_dpixels, dL_depths, dL_dfeatures, dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, dL_dopacity, dL_dcolors, dL_gradsum,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, dL_dfeatures_diffuse, render_mode);
		break;
	default:
		printf("Unsupported channel count: %d\n", C);
	}

}

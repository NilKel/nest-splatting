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
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

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
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
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

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	// float cutoff = 3.0f;
	float cutoff = 4.0f;
#endif

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors
	if (colors_precomp == nullptr) {
		// printf("Error , colors precomp is nullptr.\n");
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}
	else {
		// For hybrid_features mode: colors_precomp contains per-Gaussian learned features
		// Copy them to rgb buffer so they're accessible in the render kernel
		for(int i = 0; i < C; i++){
			rgb[idx * C + i] = colors_precomp[idx * C + i];
		}
	}
	
	// if(idx == 0 ){
	// 	printf("depth %.4f\n", p_view.z);
	// }
	depths[idx] = p_view.z;
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
	const float* __restrict__ hash_features,
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
	const float* __restrict__ hash_features_diffuse = nullptr,
	const int* __restrict__ level_offsets_diffuse = nullptr,
	const float* __restrict__ gridrange_diffuse = nullptr,
	const int render_mode = 0,
	const float* __restrict__ rgb = nullptr)
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
	
	// Shared memory for per-Gaussian baseline features (dual hashgrid mode)
	// NOTE: Disabled for baseline_double/baseline_blend_double due to shared memory limits
	// We query on-demand instead (less efficient but fits in shared memory)
	// __shared__ float collected_feat_pk[BLOCK_SIZE][6 * 4];  // 6 levels × 4 features per Gaussian

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
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
		// For hybrid_features (render_mode==4), 'level' contains hybrid_levels
		// Hashgrid levels are inferred from offsets array size
		int hashgrid_levels = level;  // Default: use level as-is

		if(render_mode == 4){
			// hybrid_features mode: 'level' = hybrid_levels (per-Gaussian feature levels)
			// Hashgrid levels = size of offsets array - 1
			// We need to copy offsets for the hashgrid query
			// Count levels from offsets array by checking how many are provided
			// This will be set properly when we query - for now just note hybrid_levels
			int hybrid_levels = level;
			// Hashgrid levels determined from offsets during query
			hashgrid_levels = 16;  // Max possible, will be clamped by actual offsets
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
		
		// NOTE: Per-Gaussian feature caching disabled due to shared memory limits
		// Features are now queried on-demand in the per-pixel loop (cases 4, 5, 12)

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
		float normal[3] = {nor_o.x, nor_o.y, nor_o.z};  // Already normalized in preprocessing
		float opa = nor_o.w;

		float power = -0.5f * rho ;
		if (power > 0.0f)
			continue;

		// Eq. (2) from 3D Gaussian splatting paper.
		// Obtain alpha by multiplying with Gaussian opacity
		// and its exponential falloff from mean.
		// Avoid numerical instabilities (see paper appendix). 
		float G = exp(power);

		if(beta > 0.0)
			G = (1.0 + beta) * G / (1.0 + beta * G);
		
		float alpha = min(0.99f, opa * G);

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
			switch (render_mode){
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
		case 2: {
			// baseline_double mode: dual 4D hashgrids
			// Hashgrid 1: 4D features per level, queried at xyz (intersection)
			// Hashgrid 2: 4D features per level, queried at pk (Gaussian center) - on-demand query
			// Output: [feat_xyz | feat_pk] concatenated

			float feat_xyz[16 * 4] = {0};   // Features from intersection point (MUST initialize!)
			float feat_pk[16 * 4] = {0};    // Features from Gaussian center (init to 0 since conditionally queried)
				
				// Query hashgrid 1 at intersection point (xyz)
				query_feature<false, 16 * 4, 4>(feat_xyz, xyz, voxel_min, voxel_max, collec_offsets, 
					appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
				
				// Determine number of levels for second hashgrid (may differ from first)
				int level_diffuse = level;  // Default to same level count
				// TODO: Could infer from level_offsets_diffuse if needed
				
				// Query hashgrid 2 at Gaussian center (pk) - on-demand instead of cached
				if(hash_features_diffuse != nullptr && level_offsets_diffuse != nullptr){
					const float3 pk_center = collected_pk[j];
					int collec_offsets_pk[17];
					for(int lv = 0; lv <= level_diffuse; lv++){
						collec_offsets_pk[lv] = level_offsets_diffuse[lv];
					}
					float voxel_min_pk = gridrange_diffuse[0];
					float voxel_max_pk = gridrange_diffuse[1];
					
					query_feature<false, 16 * 4, 4>(feat_pk, pk_center, voxel_min_pk, voxel_max_pk, collec_offsets_pk, 
						appearance_level, hash_features_diffuse, level_diffuse, l_scale, Base, align_corners, interp, contract, debug);
				}
				
				// Concatenate features: first hashgrid 1, then hashgrid 2
				// Output: [level*4 from xyz | level_diffuse*4 from pk]
				for(int i = 0; i < level * 4; i++){
					feat[i] = feat_xyz[i];
				}
				for(int i = 0; i < level_diffuse * 4; i++){
					feat[level * 4 + i] = feat_pk[i];
				}
				
				break;
			}
				case 3: {
					// baseline_blend_double mode: dual 4D hashgrids
					// Hashgrid 1: Queried at blended 3D position (AFTER alpha blending) - done outside loop
					// Hashgrid 2: 4D features per level, queried at pk (Gaussian center) - on-demand query
					// Strategy: Blend per-Gaussian features here, query spatial hashgrid after loop
					
					float feat_pk[16 * 4] = {0};  // Features from Gaussian center (init to 0)
					
					// Query hashgrid 2 at Gaussian center (pk) - on-demand instead of cached
					if(hash_features_diffuse != nullptr && level_offsets_diffuse != nullptr){
						const float3 pk_center = collected_pk[j];
						int collec_offsets_pk[17];
						for(int lv = 0; lv <= level; lv++){
							collec_offsets_pk[lv] = level_offsets_diffuse[lv];
						}
						float voxel_min_pk = gridrange_diffuse[0];
						float voxel_max_pk = gridrange_diffuse[1];
						
						query_feature<false, 16 * 4, 4>(feat_pk, pk_center, voxel_min_pk, voxel_max_pk, collec_offsets_pk, 
							appearance_level, hash_features_diffuse, level, l_scale, Base, align_corners, interp, contract, debug);
					}
					
					// Copy per-Gaussian features to feat for alpha blending
					for(int i = 0; i < level * 4; i++){
						feat[i] = feat_pk[i];
					}
					// Note: Spatial features will be added AFTER the Gaussian loop using blended position
					
					break;
				}
		case 4: {
			// hybrid_features mode: Combine per-Gaussian and hashgrid features
			// Design: Per-Gaussian features (hybrid_levels × D) + Hashgrid features (hashgrid_levels × D)
			// The hashgrid is already configured in Python to contain only the finest levels
			// Example: hybrid_levels=1, hashgrid=5 levels → per_gaussian=1×4=4D, hashgrid=5×4=20D

			// Initialize feat array to zero (use outer feat, don't shadow it!)
			for(int i = 0; i < CHANNELS; i++) feat[i] = 0.0f;

			// Decode level parameter: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
			const int hybrid_levels = level & 0xFF;  // Extract bits 0-7: per-Gaussian feature levels
			const int per_gaussian_dim = hybrid_levels * l_dim;  // e.g., 1×4 = 4D

			// Determine hashgrid levels from offsets array
			// The hashgrid passed to CUDA already contains only the finest (total - hybrid) levels
			int hashgrid_levels = 0;
			for(int i = 1; i < 17; i++){
				if(i < 17 && collec_offsets[i] > collec_offsets[i-1]){
					hashgrid_levels = i;
				}
			}

			const int hashgrid_dim = hashgrid_levels * l_dim;

			// Copy per-Gaussian features from rgb buffer (colors_precomp)
			if (hybrid_levels > 0 && rgb != nullptr) {
				int gauss_id = collected_id[j];
				const float* per_gaussian_feat = &rgb[gauss_id * per_gaussian_dim];
				for(int i = 0; i < per_gaussian_dim; i++){
					feat[i] = per_gaussian_feat[i];  // First part: per-Gaussian features
				}
			} else if (hybrid_levels > 0 && rgb == nullptr && debug) {
				printf("WARNING: hybrid_levels=%d but rgb is nullptr! Per-Gaussian features will be zero.\n", hybrid_levels);
			}

			// Query hashgrid directly - it's already configured with only the finest levels
			// No need to query full hashgrid and filter
			if (hashgrid_levels > 0) {
				float feat_hashgrid[16 * 4];  // Buffer for hashgrid features

				// Query the hashgrid (which contains only finest levels)
				if(l_dim == 2) {
					query_feature<false, 16 * 4, 2>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else if(l_dim == 4) {
					query_feature<false, 16 * 4, 4>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else if(l_dim == 8) {
					query_feature<false, 16 * 4, 8>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else if(l_dim == 12) {
					query_feature<false, 16 * 4, 12>(feat_hashgrid, xyz, voxel_min, voxel_max, collec_offsets,
						appearance_level, hash_features, hashgrid_levels, l_scale, Base, align_corners, interp, contract, debug);
				} else {
					printf("FW unsupported level dim : %d\n", l_dim);
				}

				// Concatenate hashgrid features after per-Gaussian features
				for(int i = 0; i < hashgrid_dim; i++){
					feat[per_gaussian_dim + i] = feat_hashgrid[i];
				}
			}

			break;
		}
		case 6: {
			// adaptive mode: blend per-Gaussian and hashgrid features
			// rgb buffer: [N, feat_dim] = per-Gaussian features (24D)
			// Mask is computed in Python for autograd, blending happens here
			// Output: feat = 0.5 * per_gaussian + 0.5 * hashgrid (uniform blend for now)
			
			// Initialize feat array to zero
			for(int i = 0; i < CHANNELS; i++) feat[i] = 0.0f;
			
			const int num_levels = level;
			const int feat_dim = num_levels * l_dim;
			
			// Read per-Gaussian features from rgb buffer (24D)
			int gauss_id = collected_id[j];
			const float* per_gaussian_feat = &rgb[gauss_id * CHANNELS];
			
			// Query full hashgrid at intersection point (24D)
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
			} else {
				printf("FW unsupported level dim : %d\n", l_dim);
			}
			
			// Simple blend: feat = 0.5 * per_gaussian + 0.5 * hashgrid
			// TODO: Use learnable mask from gamma parameter
			for(int i = 0; i < feat_dim && i < CHANNELS; i++) {
				feat[i] = 0.5f * per_gaussian_feat[i] + 0.5f * feat_hashgrid[i];
			}
			
			break;
		}
		case 7: {
			// adaptive_add mode: weighted sum of per-Gaussian and hashgrid features
			// rgb buffer: [N, feat_dim + 1] = [per-Gaussian features (feat_dim) | weight (1)]
			// weight = sigmoid(gamma) in [0, 1], passed from Python
			// Output: feat = weight * per_gaussian + (1 - weight) * hashgrid
			
			// Initialize feat array to zero
			for(int i = 0; i < CHANNELS; i++) feat[i] = 0.0f;
			
			// Decode level parameter: (total_levels << 16) | (active_hashgrid_levels << 8)
			const int total_levels = (level >> 16) & 0xFF;  // Extract bits 16-23
			const int active_hashgrid_levels = (level >> 8) & 0xFF;  // Extract bits 8-15
			const int feat_dim = total_levels * l_dim;  // e.g., 6 × 4 = 24D
			
			// Read per-Gaussian features and weight from rgb buffer
			// Layout: [feat_dim features | 1 weight] = feat_dim + 1 total
			int gauss_id = collected_id[j];
			const int colors_precomp_stride = feat_dim + 1;  // per-Gaussian data size
			const float* per_gaussian_data = &rgb[gauss_id * colors_precomp_stride];
			
			// Extract weight (last element)
			const float weight = per_gaussian_data[feat_dim];  // Already sigmoid'd in Python
			const float inv_weight = 1.0f - weight;
			
			// Query hashgrid at intersection point
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
				} else {
					printf("FW unsupported level dim : %d\n", l_dim);
				}
			}
			
			// Weighted blend: feat = weight * per_gaussian + (1 - weight) * hashgrid
			for(int i = 0; i < feat_dim && i < CHANNELS; i++) {
				feat[i] = weight * per_gaussian_data[i] + inv_weight * feat_hashgrid[i];
			}
			
			break;
		}
		case 1: {
			// Surface mode with optional per-Gaussian baseline features
			// Surface hashgrid: 12 features per level → dot product → 4 scalars per level
			// Baseline hashgrid (if D_DIFFUSE==4): 4 features per level, queried at pk
			// Output: baseline + surface (element-wise), concatenated across levels
			
			float feat_vec[16 * 12];  // Max 16 levels × 12 features
			
			// Query surface hashgrid at intersection point (xyz)
			query_feature<false, 16 * 12, 12>(feat_vec, xyz, voxel_min, voxel_max, collec_offsets, 
				appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
			
			// Query per-Gaussian baseline features on-demand (if dual hashgrid mode)
			float feat_baseline[16 * 4] = {0};
			if(hash_features_diffuse != nullptr && level_offsets_diffuse != nullptr){
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
			
			// Process each level: compute surface features and add baseline
			for(int lv = 0; lv < level; lv++){
				int vec_start = lv * 12;  // Start of vector potentials for this level
				int scalar_start = lv * 4;  // Start of scalar features for this level
				
				// Compute surface features: dot product with Gaussian normal
				// Layout: (f0,f1,f2), (f3,f4,f5), (f6,f7,f8), (f9,f10,f11)
				for(int i = 0; i < 4; i++){
					float dot_prod = 0.0f;
					for(int j = 0; j < 3; j++){
						dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
					}
					// ReLU activation forces vectors and normals to face opposite directions
					// Baseline features from dual hashgrid (if D_DIFFUSE == 4)
					float baseline_val = feat_baseline[lv * 4 + i];
					feat[scalar_start + i] = fmaxf(0.0f, -dot_prod + baseline_val);
				}
			}
			
			break;
		}
			case 5: {
				// Surface RGB mode: query vector potentials at xyz, RGB at Gaussian center pk
				// Buffer sized for actual number of levels: level * 15 features
				const int vec_buffer_size = level * 15;  // e.g., 6 levels × 15 = 90
				float feat_vec[16 * 15];  // Max supported: 16 levels × 15 = 240
				float feat_rgb[16 * 15];
				
				query_feature<false, 16 * 15, 15>(feat_vec, xyz, voxel_min, voxel_max, collec_offsets, 
					appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
				
				// Query RGB at Gaussian center
				query_feature<false, 16 * 15, 15>(feat_rgb, pk, voxel_min, voxel_max, collec_offsets, 
					appearance_level, hash_features, level, l_scale, Base, align_corners, interp, contract, debug);
				
				// Process each level: dot product for vector potentials + extract RGB
				// Output layout: [L0: s0,s1,s2,s3,r,g,b][L1: ...]
				for(int lv = 0; lv < level; lv++){
					int vec_start = lv * 15;  // Start of features for this level in feat_vec
					int out_start = lv * 7;   // Start of output for this level (4 scalars + 3 RGB)
					
					// Dot product for 4 base features (each is a 3D vector) with Gaussian normal
					for(int i = 0; i < 4; i++){
						float dot_prod = 0.0f;
						for(int j = 0; j < 3; j++){
							dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
						}
						// ReLU activation forces vectors and normals to face opposite directions
						feat[out_start + i] = fmaxf(0.0f, -dot_prod);
					}
					
					// Copy RGB from pk query (last 3 features of the level)
					for(int c = 0; c < 3; c++){
						feat[out_start + 4 + c] = feat_rgb[vec_start + 12 + c];
					}
				}
				break;
			}
				default: printf("FW unsupported level dim : %d\n", l_dim);
					break;
			}

				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += feat[ch] * w;
				
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

	// baseline_blend_double post-processing: Query spatial hashgrid at blended 3D position
	// This happens AFTER alpha blending all per-Gaussian features
	// NOTE: Changed from compile-time D_DIFFUSE check to runtime check
	if (inside && render_mode == 3 && level > 0)
	{
		// Use the alpha-blended 3D position
		float3 blended_pos = {pos_x, pos_y, pos_z};
		
		// Query spatial hashgrid at blended position
		float feat_spatial[16 * 4];
		
		// Need to get appearance level for this pixel (use first contributor's level)
		uint32_t pixel_ap_level = 0;
		if(ap_level != nullptr && last_contributor > 0){
			int first_gaussian_id = point_list[ranges[block.group_index().y * horizontal_blocks + block.group_index().x].x];
			pixel_ap_level = floorf(ap_level[first_gaussian_id]);
		}
		
		int collec_offsets[17];
		for(int lv = 0; lv <= level; lv++){
			collec_offsets[lv] = level_offsets[lv];
		}
		
		query_feature<false, 16 * 4, 4>(feat_spatial, blended_pos, voxel_min, voxel_max, collec_offsets, 
			pixel_ap_level, hash_features, level, l_scale, Base, align_corners, interp, if_contract, false);
		
		// Add spatial features to blended per-Gaussian features
		for(int ch = 0; ch < CHANNELS; ch++){
			C[ch] += feat_spatial[ch];
		}
	}

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
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
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
	const float* hash_features,
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
	const uint32_t D_diffuse,
	const float* hash_features_diffuse,
	const int* level_offsets_diffuse,
	const float* gridrange_diffuse,
	const int render_mode)
{
	// Determine D_DIFFUSE template parameter for kernel dispatch
	// For dual hashgrid modes (baseline_double, baseline_blend_double, surface_rgb), use D_diffuse
	// Otherwise default to 0
	const uint32_t D_DIFFUSE_TEMPLATE = D_diffuse;
	
	switch (C) {

		case 3:
			renderCUDAsurfelForward<3, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
				depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
			break;
		case 8:
			renderCUDAsurfelForward<8, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
				depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
			break;
		case 16:
			renderCUDAsurfelForward<16, 0> <<<grid, block>>>(
				ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
				depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
				hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
			break;
	case 24:
		// Always use D_DIFFUSE=0 template and handle dual hashgrids at runtime
		// This avoids shared memory issues from instantiating multiple templates
		renderCUDAsurfelForward<24, 0> <<<grid, block>>>(
			ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
			depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
			hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
		break;
	case 32:
		renderCUDAsurfelForward<32, 0> <<<grid, block>>>(
			ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
			depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
			hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
		break;
	case 42:
		renderCUDAsurfelForward<42, 0> <<<grid, block>>>(
			ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
			depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
			hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
		break;
	case 48:
		renderCUDAsurfelForward<48, 0> <<<grid, block>>>(
			ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
			depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
			hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
		break;
	case 72:
		renderCUDAsurfelForward<72, 0> <<<grid, block>>>(
			ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
			depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
			hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
		break;
	case 90:
		renderCUDAsurfelForward<90, 0> <<<grid, block>>>(
			ranges, point_list, beta, W, H, level, l_dim, l_scale, Base, align_corners, interp, if_contract, record_transmittance, scales, focal_x, focal_y, means3D, means2D, colors, transMats, homotrans, ap_level, hash_features, level_offsets, gridrange,
			depths, normal_opacity, final_T, n_contrib, bg_color, out_color, out_others, out_index, cover_pixels, trans_avg,
			hash_features_diffuse, level_offsets_diffuse, gridrange_diffuse, render_mode, colors);
		break;
	default:
		printf("Unsupported channel count: %d\n", C);
	}

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
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
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
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

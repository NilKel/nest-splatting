/*
 * Dense grid rendering submodule — forward-only.
 * SH base color (from preprocessing) + trilinear 3D grid lookup for RGB residual.
 * No hashgrid, no MLP, no backward pass.
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
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

	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into an image plane
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

	float normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
	if(normal_len > 1e-7f) {
		normal.x /= normal_len;
		normal.y /= normal_len;
		normal.z /= normal_len;
	}
}

// Computing the bounding box of the 2D Gaussian and its center
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

// Preprocessing kernel: frustum culling, SH eval, transmat computation, tile binning
// Also stores world-space surfel basis vectors (SuTu, SvTv) for xyz reconstruction.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
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
	float* world_basis,    // [N, 6] = SuTu(3) + SvTv(3)
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const float* shapes,
	const int kernel_type)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Frustum culling
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Compute transmat
	glm::mat3 T;
	float3 normal;
	compute_transmat(p_orig, scales[idx], scale_modifier, rotations[idx],
		projmatrix, viewmatrix, W, H, T, normal);

	// Flip normal if backfacing
#if BACKFACE_CULL
	float3 ray_dir = {p_view.x, p_view.y, p_view.z};
	float ndotray = normal.x * ray_dir.x + normal.y * ray_dir.y + normal.z * ray_dir.z;
	if (ndotray > 0) {
		normal.x = -normal.x;
		normal.y = -normal.y;
		normal.z = -normal.z;
	}
#if DUAL_VISIABLE
#else
	if (ndotray > 0) return;
#endif
#endif

	// Store transmat
	transMats[idx * 9 + 0] = T[0].x;
	transMats[idx * 9 + 1] = T[0].y;
	transMats[idx * 9 + 2] = T[0].z;
	transMats[idx * 9 + 3] = T[1].x;
	transMats[idx * 9 + 4] = T[1].y;
	transMats[idx * 9 + 5] = T[1].z;
	transMats[idx * 9 + 6] = T[2].x;
	transMats[idx * 9 + 7] = T[2].y;
	transMats[idx * 9 + 8] = T[2].z;

	// Compute world-space surfel basis: SuTu = scale_x * R_col0, SvTv = scale_y * R_col1
	// R*S columns: L[0] = scale_x * R_col0, L[1] = scale_y * R_col1
	glm::mat3 R = quat_to_rotmat(rotations[idx]);
	glm::mat3 S = scale_to_mat(scales[idx], scale_modifier);
	glm::mat3 L = R * S;
	world_basis[idx * 6 + 0] = L[0].x;  // SuTu.x
	world_basis[idx * 6 + 1] = L[0].y;  // SuTu.y
	world_basis[idx * 6 + 2] = L[0].z;  // SuTu.z
	world_basis[idx * 6 + 3] = L[1].x;  // SvTv.x
	world_basis[idx * 6 + 4] = L[1].y;  // SvTv.y
	world_basis[idx * 6 + 5] = L[1].z;  // SvTv.z

	// Compute AABB with appropriate cutoff
	float cutoff;
	bool use_beta_cutoff = (kernel_type >= 1 && kernel_type <= 4);
	bool use_adr_cutoff = ((kernel_type == 1 || kernel_type == 3 || kernel_type == 4) && shapes != nullptr);

	if (use_adr_cutoff) {
		float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
		float k = (kernel_type == 4) ? 3.0f : 1.0f;

		float opacity_val = opacities[idx];
		float shape = shapes[idx];

		if (opacity_val < (1.0f / 255.0f)) {
			radii[idx] = 0;
			tiles_touched[idx] = 0;
			return;
		}

		float ratio = 1.0f / (255.0f * opacity_val);
		float r_beta = 0.0f;
		float threshold = powf(ratio, 1.0f / shape);
		if (threshold < 1.0f) {
			r_beta = k * sqrtf(1.0f - threshold);
		}
		float r_lp = 0.0f;
		float log_term = logf(255.0f * opacity_val);
		if (log_term > 0.0f) {
			r_lp = sqrtf(2.0f * log_term);
		}
		cutoff = fmaxf(r_beta, r_lp);
		cutoff = fminf(cutoff, k + 2.0f);
	} else if (use_beta_cutoff) {
		float k = (kernel_type == 4) ? 3.0f : 1.0f;
		float r_lp_typical = sqrtf(2.0f * logf(127.5f));
		cutoff = fmaxf(k * 1.1f, r_lp_typical);
	} else {
		cutoff = 4.0f;
	}

	float2 point_image;
	float2 extent;
	bool ok = compute_aabb(T, cutoff, point_image, extent);
	if (!ok) return;

	float radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	uint2 rect_min, rect_max;
	getRect(point_image, (int)radius, rect_min, rect_max, grid);

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors from SH
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	} else {
		for(int i = 0; i < C; i++){
			rgb[idx * C + i] = colors_precomp[idx * C + i];
		}
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// =============================================================================
// Render kernel: SH base color + trilinear 3D dense grid lookup for RGB residual
// =============================================================================
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderDenseGridCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float beta,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,     // SH base RGB from preprocessing [N, 3]
	const float* __restrict__ transMats,    // [N, 9]
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	const float* __restrict__ shapes,
	const int kernel_type,
	// Dense 3D grid parameters
	const __half* __restrict__ dense_grid,   // [R, R, R, 3] FP16 row-major
	const int grid_resolution,               // R
	const float grid_vmin,                   // voxel_range min (e.g., -1.5)
	const float grid_vmax,                   // voxel_range max (e.g., +1.5)
	// World-space surfel basis for xyz reconstruction
	const float* __restrict__ means3D,       // [N, 3] Gaussian centers
	const float* __restrict__ world_basis)   // [N, 6] = SuTu(3) + SvTv(3) from preprocess
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Shared memory for batch loading
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float collected_shapes[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[3] = { 0 };

	// Precompute grid scale for coordinate normalization
	float grid_inv_range = 1.0f / (grid_vmax - grid_vmin);
	float grid_R = (float)grid_resolution;

#if RENDER_AXUTILITY
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	float median_contributor = {-1};
#endif

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Fetch per-Gaussian data
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
			if (shapes != nullptr) {
				collected_shapes[block.thread_rank()] = shapes[coll_id];
			}
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

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

			float rho = min(rho3d, rho2d);
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
			if (depth < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// Kernel evaluation (Gaussian or Beta)
			float alpha;
			if ((kernel_type == 1 || kernel_type == 4) && shapes != nullptr) {
				float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
				float shape = collected_shapes[j];
				if (rho3d >= k_sq + 1e-6f) continue;
				float base = fmaxf(0.0f, 1.0f - rho3d / k_sq);
				float alpha_beta = powf(base, shape);
				float alpha_lp = expf(-rho2d / 2.0f);
				float kernel_val = fmaxf(alpha_beta, alpha_lp);
				alpha = fminf(0.99f, opa * kernel_val);
			} else if (kernel_type == 2) {
				float power = -0.5f * rho;
				if (power > 0.0f) continue;
				float G = expf(power);
				float per_gaussian_beta = collected_shapes[j];
				if (per_gaussian_beta > 0.0f)
					G = (1.0f + per_gaussian_beta) * G / (1.0f + per_gaussian_beta * G);
				alpha = min(0.99f, opa * G);
			} else if (kernel_type == 3 && shapes != nullptr) {
				float beta_param = collected_shapes[j];
				float power = -0.5f * powf(rho, beta_param * 0.5f);
				if (power > 0.0f) continue;
				alpha = min(0.99f, opa * expf(power));
			} else {
				float power = -0.5f * rho;
				if (power > 0.0f) continue;
				alpha = min(0.99f, opa * expf(power));
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

#if RENDER_AXUTILITY
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				median_contributor = contributor;
			}
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// SH base color (from preprocessing)
			int gauss_id = collected_id[j];
			float feat[3];
			for (int ch = 0; ch < 3; ch++)
				feat[ch] = features[gauss_id * 3 + ch];

			// Dense 3D grid trilinear lookup for RGB residual
			if (dense_grid != nullptr) {
				// Compute world-space xyz from surfel parametric coords
				float3 pk = {means3D[gauss_id * 3 + 0],
				             means3D[gauss_id * 3 + 1],
				             means3D[gauss_id * 3 + 2]};
				float3 xyz;
				if (rho3d <= rho2d) {
					// Disk intersection: xyz = pk + s.x * SuTu + s.y * SvTv
					float3 SuTu = {world_basis[gauss_id * 6 + 0],
					               world_basis[gauss_id * 6 + 1],
					               world_basis[gauss_id * 6 + 2]};
					float3 SvTv = {world_basis[gauss_id * 6 + 3],
					               world_basis[gauss_id * 6 + 4],
					               world_basis[gauss_id * 6 + 5]};
					xyz = {s.x * SuTu.x + s.y * SvTv.x + pk.x,
					       s.x * SuTu.y + s.y * SvTv.y + pk.y,
					       s.x * SuTu.z + s.y * SvTv.z + pk.z};
				} else {
					// Center fallback
					xyz = pk;
				}

				// Normalize to voxel coordinates [0, R)
				float gx = (xyz.x - grid_vmin) * grid_inv_range * grid_R - 0.5f;
				float gy = (xyz.y - grid_vmin) * grid_inv_range * grid_R - 0.5f;
				float gz = (xyz.z - grid_vmin) * grid_inv_range * grid_R - 0.5f;

				// Clamp to valid range
				gx = fmaxf(0.0f, fminf(grid_R - 1.001f, gx));
				gy = fmaxf(0.0f, fminf(grid_R - 1.001f, gy));
				gz = fmaxf(0.0f, fminf(grid_R - 1.001f, gz));

				// Trilinear interpolation indices and weights
				int ix0 = (int)gx, iy0 = (int)gy, iz0 = (int)gz;
				float fx = gx - ix0, fy = gy - iy0, fz = gz - iz0;
				int ix1 = min(ix0 + 1, grid_resolution - 1);
				int iy1 = min(iy0 + 1, grid_resolution - 1);
				int iz1 = min(iz0 + 1, grid_resolution - 1);

				// 8 trilinear weights
				float w000 = (1-fx)*(1-fy)*(1-fz);
				float w100 = fx*(1-fy)*(1-fz);
				float w010 = (1-fx)*fy*(1-fz);
				float w110 = fx*fy*(1-fz);
				float w001 = (1-fx)*(1-fy)*fz;
				float w101 = fx*(1-fy)*fz;
				float w011 = (1-fx)*fy*fz;
				float w111 = fx*fy*fz;

				// Grid stored as [y, x, z, 3]: index = ((iy * R + ix) * R + iz) * 3 + ch
				int R = grid_resolution;
				for (int ch = 0; ch < 3; ch++) {
					float val =
						w000 * __half2float(dense_grid[((iy0 * R + ix0) * R + iz0) * 3 + ch]) +
						w100 * __half2float(dense_grid[((iy0 * R + ix1) * R + iz0) * 3 + ch]) +
						w010 * __half2float(dense_grid[((iy1 * R + ix0) * R + iz0) * 3 + ch]) +
						w110 * __half2float(dense_grid[((iy1 * R + ix1) * R + iz0) * 3 + ch]) +
						w001 * __half2float(dense_grid[((iy0 * R + ix0) * R + iz1) * 3 + ch]) +
						w101 * __half2float(dense_grid[((iy0 * R + ix1) * R + iz1) * 3 + ch]) +
						w011 * __half2float(dense_grid[((iy1 * R + ix0) * R + iz1) * 3 + ch]) +
						w111 * __half2float(dense_grid[((iy1 * R + ix1) * R + iz1) * 3 + ch]);
					feat[ch] += val;
				}
			}

			// Alpha compositing
			for (int ch = 0; ch < 3; ch++)
				C[ch] += feat[ch] * w;
			T = test_T;

			last_contributor = contributor;
		}
	}

	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < 3; ch++)
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
#endif
	}
}

// =============================================================================
// Host dispatch functions
// =============================================================================

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const float beta,
	int W, int H,
	const float2* points_xy_image,
	const float* features,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others,
	const float* shapes,
	const int kernel_type,
	const __half* dense_grid,
	const int grid_resolution,
	const float grid_vmin,
	const float grid_vmax,
	const float* means3D,
	const float* world_basis)
{
	renderDenseGridCUDA<<<grid, block>>>(
		ranges, point_list, beta, W, H,
		points_xy_image, features, transMats, depths, normal_opacity,
		final_T, n_contrib, bg_color, out_color, out_others,
		shapes, kernel_type,
		dense_grid, grid_resolution, grid_vmin, grid_vmax,
		means3D, world_basis);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* colors,
	float4* normal_opacity,
	float* world_basis,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const float* shapes,
	const int kernel_type)
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
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		points_xy_image,
		depths,
		transMats,
		colors,
		normal_opacity,
		world_basis,
		grid,
		tiles_touched,
		prefiltered,
		shapes,
		kernel_type
	);
}

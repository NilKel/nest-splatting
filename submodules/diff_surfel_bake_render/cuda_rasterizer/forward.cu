/*
 * Baked rendering submodule — forward-only.
 * Supports standard SH rendering + residual texture lookup (baked mode).
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

	// Normalize surface normals
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
// aabb_mode: 0 = square (2DGS default), 1 = square + AdR, 2 = rect, 3 = rect + AdR
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
	int* radii_x,
	int* radii_y,
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

	// Compute AABB cutoff
	// aabb_mode: 0 = square, 1 = square + AdR, 2 = rect, 3 = rect + AdR
	bool use_adr = (aabb_mode == 1 || aabb_mode == 3);
	bool use_rect = (aabb_mode >= 2);

	float cutoff;
	bool is_beta_kernel = (kernel_type >= 1 && kernel_type <= 4);

	if (use_adr && is_beta_kernel && shapes != nullptr) {
		// AdR: per-Gaussian adaptive cutoff from opacity and shape
		float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
		float k = (kernel_type == 4) ? 3.0f : 1.0f;

		float opacity_val = opacities[idx];
		float shape = shapes[idx];

		if (opacity_val < (1.0f / 255.0f)) {
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
	} else if (is_beta_kernel) {
		// Fixed conservative cutoff for beta kernels (no AdR)
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

	// Compute tile bounding box
	float filter_r = cutoff * FilterSize;
	int rx, ry;
	uint2 rect_min, rect_max;

	if (use_rect) {
		// Rectangular AABB: separate X/Y radii for tighter tile coverage
		rx = (int)ceilf(fmaxf(extent.x, filter_r));
		ry = (int)ceilf(fmaxf(extent.y, filter_r));
		getRectXY(point_image, rx, ry, rect_min, rect_max, grid);
	} else {
		// Square AABB: max(x, y) as scalar radius (original 2DGS behavior)
		float radius = ceilf(fmaxf(fmaxf(extent.x, extent.y), filter_r));
		rx = (int)radius;
		ry = (int)radius;
		getRect(point_image, (int)radius, rect_min, rect_max, grid);
	}

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
	radii[idx] = max(rx, ry);  // Non-zero signals visible
	radii_x[idx] = rx;
	radii_y[idx] = ry;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// =============================================================================
// Render kernel for baked mode: SH base color + bilinear residual texture lookup
// =============================================================================
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderBakedCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float beta,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,    // SH base RGB from preprocessing [N, 3]
	const float* __restrict__ transMats,   // [N, 9]
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	const float* __restrict__ shapes,       // Beta kernel shape [N] (nullable)
	const int kernel_type,
	const __half* __restrict__ residual_textures,  // [N, stride] FP16 (nullable), stride = 8*8*residual_dim
	const int residual_dim,                        // 3 (DC) or 48 (full SH)
	const float* __restrict__ means3D,             // [N, 3] Gaussian centers (for viewdir, nullable)
	const float* __restrict__ cam_pos,             // [3] camera position (for viewdir, nullable)
	const __half* __restrict__ atlas_texture,      // [atlas_W * atlas_W * 3] FP16 (nullable)
	const float* __restrict__ atlas_rects,         // [N, 4] (u0_px, v0_px, w_px, h_px) (nullable)
	const int atlas_width)                         // atlas dimension (e.g. 4096)
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
				// Beta kernel with max-pool handoff to Gaussian low-pass
				// kernel_type 1: k²=1 (unit disk), kernel_type 4: k²=9 (3σ scaled)
				float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
				float shape = collected_shapes[j];

				if (rho3d >= k_sq + 1e-6f) continue;

				float base = fmaxf(0.0f, 1.0f - rho3d / k_sq);
				float alpha_beta = powf(base, shape);
				float alpha_lp = expf(-rho2d / 2.0f);
				float kernel_val = fmaxf(alpha_beta, alpha_lp);
				alpha = fminf(0.99f, opa * kernel_val);
			} else if (kernel_type == 2) {
				// Flex kernel
				float power = -0.5f * rho;
				if (power > 0.0f) continue;
				float G = expf(power);
				float per_gaussian_beta = collected_shapes[j];
				if (per_gaussian_beta > 0.0f)
					G = (1.0f + per_gaussian_beta) * G / (1.0f + per_gaussian_beta * G);
				alpha = min(0.99f, opa * G);
			} else if (kernel_type == 3 && shapes != nullptr) {
				// General kernel: isotropic generalized Gaussian
				float beta_param = collected_shapes[j];
				float power = -0.5f * powf(rho, beta_param * 0.5f);
				if (power > 0.0f) continue;
				alpha = min(0.99f, opa * expf(power));
			} else {
				// Standard Gaussian kernel
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

			// Residual texture lookup (atlas or shared mode)
			if (atlas_texture != nullptr && atlas_rects != nullptr) {
				// Atlas mode: variable-resolution packed atlas
				float u0_px  = atlas_rects[gauss_id * 4 + 0];
				float v0_px  = atlas_rects[gauss_id * 4 + 1];
				float u_span = atlas_rects[gauss_id * 4 + 2];
				float v_span = atlas_rects[gauss_id * 4 + 3];

				// Surfel s → atlas pixel coords (texel-center convention)
				// Bake kernel places sample i at s = (i+0.5)*step - E
				// Inverse: tex_coord = (s+E)/(2E) * G - 0.5
				float au = u0_px + (s.x + UV_EXTENT) / (2.0f * UV_EXTENT) * u_span - 0.5f;
				float av = v0_px + (s.y + UV_EXTENT) / (2.0f * UV_EXTENT) * v_span - 0.5f;
				au = fmaxf(u0_px, fminf(u0_px + u_span - 1.001f, au));
				av = fmaxf(v0_px, fminf(v0_px + v_span - 1.001f, av));

				int au0 = (int)au, av0 = (int)av;
				float fu = au - au0, fv = av - av0;
				int au1 = min(au0 + 1, (int)(u0_px + u_span - 1));
				int av1 = min(av0 + 1, (int)(v0_px + v_span - 1));

				long long idx00 = ((long long)av0 * atlas_width + au0) * 3;
				long long idx10 = ((long long)av0 * atlas_width + au1) * 3;
				long long idx01 = ((long long)av1 * atlas_width + au0) * 3;
				long long idx11 = ((long long)av1 * atlas_width + au1) * 3;
				for (int ch = 0; ch < 3; ch++) {
					float c00 = __half2float(atlas_texture[idx00 + ch]);
					float c10 = __half2float(atlas_texture[idx10 + ch]);
					float c01 = __half2float(atlas_texture[idx01 + ch]);
					float c11 = __half2float(atlas_texture[idx11 + ch]);
					float res = (1-fu)*(1-fv)*c00 + fu*(1-fv)*c10
					          + (1-fu)*fv*c01 + fu*fv*c11;
					feat[ch] += fmaxf(0.0f, res);  // ReLU to match training kernel
				}
			}
			// Shared mode: fixed 8×8 per-Gaussian texture
			else if (residual_textures != nullptr) {
				// Texel-center convention: bake sample i at s = (i+0.5)*step - E
				// Inverse: tex_coord = (s+E)/(2E) * G - 0.5 = s + 3.5 (for G=8, E=4)
				float tex_u = fmaxf(0.0f, fminf(6.999f, s.x + 3.5f));
				float tex_v = fmaxf(0.0f, fminf(6.999f, s.y + 3.5f));
				int u0 = (int)tex_u, v0 = (int)tex_v;
				float fu = tex_u - u0, fv = tex_v - v0;
				int u1 = min(u0 + 1, 7), v1 = min(v0 + 1, 7);
				float w00 = (1-fu)*(1-fv), w10 = fu*(1-fv);
				float w01 = (1-fu)*fv, w11 = fu*fv;

				if (residual_dim == 48 && means3D != nullptr && cam_pos != nullptr) {
					// 48D SH residual: bilinear lookup + SH evaluation at viewdir
					int tex_stride = 8 * 8 * 48;  // 3072 per Gaussian
					int base = gauss_id * tex_stride;

					// Compute view direction (same convention as computeColorFromSH)
					float dx = means3D[gauss_id * 3 + 0] - cam_pos[0];
					float dy = means3D[gauss_id * 3 + 1] - cam_pos[1];
					float dz = means3D[gauss_id * 3 + 2] - cam_pos[2];
					float inv_len = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-8f);
					float dir_x = dx * inv_len, dir_y = dy * inv_len, dir_z = dz * inv_len;

					// Precompute SH basis (degree 3, 16 terms)
					float xx = dir_x*dir_x, yy = dir_y*dir_y, zz = dir_z*dir_z;
					float xy = dir_x*dir_y, yz = dir_y*dir_z, xz = dir_x*dir_z;
					float sh_basis[16];
					sh_basis[0]  = SH_C0;
					sh_basis[1]  = -SH_C1 * dir_y;
					sh_basis[2]  = SH_C1 * dir_z;
					sh_basis[3]  = -SH_C1 * dir_x;
					sh_basis[4]  = SH_C2[0] * xy;
					sh_basis[5]  = SH_C2[1] * yz;
					sh_basis[6]  = SH_C2[2] * (2.0f*zz - xx - yy);
					sh_basis[7]  = SH_C2[3] * xz;
					sh_basis[8]  = SH_C2[4] * (xx - yy);
					sh_basis[9]  = SH_C3[0] * dir_y * (3.0f*xx - yy);
					sh_basis[10] = SH_C3[1] * xy * dir_z;
					sh_basis[11] = SH_C3[2] * dir_y * (4.0f*zz - xx - yy);
					sh_basis[12] = SH_C3[3] * dir_z * (2.0f*zz - 3.0f*xx - 3.0f*yy);
					sh_basis[13] = SH_C3[4] * dir_x * (4.0f*zz - xx - yy);
					sh_basis[14] = SH_C3[5] * dir_z * (xx - yy);
					sh_basis[15] = SH_C3[6] * dir_x * (xx - 3.0f*yy);

					// Bilinear offsets into texture
					int off00 = base + (v0*8+u0)*48;
					int off10 = base + (v0*8+u1)*48;
					int off01 = base + (v1*8+u0)*48;
					int off11 = base + (v1*8+u1)*48;

					// Per-channel SH evaluation: channel layout is [R×16, G×16, B×16]
					for (int ch = 0; ch < 3; ch++) {
						int ch_off = ch * 16;
						float result = 0.0f;
						for (int k = 0; k < 16; k++) {
							float val = w00 * __half2float(residual_textures[off00 + ch_off + k])
							          + w10 * __half2float(residual_textures[off10 + ch_off + k])
							          + w01 * __half2float(residual_textures[off01 + ch_off + k])
							          + w11 * __half2float(residual_textures[off11 + ch_off + k]);
							result += sh_basis[k] * val;
						}
						feat[ch] += result;
					}
				} else {
					// 3D DC residual (legacy path)
					int base = gauss_id * 192;  // 8*8*3
					for (int ch = 0; ch < 3; ch++) {
						float c00 = __half2float(residual_textures[base + (v0*8+u0)*3 + ch]);
						float c10 = __half2float(residual_textures[base + (v0*8+u1)*3 + ch]);
						float c01 = __half2float(residual_textures[base + (v1*8+u0)*3 + ch]);
						float c11 = __half2float(residual_textures[base + (v1*8+u1)*3 + ch]);
						float res = w00*c00 + w10*c10 + w01*c01 + w11*c11;
						feat[ch] += fmaxf(0.0f, res);  // ReLU to match training kernel
					}
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
	const __half* residual_textures,
	const int residual_dim,
	const float* means3D,
	const float* cam_pos,
	const __half* atlas_texture,
	const float* atlas_rects,
	const int atlas_width)
{
	renderBakedCUDA<<<grid, block>>>(
		ranges, point_list, beta, W, H,
		points_xy_image, features, transMats, depths, normal_opacity,
		final_T, n_contrib, bg_color, out_color, out_others,
		shapes, kernel_type, residual_textures,
		residual_dim, means3D, cam_pos,
		atlas_texture, atlas_rects, atlas_width);
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
	int* radii_x,
	int* radii_y,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* colors,
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
		points_xy_image,
		depths,
		transMats,
		colors,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered,
		shapes,
		kernel_type,
		aabb_mode
	);
}

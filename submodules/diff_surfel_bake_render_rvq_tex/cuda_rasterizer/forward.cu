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

// Mirrors the training-time activation constants in diff_surfel_3D_sh_res.
// Set via FORWARD::setActivationBias / FORWARD::setCompactMult from Python.
// Defaults match the training defaults (sh_bias=0.5, res_bias=0.0 under 3D_SH_res;
// we use 0.5 here because the legacy bake pipeline always applied +0.5 to SH).
__device__ float d_sh_bias = 0.5f;
__device__ float d_res_bias = 0.0f;
__device__ float d_compact_mult = 1.0f;
// d_residual_mode mirrors the training-time global in diff_surfel_3D_sh_res /
// diff_surfel_mixed:
//   0 (3D_SH_res): color = ReLU(SH_clamped + residual + d_res_bias)
//   1 (3D_SH_add): color = SH_clamped + ReLU(residual + d_res_bias)
//   2 (mixed):     color = SH_clamped + residual + d_res_bias  (signed; the
//                  per-pixel ReLU on the blended image is applied in Python).
//                  Untextured surfels carry a zero atlas rect → residual==0 →
//                  reduce to the simple-2DGS SV baseline.
__device__ int d_residual_mode = 0;
// `--method mixed_3d --kernel2`: kernel for the UNTEXTURED EWA half at bake
// time. -1 ⇒ unset → untextured use the run's --kernel (kernel_type). Set
// from bake_meta["kernel2"] via set_untex_kernel(), mirroring d_residual_mode.
__device__ int d_untex_kernel = -1;

// ----------------------------------------------------------------------------
// RVQ atlas decode globals — set from Python via SetAtlasRVQCUDA(). When
// `d_rvq_codebooks` is non-null, the atlas-sample step in renderBakedCUDA
// does an L-stage codebook lookup instead of a tex2D / FP16-gather. Indices
// are stored surfel-major so block_id = surfel_offsets[g] + bv·(w/B) + bu.
// ----------------------------------------------------------------------------
__device__ const __half*  d_rvq_codebooks      = nullptr;   // [L, K, B*B*3] FP16
__device__ const uint8_t* d_rvq_indices        = nullptr;   // [L, N_used] uint8 (K ≤ 256)
__device__ const int64_t* d_rvq_surfel_offsets = nullptr;   // [N_gauss + 1] cumulative used-block count
__device__ int  d_rvq_L = 0;
__device__ int  d_rvq_K = 0;
__device__ int  d_rvq_B = 4;
__device__ unsigned long long d_rvq_N_used = 0;

// 2D uint8-RGBA codebook texture (codeword (l, k) at u=k*B+intra_u,
// v=l*B+intra_v). Per-stage min/max dequant: float = u8/255*scale[l] + offset[l].
// At L≤8 the device arrays comfortably fit in constant cache.
__device__ cudaTextureObject_t d_rvq_cb_tex2d = 0;
__device__ float d_rvq_cb_scale[8]  = {1,1,1,1,1,1,1,1};
__device__ float d_rvq_cb_offset[8] = {0,0,0,0,0,0,0,0};

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color. Forward-only baked
// path: no `clamped` tracking (no backward pass that needs it).
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs)
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
	result += d_sh_bias;
	return glm::max(result, 0.0f);
}

// Spherical Voronoi (SV) per-Gaussian softmax-mix evaluation — fused into
// preprocessCUDA so the rasterizer never sees an MLP / autograd graph.
// Mirrors the reference implementation in 2dgs-voronoi
// (`computeColorFromVoronoi` in submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu)
// and `eval_voronoi_sv` in nest's gaussian_renderer/__init__.py:
//
//     dir = (mean - cam_pos) / |mean - cam_pos|     (per-Gaussian view dir)
//     dist[k] = |sites[k] - dir|                    (L2 chord distance)
//     logit[k] = -tau[k] * dist[k]
//     W = softmax(logit)                             (max-subtract for stability)
//     feat = sum_k W[k] * colors[k]
//     rgb = max(0, feat + sh_bias)                  (ReLU; sh_bias from d_sh_bias)
//
// Inputs assume PRE-ACTIVATED tensors:
//   `sites_all`  : [P, K, 3]  unit vectors (normalized on the Python side).
//   `taus_all`   : [P, K]     post-exp scalars.
//   `colors_all` : [P, K, 3]  raw colors (no activation; ReLU applied here).
// Result is the same fp32 RGB the rasterizer would have gotten via
// computeColorFromSH(fake_shs) — sh_bias additive identical, max(.,0) identical.
__device__ glm::vec3 computeColorFromVoronoi(
	int idx,
	int K,
	const glm::vec3* means,
	glm::vec3 campos,
	const float* sites_all,    // [P, K, 3]
	const float* taus_all,     // [P, K]
	const float* colors_all)   // [P, K, 3]
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir * (1.0f / glm::length(dir));

	const glm::vec3* sites  = ((const glm::vec3*)sites_all)  + idx * K;
	const float*     taus   = taus_all                       + idx * K;
	const glm::vec3* colors = ((const glm::vec3*)colors_all) + idx * K;

	// Pass 1: max(logits) for numerical-stability subtract.
	float max_logit = -1e30f;
	for (int k = 0; k < K; ++k) {
		glm::vec3 d = sites[k] - dir;
		float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
		float logit = -taus[k] * dist;
		if (logit > max_logit) max_logit = logit;
	}

	// Pass 2: sum exp(logits - max).
	float sum_exp = 0.0f;
	for (int k = 0; k < K; ++k) {
		glm::vec3 d = sites[k] - dir;
		float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
		float logit = -taus[k] * dist;
		sum_exp += __expf(logit - max_logit);
	}
	float inv_sum = 1.0f / sum_exp;

	// Pass 3: weighted color sum.
	glm::vec3 feat(0.0f);
	for (int k = 0; k < K; ++k) {
		glm::vec3 d = sites[k] - dir;
		float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
		float logit = -taus[k] * dist;
		float w = __expf(logit - max_logit) * inv_sum;
		feat += w * colors[k];
	}

	// d_sh_bias is set by set_activation_bias() — same hook the SH path uses
	// (default 0.5). max(., 0) reproduces nest's relu(feat + 0.5).
	feat += d_sh_bias;
	return glm::max(feat, 0.0f);
}

// Spherical-Beta (SB) evaluation — matches eval_sb in Python (gaussian_renderer/__init__.py).
// sb_params: [K, 6] per-Gaussian block: (r, g, b, theta, phi, beta_raw).
// Returns summed RGB lobe contribution for one Gaussian under view_dir.
__device__ glm::vec3 eval_sb(const float* sb_params, int K, const glm::vec3 view_dir)
{
	const float softplus_scale = 10.0f * 0.693147f;  // 10 * ln(2) — matches Python.
	glm::vec3 rgb_sum = glm::vec3(0.0f);
	for (int k = 0; k < K; ++k) {
		const float* p = sb_params + k * 6;
		float r = p[0], g = p[1], b = p[2];
		float theta = p[3], phi = p[4], beta_raw = p[5];
		// Per-primitive beta = 4 * exp(beta_raw). Matches reference.
		float beta = 4.0f * __expf(beta_raw);
		// Steep softplus activation on RGB.
		float sr = __logf(1.0f + __expf(softplus_scale * r)) / softplus_scale;
		float sg = __logf(1.0f + __expf(softplus_scale * g)) / softplus_scale;
		float sb_ = __logf(1.0f + __expf(softplus_scale * b)) / softplus_scale;
		// Direction from (theta, phi).
		float st = __sinf(theta), ct = __cosf(theta);
		float sp = __sinf(phi),   cp = __cosf(phi);
		glm::vec3 mu = glm::vec3(st * cp, st * sp, ct);
		float dot = glm::dot(mu, view_dir);
		if (dot > 0.0f) {
			float w = __powf(dot, beta);
			rgb_sum += glm::vec3(sr * w, sg * w, sb_ * w);
		}
	}
	return rgb_sum;
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

// Preprocessing kernel: frustum culling, SH eval, transmat computation, tile binning
// aabb_mode: 0 = square (2DGS default), 1 = square + AdR, 2 = rect, 3 = rect + AdR,
//            5 = SnugBox bbox + AccuTile ellipse-tight tile emission
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
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
	__half* rgb,
	float4* normal_opacity,
	float4* conic_t,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint32_t* depth_keys_compact,
	uint32_t* prim_idx_compact,
	uint32_t* n_visible_atomic,
	uint32_t* n_instances_atomic,
	const int sort_mode,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode,
	// Spherical Voronoi (--feature SV) per-Gaussian color path. When
	// voronoi_K > 0 AND colors_precomp == nullptr, the SH evaluation is
	// replaced by `computeColorFromVoronoi(...)`. Empty ⇒ legacy SH path.
	const float* __restrict__ voronoi_sites,    // [P, K, 3]
	const float* __restrict__ voronoi_tau,      // [P, K]
	const float* __restrict__ voronoi_colors,   // [P, K, 3]
	const int voronoi_K,
	// Spherical-Beta (--feature beta). Was evaluated per-pixel in
	// renderBakedCUDA's inner loop, but view_dir is per-Gaussian, so the
	// same RGB was redundantly recomputed for every pixel. Move to here
	// (per-Gauss, fused into preprocess), write fp16 to `sb_rgb_out`,
	// renderBakedCUDA just adds it to feat[]. Same chain math, ~10-25%
	// FPS lift on SB scenes. When sb_number == 0, no work happens.
	const float* __restrict__ sb_params,        // [P, K, 6]
	const int sb_number,
	__half* __restrict__ sb_rgb_out,            // [P, 3]
	// `--method mixed_3d`: untextured surfels render as EWA 3D ellipsoids
	// (FastGS-verbatim conic). nullptr scaling_z → pure 2DGS bake (unchanged).
	const bool* __restrict__ is_textured = nullptr,
	const float* __restrict__ scaling_z = nullptr,
	float4* __restrict__ ewa_conic = nullptr)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize. tiles_touched is needed by both modes (legacy uses it for
	// inclusive prefix sum; FastGS uses it as per-id tile-count lookup table
	// for apply_depth_ordering).
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	if (ewa_conic != nullptr)
		ewa_conic[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// Frustum culling
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// `--method mixed_3d`: UNTEXTURED surfels use EWA 3D-ellipsoid geometry
	// (FastGS conic) instead of the 2DGS transmat. Color path is unchanged
	// (per-Gauss SV/SH baseline; untextured carry a zero atlas rect → no
	// residual). Geometry vars (point_image, rx/ry, rect_*, normal) are set
	// here; the shared color + store code below is reused verbatim.
	const bool untex_ewa = (scaling_z != nullptr) && (is_textured != nullptr)
	                       && (ewa_conic != nullptr) && (!is_textured[idx]);

	// Compute transmat
	glm::mat3 T;
	float3 normal;
	if (!untex_ewa)
	compute_transmat(p_orig, scales[idx], scale_modifier, rotations[idx],
		projmatrix, viewmatrix, W, H, T, normal);

	// Geometry outputs (set by the 2DGS path OR the mixed_3d EWA path below);
	// consumed by the shared color + store code that follows.
	float2 point_image;
	float2 extent = {0.0f, 0.0f};
	int rx = 0, ry = 0;
	uint2 rect_min, rect_max;
	uint32_t rect_tiles = 0;
	bool snugbox_succeeded = false;
	float A_c = 0.0f, B_c = 0.0f, E_c = 0.0f, t_c = 0.0f;
	float2 p_c;
	uint32_t n_tiles_sb = 0;
	if (!untex_ewa) {
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
		// aabb_mode: 0 = square, 1 = square + AdR, 2 = rect, 3 = rect + AdR,
		//            5 = SnugBox+AccuTile (uses the same fixed 4σ cutoff as mode 2,
		//                which is what `--aabb rect` produces in training)
		bool use_adr = (aabb_mode == 1 || aabb_mode == 3);
		bool use_rect = (aabb_mode >= 2);

		float cutoff;
		bool is_beta_kernel = (kernel_type >= 1 && kernel_type <= 4);

		// Cutoff selection mirrors diff_surfel_3D_sh_res/forward.cu ~L600 branches.
		// Matching training exactly is critical: even subtle cutoff differences cause
		// multiplicative dimming (we've observed ~16% at mode=2 + beta_scaled).
		bool use_beta_fixed = (aabb_mode == 4);  // aabb=beta: deliberate tight support
		if (use_adr && is_beta_kernel && shapes != nullptr) {
			// Mode 1/3 + beta: AdR for beta kernel with Gaussian-low-pass max-pool.
			float k = (kernel_type == 4) ? 3.0f : 1.0f;

			float opacity_val = opacities[idx];
			float shape = shapes[idx];

			if (opacity_val < (1.0f / 255.0f)) return;

			float ratio = 1.0f / (255.0f * opacity_val);
			float r_beta = 0.0f;
			float threshold = powf(ratio, 1.0f / shape);
			if (threshold < 1.0f) r_beta = k * sqrtf(1.0f - threshold);

			// IMPORTANT: training's beta-AdR branch leaves r_lp unmodified.
			// d_compact_mult is only applied to the pure-Gaussian branch below.
			float r_lp = 0.0f;
			float log_term = logf(255.0f * opacity_val);
			if (log_term > 0.0f) r_lp = sqrtf(2.0f * log_term);

			cutoff = fmaxf(r_beta, r_lp);
			cutoff = fminf(cutoff, k + 2.0f);
		} else if (use_adr) {
			// Mode 1/3 + Gaussian (no shapes): FastGS Compact Box — only branch
			// that uses d_compact_mult.
			float opacity_val = opacities[idx];
			if (opacity_val < (1.0f / 255.0f)) return;

			float log_term = logf(255.0f * opacity_val);
			cutoff = (log_term > 0.0f)
				? sqrtf(2.0f * log_term * d_compact_mult)
				: 0.1f;
			cutoff = fminf(cutoff, 4.0f);
		} else if (use_beta_fixed) {
			// Mode 4 (aabb=beta) only: deliberate tight cutoff for max-pool compact support.
			float k = (kernel_type == 4) ? 3.0f : 1.0f;
			float r_lp_typical = sqrtf(2.0f * logf(127.5f));
			cutoff = fmaxf(k * 1.1f, r_lp_typical);
		} else if (!is_beta_kernel) {
			// Gaussian kernel + any binning mode (2/5 in practice): opacity-aware
			// cutoff = √(2·log(255·α)). The visible iso-line for α·exp(-ρ²/2) at
			// the 1/255 floor. For α=1 this is ~3.33; α=0.5 ~3.0; α=0.1 ~2.5.
			// Replaces the prior fixed 4.0 fallback which was binning ~50% more
			// tiles than necessary for Gaussian bakes (room_gauss was the
			// triggering case — 169 k all-textured Gausses at fixed 4σ).
			float opacity_val = opacities[idx];
			if (opacity_val < (1.0f / 255.0f)) return;
			float log_term = logf(255.0f * opacity_val);
			cutoff = (log_term > 0.0f) ? sqrtf(2.0f * log_term) : 0.1f;
			cutoff = fminf(cutoff, 4.0f);
		} else {
			// Mode 0 (square) / mode 2 (rect) + beta_scaled: fixed 4σ (2DGS
			// default). Training uses 4.0 for these modes; the bake-render
			// must match to avoid multiplicative dimming (~16 % at mode=2
			// + beta_scaled was observed when this was 3.3).
			cutoff = 4.0f;
		}

		// Project the surfel disk to a screen-space ellipse and take its bbox.
		// Two paths:
		//   aabb_mode 0..3 → rect AABB from compute_aabb (existing, byte-identical).
		//   aabb_mode 5    → SnugBox bbox + AccuTile tile count (ellipse-tight).
		// The SnugBox path falls back to rect AABB on degenerate conic (rare,
		// e.g. near-edge-on surfels) so we never lose coverage.

		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;

		float filter_r = cutoff * FilterSize;

		// Always compute the rect AABB. We need its tile count as an upper bound
		// for SnugBox+AccuTile (a numerically-sound ellipse can never touch more
		// tiles than its bounding rect; if AccuTile reports more, the conic is
		// near-degenerate and its bbox blew up — fall through to rect).
		bool use_rect_aabb = (aabb_mode >= 2);
		if (use_rect_aabb) {
			rx = (int)ceilf(fmaxf(extent.x, filter_r));
			ry = (int)ceilf(fmaxf(extent.y, filter_r));
			getRectXY(point_image, rx, ry, rect_min, rect_max, grid);
		} else {
			float radius = ceilf(fmaxf(fmaxf(extent.x, extent.y), filter_r));
			rx = (int)radius;
			ry = (int)radius;
			getRect(point_image, (int)radius, rect_min, rect_max, grid);
		}
		rect_tiles = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
		if (rect_tiles == 0) return;

		// SnugBox+AccuTile is the default for rect modes (2 and 5; aabb_mode 5 is
		// kept as an alias for backward compatibility). Verified +4..14% FPS on
		// 18/18 scene-config pairs vs the rect-AABB-only enumeration with bit-
		// identical PSNR. The rect-AABB code path below is still reached as a
		// fallback for numerically-degenerate conics (disc → 0⁻).
		bool use_snugbox = (aabb_mode == 2 || aabb_mode == 5);

		if (use_snugbox) {
			if (compute_conic_from_transmat(T, cutoff, A_c, B_c, E_c, t_c, p_c)) {
				n_tiles_sb = duplicateToTilesTouched(
					A_c, B_c, E_c, t_c, p_c, grid,
					0, 0, 0.0f, nullptr, nullptr);
				// Hard upper bound: AccuTile's count must be ≤ rect's count.
				// If it isn't, the conic was near-degenerate (disc → 0⁻) and the
				// scan inflated outside the rect — fall through to rect.
				if (n_tiles_sb > 0 && n_tiles_sb <= rect_tiles) {
					snugbox_succeeded = true;
					point_image = p_c;  // conic center == compute_aabb's center
				}
			}
		}

	}
	else {
		// `--method mixed_3d` UNTEXTURED: FastGS EWA 3D-ellipsoid geometry.
		// AdR + AccuTile parity with the textured 2DGS+SnugBox path:
		//   * Mahalanobis cutoff is OPACITY-AWARE — t_cut = 2·log(255·α) so the
		//     binning iso-line is exactly where α·exp(-m/2) hits the per-Gauss
		//     1/255 alpha-floor. Anything outside would be culled in the inner
		//     loop's `alpha_e < 1/255` check anyway, so the tighter bound
		//     loses zero renderable contribution.
		//   * Tile count is the **ellipse-tight** AccuTile emission via
		//     duplicateToTilesTouched — same helper the textured SnugBox path
		//     uses. The conic format (A, B, E, t, p) is exactly the EWA path's
		//     (inverse-cov, Mahalanobis-sq cutoff, projected center), so no
		//     adapter / separate implementation is needed; the downstream
		//     conic_t / create_instances dispatch (lines 684+) handles both
		//     producers transparently.
		//   * Rect fallback (only when AccuTile reports degenerate) is per-axis
		//     tight from Σ = (Σ⁻¹)⁻¹  ⇒  Σ.xx = E/det, Σ.yy = A/det — NOT the
		//     legacy 3σ-of-maxλ square that was overshoooting rotated /
		//     elongated ellipsoids by up to 1.5×.
		glm::vec3 scale3 = glm::vec3(scales[idx].x, scales[idx].y, scaling_z[idx]);
		float3 conic_e; float my_radius_legacy;
		if (!compute_ewa_conic(p_orig, scale3, scale_modifier, rotations[idx],
				viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy,
				W, H, conic_e, point_image, my_radius_legacy))
			return;

		// Opacity-aware Mahalanobis-squared cutoff (≈ FastGS Compact Box for
		// the textured path, but in the EWA conic's natural form).
		float opa = opacities[idx];
		if (opa < (1.0f / 255.0f)) return;
		float t_cut = fmaxf(0.5f, 2.0f * logf(255.0f * opa));

		// Per-axis tight rect AABB (fallback + AccuTile upper-bound).
		// Σ = K⁻¹ where K = (a,b,c) is the inverse cov; det(K) = ac − b².
		// Σ.xx = c / det, Σ.yy = a / det. Half-widths at Mahalanobis = √t_cut.
		float det = conic_e.x * conic_e.z - conic_e.y * conic_e.y;
		if (det <= 0.0f) return;  // degenerate (near-edge-on or numerically dead)
		float r_cut = sqrtf(t_cut);
		rx = (int)ceilf(r_cut * sqrtf(fmaxf(0.0f, conic_e.z / det)));
		ry = (int)ceilf(r_cut * sqrtf(fmaxf(0.0f, conic_e.x / det)));
		getRectXY(point_image, rx, ry, rect_min, rect_max, grid);
		rect_tiles = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
		if (rect_tiles == 0) return;

		// AccuTile ellipse-tight tile count. EWA conic IS the (A, B, E)
		// quadratic-form coefficients; t = Mahalanobis² cutoff; p = center.
		A_c = conic_e.x; B_c = conic_e.y; E_c = conic_e.z; t_c = t_cut;
		p_c = point_image;
		n_tiles_sb = duplicateToTilesTouched(
			A_c, B_c, E_c, t_c, p_c, grid,
			0, 0, 0.0f, nullptr, nullptr);
		// Same upper-bound sanity check the textured path uses: AccuTile's
		// count must be ≤ rect's count, else the conic was near-degenerate
		// (disc → 0⁻) and the scan inflated — fall through to rect.
		if (n_tiles_sb > 0 && n_tiles_sb <= rect_tiles) {
			snugbox_succeeded = true;
			// EWA's center == conic center == point_image already; nothing
			// to fix up (unlike the 2DGS path where compute_aabb's center
			// drifts and p_c is the corrected one).
		}

		ewa_conic[idx] = make_float4(conic_e.x, conic_e.y, conic_e.z, opa);
		normal = make_float3(0.0f, 0.0f, 1.0f);  // untextured: no surfel normal
	}

	// FP16 SH-color storage: range is [0, ~5] post-bias+clamp — well within
	// FP16 precision and halves the global-mem bandwidth of the inner-loop fetch
	// in renderBakedCUDA.
	if (colors_precomp == nullptr) {
		glm::vec3 result;
		if (voronoi_K > 0 && voronoi_sites != nullptr
		    && voronoi_tau != nullptr && voronoi_colors != nullptr) {
			// --feature SV: per-Gaussian softmax-mix in pre-activated form.
			// Bit-equivalent to nest's torch eval_voronoi_sv_feat → fake-SH-DC
			// → computeColorFromSH path, but skips the autograd graph + fake-SH
			// memcpy roundtrip (~1 ms/frame on a 5090 saved at this scale).
			result = computeColorFromVoronoi(
				idx, voronoi_K,
				(const glm::vec3*)orig_points, *cam_pos,
				voronoi_sites, voronoi_tau, voronoi_colors);
		} else {
			result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs);
		}
		rgb[idx * C + 0] = __float2half(result.x);
		rgb[idx * C + 1] = __float2half(result.y);
		rgb[idx * C + 2] = __float2half(result.z);
	} else {
		for(int i = 0; i < C; i++){
			rgb[idx * C + i] = __float2half(colors_precomp[idx * C + i]);
		}
	}

	// SB per-Gaussian color, computed once. Was per-pixel in renderBakedCUDA
	// but view_dir only depends on the Gaussian's mean → recomputing inside
	// the inner loop wasted ~K*4 transcendentals per touched pixel. The
	// activation chain stays correct because SB is added linearly to feat[]
	// before the outer ReLU; pre-evaluating it here is associative.
	if (sb_number > 0 && sb_params != nullptr && sb_rgb_out != nullptr) {
		const float* sbp = sb_params + idx * sb_number * 6;
		glm::vec3 gc = glm::vec3(orig_points[idx * 3 + 0],
		                         orig_points[idx * 3 + 1],
		                         orig_points[idx * 3 + 2]);
		glm::vec3 cp = *cam_pos;
		glm::vec3 vd = gc - cp;
		vd = vd / (glm::length(vd) + 1e-8f);
		glm::vec3 sb = eval_sb(sbp, sb_number, vd);
		sb_rgb_out[idx * 3 + 0] = __float2half(sb.x);
		sb_rgb_out[idx * 3 + 1] = __float2half(sb.y);
		sb_rgb_out[idx * 3 + 2] = __float2half(sb.z);
	}

	depths[idx] = p_view.z;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};

	uint32_t my_tile_count;
	if (snugbox_succeeded) {
		// Cache conic for create_instances' AccuTile re-walk.
		conic_t[idx] = make_float4(A_c, B_c, E_c, t_c);
		radii[idx] = 1;        // visible flag (SnugBox path doesn't produce a single radius)
		radii_x[idx] = 0;      // unused on SnugBox path
		radii_y[idx] = 0;
		my_tile_count = n_tiles_sb;
	} else {
		conic_t[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // mark as "use rect AABB"
		radii[idx] = max(rx, ry);
		radii_x[idx] = rx;
		radii_y[idx] = ry;
		my_tile_count = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	}

	tiles_touched[idx] = my_tile_count;  // per-id (used by both modes)
	if (sort_mode == 1 && my_tile_count > 0) {
		// FastGS compaction: push visible Gaussians into the depth-sort input.
		const uint32_t v_idx = atomicAdd(n_visible_atomic, 1u);
		depth_keys_compact[v_idx] = __float_as_uint(p_view.z);  // 32-bit float-bit key
		prim_idx_compact[v_idx] = (uint32_t)idx;
		atomicAdd(n_instances_atomic, my_tile_count);
	}
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
	const __half* __restrict__ features,    // SH base RGB from preprocessing [N, 3] FP16
	const float* __restrict__ transMats,   // [N, 9]
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ shapes,       // Beta kernel shape [N] (nullable)
	const int kernel_type,
	const float* __restrict__ means3D,             // [N, 3] Gaussian centers (for SB viewdir, nullable when sb_number==0)
	const float* __restrict__ cam_pos,             // [3] camera position (for SB viewdir, nullable when sb_number==0)
	const __half* __restrict__ atlas_texture,      // [atlas_h * atlas_width * 3] FP16 — used by SW fallback when atlas_tex_obj==0
	const float* __restrict__ atlas_rects,         // [N, 4] (u0_px, v0_px, w_px, h_px) (required for atlas mode)
	const int atlas_width,                         // atlas dimension (e.g. 4096)
	const float* __restrict__ sb_params,           // [N, K, 6] SB params (legacy; unused — sb_rgb_in is per-Gauss)
	const int sb_number,                           // K lobes per Gaussian (0 = SB disabled)
	const __half* __restrict__ sb_rgb_in,          // [N, 3] SB RGB precomputed in preprocessCUDA (read when sb_number > 0)
	cudaTextureObject_t atlas_tex_obj,             // hardware texture object for atlas (required)
	float atlas_offset,                            // dequantization offset (uint8 atlas)
	float atlas_scale,                             // dequantization scale  (uint8 atlas)
	// `--method mixed_3d`: per-Gauss textured flag + EWA conic. nullptr
	// ewa_conic → pure 2DGS (unchanged). Untextured rows use the EWA conic
	// falloff; the per-Gauss SV/SH color path below is shared (untextured
	// carry a zero atlas rect → SV/SH-only, same as `--method mixed`).
	const bool* __restrict__ is_textured = nullptr,
	const float4* __restrict__ ewa_conic = nullptr,
	const float* __restrict__ ewa_depths = nullptr)  // geomState.depths (camera z)
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
	__shared__ bool collected_is_textured[BLOCK_SIZE];   // `--method mixed_3d`
	__shared__ float4 collected_ewa_conic[BLOCK_SIZE];   // `--method mixed_3d`

	float T = 1.0f;
	float C[3] = { 0 };

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
			collected_is_textured[block.thread_rank()] =
				(is_textured == nullptr) ? true : is_textured[coll_id];
			collected_ewa_conic[block.thread_rank()] =
				(ewa_conic == nullptr) ? make_float4(0.0f, 0.0f, 0.0f, 0.0f)
				                       : ewa_conic[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// ============================================================
			// `--method mixed_3d` — UNTEXTURED EWA 3D-ellipsoid splat.
			// Self-contained (mirrors the verified mixed_3d training-forward
			// branch): EWA conic falloff + the SV/SH baseline colour path
			// (untextured carry a zero atlas rect → NO residual lookup, so
			// the colour reduces to exactly the `--method mixed` skip-texture
			// behaviour — only the geometry differs). `continue` past the
			// 2DGS code.
			// ============================================================
			if (ewa_conic != nullptr && !collected_is_textured[j]) {
				const float4 con_o = collected_ewa_conic[j];
				if (con_o.w <= 0.0f) continue;            // culled in preprocess
				const float2 xy_e = collected_xy[j];
				const float2 de = { xy_e.x - pixf.x, xy_e.y - pixf.y };
				const float me = con_o.x * de.x * de.x
				               + 2.0f * con_o.y * de.x * de.y
				               + con_o.z * de.y * de.y;
				// Untextured kernel = `--kernel2` (d_untex_kernel, set from
				// bake_meta) if ≥0, else the run's --kernel. Matches the
				// mixed_3d training forward EWA decode.
				const int ut_kt = (d_untex_kernel >= 0) ? d_untex_kernel : kernel_type;
				float G_e;
				if (ut_kt == 1 || ut_kt == 4) {
					const float k_sq = (ut_kt == 4) ? 9.0f : 1.0f;
					if (me >= k_sq + 1e-6f) continue;     // compact support
					const float beta_e = collected_shapes[j];
					const float base_e = fmaxf(0.0f, 1.0f - me / k_sq);
					G_e = powf(base_e, beta_e);
				} else {
					const float power = -0.5f * me;
					if (power > 0.0f) continue;           // me < 0 (degenerate)
					G_e = expf(power);
				}
				const float alpha_e = fminf(0.99f, con_o.w * G_e);
				if (alpha_e < 1.0f / 255.0f) continue;
				const float test_T = T * (1.0f - alpha_e);
				if (test_T < 0.0001f) { done = true; continue; }
				const float w = alpha_e * T;
				const int gid_e = collected_id[j];
				float feat[3] = {
					__half2float(features[gid_e * 3 + 0]),
					__half2float(features[gid_e * 3 + 1]),
					__half2float(features[gid_e * 3 + 2]),
				};
				const float sh_feat[3] = { feat[0], feat[1], feat[2] };
				// untextured: zero atlas rect → residual == 0 (skip atlas).
				if (sb_number > 0 && sb_rgb_in != nullptr) {
					feat[0] += __half2float(sb_rgb_in[gid_e * 3 + 0]);
					feat[1] += __half2float(sb_rgb_in[gid_e * 3 + 1]);
					feat[2] += __half2float(sb_rgb_in[gid_e * 3 + 2]);
				}
				// Same activation as the shared path with residual == 0.
				if (d_residual_mode == 1) {
					for (int ch = 0; ch < 3; ch++) {
						float residual_part = feat[ch] - sh_feat[ch];
						feat[ch] = sh_feat[ch] + fmaxf(0.0f, residual_part + d_res_bias);
					}
				} else if (d_residual_mode == 2) {
					for (int ch = 0; ch < 3; ch++)
						feat[ch] = feat[ch] + d_res_bias;
				} else {
					for (int ch = 0; ch < 3; ch++)
						feat[ch] = fmaxf(0.0f, feat[ch] + d_res_bias);
				}
				for (int ch = 0; ch < 3; ch++)
					C[ch] += feat[ch] * w;
				T = test_T;
				continue;
			}

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

			// SH base color (from preprocessing). FP16 storage halves the
			// global-memory bandwidth of this hot read (called 256× per tile).
			int gauss_id = collected_id[j];
			float feat[3] = {
				__half2float(features[gauss_id * 3 + 0]),
				__half2float(features[gauss_id * 3 + 1]),
				__half2float(features[gauss_id * 3 + 2]),
			};
			// Snapshot SH-only feat for d_residual_mode == 1 (3D_SH_add), where
			// SH bypasses the residual ReLU. Cheap (3 regs); ignored under mode 0.
			float sh_feat[3] = { feat[0], feat[1], feat[2] };

			// Residual texture lookup (atlas mode). Skip when no atlas was
			// bound (SH-only render path) — atlas_rects is nullptr and
			// atlas_tex_obj is 0; dereferencing either is illegal.
			// Also skip per-Gaussian when the rect is zero-area
			// (--bake_skip_texture_low_contrib survivors get SH-only).
			if (atlas_rects != nullptr) {
				float u0_px  = atlas_rects[gauss_id * 4 + 0];
				float v0_px  = atlas_rects[gauss_id * 4 + 1];
				float u_span = atlas_rects[gauss_id * 4 + 2];
				float v_span = atlas_rects[gauss_id * 4 + 3];
				if (u_span > 0.0f && v_span > 0.0f) {

				// Surfel s → atlas pixel coords (texel-center convention).
				// Bake kernel places sample i at s = (i+0.5)*step - E
				// Inverse: tex_coord = (s+E)/(2E) * G - 0.5
				float au = u0_px + (s.x + UV_EXTENT) / (2.0f * UV_EXTENT) * u_span - 0.5f;
				float av = v0_px + (s.y + UV_EXTENT) / (2.0f * UV_EXTENT) * v_span - 0.5f;
				au = fmaxf(u0_px, fminf(u0_px + u_span - 1.001f, au));
				av = fmaxf(v0_px, fminf(v0_px + v_span - 1.001f, av));

				{
					// RVQ atlas — L-stage codebook lookup + sum via 2D uint8
					// codebook texture with per-stage min/max dequant. The
					// texture is laid out so codeword (l, k) lives at the
					// 4×4 region at (k*B + intra_u, l*B + intra_v).
					const int B = d_rvq_B;
					const int bw_g = (int)u_span / B;
					const int u0i = (int)u0_px;
					const int v0i = (int)v0_px;
					const long long surfel_base = d_rvq_surfel_offsets[gauss_id];
					const long long N_used = (long long)d_rvq_N_used;
					const cudaTextureObject_t cb_tex = d_rvq_cb_tex2d;

					auto sample_at = [&](float au_in, float av_in, float* rgb) {
						int local_u = (int)floorf(au_in) - u0i;
						int local_v = (int)floorf(av_in) - v0i;
						local_u = max(0, min(local_u, (int)u_span - 1));
						local_v = max(0, min(local_v, (int)v_span - 1));
						int bu_l = local_u / B;
						int bv_l = local_v / B;
						int intra_u = local_u - bu_l * B;
						int intra_v = local_v - bv_l * B;
						long long bid = surfel_base + (long long)bv_l * bw_g + bu_l;
						float r = 0.f, g_ = 0.f, b = 0.f;
						#pragma unroll
						for (int l = 0; l < 4; ++l) {
							if (l >= d_rvq_L) break;
							int code = (int)d_rvq_indices[(long long)l * N_used + bid];
							float4 rgba = tex2D<float4>(cb_tex,
								(float)(code * B + intra_u) + 0.5f,
								(float)(l    * B + intra_v) + 0.5f);
							float s = d_rvq_cb_scale[l];
							float o = d_rvq_cb_offset[l];
							r  += rgba.x * s + o;
							g_ += rgba.y * s + o;
							b  += rgba.z * s + o;
						}
						rgb[0] = r; rgb[1] = g_; rgb[2] = b;
					};
					float au0_f = floorf(au); float av0_f = floorf(av);
					float fu = au - au0_f; float fv = av - av0_f;
					float c00[3], c01[3], c10[3], c11[3];
					sample_at(au0_f,       av0_f,       c00);
					sample_at(au0_f + 1.f, av0_f,       c01);
					sample_at(au0_f,       av0_f + 1.f, c10);
					sample_at(au0_f + 1.f, av0_f + 1.f, c11);
					#pragma unroll
					for (int ch = 0; ch < 3; ++ch) {
						float top = c00[ch] * (1 - fu) + c01[ch] * fu;
						float bot = c10[ch] * (1 - fu) + c11[ch] * fu;
						feat[ch] += top * (1 - fv) + bot * fv;
					}
				}
				}  // closes `if (u_span > 0 && v_span > 0)`
			}

			// Spherical-Beta (SB): per-Gauss precomputed in preprocessCUDA
			// (was per-pixel here, recomputed redundantly for every pixel a
			// Gaussian touched). Just an FP16 load + add now.
			if (sb_number > 0 && sb_rgb_in != nullptr) {
				feat[0] += __half2float(sb_rgb_in[gauss_id * 3 + 0]);
				feat[1] += __half2float(sb_rgb_in[gauss_id * 3 + 1]);
				feat[2] += __half2float(sb_rgb_in[gauss_id * 3 + 2]);
			}

			// Final activation matches training's outer ReLU site:
			//   mode 0 (3D_SH_res): color = ReLU(SH_clamped + residual + d_res_bias)
			//   mode 1 (3D_SH_add): color = SH_clamped + ReLU(residual + d_res_bias)
			// SH_clamped is already in feat[] (precomputed via computeColorFromSH
			// which uses d_sh_bias). residual = feat[] - sh_feat[] (atlas + SB).
			if (d_residual_mode == 1) {
				for (int ch = 0; ch < 3; ch++) {
					float residual_part = feat[ch] - sh_feat[ch];
					feat[ch] = sh_feat[ch] + fmaxf(0.0f, residual_part + d_res_bias);
				}
			} else if (d_residual_mode == 2) {
				// mixed: feat = ReLU(SV) + residual + res_bias (signed; NO per-Gauss
				// outer ReLU — the per-pixel ReLU on the blended color is applied in
				// Python, matching diff_surfel_mixed training). Untextured surfels
				// hit this with residual==0 (zero rect → no atlas add) so they
				// reduce to ReLU(SV) + res_bias = the simple-2DGS SV baseline.
				for (int ch = 0; ch < 3; ch++)
					feat[ch] = feat[ch] + d_res_bias;
			} else {
				for (int ch = 0; ch < 3; ch++)
					feat[ch] = fmaxf(0.0f, feat[ch] + d_res_bias);
			}

			// Alpha compositing
			for (int ch = 0; ch < 3; ch++)
				C[ch] += feat[ch] * w;
			T = test_T;
		}
	}

	if (inside)
	{
		for (int ch = 0; ch < 3; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
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
	const __half* features,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	const float* bg_color,
	float* out_color,
	const float* shapes,
	const int kernel_type,
	const float* means3D,
	const float* cam_pos,
	const __half* atlas_texture,
	const float* atlas_rects,
	const int atlas_width,
	const float* sb_params,
	const int sb_number,
	const __half* sb_rgb_in,
	cudaTextureObject_t atlas_tex_obj,
	float atlas_offset,
	float atlas_scale,
	const bool* is_textured,
	const float4* ewa_conic)
{
	renderBakedCUDA<<<grid, block>>>(
		ranges, point_list, beta, W, H,
		points_xy_image, features, transMats, depths, normal_opacity,
		bg_color, out_color,
		shapes, kernel_type,
		means3D, cam_pos,
		atlas_texture, atlas_rects, atlas_width,
		sb_params, sb_number, sb_rgb_in, atlas_tex_obj,
		atlas_offset, atlas_scale,
		is_textured, ewa_conic, depths);
}

// Device-global setters (mirror training-time setters in diff_surfel_3D_sh_res).
__global__ void setBakeActivationBiasKernel(float sh, float res) {
	d_sh_bias = sh;
	d_res_bias = res;
}
void FORWARD::setActivationBias(float sh_bias, float res_bias) {
	setBakeActivationBiasKernel<<<1, 1>>>(sh_bias, res_bias);
}

__global__ void setBakeCompactMultKernel(float val) { d_compact_mult = val; }
void FORWARD::setCompactMult(float val) {
	setBakeCompactMultKernel<<<1, 1>>>(val);
}

__global__ void setBakeResidualModeKernel(int mode) { d_residual_mode = mode; }
void FORWARD::setResidualMode(int mode) {
	setBakeResidualModeKernel<<<1, 1>>>(mode);
}

__global__ void setBakeUntexKernelKernel(int v) { d_untex_kernel = v; }
void FORWARD::setUntexKernel(int v) {
	setBakeUntexKernelKernel<<<1, 1>>>(v);
}

__global__ void setAtlasRVQKernel(
	const __half* cb, const uint8_t* idx, const int64_t* off,
	int L, int K, int B, unsigned long long N_used
) {
	d_rvq_codebooks      = cb;
	d_rvq_indices        = idx;
	d_rvq_surfel_offsets = off;
	d_rvq_L = L; d_rvq_K = K; d_rvq_B = B;
	d_rvq_N_used = N_used;
}
void FORWARD::setAtlasRVQ(
	const __half* codebooks, const uint8_t* indices, const int64_t* surfel_offsets,
	int L, int K, int B, unsigned long long N_used
) {
	setAtlasRVQKernel<<<1, 1>>>(codebooks, indices, surfel_offsets, L, K, B, N_used);
}

__global__ void clearAtlasRVQKernel() {
	d_rvq_codebooks      = nullptr;
	d_rvq_indices        = nullptr;
	d_rvq_surfel_offsets = nullptr;
	d_rvq_L = 0; d_rvq_K = 0; d_rvq_N_used = 0;
	d_rvq_cb_tex2d = 0;
}
void FORWARD::clearAtlasRVQ() { clearAtlasRVQKernel<<<1, 1>>>(); }

__global__ void setAtlasRVQTex2DKernel(cudaTextureObject_t tex) { d_rvq_cb_tex2d = tex; }
void FORWARD::setAtlasRVQTex2D(cudaTextureObject_t tex) {
	setAtlasRVQTex2DKernel<<<1, 1>>>(tex);
}

__global__ void setAtlasRVQDequantKernel(int l, float scale, float offset) {
	if (l >= 0 && l < 8) {
		d_rvq_cb_scale[l]  = scale;
		d_rvq_cb_offset[l] = offset;
	}
}
void FORWARD::setAtlasRVQDequant(int l, float scale, float offset) {
	setAtlasRVQDequantKernel<<<1, 1>>>(l, scale, offset);
}


void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
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
	__half* colors,
	float4* normal_opacity,
	float4* conic_t,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint32_t* depth_keys_compact,
	uint32_t* prim_idx_compact,
	uint32_t* n_visible_atomic,
	uint32_t* n_instances_atomic,
	const int sort_mode,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode,
	const float* voronoi_sites,
	const float* voronoi_tau,
	const float* voronoi_colors,
	const int voronoi_K,
	const float* sb_params,
	const int sb_number,
	__half* sb_rgb_out,
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
		conic_t,
		grid,
		tiles_touched,
		depth_keys_compact,
		prim_idx_compact,
		n_visible_atomic,
		n_instances_atomic,
		sort_mode,
		prefiltered,
		shapes,
		kernel_type,
		aabb_mode,
		voronoi_sites,
		voronoi_tau,
		voronoi_colors,
		voronoi_K,
		sb_params,
		sb_number,
		sb_rgb_out,
		is_textured,
		scaling_z,
		ewa_conic
	);
}

/*
 * Mode-specific render functions for diff-surfel-rasterization
 * Each mode is in its own .cu file for faster incremental builds
 */

#ifndef CUDA_RASTERIZER_MODES_H
#define CUDA_RASTERIZER_MODES_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

// Forward declarations for mode-specific render functions
// These are called from the main renderCUDA kernel in forward.cu

namespace MODES {

// ============================================================================
// Mode 5: 3D_direct_fused - In-kernel MLP evaluation
// ============================================================================

// Forward pass: Compute RGB from per-Gaussian features + hash + view encoding + MLP
// Returns RGB in feat[3]
__device__ void forward_3d_direct_fused(
    // Intersection data
    const float3& xyz,              // Intersection point in world space
    const float3& pk,               // Gaussian center
    const float3& sutu,             // Su, Tu for disk parametrization
    const float3& svtv,             // Sv, Tv for disk parametrization
    const float2& s,                // Disk local coordinates
    const float rho3d,              // Disk distance
    const float rho2d,              // Center fallback distance

    // Per-Gaussian data
    const int gauss_id,             // Gaussian index
    const float* colors,            // Per-Gaussian features [N, feat_dim]

    // Hash grid data
    const float* hash_features,     // Hash table
    const int* level_offsets,       // Level offsets in hash table
    const float3& voxel_min,        // Scene bounds min
    const float3& voxel_max,        // Scene bounds max
    const float appearance_level,   // Appearance level (LoD)
    const int level,                // Encoded: (total << 16) | (active_hash << 8) | hybrid
    const int l_dim,                // Feature dimension per level
    const float l_scale,            // Level scale
    const int Base,                 // Hash base
    const bool align_corners,
    const int interp,
    const bool contract,

    // Camera data
    const glm::vec3* cam_pos,       // Camera position

    // Output
    float* feat,                    // Output RGB [3]
    const bool debug
);

// Backward pass: Compute gradients for MLP weights, Gaussian features, hash grid
// Uses per-tile shared memory accumulation for MLP gradients
// NOTE: Declaration removed - function defined in mode_3d_direct_fused.cu with MlpWeights parameter

// Flush tile-local MLP gradients to global memory
// Called once per tile after processing all Gaussians
__device__ void flush_tile_mlp_grads(
    // Tile-local shared memory buffers
    const float* tile_dL_dW1,
    const float* tile_dL_db1,
    const float* tile_dL_dW2,
    const float* tile_dL_db2,
    const float* tile_dL_dW3,
    const float* tile_dL_db3,

    // Global gradient buffers
    float* global_dL_dW1,
    float* global_dL_db1,
    float* global_dL_dW2,
    float* global_dL_db2,
    float* global_dL_dW3,
    float* global_dL_db3,

    const int thread_id             // threadIdx.x for parallel writes
);

} // namespace MODES

#endif // CUDA_RASTERIZER_MODES_H

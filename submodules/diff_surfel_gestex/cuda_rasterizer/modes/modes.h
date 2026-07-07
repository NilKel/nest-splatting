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
// Forward pass is inline in forward.cu case 5
// Backward pass is in mode_3d_direct_fused.cu
// ============================================================================

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

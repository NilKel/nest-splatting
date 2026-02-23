/*
 * mma_utils.h - WMMA Tensor Core utilities for MLP weight gradient GEMM
 *
 * Provides fragment types, dimension constants, and tile counts for
 * replacing scalar collaborative GEMM with WMMA mma_sync operations.
 *
 * All MLP dimensions are padded to multiples of 16 for WMMA alignment:
 *   Input: 41 -> 48, Hidden: 32 (unchanged), Output: 3 -> 16
 */

#ifndef MMA_UTILS_H_INCLUDED
#define MMA_UTILS_H_INCLUDED

#include <mma.h>
using namespace nvcuda;

// ============================================================================
// WMMA tile dimensions (fixed by hardware)
// ============================================================================
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ============================================================================
// Padded MLP dimensions for WMMA alignment
// ============================================================================
constexpr int TC_INPUT_DIM  = 48;   // 41 -> 48 (pad 7 zeros after bias at position 40)
constexpr int TC_HIDDEN_DIM = 32;   // already aligned
constexpr int TC_OUTPUT_DIM = 16;   // 3 -> 16 (pad 13 zeros, only 3 used for RGB)
constexpr int TC_BATCH      = 256;  // pixels per tile (16x16 block)

// Original (unpadded) dimensions for masking/trimming
constexpr int ORIG_INPUT_DIM  = 41; // [gauss(20) | hash(4) | view(16) | bias(1)]
constexpr int ORIG_OUTPUT_DIM = 3;  // RGB

// ============================================================================
// Weight matrix dimensions: [rows, cols]
// ============================================================================
constexpr int W1_ROWS = TC_HIDDEN_DIM;  // 32
constexpr int W1_COLS = TC_INPUT_DIM;   // 48
constexpr int W2_ROWS = TC_HIDDEN_DIM;  // 32
constexpr int W2_COLS = TC_HIDDEN_DIM;  // 32
constexpr int W3_ROWS = TC_OUTPUT_DIM;  // 16
constexpr int W3_COLS = TC_HIDDEN_DIM;  // 32

// Total weight elements per layer
constexpr int W1_SIZE = W1_ROWS * W1_COLS;  // 32*48 = 1536
constexpr int W2_SIZE = W2_ROWS * W2_COLS;  // 32*32 = 1024
constexpr int W3_SIZE = W3_ROWS * W3_COLS;  // 16*32 = 512

// ============================================================================
// WMMA tile counts for backward weight GEMM
// dL_dW[M,N] = dL_dz^T[M,K] @ activation[K,N], where K = TC_BATCH = 256 pixels
// ============================================================================

// Layer 1: dL_dW1[32, 48] = dL_dz1^T[32, 256] @ input[256, 48]
constexpr int BW_L1_M_TILES = W1_ROWS / WMMA_M;   // 2
constexpr int BW_L1_N_TILES = W1_COLS / WMMA_N;    // 3
constexpr int BW_L1_TILES   = BW_L1_M_TILES * BW_L1_N_TILES;  // 6

// Layer 2: dL_dW2[32, 32] = dL_dz2^T[32, 256] @ h1[256, 32]
constexpr int BW_L2_M_TILES = W2_ROWS / WMMA_M;   // 2
constexpr int BW_L2_N_TILES = W2_COLS / WMMA_N;    // 2
constexpr int BW_L2_TILES   = BW_L2_M_TILES * BW_L2_N_TILES;  // 4

// Layer 3: dL_dW3[16, 32] = dL_dz3^T[16, 256] @ h2[256, 32]
constexpr int BW_L3_M_TILES = W3_ROWS / WMMA_M;   // 1
constexpr int BW_L3_N_TILES = W3_COLS / WMMA_N;    // 2
constexpr int BW_L3_TILES   = BW_L3_M_TILES * BW_L3_N_TILES;  // 2

// K dimension chunks (256 pixels / 16 = 16 chunks)
constexpr int BW_K_CHUNKS = TC_BATCH / WMMA_K;     // 16

// ============================================================================
// WMMA tile counts for forward batched MLP
// H[M,N] = Input[M,K] @ W^T[K,N], where M = TC_BATCH = 256 pixels
// ============================================================================

// Layer 1: H1[256, 32] = Input[256, 48] @ W1^T[48, 32]
constexpr int FW_L1_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L1_N_TILES = TC_HIDDEN_DIM / WMMA_N;  // 2
constexpr int FW_L1_K_CHUNKS = TC_INPUT_DIM / WMMA_K;  // 3
constexpr int FW_L1_TILES = FW_L1_M_TILES * FW_L1_N_TILES;  // 32

// Layer 2: H2[256, 32] = H1[256, 32] @ W2^T[32, 32]
constexpr int FW_L2_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L2_N_TILES = TC_HIDDEN_DIM / WMMA_N;  // 2
constexpr int FW_L2_K_CHUNKS = TC_HIDDEN_DIM / WMMA_K; // 2
constexpr int FW_L2_TILES = FW_L2_M_TILES * FW_L2_N_TILES;  // 32

// Layer 3: Out[256, 16] = H2[256, 32] @ W3^T[32, 16]
constexpr int FW_L3_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L3_N_TILES = TC_OUTPUT_DIM / WMMA_N;  // 1
constexpr int FW_L3_K_CHUNKS = TC_HIDDEN_DIM / WMMA_K; // 2
constexpr int FW_L3_TILES = FW_L3_M_TILES * FW_L3_N_TILES;  // 16

// Total warps per block: 256 threads / 32 threads-per-warp = 8 warps
constexpr int WARPS_PER_BLOCK = TC_BATCH / 32;  // 8

// ============================================================================
// Fragment type aliases
// ============================================================================

// Backward GEMM: C[M,N] = A^T[M,K] @ B[K,N]
//   dL_dz stored as [K,M] row-major -> matrix_a col_major gives transposed view
//   activation stored as [K,N] row-major -> matrix_b row_major
using bw_frag_a = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;
using bw_frag_b = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using frag_acc  = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

// Forward GEMM: H[M,N] = Input[M,K] @ W^T[K,N]
//   Input stored as [M,K] row-major -> matrix_a row_major
//   W stored as [N,K] row-major = W^T[K,N] col-major -> matrix_b col_major
using fw_frag_a = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using fw_frag_b = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;

// ============================================================================
// Shared memory size constants
// ============================================================================

// Backward GEMM: largest layer is L1 = 256*(48+32)*sizeof(half) = 40,960 bytes
constexpr int TC_COLLABORATIVE_SMEM_SIZE = TC_BATCH * (TC_INPUT_DIM + TC_HIDDEN_DIM) * sizeof(half);

// Forward WMMA: half input buffer + float accumulator output
// Half region: 256*48*2 = 24,576 bytes (L1 reads full 48D input, then reused for 32D hidden)
// Float region: 256*32*4 = 32,768 bytes (WMMA output, starts after half region)
// Total: 57,344 bytes — fits with ~35KB static shared memory under 100KB opt-in limit
constexpr int TC_FORWARD_SMEM_HALF = TC_BATCH * TC_INPUT_DIM * sizeof(half);
constexpr int TC_FORWARD_SMEM_FLOAT = TC_BATCH * TC_HIDDEN_DIM * sizeof(float);
constexpr int TC_FORWARD_SMEM_SIZE = TC_FORWARD_SMEM_HALF + TC_FORWARD_SMEM_FLOAT;

#endif // MMA_UTILS_H_INCLUDED

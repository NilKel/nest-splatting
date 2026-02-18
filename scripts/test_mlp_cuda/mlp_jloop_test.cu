/**
 * Test MLP backward with j-loop pattern (replicating rasterizer structure)
 *
 * Hypothesis: The bug is in how the rasterizer's j-loop calls collaborative_mlp_backward_all
 * multiple times, accumulating to tile_dL_dW/tile_dL_db.
 *
 * Compile:
 *   nvcc -o mlp_jloop_test mlp_jloop_test.cu -arch=sm_86
 * Run:
 *   ./mlp_jloop_test
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// MLP dimensions (matching rasterizer)
#define IN_DIM 40
#define HIDDEN_DIM 32
#define OUT_DIM 3
#define BLOCK_SIZE 256

// Tile dimensions for collaborative GEMM (matching rasterizer)
#define TILE_M 16
#define TILE_N 16
#define SUB_TILE_K 16
#define THREADS_PER_GEMM 128

// Shared memory structure (matching rasterizer)
struct TileStorage {
    float W1[IN_DIM * HIDDEN_DIM];     // 40 * 32 = 1280
    float b1[HIDDEN_DIM];               // 32
    float W2[HIDDEN_DIM * HIDDEN_DIM]; // 32 * 32 = 1024
    float b2[HIDDEN_DIM];               // 32
    float W3[HIDDEN_DIM * OUT_DIM];    // 32 * 3 = 96
    float b3[OUT_DIM];                  // 3
};

// Weight memory layout
__constant__ float const_mlp_weights[IN_DIM * HIDDEN_DIM + HIDDEN_DIM +
                                      HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM +
                                      HIDDEN_DIM * OUT_DIM + OUT_DIM];

// Offsets into weight array
#define W1_OFFSET 0
#define B1_OFFSET (IN_DIM * HIDDEN_DIM)
#define W2_OFFSET (B1_OFFSET + HIDDEN_DIM)
#define B2_OFFSET (W2_OFFSET + HIDDEN_DIM * HIDDEN_DIM)
#define W3_OFFSET (B2_OFFSET + HIDDEN_DIM)
#define B3_OFFSET (W3_OFFSET + HIDDEN_DIM * OUT_DIM)

// ============================================================================
// collaborative_mlp_backward_all - EXACT COPY FROM RASTERIZER
// ============================================================================
__device__ void collaborative_mlp_backward_all(
    const float* my_input,    // [40] this thread's input (zeros if non-participating)
    const float* my_h1,       // [32] this thread's h1 activations
    const float* my_h2,       // [32] this thread's h2 activations
    const float* my_dL_dz1,   // [32] gradient w.r.t. pre-activation z1
    const float* my_dL_dz2,   // [32] gradient w.r.t. pre-activation z2
    const float* my_dL_dz3,   // [3]  gradient w.r.t. pre-activation z3
    float* tile_dL_dW1,       // [40 x 32] accumulated in shared memory
    float* tile_dL_db1,       // [32]
    float* tile_dL_dW2,       // [32 x 32]
    float* tile_dL_db2,       // [32]
    float* tile_dL_dW3,       // [32 x 3]
    float* tile_dL_db3,       // [3]
    float* dynamic_smem       // Scratch space for sub-tile accumulation
) {
    const int tid = threadIdx.x;

    // ======== LAYER 3: dL_dW3 [32 x 3] = h2^T @ dL_dz3 ========
    // Simple approach: each thread contributes its outer product
    {
        const int gemm_tid = tid % THREADS_PER_GEMM;  // 0-127
        const int gemm_id = tid / THREADS_PER_GEMM;   // 0 or 1

        // Split work: gemm_id=0 handles first half of rows, gemm_id=1 handles second
        const int row_start = gemm_id * 16;  // 0 or 16
        const int row_end = row_start + 16;  // 16 or 32

        // Thread assignment within the 16x3 output tile
        const int elements_per_gemm = 16 * 3;  // 48 elements

        // First, each thread computes its contribution to certain elements
        // Then we do a block-wide reduction
        float* scratch = dynamic_smem;  // [256 * 48] for storing all threads' contributions

        // Zero scratch space
        for (int e = gemm_tid; e < elements_per_gemm; e += THREADS_PER_GEMM) {
            scratch[gemm_id * THREADS_PER_GEMM * elements_per_gemm + tid * elements_per_gemm + e] = 0.0f;
        }

        // Actually, simpler approach: just use atomicAdd to tile_dL_dW3
        for (int r = row_start; r < row_end; r++) {
            for (int c = 0; c < 3; c++) {
                float contrib = my_h2[r] * my_dL_dz3[c];
                atomicAdd(&tile_dL_dW3[r * 3 + c], contrib);
            }
        }

        // Bias: simpler - each thread adds its contribution
        if (tid < 3) {
            float sum = 0.0f;
            // Actually need reduction across all threads
            // For now, use atomicAdd
        }
        for (int c = 0; c < 3; c++) {
            atomicAdd(&tile_dL_db3[c], my_dL_dz3[c]);
        }
    }

    __syncthreads();

    // ======== LAYER 2: dL_dW2 [32 x 32] = h1^T @ dL_dz2 ========
    {
        const int gemm_tid = tid % THREADS_PER_GEMM;
        const int gemm_id = tid / THREADS_PER_GEMM;

        const int rows_per_gemm = 16;
        const int row_start = gemm_id * rows_per_gemm;
        const int row_end = row_start + rows_per_gemm;

        for (int r = row_start; r < row_end; r++) {
            for (int c = 0; c < 32; c++) {
                float contrib = my_h1[r] * my_dL_dz2[c];
                atomicAdd(&tile_dL_dW2[r * 32 + c], contrib);
            }
        }

        // Bias
        for (int c = 0; c < 32; c++) {
            atomicAdd(&tile_dL_db2[c], my_dL_dz2[c]);
        }
    }

    __syncthreads();

    // ======== LAYER 1: dL_dW1 [40 x 32] = input^T @ dL_dz1 ========
    {
        const int gemm_tid = tid % THREADS_PER_GEMM;
        const int gemm_id = tid / THREADS_PER_GEMM;

        const int rows_per_gemm = 20;
        const int row_start = gemm_id * rows_per_gemm;
        const int row_end = row_start + rows_per_gemm;

        for (int r = row_start; r < row_end; r++) {
            for (int c = 0; c < 32; c++) {
                float contrib = my_input[r] * my_dL_dz1[c];
                atomicAdd(&tile_dL_dW1[r * 32 + c], contrib);
            }
        }

        // Bias
        for (int c = 0; c < 32; c++) {
            atomicAdd(&tile_dL_db1[c], my_dL_dz1[c]);
        }
    }

    __syncthreads();
}

// ============================================================================
// MLP Forward (for recomputing during backward)
// ============================================================================
__device__ void mlp_forward(
    const float* input,    // [40]
    float* output,         // [3]
    float* h1_pre,         // [32] pre-ReLU
    float* h1_post,        // [32] post-ReLU
    float* h2_pre,         // [32]
    float* h2_post,        // [32]
    const float* weights   // Pointer to weight array
) {
    const float* W1 = weights + W1_OFFSET;
    const float* b1 = weights + B1_OFFSET;
    const float* W2 = weights + W2_OFFSET;
    const float* b2 = weights + B2_OFFSET;
    const float* W3 = weights + W3_OFFSET;
    const float* b3 = weights + B3_OFFSET;

    // Layer 1
    for (int h = 0; h < 32; h++) {
        float acc = b1[h];
        for (int i = 0; i < 40; i++) {
            acc += input[i] * W1[h * 40 + i];  // W1 is [32 x 40]
        }
        h1_pre[h] = acc;
        h1_post[h] = fmaxf(0.0f, acc);
    }

    // Layer 2
    for (int h = 0; h < 32; h++) {
        float acc = b2[h];
        for (int i = 0; i < 32; i++) {
            acc += h1_post[i] * W2[h * 32 + i];
        }
        h2_pre[h] = acc;
        h2_post[h] = fmaxf(0.0f, acc);
    }

    // Layer 3 (with sigmoid)
    for (int o = 0; o < 3; o++) {
        float acc = b3[o];
        for (int i = 0; i < 32; i++) {
            acc += h2_post[i] * W3[o * 32 + i];
        }
        output[o] = 1.0f / (1.0f + expf(-acc));
    }
}

// ============================================================================
// compute_dL_dz_all - Compute gradients for all pre-activations
// ============================================================================
__device__ void compute_dL_dz_all(
    const float* h1_pre,
    const float* h1_post,
    const float* h2_pre,
    const float* h2_post,
    const float* output,
    const float* dL_dout,
    float* dL_dz3,
    float* dL_dz2,
    float* dL_dz1,
    const float* weights
) {
    const float* W2 = weights + W2_OFFSET;
    const float* W3 = weights + W3_OFFSET;

    // dL_dz3 = dL_dout * sigmoid'(z3) = dL_dout * output * (1 - output)
    for (int o = 0; o < 3; o++) {
        dL_dz3[o] = dL_dout[o] * output[o] * (1.0f - output[o]);
    }

    // dL_dh2 = dL_dz3 @ W3  (W3 is [3 x 32])
    // dL_dz2 = dL_dh2 * relu'(h2_pre)
    for (int h = 0; h < 32; h++) {
        float dL_dh2 = 0.0f;
        for (int o = 0; o < 3; o++) {
            dL_dh2 += dL_dz3[o] * W3[o * 32 + h];
        }
        dL_dz2[h] = (h2_pre[h] > 0.0f) ? dL_dh2 : 0.0f;
    }

    // dL_dh1 = dL_dz2 @ W2  (W2 is [32 x 32])
    // dL_dz1 = dL_dh1 * relu'(h1_pre)
    for (int h = 0; h < 32; h++) {
        float dL_dh1 = 0.0f;
        for (int i = 0; i < 32; i++) {
            dL_dh1 += dL_dz2[i] * W2[i * 32 + h];
        }
        dL_dz1[h] = (h1_pre[h] > 0.0f) ? dL_dh1 : 0.0f;
    }
}

// ============================================================================
// TEST KERNEL: Replicate rasterizer j-loop pattern (BUGGY VERSION)
// ============================================================================
__global__ void test_jloop_buggy(
    const float* __restrict__ input,      // [256 x 40]
    const float* __restrict__ dL_dpixel,  // [256 x 3]
    float* __restrict__ dL_dW1,           // [40 x 32]
    float* __restrict__ dL_db1,           // [32]
    float* __restrict__ dL_dW2,           // [32 x 32]
    float* __restrict__ dL_db2,           // [32]
    float* __restrict__ dL_dW3,           // [32 x 3]
    float* __restrict__ dL_db3,           // [3]
    int num_gaussians                      // Number of j-loop iterations
) {
    const int tid = threadIdx.x;

    // Shared memory for tile accumulators (matching rasterizer)
    __shared__ float tile_dL_dW1[40 * 32];
    __shared__ float tile_dL_db1[32];
    __shared__ float tile_dL_dW2[32 * 32];
    __shared__ float tile_dL_db2[32];
    __shared__ float tile_dL_dW3[32 * 3];
    __shared__ float tile_dL_db3[3];

    // Zero tile accumulators (done once at start)
    for (int i = tid; i < 40 * 32; i += BLOCK_SIZE) tile_dL_dW1[i] = 0.0f;
    for (int i = tid; i < 32; i += BLOCK_SIZE) tile_dL_db1[i] = 0.0f;
    for (int i = tid; i < 32 * 32; i += BLOCK_SIZE) tile_dL_dW2[i] = 0.0f;
    for (int i = tid; i < 32; i += BLOCK_SIZE) tile_dL_db2[i] = 0.0f;
    for (int i = tid; i < 32 * 3; i += BLOCK_SIZE) tile_dL_dW3[i] = 0.0f;
    for (int i = tid; i < 3; i += BLOCK_SIZE) tile_dL_db3[i] = 0.0f;
    __syncthreads();

    // ======== J-LOOP (simulating multiple Gaussians) ========
    // In real rasterizer, each thread processes a different pixel but same Gaussian
    // Here, we simulate by processing the same input multiple times
    for (int j = 0; j < num_gaussians; j++) {
        // BUGGY: Determine participation (simulating rasterizer logic)
        // In real rasterizer, not all threads participate in every j iteration
        bool participates = (j == 0);  // Only participate on first iteration (simulating real case)

        // Initialize arrays (inside j-loop, matching rasterizer)
        float my_input[40] = {0};
        float my_h1_pre[32] = {0}, my_h1_post[32] = {0};
        float my_h2_pre[32] = {0}, my_h2_post[32] = {0};
        float my_output[3] = {0};
        float my_dL_dout[3] = {0};
        float my_dL_dz1[32] = {0}, my_dL_dz2[32] = {0}, my_dL_dz3[3] = {0};

        if (participates) {
            // Load input
            for (int i = 0; i < 40; i++) {
                my_input[i] = input[tid * 40 + i];
            }

            // Forward pass
            mlp_forward(my_input, my_output, my_h1_pre, my_h1_post, my_h2_pre, my_h2_post, const_mlp_weights);

            // dL_dout = dL_dpixel * w (w=1 for simplicity)
            for (int c = 0; c < 3; c++) {
                my_dL_dout[c] = dL_dpixel[tid * 3 + c];
            }

            // Compute dL_dz for all layers
            compute_dL_dz_all(my_h1_pre, my_h1_post, my_h2_pre, my_h2_post,
                              my_output, my_dL_dout, my_dL_dz3, my_dL_dz2, my_dL_dz1, const_mlp_weights);
        }

        __syncthreads();

        // ======== BUG: collaborative_mlp_backward_all is called EVERY j iteration ========
        // Even when most threads have zeros, this accumulates garbage/zeros to the tile
        extern __shared__ float dynamic_smem[];
        collaborative_mlp_backward_all(
            my_input, my_h1_post, my_h2_post,
            my_dL_dz1, my_dL_dz2, my_dL_dz3,
            tile_dL_dW1, tile_dL_db1,
            tile_dL_dW2, tile_dL_db2,
            tile_dL_dW3, tile_dL_db3,
            dynamic_smem
        );
    }

    __syncthreads();

    // Write tile to global memory
    for (int i = tid; i < 40 * 32; i += BLOCK_SIZE) atomicAdd(&dL_dW1[i], tile_dL_dW1[i]);
    for (int i = tid; i < 32; i += BLOCK_SIZE) atomicAdd(&dL_db1[i], tile_dL_db1[i]);
    for (int i = tid; i < 32 * 32; i += BLOCK_SIZE) atomicAdd(&dL_dW2[i], tile_dL_dW2[i]);
    for (int i = tid; i < 32; i += BLOCK_SIZE) atomicAdd(&dL_db2[i], tile_dL_db2[i]);
    for (int i = tid; i < 32 * 3; i += BLOCK_SIZE) atomicAdd(&dL_dW3[i], tile_dL_dW3[i]);
    for (int i = tid; i < 3; i += BLOCK_SIZE) atomicAdd(&dL_db3[i], tile_dL_db3[i]);
}

// ============================================================================
// TEST KERNEL: Single call (working version, matching standalone test)
// ============================================================================
__global__ void test_single_call(
    const float* __restrict__ input,
    const float* __restrict__ dL_dpixel,
    float* __restrict__ dL_dW1,
    float* __restrict__ dL_db1,
    float* __restrict__ dL_dW2,
    float* __restrict__ dL_db2,
    float* __restrict__ dL_dW3,
    float* __restrict__ dL_db3
) {
    const int tid = threadIdx.x;

    __shared__ float tile_dL_dW1[40 * 32];
    __shared__ float tile_dL_db1[32];
    __shared__ float tile_dL_dW2[32 * 32];
    __shared__ float tile_dL_db2[32];
    __shared__ float tile_dL_dW3[32 * 3];
    __shared__ float tile_dL_db3[3];

    for (int i = tid; i < 40 * 32; i += BLOCK_SIZE) tile_dL_dW1[i] = 0.0f;
    for (int i = tid; i < 32; i += BLOCK_SIZE) tile_dL_db1[i] = 0.0f;
    for (int i = tid; i < 32 * 32; i += BLOCK_SIZE) tile_dL_dW2[i] = 0.0f;
    for (int i = tid; i < 32; i += BLOCK_SIZE) tile_dL_db2[i] = 0.0f;
    for (int i = tid; i < 32 * 3; i += BLOCK_SIZE) tile_dL_dW3[i] = 0.0f;
    for (int i = tid; i < 3; i += BLOCK_SIZE) tile_dL_db3[i] = 0.0f;
    __syncthreads();

    // All threads participate (single iteration)
    float my_input[40], my_h1_pre[32], my_h1_post[32], my_h2_pre[32], my_h2_post[32];
    float my_output[3], my_dL_dout[3];
    float my_dL_dz1[32], my_dL_dz2[32], my_dL_dz3[3];

    for (int i = 0; i < 40; i++) my_input[i] = input[tid * 40 + i];

    mlp_forward(my_input, my_output, my_h1_pre, my_h1_post, my_h2_pre, my_h2_post, const_mlp_weights);

    for (int c = 0; c < 3; c++) my_dL_dout[c] = dL_dpixel[tid * 3 + c];

    compute_dL_dz_all(my_h1_pre, my_h1_post, my_h2_pre, my_h2_post,
                      my_output, my_dL_dout, my_dL_dz3, my_dL_dz2, my_dL_dz1, const_mlp_weights);

    __syncthreads();

    extern __shared__ float dynamic_smem[];
    collaborative_mlp_backward_all(
        my_input, my_h1_post, my_h2_post,
        my_dL_dz1, my_dL_dz2, my_dL_dz3,
        tile_dL_dW1, tile_dL_db1,
        tile_dL_dW2, tile_dL_db2,
        tile_dL_dW3, tile_dL_db3,
        dynamic_smem
    );

    __syncthreads();

    for (int i = tid; i < 40 * 32; i += BLOCK_SIZE) atomicAdd(&dL_dW1[i], tile_dL_dW1[i]);
    for (int i = tid; i < 32; i += BLOCK_SIZE) atomicAdd(&dL_db1[i], tile_dL_db1[i]);
    for (int i = tid; i < 32 * 32; i += BLOCK_SIZE) atomicAdd(&dL_dW2[i], tile_dL_dW2[i]);
    for (int i = tid; i < 32; i += BLOCK_SIZE) atomicAdd(&dL_db2[i], tile_dL_db2[i]);
    for (int i = tid; i < 32 * 3; i += BLOCK_SIZE) atomicAdd(&dL_dW3[i], tile_dL_dW3[i]);
    for (int i = tid; i < 3; i += BLOCK_SIZE) atomicAdd(&dL_db3[i], tile_dL_db3[i]);
}

// ============================================================================
// Setup unit weights
// ============================================================================
void setup_unit_weights(float* weights) {
    memset(weights, 0, (40*32 + 32 + 32*32 + 32 + 32*3 + 3) * sizeof(float));

    // W1: [32 x 40] - diagonal for first 32 cols
    float* W1 = weights + W1_OFFSET;
    for (int i = 0; i < 32; i++) {
        W1[i * 40 + i] = 1.0f;
    }

    // W2: [32 x 32] - identity
    float* W2 = weights + W2_OFFSET;
    for (int i = 0; i < 32; i++) {
        W2[i * 32 + i] = 1.0f;
    }

    // W3: [3 x 32] - diagonal for first 3 rows
    float* W3 = weights + W3_OFFSET;
    for (int i = 0; i < 3; i++) {
        W3[i * 32 + i] = 1.0f;
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== MLP J-Loop Test ===\n");
    printf("Testing whether j-loop pattern causes gradient issues\n\n");

    // Setup weights (unit weights)
    float h_weights[40*32 + 32 + 32*32 + 32 + 32*3 + 3];
    setup_unit_weights(h_weights);
    cudaMemcpyToSymbol(const_mlp_weights, h_weights, sizeof(h_weights));

    // Setup input (all ones)
    float h_input[256 * 40];
    for (int i = 0; i < 256 * 40; i++) h_input[i] = 1.0f;

    // Setup dL_dpixel (all ones)
    float h_dL_dpixel[256 * 3];
    for (int i = 0; i < 256 * 3; i++) h_dL_dpixel[i] = 1.0f;

    // Allocate device memory
    float *d_input, *d_dL_dpixel;
    float *d_dL_dW1, *d_dL_db1, *d_dL_dW2, *d_dL_db2, *d_dL_dW3, *d_dL_db3;

    cudaMalloc(&d_input, 256 * 40 * sizeof(float));
    cudaMalloc(&d_dL_dpixel, 256 * 3 * sizeof(float));
    cudaMalloc(&d_dL_dW1, 40 * 32 * sizeof(float));
    cudaMalloc(&d_dL_db1, 32 * sizeof(float));
    cudaMalloc(&d_dL_dW2, 32 * 32 * sizeof(float));
    cudaMalloc(&d_dL_db2, 32 * sizeof(float));
    cudaMalloc(&d_dL_dW3, 32 * 3 * sizeof(float));
    cudaMalloc(&d_dL_db3, 3 * sizeof(float));

    cudaMemcpy(d_input, h_input, 256 * 40 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_dpixel, h_dL_dpixel, 256 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // ======== TEST 1: Single call (should PASS) ========
    printf("--- Test 1: Single call (all threads participate) ---\n");
    cudaMemset(d_dL_dW1, 0, 40 * 32 * sizeof(float));
    cudaMemset(d_dL_db1, 0, 32 * sizeof(float));
    cudaMemset(d_dL_dW2, 0, 32 * 32 * sizeof(float));
    cudaMemset(d_dL_db2, 0, 32 * sizeof(float));
    cudaMemset(d_dL_dW3, 0, 32 * 3 * sizeof(float));
    cudaMemset(d_dL_db3, 0, 3 * sizeof(float));

    test_single_call<<<1, 256, 64 * 1024>>>(
        d_input, d_dL_dpixel,
        d_dL_dW1, d_dL_db1, d_dL_dW2, d_dL_db2, d_dL_dW3, d_dL_db3
    );
    cudaDeviceSynchronize();

    float h_db2[32];
    cudaMemcpy(h_db2, d_dL_db2, 32 * sizeof(float), cudaMemcpyDeviceToHost);

    int nz_count = 0;
    printf("dL_db2 nonzeros: ");
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_db2[i]) > 1e-10f) {
            printf("%d ", i);
            nz_count++;
        }
    }
    printf("\nTotal: %d/32 (expected: 3 at 0,1,2)\n\n", nz_count);

    // ======== TEST 2: J-loop with 4 iterations (should show bug) ========
    printf("--- Test 2: J-loop with 4 iterations (only j=0 participates) ---\n");
    cudaMemset(d_dL_dW1, 0, 40 * 32 * sizeof(float));
    cudaMemset(d_dL_db1, 0, 32 * sizeof(float));
    cudaMemset(d_dL_dW2, 0, 32 * 32 * sizeof(float));
    cudaMemset(d_dL_db2, 0, 32 * sizeof(float));
    cudaMemset(d_dL_dW3, 0, 32 * 3 * sizeof(float));
    cudaMemset(d_dL_db3, 0, 3 * sizeof(float));

    test_jloop_buggy<<<1, 256, 64 * 1024>>>(
        d_input, d_dL_dpixel,
        d_dL_dW1, d_dL_db1, d_dL_dW2, d_dL_db2, d_dL_dW3, d_dL_db3,
        4  // num_gaussians
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_db2, d_dL_db2, 32 * sizeof(float), cudaMemcpyDeviceToHost);

    nz_count = 0;
    printf("dL_db2 nonzeros: ");
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_db2[i]) > 1e-10f) {
            printf("%d ", i);
            nz_count++;
        }
    }
    printf("\nTotal: %d/32 (expected: 3 at 0,1,2 if correct, more if buggy)\n\n", nz_count);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_dL_dpixel);
    cudaFree(d_dL_dW1);
    cudaFree(d_dL_db1);
    cudaFree(d_dL_dW2);
    cudaFree(d_dL_db2);
    cudaFree(d_dL_dW3);
    cudaFree(d_dL_db3);

    return 0;
}

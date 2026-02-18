/**
 * Minimal test to check CUDA execution
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel(float* out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        printf("Kernel running! tid=%d\n", tid);
    }
    __syncthreads();

    extern __shared__ float smem[];
    smem[tid] = (float)tid;
    __syncthreads();

    if (tid == 0) {
        printf("smem[0]=%.1f smem[1]=%.1f\n", smem[0], smem[1]);
        out[0] = 42.0f;
    }
}

int main() {
    printf("Starting...\n");

    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));

    printf("Launching kernel with 64KB smem...\n");

    // Try to set max shared memory
    cudaFuncSetAttribute(test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

    test_kernel<<<1, 256, 65536>>>(d_out);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel completed successfully\n");
    }

    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Output: %.1f (expected: 42.0)\n", h_out);

    cudaFree(d_out);
    return 0;
}

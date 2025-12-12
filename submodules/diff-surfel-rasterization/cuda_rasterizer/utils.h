#ifndef CUDA_RASTERIZER_UTILS_H_INCLUDED
#define CUDA_RASTERIZER_UTILS_H_INCLUDED

#include <torch/extension.h>
#include <tuple>
#include <c10/cuda/CUDAGuard.h>

#define N_THREADS 256

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                 \
    CHECK_CUDA(x);                                                                     \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten)                                                             \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));


std::tuple<torch::Tensor, torch::Tensor>
compute_relocation_tensor(
    torch::Tensor& opacities,
    torch::Tensor& scales,
    torch::Tensor& ratios,
    torch::Tensor& binoms,
    const int n_max
);

#endif


#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("mark_visible", &markVisible);
  m.def("set_activation_bias", &SetActivationBiasBakeCUDA);
  m.def("set_compact_mult", &SetCompactMultBakeCUDA);
  m.def("set_beta_mult", &SetBetaMultBakeCUDA,
        "EXPERIMENT: scale the beta_scaled footprint cutoff (1.0 = 4-sigma baseline).");
  m.def("set_drop_lowpass", &SetDropLowpassBakeCUDA,
        "EXPERIMENT: drop the Gaussian low-pass (alpha max-pool + filter_r screen extension).");
  m.def("set_opacity_aware_beta", &SetOpacityAwareBetaBakeCUDA,
        "EXPERIMENT: mode-5/0/2 beta cutoff = max(r_beta, r_lp) (1/255 iso) instead of fixed 4σ.");
  m.def("set_residual_mode", &SetResidualModeBakeCUDA,
        "0 = 3D_SH_res outer ReLU (default). 1 = 3D_SH_add separate ReLUs.");
  m.def("set_untex_kernel", &SetUntexKernelBakeCUDA,
        "mixed_3d --kernel2: untextured-EWA kernel int (-1 = unset → use --kernel).");
  m.def("clear_atlas_cache", &ClearAtlasCacheCUDA);
  m.def("set_atlas_use_uint8", &SetAtlasUseUint8CUDA);
  m.def("set_use_atlas_tex_object", &SetUseAtlasTexObjectCUDA);
  m.def("set_atlas_bc7", &SetAtlasBC7CUDA,
        "Install BC7-encoded atlas (W, H multiples of 4). "
        "After this call, all renders sample from the BC7 cudaArray directly.");
  m.def("clear_atlas_bc7", &ClearAtlasBC7CUDA,
        "Clear BC7 atlas — reverts to FP16/uint8 path on next render.");
  m.def("set_atlas_rvq", &SetAtlasRVQCUDA,
        "Install RVQ atlas (codebooks FP16, indices uint8 surfel-major, "
        "surfel_offsets int64 cumulative). Next render uses per-fragment "
        "codebook decode.");
  m.def("clear_atlas_rvq", &ClearAtlasRVQCUDA,
        "Clear RVQ atlas — reverts to BC7/uint8/FP16 path on next render.");
  m.def("set_atlas_rvq_bilinear", &SetAtlasRVQBilinearCUDA,
        "True = 4-tap bilinear (default, matches BC7 quality). "
        "False = nearest-neighbour (~4× fewer codebook reads, "
        "atlas-PSNR drops ~2 dB).");
  m.def("set_atlas_rvq_use_shared_cb", &SetAtlasRVQUseSharedCBCUDA,
        "True = load codebook into __shared__ at kernel start (faster reads, "
        "may reduce occupancy). False (default) = read codebook from global.");
  m.def("set_atlas_rvq_use_tex_cb", &SetAtlasRVQUseTexCBCUDA,
        "True = read codebook via cudaTextureObject (separate cache from L1, "
        "FP16→FP32 done by texture unit). False (default) = read from global.");
  m.def("set_atlas_rvq_use_tex_idx", &SetAtlasRVQUseTexIdxCUDA,
        "True = read indices via cudaTextureObject (uint8). False (default) "
        "= read from global memory.");
}

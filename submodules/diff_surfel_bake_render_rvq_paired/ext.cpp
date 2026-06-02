#include <torch/extension.h>
#include "rasterize_points.h"

// Pure-RVQ fork: no BC7 / uint8-tex / shared-CB / texture-CB toggles. The
// atlas-sample step in the kernel is the RVQ-bilinear-global-memory branch
// only, fully dead-code-eliminated otherwise. Use set_atlas_rvq() to install
// codebooks + indices before the first render.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("mark_visible", &markVisible);
  m.def("set_activation_bias", &SetActivationBiasBakeCUDA);
  m.def("set_compact_mult", &SetCompactMultBakeCUDA);
  m.def("set_residual_mode", &SetResidualModeBakeCUDA,
        "0 = 3D_SH_res outer ReLU (default). 1 = 3D_SH_add separate ReLUs.");
  m.def("set_untex_kernel", &SetUntexKernelBakeCUDA,
        "mixed_3d --kernel2: untextured-EWA kernel int (-1 = unset → use --kernel).");
  m.def("set_atlas_rvq", &SetAtlasRVQCUDA,
        "Install RVQ atlas (codebooks FP16, indices uint8 surfel-major, "
        "surfel_offsets int64 cumulative). The render kernel will do an "
        "L-stage codebook lookup at each fragment.");
  m.def("clear_atlas_rvq", &ClearAtlasRVQCUDA);
}

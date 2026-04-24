#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("mark_visible", &markVisible);
  m.def("set_activation_bias", &SetActivationBiasBakeCUDA);
  m.def("set_compact_mult", &SetCompactMultBakeCUDA);
  m.def("clear_atlas_cache", &ClearAtlasCacheCUDA);
  m.def("set_atlas_use_uint8", &SetAtlasUseUint8CUDA);
  m.def("set_use_atlas_tex_object", &SetUseAtlasTexObjectCUDA);
}

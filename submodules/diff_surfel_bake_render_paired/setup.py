from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

_src_path = os.path.dirname(os.path.abspath(__file__))
conda_include = os.path.join(sys.prefix, 'include')
glm_include = os.path.join(_src_path, 'third_party', 'glm')

setup(
    name="diff_surfel_bake_render_paired",
    packages=['diff_surfel_bake_render_paired'],
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_bake_render_paired._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + conda_include,
                    "-I" + glm_include,
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
                "cxx": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

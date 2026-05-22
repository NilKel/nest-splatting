from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

# Get the directory containing this setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))

# Get conda environment include path
conda_include = os.path.join(sys.prefix, 'include')
# Get glm include path from the main rasterizer (sibling directory)
glm_include = os.path.join(this_dir, '..', 'diff-surfel-rasterization', 'third_party', 'glm')

setup(
    name="diff_surfel_mixed",
    packages=['diff_surfel_mixed'],
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_mixed._C",
            sources=[
                "ext.cpp",
                "rasterize_points.cu",
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/utils.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + conda_include,
                    "-I" + glm_include,
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--ptxas-options=-v",
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

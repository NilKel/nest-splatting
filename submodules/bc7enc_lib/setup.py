from setuptools import setup, Extension
import pybind11
import os

src_dir = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    "bc7encoder",
    sources=[
        os.path.join(src_dir, "bc7_module.cpp"),
        os.path.join(src_dir, "src", "bc7enc.cpp"),
    ],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(src_dir, "src"),
    ],
    extra_compile_args=["-O3", "-std=c++17", "-pthread", "-fPIC"],
    extra_link_args=["-pthread"],
    language="c++",
)

setup(
    name="bc7encoder",
    version="0.1",
    ext_modules=[ext],
)

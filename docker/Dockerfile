FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglm-dev ffmpeg git build-essential cmake libgl1-mesa-dev libglib2.0-0 gcc g++ wget \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# RUN pip install git+https://github.com/NVlabs/nvdiffrast.git

CMD ["/bin/bash"]

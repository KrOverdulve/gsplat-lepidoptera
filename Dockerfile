FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL org.opencontainers.image.version="0.0.1"
LABEL org.opencontainers.image.source="https://github.com/KrOverdulve/gsplat-lepidoptera"
LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04"

# No prompts during apt installs.
ENV DEBIAN_FRONTEND=noninteractive
# gsplat compiles its CUDA code at install time, without a GPU to detect: build
# for all common GPU architectures (8.9 = RTX 40 series, 12.0 = RTX 50 series).
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="12.0;9.0;8.9;8.6;8.0;7.5;7.0"

# Build tools, Python, ffmpeg (trajectory videos) and the shared libraries
# OpenCV and Open3D need.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libx11-6 \
    ninja-build \
    python-is-python3 \
    python3.10-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (uid 1000, so files written to mounted folders belong to a normal host user).
RUN useradd -m -d /home/user -u 1000 user
USER 1000
WORKDIR /home/user
ENV PATH="${PATH}:/home/user/.local/bin"
ENV PYTHONDONTWRITEBYTECODE=1

# The same steps as "Step by step" in the README.
RUN python3.10 -m pip install --upgrade pip setuptools==78.1.0 pybind11==3.1.0
RUN python3.10 -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN python3.10 -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3
RUN python3.10 -m pip install plyfile==0.8.1 tqdm==4.70.1 numpy==1.26.4 opencv-python==4.11.0.86 open3d==0.20.0

COPY --chown=user requirements.txt /home/user/code/requirements.txt
RUN python3.10 -m pip install --no-build-isolation -r /home/user/code/requirements.txt

COPY --chown=user . /home/user/code
WORKDIR /home/user/code

# Viewer port.
EXPOSE 8080

CMD ["/bin/bash"]

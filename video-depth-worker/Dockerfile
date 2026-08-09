# Use an official NVIDIA PyTorch image with GPU & CUDA support
# NOTE: pinned to 2.7.0/cuda12.8 (not the older 2.1.2/cuda12.1) because
# Blackwell GPUs (RTX 50-series, RTX PRO 6000 Blackwell — compute capability
# sm_120) are only supported starting with PyTorch 2.7.0 + CUDA 12.8. Older
# builds fail every job with "CUDA error: no kernel image is available for
# execution on the device" on that hardware.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by OpenCV, OpenEXR, and Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenexr-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages (CRITICAL: Pin numpy<2 to avoid PyTorch NumPy 2.0 crashes!)
# NOTE: easydict and xformers are hard, unconditional imports inside
# Video-Depth-Anything's dpt_temporal.py / motion_module.py — without them
# `from video_depth_anything.video_depth import VideoDepthAnything` raises
# ImportError and the worker never starts. einops is pinned to match the
# version Video-Depth-Anything's own requirements.txt was built against.
RUN pip install --no-cache-dir \
    "einops==0.4.1" \
    "numpy<2.0.0" \
    runpod \
    supabase \
    opencv-python-headless \
    OpenEXR \
    Imath \
    pillow \
    easydict \
    "xformers==0.0.30" \
    "imageio==2.37.0" \
    "imageio-ffmpeg==0.4.7" \
    matplotlib \
    tqdm \
    decord

# Clone Video Depth Anything repository (pinned to a known-good commit so
# upstream API/dependency changes can't silently break this image on rebuild)
RUN git clone https://github.com/DepthAnything/Video-Depth-Anything.git /app/Video-Depth-Anything && \
    cd /app/Video-Depth-Anything && \
    git checkout 4f5ae23172ba60fd7bc11ef671cca678842c7072

# Pre-download Base Model Weights with exact filename expected by handler.py
RUN mkdir -p /app/checkpoints && \
    wget -O /app/checkpoints/video-depth-anything-base.pth \
    https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth

# Copy application files
COPY handler.py /app/handler.py

# Entry point command with unbuffered Python logs
CMD ["python", "-u", "/app/handler.py"]
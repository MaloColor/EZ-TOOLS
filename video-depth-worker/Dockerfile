# Use an official NVIDIA PyTorch image with GPU & CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

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
RUN pip install --no-cache-dir \
    einops \
    "numpy<2.0.0" \
    runpod \
    supabase \
    opencv-python-headless \
    OpenEXR \
    Imath \
    pillow

# Clone Video Depth Anything repository
RUN git clone https://github.com/DepthAnything/Video-Depth-Anything.git /app/Video-Depth-Anything

# Pre-download Base Model Weights with exact filename expected by handler.py
RUN mkdir -p /app/checkpoints && \
    wget -O /app/checkpoints/video-depth-anything-base.pth \
    https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth

# Copy application files
COPY handler.py /app/handler.py

# Entry point command with unbuffered Python logs
CMD ["python", "-u", "/app/handler.py"]
# Use an official NVIDIA PyTorch image with GPU & CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by OpenCV and Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages (Supabase, OpenCV, OpenEXR support)
RUN pip install --no-cache-dir \
    supabase \
    opencv-python-headless \
    OpenEXR \
    numpy \
    pillow

# Clone Video Depth Anything repository (or copy local code)
RUN git clone https://github.com/DepthAnything/Video-Depth-Anything.git /app/Video-Depth-Anything

# Copy application files
COPY handler.py /app/handler.py

# Entry point command
CMD ["python", "handler.py"]
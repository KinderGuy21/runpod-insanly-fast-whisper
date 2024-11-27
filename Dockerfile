# Base image with CUDA, Python 3.10, and PyTorch pre-installed
FROM runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel

# Arguments and Environment Variables
ARG HUGGING_FACE_HUB_WRITE_TOKEN
ENV HUGGING_FACE_HUB_WRITE_TOKEN=$HUGGING_FACE_HUB_WRITE_TOKEN

ENV HF_HOME="/cache/huggingface"
ENV HF_DATASETS_CACHE="/cache/huggingface/datasets"
ENV DEFAULT_HF_METRICS_CACHE="/cache/huggingface/metrics"
ENV DEFAULT_HF_MODULES_CACHE="/cache/huggingface/modules"

ENV HUGGINFACE_HUB_CACHE="/cache/huggingface/hub"
ENV HUGGINGFACE_ASSETS_CACHE="/cache/huggingface/assets"

# Use bash shell with pipefail option for better debugging
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libopenblas-dev \
    libomp-dev \
    ninja-build \
    git \
    curl \
    ffmpeg \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Upgrade pip and set up Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Install FlashAttention
RUN git clone https://github.com/HazyResearch/flash-attention.git /workspace/flash-attention && \
    cd /workspace/flash-attention && \
    pip install .

# Cache Models for Hugging Face
COPY builder/cache_model.py /cache_model.py
RUN python /cache_model.py && \
    rm /cache_model.py

# Copy the source code
ADD src .

# Verify that the cache folder is not empty
RUN test -n "$(ls -A /cache/huggingface)"

# Set the default command
CMD ["python", "-u", "handler.py"]
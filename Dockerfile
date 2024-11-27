# Base image
FROM runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel

# Arguments and environment variables
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

# Upgrade pip
RUN pip install --upgrade pip

# Clone and install FlashAttention
RUN git clone https://github.com/HazyResearch/flash-attention.git /workspace/flash-attention && \
    cd /workspace/flash-attention && \
    pip install --no-build-isolation . && \
    pip install -e .

# Copy Python dependencies
COPY builder/requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Cache models
COPY builder/cache_model.py /cache_model.py
RUN python /cache_model.py && \
    rm /cache_model.py

# Copy the source code
ADD src .

# Verify the cache folder is not empty
RUN test -n "$(ls -A /cache/huggingface)"

# Default command
CMD ["python", "-u", "handler.py"]

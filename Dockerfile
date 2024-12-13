# Base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG HUGGING_FACE_HUB_WRITE_TOKEN
ENV HUGGING_FACE_HUB_WRITE_TOKEN=$HUGGING_FACE_HUB_WRITE_TOKEN

ARG DEBIAN_FRONTEND=noninteractive

ENV HF_HOME="/cache/huggingface"
ENV HF_DATASETS_CACHE="/cache/huggingface/datasets"
ENV DEFAULT_HF_METRICS_CACHE="/cache/huggingface/metrics"
ENV DEFAULT_HF_MODULES_CACHE="/cache/huggingface/modules"

ENV HUGGINFACE_HUB_CACHE="/cache/huggingface/hub"
ENV HUGGINGFACE_ASSETS_CACHE="/cache/huggingface/assets"

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

# install system package ffmpeg
RUN apt-get update && apt-get install -y ffmpeg build-essential cmake git

# Install Python Dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install flash-attn==2.6.2 && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_model.py /cache_model.py
RUN python /cache_model.py && \
    rm /cache_model.py

# Copy Source Code
ADD src .

# Basic validation
# Verify that the cache folder is not empty
RUN test -n "$(ls -A /cache/huggingface)"


CMD ["python", "-u", "handler.py"]
# OpenClaw Voice - GPU-enabled Docker image
# Supports NVIDIA GPUs for fast Whisper + TTS inference

ARG CUDA_BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_VERSION=
ARG TORCHAUDIO_VERSION=
ARG FASTER_WHISPER_VERSION=
ARG CTRANSLATE2_VERSION=

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    pkg-config \
    ffmpeg \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libsndfile1 \
    libswresample-dev \
    libswscale-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Install uv into a standard system path so later layers can invoke it reliably.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Copy dependency manifests first for caching
COPY requirements.txt pyproject.toml README.md ./

# Create venv and install dependencies.
# Use requirements.txt here instead of editable package install because the repo
# is not packaged as a standard installable src-layout project.
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt && \
    if [ -n "${TORCH_VERSION}" ] && [ -n "${TORCHAUDIO_VERSION}" ]; then \
        uv pip install --upgrade "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url ${TORCH_INDEX_URL}; \
    else \
        uv pip install --upgrade torch torchaudio --index-url ${TORCH_INDEX_URL}; \
    fi && \
    if [ -n "${CTRANSLATE2_VERSION}" ]; then uv pip install --upgrade "ctranslate2==${CTRANSLATE2_VERSION}"; fi && \
    if [ -n "${FASTER_WHISPER_VERSION}" ]; then uv pip install --upgrade "faster-whisper==${FASTER_WHISPER_VERSION}"; fi && \
    uv pip install --upgrade "numpy<2"

# Copy application code
COPY src/ ./src/
COPY .env.example ./.env.example

# Create directories for models (will be mounted or downloaded)
RUN mkdir -p /app/models /app/voices

# Environment variables
ENV OPENCLAW_HOST=0.0.0.0
ENV OPENCLAW_PORT=8765
ENV OPENCLAW_STT_MODEL=large-v3-turbo
ENV OPENCLAW_STT_DEVICE=cuda
ENV OPENCLAW_REQUIRE_AUTH=false

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8765/ || exit 1

# Run server
CMD [".venv/bin/python", "-m", "uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8765"]

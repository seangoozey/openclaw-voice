# OpenClaw Voice - P40-targeted GPU-enabled Docker image
# Uses a CUDA 11.8 / Pascal-friendly dependency stack.

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
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

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create app directory
WORKDIR /app

# Install uv into a standard system path so later layers can invoke it reliably.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Copy dependency manifests first for caching
COPY requirements.txt pyproject.toml README.md constraints.txt ./

# Create venv and install dependencies.
# Use requirements.txt here instead of editable package install because the repo
# is not packaged as a standard installable src-layout project.
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -c constraints.txt -r requirements.txt && \
    uv pip install -c constraints.txt --upgrade \
        "torch==2.0.1" \
        "torchaudio==2.0.2" \
        --index-url https://download.pytorch.org/whl/cu118 && \
    uv pip install -c constraints.txt --upgrade --force-reinstall --no-deps \
        "faster-whisper==0.9.0" \
        "ctranslate2==3.24.0" && \
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

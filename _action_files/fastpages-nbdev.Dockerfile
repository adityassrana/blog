# Updated to use uv for ultra-fast Python package management
FROM python:3.12-slim-bookworm

# Install uv from the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies in a single layer for efficiency
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    jq \
    dos2unix \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install latest Pandoc (architecture-aware and more efficient)
RUN PANDOC_VERSION="3.1.9" && \
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then PANDOC_ARCH="arm64"; else PANDOC_ARCH="amd64"; fi && \
    wget -q "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-1-${PANDOC_ARCH}.deb" && \
    dpkg -i "pandoc-${PANDOC_VERSION}-1-${PANDOC_ARCH}.deb" && \
    rm "pandoc-${PANDOC_VERSION}-1-${PANDOC_ARCH}.deb"

# Set environment for uv to use system Python and enable caching
ENV UV_SYSTEM_PYTHON=1
ENV UV_CACHE_DIR=/root/.cache/uv

# Install Python packages with uv (much faster than pip)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    jupyter==1.0.0 \
    watchdog[watchmedo]==3.0.0 \
    jupyter_client==8.6.0 \
    ipykernel==6.29.0 \
    nbdev==2.3.12

# Setup Jupyter kernel
RUN python3 -m ipykernel install --user

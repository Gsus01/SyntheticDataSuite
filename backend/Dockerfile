# syntax=docker/dockerfile:1
# =============================================================================
# SyntheticDataSuite Backend
# =============================================================================
# FastAPI backend with Argo CLI for workflow orchestration
#
# Build:
#   eval $(minikube docker-env)
#   docker build -t syntheticdata-backend:latest ./backend
#
# Run locally (for testing):
#   docker run -p 8000:8000 \
#     -e MINIO_ENDPOINT=host.docker.internal:9000 \
#     syntheticdata-backend:latest
# =============================================================================

FROM python:3.13-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install Argo CLI
# -----------------------------------------------------------------------------
ARG ARGO_VERSION=v3.5.0
RUN curl -sLO "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_VERSION}/argo-linux-amd64.gz" \
    && gunzip argo-linux-amd64.gz \
    && chmod +x argo-linux-amd64 \
    && mv argo-linux-amd64 /usr/local/bin/argo \
    && argo version --short

# -----------------------------------------------------------------------------
# Install uv for fast Python dependency management
# -----------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------
# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without dev dependencies)
RUN uv sync --frozen --no-dev

# -----------------------------------------------------------------------------
# Copy application code
# -----------------------------------------------------------------------------
COPY . .

# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Skip image validation since minikube/docker commands aren't available
    SKIP_IMAGE_VALIDATION=true \
    # Argo CLI location
    ARGO_CLI_PATH=/usr/local/bin/argo

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

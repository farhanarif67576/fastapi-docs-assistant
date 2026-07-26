# Dockerfile for the FastAPI Docs Assistant backend.
# Multi-stage build: installs core dependencies with uv (NO torch/onnx),
# runs with uvicorn. Heavy models (sentence-transformers, cross-encoder)
# are lazy-loaded at runtime only if extra packages are installed.
#
# Build optimization: Only ~300MB (vs ~5.5GB with torch).
# Rebuilds from cache in ~30 seconds if deps haven't changed.

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv (extremely fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (for Docker layer caching)
# pyproject.toml and uv.lock change rarely → this layer is cached between builds
COPY pyproject.toml uv.lock ./

# Install ONLY core production dependencies.
# Heavy packages (torch ~2.5GB, onnxruntime ~300MB, sentence-transformers,
# jupyter ~400MB) are in [tool.uv.dev-dependencies] and are skipped by --no-dev.
# They only install when running `uv sync` (without --no-dev) for local dev.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Copy application source code AFTER dependencies are installed.
# This layer invalidates only when source code changes (often).
COPY app/ app/
COPY server.py .

# Copy pre-generated data (changes only on re-ingestion, rarely)
COPY data/ data/

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Copy application code from builder
COPY --from=builder /app/app/ app/
COPY --from=builder /app/server.py .
COPY --from=builder /app/data/ data/

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run with uvicorn
# Note: server.py contains `app = FastAPI()` so we import it as `server:app`
# This avoids the naming conflict with the `app/` package directory.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

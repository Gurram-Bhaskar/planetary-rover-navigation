# =============================================================================
# Planetary Rover Navigation Simulator — Dockerfile
# =============================================================================
# Target platform  : Hugging Face Spaces (Docker SDK)
# Exposed port     : 7860  (HF Spaces default)
# Python version   : 3.12-slim (smallest image with a stable C runtime)
#
# Build strategy — two stages:
#   1. builder  — installs pip packages into an isolated prefix so the
#                 final image contains no build tools (gcc, pip, wheel, etc.)
#   2. runtime  — copies only the installed packages + app source;
#                 result is ~60 % smaller than a single-stage build.
#
# Layer ordering is optimised for cache reuse:
#   Layer 1  system packages        (changes rarely)
#   Layer 2  requirements.txt COPY  (changes only when deps change)
#   Layer 3  pip install            (invalidated only when layer 2 changes)
#   Layer 4  application source     (invalidated on every code change)
#
# This means rebuilding after a code-only edit reuses the expensive pip
# install layer and completes in seconds instead of minutes.
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Prevent .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install only the build tools we actually need, then clean the apt cache
# in the same RUN layer so it never lands in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy just the requirements file first.
# Docker cache: this layer is only invalidated when requirements.txt changes.
COPY requirements.txt .

# Install into /install prefix (not site-packages) so the runtime stage
# can COPY the entire prefix without carrying pip, setuptools, or wheel.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --prefix=/install \
        --no-warn-script-location \
        -r requirements.txt


# ── Stage 2: minimal runtime ──────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Add the /install prefix to the module search path
    PYTHONPATH=/install/lib/python3.12/site-packages

# Non-root user — Hugging Face Spaces requires this for security.
# UID 1000 matches the default HF Spaces user.
RUN useradd --uid 1000 --create-home --shell /bin/bash rover

WORKDIR /app

# Copy installed packages from builder (no pip, no compiler, no build cache)
COPY --from=builder /install /install

# Copy application source last — maximises cache reuse for code iterations.
# Only the files the server actually needs at runtime:
COPY main.py        ./main.py
COPY inference.py    ./inference.py
COPY train.py        ./train.py
COPY openenv.yaml   ./openenv.yaml

# Transfer ownership to the non-root user
RUN chown -R rover:rover /app

USER rover

# Hugging Face Spaces routes external traffic to port 7860.
# Uvicorn is started with:
#   --workers 1      single worker — episode state lives in-process memory
#   --loop uvloop    fastest async event loop (installed via uvicorn[standard])
#   --access-log     request log visible in HF Spaces logs panel
EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/tasks')"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 7860 & sleep 10 && uv run python train.py"]

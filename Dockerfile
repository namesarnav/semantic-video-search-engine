# syntax=docker/dockerfile:1
#
# Single container: the API, the built UI, and the CLIP checkpoint.
# `docker compose up` is meant to be the only setup step, so the image is
# self-contained -- no model download on first run, no npm install on the host.
#
# This container is **CPU-only, deliberately**. Docker on macOS has no MPS
# passthrough, so CLIP inference here is slower than a native `uv run`. Use it
# for reproducibility and the demo path; do embedding work on the host.


# ---- stage 1: the React UI ------------------------------------------------
#
# Built here rather than on the host so the image does not depend on the
# developer having run `npm run build`. FastAPI serves the output at /.
FROM node:22-slim AS web

WORKDIR /web
# Lockfile first: this layer is rebuilt only when dependencies actually change.
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build


# ---- stage 2: python dependencies ----------------------------------------
FROM python:3.12-slim-bookworm AS deps

# uv, pinned to the same minor the project builds with.
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies before source, so editing a .py does not reinstall torch.
# --frozen: build from uv.lock exactly, never re-resolve. A container that
# quietly resolved something different from the host would defeat the point.
# On linux the lock pins the CPU torch wheels (see pyproject.toml) -- the
# default ones drag in gigabytes of CUDA for hardware this image cannot reach.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---- stage 3: the checkpoint ----------------------------------------------
#
# Baked in rather than downloaded on first run. It is ~600MB, which is most of
# this image's size, and it buys two things worth more than the bytes: the
# container starts offline, and `docker compose up` stays a single step that
# does not silently spend minutes on a download.
FROM deps AS weights

ENV HF_HOME=/opt/hf \
    PATH="/app/.venv/bin:$PATH"

RUN python -c "\
import open_clip; \
open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')" \
    && find /opt/hf -name '*.lock' -delete


# ---- stage 4: runtime -----------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# libgomp: faiss and torch both need an OpenMP runtime. On linux there is one
# system copy and they share it, so the macOS duplicate-runtime problem that
# scripts/fix_openmp.py exists for does not arise here.
# libglib2.0-0: opencv-python-headless links against it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Runs unprivileged: nothing here needs root, and the mounted /data volume
# should not end up owned by it.
RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY --from=deps    --chown=app:app /app/.venv  /app/.venv
COPY --from=deps    --chown=app:app /app/src    /app/src
COPY --from=web     --chown=app:app /web/dist   /app/web/dist
COPY --from=weights --chown=app:app /opt/hf     /opt/hf

# Ground truth, so `sv-engine eval` works in the container rather than dying
# on a missing labels file. Tiny, and it is source: the metric is only
# reproducible if the labels ship with the code that scores them.
COPY --chown=app:app eval/ /app/eval/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    # No MPS or CUDA in here; say so explicitly rather than letting device
    # selection fall through and look like a choice.
    SV_DEVICE=cpu \
    # The store lives on the volume, not in the image layers.
    SV_DATA_DIR=/data \
    # Set explicitly: config.py derives this from the source tree layout,
    # which is an assumption worth not relying on inside an image.
    SV_WEB_DIST=/app/web/dist

# Created here, owned by the runtime user, *before* VOLUME. Docker seeds a
# fresh named volume from whatever the image has at that path, ownership
# included; without this the volume arrives root-owned and the unprivileged
# process cannot create /data/videos, which ensure_dirs() does at startup.
RUN mkdir -p /data && chown app:app /data

# Declared so `docker run` without -v still works; compose mounts a named
# volume over it so an ingested corpus survives `down` and `up`.
VOLUME ["/data"]
EXPOSE 8000

USER app

# /health reports corpus size and device, so a passing check means CLIP loaded
# and the index opened -- not merely that a socket is listening. start-period
# covers the checkpoint load, which is slow on CPU.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# The factory runs the lifespan hook, which repairs a store left inconsistent
# by a crash before any traffic is served (M4).
CMD ["uvicorn", "sv_engine.api:create_default_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

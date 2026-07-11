"""Runtime configuration and device selection."""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Repo-root-relative data locations. Overridable so tests and containers can
# point at a scratch directory.
DATA_DIR = Path(os.environ.get("SV_DATA_DIR", "data")).resolve()
VIDEO_DIR = DATA_DIR / "videos"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "sv_engine.sqlite"

# CLIP checkpoint. ViT-B-32 is the fast/cheap baseline; ViT-L-14 is slower but
# better. Benchmark against Recall@K before switching (see CLAUDE.md).
CLIP_MODEL = os.environ.get("SV_CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.environ.get("SV_CLIP_PRETRAINED", "laion2b_s34b_b79k")

# Sampling defaults. See sampler.py for the trade-off these encode.
BASELINE_FPS = float(os.environ.get("SV_BASELINE_FPS", "1.0"))
SCENE_THRESHOLD = float(os.environ.get("SV_SCENE_THRESHOLD", "0.35"))
MIN_SAMPLE_GAP_SEC = float(os.environ.get("SV_MIN_SAMPLE_GAP", "0.4"))

# The built React UI (M5). Anchored to the checkout rather than DATA_DIR: it is
# code that ships with the repo, not data the user accumulates. Absent until
# `npm run build` has run, which the API treats as "no UI", not as an error.
WEB_DIST_DIR = Path(
    os.environ.get("SV_WEB_DIST", Path(__file__).resolve().parents[2] / "web" / "dist")
)

EMBED_BATCH_SIZE = int(os.environ.get("SV_EMBED_BATCH", "32"))
THUMBNAIL_WIDTH = int(os.environ.get("SV_THUMBNAIL_WIDTH", "384"))


def resolve_device() -> str:
    """Pick the best available torch device.

    Order is mps -> cuda -> cpu. Overridable via SV_DEVICE because Docker on
    macOS has no MPS passthrough and tests want a deterministic cpu run.
    """
    override = os.environ.get("SV_DEVICE")
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dirs() -> None:
    for path in (VIDEO_DIR, THUMBNAIL_DIR, INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)

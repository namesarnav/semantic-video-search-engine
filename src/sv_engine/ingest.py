"""Ingestion: video file -> sampled frames -> embeddings -> index.

M1 keys videos by a content hash already, even though there is no status table
to dedup against yet. Doing it now means M4's idempotency requirement is a
lookup, not a re-architecture.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import config, sampler
from .embedder import ClipEmbedder, get_embedder
from .index import FrameIndex, FrameRecord

# Hash in chunks: these files are hundreds of MB and do not belong in memory.
_HASH_CHUNK_BYTES = 1024 * 1024


def content_hash(path: Path | str) -> str:
    """SHA-256 of the file's bytes, truncated to 16 hex chars.

    Keyed on content, not filename, so the same video under two names is one
    video and a renamed file does not re-ingest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@dataclass
class IngestResult:
    video_id: str
    filename: str
    duration_sec: float
    frames_indexed: int
    skipped: bool = False


def _write_thumbnail(frame, video_id: str, timestamp_sec: float) -> Path:
    config.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    height = max(1, int(frame.shape[0] * (config.THUMBNAIL_WIDTH / frame.shape[1])))
    small = cv2.resize(
        frame, (config.THUMBNAIL_WIDTH, height), interpolation=cv2.INTER_AREA
    )
    # Milliseconds in the name so two samples in the same second cannot collide.
    path = config.THUMBNAIL_DIR / f"{video_id}_{int(timestamp_sec * 1000):09d}.jpg"
    cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def ingest_video(
    path: Path | str,
    index: FrameIndex,
    embedder: ClipEmbedder | None = None,
    *,
    scene_threshold: float | None = config.SCENE_THRESHOLD,
    baseline_fps: float = config.BASELINE_FPS,
    skip_if_present: bool = True,
) -> IngestResult:
    """Sample, embed and index one video. Mutates ``index`` in place."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    embedder = embedder or get_embedder()
    video_id = content_hash(path)

    if skip_if_present and video_id in index.video_ids():
        return IngestResult(
            video_id=video_id,
            filename=path.name,
            duration_sec=sampler.video_duration_sec(path),
            frames_indexed=0,
            skipped=True,
        )

    frames = list(
        sampler.sample_video(
            path, baseline_fps=baseline_fps, scene_threshold=scene_threshold
        )
    )
    if not frames:
        return IngestResult(video_id, path.name, 0.0, 0)

    vectors = embedder.encode_images([f.image for f in frames])
    records = [
        FrameRecord(
            video_id=video_id,
            filename=path.name,
            timestamp_sec=f.timestamp_sec,
            thumbnail_path=str(_write_thumbnail(f.image, video_id, f.timestamp_sec)),
            reason=f.reason,
        )
        for f in frames
    ]
    index.add(vectors, records)

    return IngestResult(
        video_id=video_id,
        filename=path.name,
        duration_sec=sampler.video_duration_sec(path),
        frames_indexed=len(records),
    )

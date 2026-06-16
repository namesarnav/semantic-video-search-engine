"""Ingestion: video file -> sampled frames -> embeddings -> index + database.

Ordering matters here. A video's frames are sampled and embedded *entirely*
before anything is written, so a failure part-way through leaves no vectors in
the index and no rows in the database. That gives atomicity at video
granularity, which a flat FAISS index cannot otherwise provide -- it has no way
to remove vectors without shifting every id after them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import config, db, sampler
from .db import Database
from .embedder import ClipEmbedder, get_embedder
from .index import VectorIndex

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
    status: str
    skipped: bool = False
    error: str | None = None


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
    index: VectorIndex,
    database: Database,
    embedder: ClipEmbedder | None = None,
    *,
    scene_threshold: float | None = config.SCENE_THRESHOLD,
    baseline_fps: float = config.BASELINE_FPS,
    force: bool = False,
) -> IngestResult:
    """Sample, embed and index one video. Mutates ``index`` and ``database``.

    Re-ingesting an already-``done`` video is a no-op unless ``force`` is set
    (FR5). A video left in ``processing`` by an earlier crash is retried.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    embedder = embedder or get_embedder()
    video_id = content_hash(path)

    existing = database.get_video(video_id)
    if existing and existing.status == db.DONE and not force:
        return IngestResult(
            video_id=video_id,
            filename=path.name,
            duration_sec=existing.duration_sec,
            frames_indexed=0,
            status=db.DONE,
            skipped=True,
        )

    # Register the video before anything that can fail. Probing a corrupt file
    # raises, and a video that vanishes with no row at all is exactly the
    # silent failure FR6 exists to prevent.
    database.upsert_video(
        video_id, path.name, str(path.resolve()), 0.0, status=db.PROCESSING
    )

    duration = 0.0
    try:
        duration = sampler.video_duration_sec(path)
        database.upsert_video(
            video_id, path.name, str(path.resolve()), duration, status=db.PROCESSING
        )

        frames = list(
            sampler.sample_video(
                path, baseline_fps=baseline_fps, scene_threshold=scene_threshold
            )
        )
        if not frames:
            database.set_status(video_id, db.FAILED, error="no frames sampled")
            return IngestResult(
                video_id, path.name, duration, 0, db.FAILED, error="no frames sampled"
            )

        # Embed everything before writing anything.
        vectors = embedder.encode_images([f.image for f in frames])
        thumbnails = [
            str(_write_thumbnail(f.image, video_id, f.timestamp_sec)) for f in frames
        ]

        vector_ids = index.add(vectors)
        database.add_frames(
            video_id,
            [
                {
                    "timestamp_sec": f.timestamp_sec,
                    "thumbnail_path": thumb,
                    "reason": f.reason,
                    "vector_index_id": vid,
                }
                for f, thumb, vid in zip(frames, thumbnails, vector_ids)
            ],
        )
        database.set_status(video_id, db.DONE)

        return IngestResult(
            video_id=video_id,
            filename=path.name,
            duration_sec=duration,
            frames_indexed=len(frames),
            status=db.DONE,
        )
    except Exception as exc:  # noqa: BLE001 - status must reflect any failure
        # FR6: a crash must leave a visible `failed`, never a silent `processing`.
        database.set_status(video_id, db.FAILED, error=str(exc))
        raise

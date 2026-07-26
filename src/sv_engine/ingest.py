"""Ingestion: video file -> sampled frames -> embeddings -> index + database.

Ordering matters here. A video's frames are sampled and embedded *entirely*
before anything is written, so a failure part-way through leaves no vectors in
the index and no rows in the database. That gives atomicity at video
granularity, which a flat FAISS index cannot otherwise provide -- it has no way
to remove vectors without shifting every id after them.

The write itself is ordered for the benefit of crash recovery (M4). Under
``index.appending`` -- so no other ingest can interleave its vectors -- the
index is persisted *first*, and only then are the rows and the ``done`` status
committed in one transaction. Every crash window therefore leaves at most a run
of surplus vectors at the tail of the index, which ``recovery.reconcile`` can
drop without moving a single surviving ``vector_index_id``. The opposite order
would lose vectors the database still points at, which nothing short of
re-embedding can repair.
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


def _write_thumbnail(
    frame, video_id: str, timestamp_sec: float, thumbnail_dir: Path | None = None
) -> Path:
    directory = Path(thumbnail_dir) if thumbnail_dir else config.THUMBNAIL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    height = max(1, int(frame.shape[0] * (config.THUMBNAIL_WIDTH / frame.shape[1])))
    small = cv2.resize(
        frame, (config.THUMBNAIL_WIDTH, height), interpolation=cv2.INTER_AREA
    )
    # Milliseconds in the name so two samples in the same second cannot collide.
    path = directory / f"{video_id}_{int(timestamp_sec * 1000):09d}.jpg"
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
    index_dir: Path | None = None,
    thumbnail_dir: Path | None = None,
) -> IngestResult:
    """Sample, embed and index one video. Mutates ``index`` and ``database``.

    Re-ingesting an already-``done`` video is a no-op unless ``force`` is set
    (FR5). A video left in ``processing`` by an earlier crash is retried; see
    ``recovery.py`` for how such a video is swept to ``failed`` at startup.

    Passing ``index_dir`` makes each video durable as it completes, rather than
    only when the caller eventually saves. Both front ends do so: an unsaved
    index is lost on restart even though the database says ``done``.
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

    if existing is not None and database.frame_count(video_id) > 0:
        # The previous run's frames are still indexed, and their vectors sit in
        # the middle of a flat index. Adding a second set would silently double
        # every hit for this video, so the old ones are compacted out first --
        # index rebuilt and every surviving vector_index_id renumbered, as one
        # crash-repairable unit. See compaction.py for why that needs a marker.
        from .compaction import drop_video

        if index_dir is None:
            # Compaction swaps a file on disk, so it needs to know which one.
            # Refusing beats guessing at config.INDEX_DIR: that guess once let
            # a unit test overwrite a real index.
            raise ValueError(
                f"{path.name} already has {database.frame_count(video_id)} indexed "
                "frames, so re-ingesting must first compact the old ones out -- "
                "which needs an explicit index_dir. Pass one, or use --rebuild."
            )
        drop_video(video_id, index, database, index_dir=index_dir)
        existing = None

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

        # Embed everything before writing anything. Deliberately outside the
        # append lock below: embedding is the slow part, and holding a lock
        # across it would make one ingest block every other.
        vectors = embedder.encode_images([f.image for f in frames])
        thumbnails = [
            str(_write_thumbnail(f.image, video_id, f.timestamp_sec, thumbnail_dir))
            for f in frames
        ]

        with index.appending():
            before = len(index)
            vector_ids = index.add(vectors)
            try:
                if index_dir is not None:
                    index.save(index_dir)
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
                    status=db.DONE,
                )
            except Exception:
                # Roll the in-memory index back to where it started. The lock
                # guarantees these vectors are still the tail, so dropping them
                # is exact -- and leaving them would strand vectors no row
                # points at, which reports the whole store as out of sync.
                index.truncate(before)
                if index_dir is not None:
                    index.save(index_dir)
                raise

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

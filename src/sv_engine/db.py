"""SQLite metadata store.

FAISS holds vectors and nothing else. Everything needed to turn a vector hit
back into something meaningful -- which video, which timestamp, which thumbnail
-- lives here, joined by ``frames.vector_index_id``: the vector's position in
the FAISS index.

Keeping the two consistent is the sharpest failure mode in the system. A
mismatch does not raise, it silently answers with the wrong timestamp. So:

* a video's frames are written in one transaction, after its vectors are
  already computed, so a mid-ingest failure leaves neither behind;
* ``check_consistency`` compares row count against index size and is called on
  every load.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from . import config

# Ingestion status values (FR6). Persisted rather than held in memory so a
# crash cannot leave a video silently stuck with no way to notice.
QUEUED = "queued"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"
VALID_STATUSES = frozenset({QUEUED, PROCESSING, DONE, FAILED})

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id           TEXT PRIMARY KEY,        -- content hash, not filename
    filename     TEXT NOT NULL,
    path         TEXT NOT NULL,
    duration_sec REAL NOT NULL DEFAULT 0,
    status       TEXT NOT NULL,
    error        TEXT,
    ingested_at  TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    timestamp_sec   REAL NOT NULL,
    thumbnail_path  TEXT NOT NULL,
    reason          TEXT NOT NULL,
    vector_index_id INTEGER NOT NULL UNIQUE   -- position in the FAISS index
);

CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id);
CREATE INDEX IF NOT EXISTS idx_frames_vector ON frames(vector_index_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
"""


@dataclass(frozen=True)
class VideoRow:
    id: str
    filename: str
    path: str
    duration_sec: float
    status: str
    error: str | None
    ingested_at: str | None


@dataclass(frozen=True)
class FrameRow:
    id: int
    video_id: str
    filename: str
    timestamp_sec: float
    thumbnail_path: str
    reason: str
    vector_index_id: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thin wrapper over sqlite3. Not thread-safe; open one per thread."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else config.DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        # ON DELETE CASCADE is off by default in sqlite and must be asked for.
        self.conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets a reader (search) proceed while a writer (ingestion) works,
        # which M3 needs once ingestion runs as a background task.
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- videos -------------------------------------------------------

    def upsert_video(
        self,
        video_id: str,
        filename: str,
        path: str,
        duration_sec: float = 0.0,
        status: str = QUEUED,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown status: {status!r}")
        self.conn.execute(
            """
            INSERT INTO videos (id, filename, path, duration_sec, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename = excluded.filename,
                path = excluded.path,
                duration_sec = excluded.duration_sec,
                status = excluded.status
            """,
            (video_id, filename, path, duration_sec, status),
        )
        self.conn.commit()

    def set_status(
        self, video_id: str, status: str, error: str | None = None
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown status: {status!r}")
        self.conn.execute(
            "UPDATE videos SET status = ?, error = ?, ingested_at = ? WHERE id = ?",
            (status, error, _utc_now() if status == DONE else None, video_id),
        )
        self.conn.commit()

    def get_video(self, video_id: str) -> VideoRow | None:
        row = self.conn.execute(
            "SELECT * FROM videos WHERE id = ?", (video_id,)
        ).fetchone()
        return VideoRow(**dict(row)) if row else None

    def list_videos(self, status: str | None = None) -> list[VideoRow]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM videos ORDER BY filename"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM videos WHERE status = ? ORDER BY filename", (status,)
            ).fetchall()
        return [VideoRow(**dict(r)) for r in rows]

    def searchable_filenames(self) -> set[str]:
        """Filenames of videos that actually have frames in the index.

        Distinct from `list_videos()`: a video can exist as a row while being
        `queued` or `failed`, and no query will ever return it. The eval
        harness validates labels against this, so "your label names a video
        that cannot be found" and "the video is there but was never ingested"
        are the same answer -- which is the honest one.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT v.filename FROM videos v JOIN frames f ON f.video_id = v.id"
        ).fetchall()
        return {r["filename"] for r in rows}

    def frame_reason_counts(self) -> dict[str, int]:
        """How many frames came from each sampling reason.

        Lets an eval report say which sampling arm produced the store it
        measured: zero `scene_cut` frames means the corpus was built with
        `--fixed-interval`. Without that, two A/B reports are two numbers with
        no record of what they are comparing.
        """
        rows = self.conn.execute(
            "SELECT reason, COUNT(*) AS n FROM frames GROUP BY reason"
        ).fetchall()
        return {r["reason"]: r["n"] for r in rows}

    def delete_video(self, video_id: str) -> None:
        """Remove a video and its frames. Does not touch the FAISS index --
        callers must rebuild, since a flat index cannot remove vectors without
        invalidating every id after the removed one."""
        self.conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        self.conn.commit()

    # ---- frames -------------------------------------------------------

    def add_frames(
        self, video_id: str, frames: Sequence[dict], *, status: str | None = None
    ) -> None:
        """Insert a video's frames in a single transaction.

        Called only after the vectors are computed, so a failure during
        embedding leaves no half-written rows.

        ``status`` is applied inside that same transaction. Ingestion passes
        ``done``, which closes the window a separate ``set_status`` would leave
        open: a crash between the two commits would leave a complete set of
        frames on a video still marked ``processing``, and recovery -- unable
        to tell that from a half-written one -- would re-ingest it and double
        every frame.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"unknown status: {status!r}")
        with self.conn:  # commits on success, rolls back on exception
            self.conn.executemany(
                """
                INSERT INTO frames
                    (video_id, timestamp_sec, thumbnail_path, reason, vector_index_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        video_id,
                        f["timestamp_sec"],
                        f["thumbnail_path"],
                        f["reason"],
                        f["vector_index_id"],
                    )
                    for f in frames
                ],
            )
            if status is not None:
                self.conn.execute(
                    "UPDATE videos SET status = ?, error = NULL, ingested_at = ? "
                    "WHERE id = ?",
                    (status, _utc_now() if status == DONE else None, video_id),
                )

    def delete_frames(self, video_id: str) -> int:
        """Drop one video's frames while keeping the video row.

        Used by crash recovery: a video whose vectors did not survive is failed
        wholesale rather than left half-searchable, but the row must stay so the
        failure is visible and the file can be re-ingested.
        """
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM frames WHERE video_id = ?", (video_id,)
            )
        return cursor.rowcount

    def video_ids_with_vectors_at_or_above(self, threshold: int) -> list[str]:
        """Videos owning a frame whose vector is past the end of the index.

        After a crash the index can be shorter than the metadata; every video
        listed here is missing at least one vector and cannot be trusted.
        """
        rows = self.conn.execute(
            """
            SELECT DISTINCT video_id FROM frames
            WHERE vector_index_id >= ?
            ORDER BY video_id
            """,
            (threshold,),
        ).fetchall()
        return [r["video_id"] for r in rows]

    def frames_by_vector_ids(self, vector_ids: Iterable[int]) -> dict[int, FrameRow]:
        """Look up frames by FAISS position. Keyed by vector_index_id so the
        caller can preserve FAISS's ranking rather than SQL's row order."""
        ids = list(vector_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"""
            SELECT f.*, v.filename
            FROM frames f
            JOIN videos v ON v.id = f.video_id
            WHERE f.vector_index_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        return {r["vector_index_id"]: FrameRow(**dict(r)) for r in rows}

    def get_frame(self, frame_id: int) -> FrameRow | None:
        """Fetch one frame by primary key, for serving its thumbnail."""
        row = self.conn.execute(
            """
            SELECT f.*, v.filename
            FROM frames f
            JOIN videos v ON v.id = f.video_id
            WHERE f.id = ?
            """,
            (frame_id,),
        ).fetchone()
        return FrameRow(**dict(row)) if row else None

    def frame_count(self, video_id: str | None = None) -> int:
        """Frames across the corpus, or for one video when given an id."""
        if video_id is None:
            return int(self.conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM frames WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
        )

    def max_vector_id(self) -> int:
        """Highest assigned FAISS position, or -1 when empty."""
        value = self.conn.execute(
            "SELECT MAX(vector_index_id) FROM frames"
        ).fetchone()[0]
        return -1 if value is None else int(value)

    def check_consistency(self, index_size: int) -> None:
        """Fail loudly if the metadata and the vector index disagree."""
        count = self.frame_count()
        if count != index_size:
            raise ValueError(
                "database and vector index are out of sync: "
                f"{count} frame rows vs {index_size} vectors. "
                "Run `sv-engine recover`, or re-ingest with --rebuild."
            )
        highest = self.max_vector_id()
        if count and highest != count - 1:
            raise ValueError(
                "vector ids are not contiguous: "
                f"{count} rows but highest id is {highest}. "
                "Run `sv-engine recover`, or re-ingest with --rebuild."
            )

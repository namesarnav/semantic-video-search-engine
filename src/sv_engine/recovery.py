"""Crash recovery (M4).

Every guarantee ``ingest.py`` makes on the way down is best-effort: a clean
exception marks the video ``failed``, but ``kill -9``, an OOM kill or a power
cut run no ``except`` block at all. What is left on disk afterwards has to be
repairable from the disk alone. That is this module's whole job, and it runs at
startup -- before the CLI ingests anything and before the API accepts a
request.

Two kinds of damage are possible.

**A video stuck mid-flight.** ``processing`` means "a worker is on it", and
``queued`` means "a worker is about to be". After a restart neither is true:
the worker died with the process and nothing re-queues it. Both are swept to
``failed`` with the reason recorded, which is what makes the status column
honest -- FR6 exists so a dead ingest is *noticeable*.

**The index and the database disagreeing.** ``ingest_video`` persists vectors
before committing rows (and holds ``VectorIndex.appending`` across both), so a
crash can only ever leave surplus vectors at the *tail* of the index. That
shape is repairable: dropping a tail shifts no surviving ``vector_index_id``.
The reverse -- rows whose vectors never reached disk -- is not repairable
without re-embedding, so those videos are failed wholesale and re-ingested.
Half a video in the index is worse than none: it answers queries confidently
with the frames that happen to have survived.

Recovery assumes it is the only thing running. It is startup-only for that
reason: sweeping while another process ingests would fail a live job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import config, db
from .db import Database
from .index import VectorIndex

# Recorded in `videos.error`, so `sv-engine videos` says why a video is failed
# rather than leaving a bare status.
def _interrupted_reason(status: str) -> str:
    return (
        f"interrupted: the process exited while this video was {status}. "
        "Re-ingest to retry."
    )


LOST_VECTORS_REASON = (
    "interrupted: vectors for this video did not reach disk, so its frames "
    "were discarded. Re-ingest to retry."
)


@dataclass(frozen=True)
class RecoveryReport:
    """What a recovery pass had to repair. Empty is the healthy case."""

    swept: list[str] = field(default_factory=list)
    dropped_videos: list[str] = field(default_factory=list)
    dropped_frames: int = 0
    dropped_vectors: int = 0
    # What compaction repair did, if anything -- a compaction interrupted by
    # a crash is finished or discarded before the rest of the pass runs.
    compaction: str | None = None

    @property
    def clean(self) -> bool:
        return not (
            self.swept
            or self.dropped_videos
            or self.dropped_frames
            or self.dropped_vectors
            or self.compaction
        )

    def describe(self) -> str:
        if self.clean:
            return "store is consistent; nothing to recover"
        parts = []
        if self.compaction:
            parts.append(self.compaction)
        if self.swept:
            parts.append(f"{len(self.swept)} interrupted video(s) marked failed")
        if self.dropped_videos:
            parts.append(
                f"{len(self.dropped_videos)} video(s) discarded "
                f"({self.dropped_frames} frame rows) -- their vectors were lost"
            )
        if self.dropped_vectors:
            parts.append(f"{self.dropped_vectors} orphaned vector(s) dropped")
        return "recovered: " + "; ".join(parts)


def sweep_interrupted(database: Database) -> list[str]:
    """Fail every video the previous process left mid-flight.

    Returns the ids that were swept, so the caller can report them.
    """
    swept: list[str] = []
    for status in (db.PROCESSING, db.QUEUED):
        for row in database.list_videos(status=status):
            database.set_status(row.id, db.FAILED, error=_interrupted_reason(status))
            swept.append(row.id)
    return sorted(swept)


def reconcile(
    index: VectorIndex,
    database: Database,
    index_dir: Path | str | None = None,
) -> RecoveryReport:
    """Bring the vector index and the metadata back into agreement.

    Repairs in one direction only -- by discarding. Nothing here re-embeds, and
    nothing here moves a vector: the surviving ids must keep pointing at the
    same frames they did before.
    """
    dropped_videos: list[str] = []
    dropped_frames = 0

    # 1. Rows whose vectors are past the end of the index. Those frames are
    #    unrecoverable, and a video is all-or-nothing, so drop all of its rows.
    for video_id in database.video_ids_with_vectors_at_or_above(len(index)):
        dropped_frames += database.delete_frames(video_id)
        database.set_status(video_id, db.FAILED, error=LOST_VECTORS_REASON)
        dropped_videos.append(video_id)

    # 2. Vectors no row points at -- either surplus from a crash between the
    #    index write and the commit, or the partial video just discarded above.
    #    Both are at the tail, because writers hold `appending` across the
    #    whole unit, so a plain truncation is exact.
    dropped_vectors = len(index) - database.frame_count()
    if dropped_vectors > 0:
        index.truncate(database.frame_count())
        if index_dir is not None:
            # Persist, or the next startup repeats this repair -- and worse,
            # a search in between would see the unrepaired file.
            index.save(index_dir)
    else:
        dropped_vectors = 0

    report = RecoveryReport(
        dropped_videos=dropped_videos,
        dropped_frames=dropped_frames,
        dropped_vectors=dropped_vectors,
    )
    # If this still fails the damage is not a tail, which nothing here can fix.
    database.check_consistency(len(index))
    return report


def recover(
    index: VectorIndex,
    database: Database,
    index_dir: Path | str | None = None,
) -> RecoveryReport:
    """Full startup pass: finish any interrupted compaction, sweep abandoned
    videos, then reconcile the store.

    Compaction repair runs **first and reloads the index**. It decides which
    index file is the live one, so reconciling before it would compare the
    database against a file that is about to be replaced -- and conclude,
    correctly but uselessly, that they disagree.
    """
    from . import compaction  # local: compaction imports VectorIndex

    directory = Path(index_dir) if index_dir is not None else config.INDEX_DIR
    compacted = compaction.repair(database, index_dir=directory)
    if compacted:
        # The file on disk may have just changed underneath this process.
        try:
            index.index = VectorIndex.load(directory).index
        except FileNotFoundError:
            pass

    swept = sweep_interrupted(database)
    reconciled = reconcile(index, database, index_dir=index_dir)
    return RecoveryReport(
        swept=swept,
        dropped_videos=reconciled.dropped_videos,
        dropped_frames=reconciled.dropped_frames,
        dropped_vectors=reconciled.dropped_vectors,
        compaction=compacted,
    )

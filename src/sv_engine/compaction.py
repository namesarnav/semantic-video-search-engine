"""Removing a video from the store without corrupting every frame after it.

A flat FAISS index cannot delete a vector. Removing position 3 shifts 4, 5, 6
down by one, and every `frames.vector_index_id` above it then points at the
wrong video and timestamp -- silently, with confident wrong answers. So the
index is rebuilt without the dropped video's vectors and the mapping is
rewritten to match, as one unit.

**Why this needs a write-ahead marker.** Unlike an append, a compaction has no
safe write order. Saving the index first leaves a compacted index against
stale ids; committing the rows first leaves new ids against the old index.
Both are silent corruption, and neither is what M4's "surplus vectors at the
tail" invariant can repair -- `truncate` drops from the end, and compaction
renumbers from the middle.

So the operation announces itself before it swaps:

1. build the compacted index and stage it beside the live one;
2. in **one** SQLite transaction, renumber the survivors, delete the video,
   and record that a swap is owed;
3. `os.replace` the staged file over the live one;
4. clear the marker.

Every interruption then lands somewhere repairable, and `recovery.recover`
finishes the job at startup:

* crash before (2) -- nothing is committed and the marker is absent, so the
  staged file is an orphan and is deleted. The old store is intact.
* crash between (2) and (3) -- the database already describes the compacted
  index, and the staged file holds it. Recovery completes the swap.
* crash between (3) and (4) -- the swap already happened and the staged file
  is gone. Recovery clears the marker and stops. Idempotent.

The marker holds the staged filename rather than a boolean so repair can tell
those last two apart by asking whether the file is still there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from . import config
from .db import Database
from .index import VectorIndex

PENDING_SWAP_KEY = "compaction.pending_swap"
_STAGED_FILENAME = "frames.faiss.compacting"


def staged_path(index_dir: Path | str = config.INDEX_DIR) -> Path:
    return Path(index_dir) / _STAGED_FILENAME


@dataclass(frozen=True)
class CompactionResult:
    video_id: str
    frames_removed: int
    vectors_before: int
    vectors_after: int

    def describe(self) -> str:
        return (
            f"dropped {self.video_id}: {self.frames_removed} frames, "
            f"index {self.vectors_before} -> {self.vectors_after} vectors"
        )


def drop_video(
    video_id: str,
    index: VectorIndex,
    database: Database,
    index_dir: Path | str,
    *,
    _stop_before_swap: bool = False,
) -> CompactionResult:
    """Remove one video's frames and vectors, renumbering the survivors.

    ``index_dir`` is **required and deliberately has no default.** An earlier
    version defaulted it to ``config.INDEX_DIR``, and a unit test that did not
    pass one duly compacted the developer's real index out from under them.
    A destructive file operation should never be able to reach a global
    location because a caller left an argument off.

    ``_stop_before_swap`` exists only so a test can occupy the crash window
    between the commit and the rename; nothing in production passes it.
    """
    if database.get_video(video_id) is None:
        raise KeyError(f"unknown video: {video_id}")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Hold the writer lock for the whole operation: a concurrent ingest would
    # be appending to the index this is rebuilding, and its vectors would be
    # dropped on the floor by the swap.
    with index.appending():
        survivors = database.conn.execute(
            "SELECT id, vector_index_id FROM frames "
            "WHERE video_id != ? ORDER BY vector_index_id",
            (video_id,),
        ).fetchall()
        doomed = database.frame_count(video_id=video_id)
        before = len(index)

        # Pull the surviving vectors out of the live index in their current
        # order, so the rebuilt index holds exactly them, renumbered 0..n-1.
        rebuilt = faiss.IndexFlatIP(index.dim)
        if survivors:
            vectors = np.vstack(
                [index.index.reconstruct(int(r["vector_index_id"])) for r in survivors]
            ).astype(np.float32)
            rebuilt.add(vectors)

        staged = staged_path(index_dir)
        faiss.write_index(rebuilt, str(staged))

        try:
            _commit_remap(database, video_id, survivors)
        except Exception:
            staged.unlink(missing_ok=True)
            raise

        if not _stop_before_swap:
            _finish_swap(database, index_dir)

        # Adopt the rebuilt index in memory too, or this process keeps serving
        # from the pre-compaction one.
        index.index = rebuilt

    return CompactionResult(
        video_id=video_id,
        frames_removed=doomed,
        vectors_before=before,
        vectors_after=rebuilt.ntotal,
    )


def _commit_remap(database: Database, video_id: str, survivors) -> None:
    """Renumber survivors and drop the video, atomically, marking the swap.

    Renumbering runs in two passes through negative ids. `vector_index_id` is
    UNIQUE, so assigning final values directly would collide with rows that
    have not moved yet; negatives cannot collide with positives.
    """
    conn = database.conn
    with conn:  # BEGIN ... COMMIT, or ROLLBACK on exception
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        conn.executemany(
            "UPDATE frames SET vector_index_id = ? WHERE id = ?",
            [(-(new + 1), row["id"]) for new, row in enumerate(survivors)],
        )
        conn.execute(
            "UPDATE frames SET vector_index_id = -vector_index_id - 1 "
            "WHERE vector_index_id < 0"
        )
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (PENDING_SWAP_KEY, _STAGED_FILENAME),
        )


def _finish_swap(database: Database, index_dir: Path) -> None:
    staged = staged_path(index_dir)
    if staged.exists():
        os.replace(staged, VectorIndex._path(index_dir))
    database.delete_meta(PENDING_SWAP_KEY)


def repair(database: Database, index_dir: Path | str = config.INDEX_DIR) -> str | None:
    """Finish or discard a compaction interrupted by a crash.

    Called from `recovery.recover`, before anything reads the index. Returns a
    description of what it did, or None if there was nothing to do.
    """
    index_dir = Path(index_dir)
    staged = staged_path(index_dir)
    pending = database.get_meta(PENDING_SWAP_KEY)

    if pending is None:
        if staged.exists():
            # Staged but never committed: the database still describes the old
            # index, so this file is an orphan from a crash before the commit.
            staged.unlink()
            return "discarded an unfinished index compaction"
        return None

    if staged.exists():
        os.replace(staged, VectorIndex._path(index_dir))
        database.delete_meta(PENDING_SWAP_KEY)
        return "completed an interrupted index compaction"

    # Marker set, staged file gone: the swap already happened and only the
    # marker outlived it.
    database.delete_meta(PENDING_SWAP_KEY)
    return "cleared a stale compaction marker (swap had already completed)"

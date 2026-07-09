"""Crash recovery (M4).

A kill -9 does not run ``except`` blocks. Everything ``ingest.py`` does to
report failure is therefore best-effort only, and the state left behind after a
hard stop has to be repairable from what is on disk alone.

Two kinds of damage are possible, and only one of them is recoverable:

* the index holds vectors the database has no rows for -- repairable, because
  those vectors are always at the *tail* (see ``VectorIndex.appending``), and a
  tail can be dropped without shifting any surviving ``vector_index_id``;
* the database holds rows whose vectors never reached disk -- those frames are
  gone, so the video they belong to is failed wholesale and re-ingested.

The tests below pin both directions, plus the sweep that turns an abandoned
``processing`` row into a visible ``failed`` one.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_engine import db, recovery
from sv_engine.db import Database
from sv_engine.index import VectorIndex
from sv_engine.ingest import ingest_video

DIM = 4


class StubEmbedder:
    """Stands in for CLIP so these stay in the fast suite."""

    dim = DIM
    device = "cpu"

    def encode_images(self, images):
        arr = np.linspace(1.0, 2.0, num=len(images) * DIM, dtype=np.float32)
        arr = arr.reshape(len(images), DIM)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


@pytest.fixture
def database(tmp_path):
    with Database(tmp_path / "test.sqlite") as conn:
        yield conn


@pytest.fixture
def index():
    return VectorIndex(dim=DIM)


def _seed(database, index, video_id: str, frames: int, status: str = db.DONE) -> None:
    """Add one video's worth of vectors and matching rows, the way a
    successful ingest would."""
    vectors = np.random.default_rng(abs(hash(video_id)) % 2**32).normal(
        size=(frames, DIM)
    ).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = index.add(vectors)

    database.upsert_video(video_id, f"{video_id}.mp4", f"/v/{video_id}.mp4", 10.0)
    database.add_frames(
        video_id,
        [
            {
                "timestamp_sec": float(i),
                "thumbnail_path": f"/t/{video_id}_{i}.jpg",
                "reason": "baseline",
                "vector_index_id": vid,
            }
            for i, vid in enumerate(ids)
        ],
    )
    database.set_status(video_id, status)


# ---- the sweep -----------------------------------------------------------


def test_sweep_marks_an_abandoned_processing_video_failed(database):
    """The exact state a kill -9 leaves: mid-ingest, no except block ran."""
    database.upsert_video("v1", "v1.mp4", "/v/v1.mp4", status=db.PROCESSING)

    swept = recovery.sweep_interrupted(database)

    assert swept == ["v1"]
    row = database.get_video("v1")
    assert row.status == db.FAILED
    assert row.error and "interrupted" in row.error


def test_sweep_marks_a_never_started_queued_video_failed(database):
    """``queued`` is just as stuck: the task that would have run it died with
    the process, and nothing re-queues it on restart."""
    database.upsert_video("v1", "v1.mp4", "/v/v1.mp4", status=db.QUEUED)

    assert recovery.sweep_interrupted(database) == ["v1"]
    assert database.get_video("v1").status == db.FAILED


def test_sweep_records_which_status_was_abandoned(database):
    database.upsert_video("v1", "v1.mp4", "/v/v1.mp4", status=db.PROCESSING)
    recovery.sweep_interrupted(database)
    assert db.PROCESSING in database.get_video("v1").error


def test_sweep_leaves_done_and_failed_alone(database):
    database.upsert_video("ok", "ok.mp4", "/v/ok.mp4", status=db.DONE)
    database.upsert_video("bad", "bad.mp4", "/v/bad.mp4")
    database.set_status("bad", db.FAILED, error="unreadable file")

    assert recovery.sweep_interrupted(database) == []
    assert database.get_video("ok").status == db.DONE
    # An existing failure must keep its own reason, not be overwritten.
    assert database.get_video("bad").error == "unreadable file"


# ---- reconciling the index against the database --------------------------


def test_reconcile_leaves_a_healthy_store_untouched(database, index):
    _seed(database, index, "a", frames=3)

    report = recovery.reconcile(index, database)

    assert report.clean
    assert len(index) == database.frame_count() == 3
    database.check_consistency(len(index))


def test_reconcile_drops_vectors_the_database_has_no_rows_for(database, index):
    """Crash after the index was persisted but before the rows were committed.

    The orphans are the tail, so dropping them is exact -- no surviving
    vector_index_id moves.
    """
    _seed(database, index, "a", frames=3)
    extra = np.eye(DIM, dtype=np.float32)[:2]
    index.add(extra)
    assert len(index) == 5

    report = recovery.reconcile(index, database)

    assert report.dropped_vectors == 2
    assert len(index) == 3
    database.check_consistency(len(index))


def test_reconcile_persists_the_truncated_index(database, index, tmp_path):
    _seed(database, index, "a", frames=3)
    index.add(np.eye(DIM, dtype=np.float32)[:2])

    recovery.reconcile(index, database, index_dir=tmp_path)

    # The repair has to survive the next restart, or every startup redoes it.
    assert len(VectorIndex.load(tmp_path)) == 3


def test_reconcile_fails_a_video_whose_vectors_never_reached_disk(database, index):
    """The other direction: rows committed, index write lost.

    Those frames cannot be recovered without re-embedding, so the whole video
    goes back to ``failed`` and is re-ingested rather than half-searched.
    """
    _seed(database, index, "a", frames=3)
    _seed(database, index, "b", frames=2)
    # Simulate an index that only made it as far as video a's vectors.
    index.truncate(3)

    report = recovery.reconcile(index, database)

    assert report.dropped_videos == ["b"]
    assert report.dropped_frames == 2
    assert database.get_video("b").status == db.FAILED
    assert "vectors" in database.get_video("b").error
    database.check_consistency(len(index))


def test_reconcile_keeps_the_videos_that_did_survive(database, index):
    _seed(database, index, "a", frames=3)
    _seed(database, index, "b", frames=2)
    index.truncate(3)

    recovery.reconcile(index, database)

    assert database.get_video("a").status == db.DONE
    assert database.frame_count("a") == 3
    assert database.frame_count("b") == 0


def test_reconcile_truncates_a_partially_written_video(database, index):
    """The messy middle: some of video b's vectors landed, some did not.

    Dropping b's rows leaves b's surviving vectors orphaned, so the index has
    to come down to the new row count too.
    """
    _seed(database, index, "a", frames=3)
    _seed(database, index, "b", frames=4)
    index.truncate(5)  # a's 3, plus 2 of b's 4

    report = recovery.reconcile(index, database)

    assert report.dropped_frames == 4
    assert report.dropped_vectors == 2
    assert len(index) == database.frame_count() == 3
    database.check_consistency(len(index))


def test_recover_sweeps_and_reconciles_together(database, index):
    _seed(database, index, "a", frames=3)
    _seed(database, index, "b", frames=2, status=db.PROCESSING)
    index.truncate(3)

    report = recovery.recover(index, database)

    assert not report.clean
    assert database.get_video("b").status == db.FAILED
    database.check_consistency(len(index))


def test_recover_on_an_empty_store_is_clean(database, index):
    assert recovery.recover(index, database).clean


def test_report_describes_what_it_repaired(database, index):
    _seed(database, index, "a", frames=2, status=db.PROCESSING)
    index.add(np.eye(DIM, dtype=np.float32)[:1])

    description = recovery.recover(index, database).describe()

    assert "1" in description and "vector" in description


# ---- the ingest-side half of the invariant --------------------------------


def test_a_failure_after_adding_vectors_rolls_the_index_back(
    database, index, synthetic_video, monkeypatch, tmp_path
):
    """A failure between ``index.add`` and the row commit must not strand
    vectors, or the very next search reports the store as out of sync."""
    path = synthetic_video(duration_sec=3.0, cut_at_sec=1.5)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Database, "add_frames", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        ingest_video(
            path,
            index,
            database,
            StubEmbedder(),
            baseline_fps=1.0,
            thumbnail_dir=tmp_path / "thumbs",
        )

    assert len(index) == 0
    assert database.frame_count() == 0
    database.check_consistency(len(index))


def test_a_failed_ingest_is_visible_as_failed(
    database, index, synthetic_video, monkeypatch, tmp_path
):
    path = synthetic_video(duration_sec=3.0, cut_at_sec=1.5)
    monkeypatch.setattr(
        Database, "add_frames", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )

    with pytest.raises(RuntimeError):
        ingest_video(path, index, database, StubEmbedder(), baseline_fps=1.0,
                     thumbnail_dir=tmp_path / "thumbs")

    rows = database.list_videos(status=db.FAILED)
    assert len(rows) == 1 and rows[0].error == "x"


def test_the_index_is_durable_before_the_rows_are_committed(
    database, index, synthetic_video, tmp_path
):
    """The ordering that makes crash damage repairable at all.

    If the rows were committed first, a crash would lose vectors the database
    still points at -- unrecoverable without re-embedding. Persisting the index
    first means the worst case is surplus vectors, which reconcile can drop.
    """
    path = synthetic_video(duration_sec=3.0, cut_at_sec=1.5)
    index_dir = tmp_path / "index"
    seen: list[int] = []

    original = Database.add_frames

    def spy(self, video_id, frames, **kwargs):
        # At commit time the index must already be on disk with every vector.
        seen.append(len(VectorIndex.load(index_dir)))
        return original(self, video_id, frames, **kwargs)

    Database.add_frames = spy
    try:
        result = ingest_video(
            path,
            index,
            database,
            StubEmbedder(),
            baseline_fps=1.0,
            index_dir=index_dir,
            thumbnail_dir=tmp_path / "thumbs",
        )
    finally:
        Database.add_frames = original

    assert seen == [result.frames_indexed]


def test_force_will_not_duplicate_an_already_indexed_video(
    database, index, synthetic_video, tmp_path
):
    """``--force`` re-ingests, but a flat index cannot drop the old vectors --
    they sit below newer ones and removing them would shift every id after.
    Adding a second set would silently double every hit for this video, so the
    only honest answer is to refuse and point at ``--rebuild``.
    """
    path = synthetic_video(duration_sec=3.0, cut_at_sec=1.5)
    thumbs = tmp_path / "thumbs"
    first = ingest_video(
        path, index, database, StubEmbedder(), baseline_fps=1.0, thumbnail_dir=thumbs
    )

    with pytest.raises(ValueError, match="rebuild"):
        ingest_video(
            path,
            index,
            database,
            StubEmbedder(),
            baseline_fps=1.0,
            force=True,
            thumbnail_dir=thumbs,
        )

    assert database.frame_count() == first.frames_indexed
    assert database.get_video(first.video_id).status == db.DONE
    database.check_consistency(len(index))

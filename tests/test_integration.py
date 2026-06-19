"""End-to-end tests (§6.2). These load the real CLIP checkpoint.

Marked ``slow`` so the fast unit suite stays under two seconds:

    uv run pytest -m "not slow"     # skip these
    uv run pytest -m slow           # only these
"""

from __future__ import annotations

import pytest

from sv_engine import config, db
from sv_engine.db import Database
from sv_engine.embedder import get_embedder
from sv_engine.index import VectorIndex
from sv_engine.ingest import ingest_video
from sv_engine.search import search

pytestmark = pytest.mark.slow


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A scratch index/database/thumbnail directory per test."""
    monkeypatch.setattr(config, "THUMBNAIL_DIR", tmp_path / "thumbs")
    database = Database(tmp_path / "test.sqlite")
    embedder = get_embedder()
    yield VectorIndex(dim=embedder.dim), database, embedder
    database.close()


def test_end_to_end_ingest_reaches_done(store, synthetic_video):
    index, database, embedder = store
    path = synthetic_video(duration_sec=6.0, cut_at_sec=3.0)

    result = ingest_video(path, index, database, embedder, baseline_fps=1.0)

    assert result.status == db.DONE
    assert result.frames_indexed > 0
    assert database.get_video(result.video_id).status == db.DONE
    # Index and database must agree on how many frames exist.
    assert len(index) == database.frame_count() == result.frames_indexed
    database.check_consistency(len(index))


def test_thumbnails_are_written(store, synthetic_video):
    index, database, embedder = store
    path = synthetic_video()

    ingest_video(path, index, database, embedder, baseline_fps=1.0)

    for row in database.frames_by_vector_ids(range(len(index))).values():
        assert config.THUMBNAIL_DIR.joinpath(
            row.thumbnail_path.rsplit("/", 1)[-1]
        ).exists()


def test_reingesting_is_a_noop(store, synthetic_video):
    """FR5: the same video twice must not duplicate frames."""
    index, database, embedder = store
    path = synthetic_video()

    first = ingest_video(path, index, database, embedder, baseline_fps=1.0)
    frames_after_first = database.frame_count()

    second = ingest_video(path, index, database, embedder, baseline_fps=1.0)

    assert second.skipped is True
    assert second.frames_indexed == 0
    assert database.frame_count() == frames_after_first
    assert len(index) == frames_after_first
    assert len(database.list_videos()) == 1
    assert first.video_id == second.video_id


def test_renamed_file_is_recognised_as_the_same_video(store, synthetic_video, tmp_path):
    """Identity is content, not filename."""
    index, database, embedder = store
    original = synthetic_video("original.mp4")
    copy = tmp_path / "renamed.mp4"
    copy.write_bytes(original.read_bytes())

    ingest_video(original, index, database, embedder, baseline_fps=1.0)
    second = ingest_video(copy, index, database, embedder, baseline_fps=1.0)

    assert second.skipped is True
    assert len(database.list_videos()) == 1


def test_failed_ingest_is_marked_failed_not_left_processing(store, tmp_path, monkeypatch):
    """FR6: a crash mid-ingest must be visible, not a silent `processing`."""
    index, database, embedder = store
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video at all")

    with pytest.raises(Exception):
        ingest_video(path, index, database, embedder, baseline_fps=1.0)

    rows = database.list_videos()
    assert len(rows) == 1
    assert rows[0].status == db.FAILED
    assert rows[0].error


def test_failed_ingest_leaves_no_orphan_vectors(store, synthetic_video, monkeypatch):
    """Embedding failure must write neither vectors nor rows -- otherwise the
    two stores diverge and every later lookup is wrong."""
    index, database, embedder = store
    path = synthetic_video()

    def explode(*args, **kwargs):
        raise RuntimeError("simulated embedding failure")

    monkeypatch.setattr(embedder, "encode_images", explode)

    with pytest.raises(RuntimeError, match="simulated"):
        ingest_video(path, index, database, embedder, baseline_fps=1.0)

    assert len(index) == 0
    assert database.frame_count() == 0
    assert database.list_videos()[0].status == db.FAILED
    database.check_consistency(len(index))


def test_cross_video_search_maps_back_correctly(store, synthetic_video):
    """FR4 + FR3: search spans videos and each hit resolves to the right one."""
    index, database, embedder = store
    red = synthetic_video("red.mp4", duration_sec=4.0, cut_at_sec=99.0)
    # Second video starts on the palette's blue entry by cutting at t=0.
    blue = synthetic_video("blue.mp4", duration_sec=4.0, cut_at_sec=0.0)

    ingest_video(red, index, database, embedder, baseline_fps=1.0)
    ingest_video(blue, index, database, embedder, baseline_fps=1.0)

    assert len(database.list_videos(status=db.DONE)) == 2

    results = search("a bright red screen", index, database, embedder, top_k=8)
    assert len({r.video_id for r in results}) == 2

    top = results[0]
    row = database.get_video(top.video_id)
    assert top.filename == row.filename
    assert 0.0 <= top.timestamp_sec <= 4.0

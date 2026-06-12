from __future__ import annotations

import pytest

from sv_engine import db
from sv_engine.db import Database


@pytest.fixture
def database(tmp_path):
    with Database(tmp_path / "test.sqlite") as conn:
        yield conn


def _frames(*specs) -> list[dict]:
    return [
        {
            "timestamp_sec": ts,
            "thumbnail_path": f"/thumbs/{vid}.jpg",
            "reason": "baseline",
            "vector_index_id": vid,
        }
        for ts, vid in specs
    ]


def test_video_starts_absent(database):
    assert database.get_video("nope") is None


def test_upsert_then_fetch(database):
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4", 12.5)
    row = database.get_video("abc")

    assert row.filename == "clip.mp4"
    assert row.duration_sec == 12.5
    assert row.status == db.QUEUED


def test_upsert_is_idempotent(database):
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4", 12.5)
    database.upsert_video("abc", "renamed.mp4", "/tmp/renamed.mp4", 12.5)

    # Same content hash means one row, updated -- never a duplicate.
    assert len(database.list_videos()) == 1
    assert database.get_video("abc").filename == "renamed.mp4"


def test_status_transitions_are_recorded(database):
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.set_status("abc", db.PROCESSING)
    assert database.get_video("abc").status == db.PROCESSING

    database.set_status("abc", db.DONE)
    row = database.get_video("abc")
    assert row.status == db.DONE
    assert row.ingested_at is not None


def test_failure_records_the_error(database):
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.set_status("abc", db.FAILED, error="decoder exploded")

    row = database.get_video("abc")
    assert row.status == db.FAILED
    assert row.error == "decoder exploded"


def test_unknown_status_is_rejected(database):
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    with pytest.raises(ValueError, match="unknown status"):
        database.set_status("abc", "definitely-not-a-status")


def test_list_videos_can_filter_by_status(database):
    database.upsert_video("a", "a.mp4", "/a.mp4", status=db.DONE)
    database.upsert_video("b", "b.mp4", "/b.mp4", status=db.FAILED)

    assert [v.id for v in database.list_videos(status=db.DONE)] == ["a"]
    assert [v.id for v in database.list_videos(status=db.FAILED)] == ["b"]
    assert len(database.list_videos()) == 2


def test_frames_are_looked_up_by_vector_id(database):
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.add_frames("abc", _frames((1.0, 0), (2.0, 1)))

    rows = database.frames_by_vector_ids([1])
    assert set(rows) == {1}
    assert rows[1].timestamp_sec == 2.0
    assert rows[1].filename == "clip.mp4"


def test_lookup_of_unknown_vector_ids_returns_nothing(database):
    assert database.frames_by_vector_ids([99]) == {}


def test_lookup_of_empty_list_returns_nothing(database):
    assert database.frames_by_vector_ids([]) == {}


def test_duplicate_vector_id_is_rejected(database):
    """Two frames pointing at one vector means the mapping is broken."""
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.add_frames("abc", _frames((1.0, 0)))

    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        database.add_frames("abc", _frames((2.0, 0)))


def test_add_frames_is_atomic(database):
    """A batch containing a conflict must write none of its rows, or a failed
    ingest would leave partial frames behind."""
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.add_frames("abc", _frames((1.0, 0)))

    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        # Second entry collides with the existing vector id 0.
        database.add_frames("abc", _frames((2.0, 1), (3.0, 0)))

    assert database.frame_count() == 1


def test_deleting_a_video_cascades_to_its_frames(database):
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.add_frames("abc", _frames((1.0, 0), (2.0, 1)))
    assert database.frame_count() == 2

    database.delete_video("abc")
    assert database.frame_count() == 0


def test_frames_require_an_existing_video(database):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        database.add_frames("ghost", _frames((1.0, 0)))


def test_max_vector_id_on_empty_database(database):
    assert database.max_vector_id() == -1


def test_consistency_passes_when_counts_agree(database):
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.add_frames("abc", _frames((1.0, 0), (2.0, 1)))

    database.check_consistency(index_size=2)  # must not raise


def test_consistency_detects_a_count_mismatch(database):
    """The failure this guard exists for: silently answering with the wrong
    timestamp because index and database disagree."""
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.add_frames("abc", _frames((1.0, 0), (2.0, 1)))

    with pytest.raises(ValueError, match="out of sync"):
        database.check_consistency(index_size=5)


def test_consistency_detects_non_contiguous_ids(database):
    """Row count can match while the ids themselves are wrong."""
    database.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4")
    database.add_frames("abc", _frames((1.0, 0), (2.0, 7)))

    with pytest.raises(ValueError, match="not contiguous"):
        database.check_consistency(index_size=2)


def test_schema_survives_reopening(tmp_path):
    path = tmp_path / "persist.sqlite"
    with Database(path) as first:
        first.upsert_video("abc", "clip.mp4", "/tmp/clip.mp4", status=db.DONE)
        first.add_frames("abc", _frames((1.0, 0)))

    with Database(path) as second:
        assert second.get_video("abc").status == db.DONE
        assert second.frame_count() == 1

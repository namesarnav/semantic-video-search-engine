"""Dropping a video from the index without corrupting every later frame.

A flat index cannot remove a vector: deleting position 3 shifts 4, 5, 6...
down by one, and every `frames.vector_index_id` above it silently starts
pointing at the wrong video and timestamp. That is the sharpest failure mode
in the system, so the fix rebuilds the index and rewrites the mapping as one
unit.

Neither write order is crash-safe on its own -- index-then-rows leaves a
compacted index against stale ids, rows-then-index leaves new ids against the
old index, and both are silent. So the operation records its intent in the
database before the swap, and every crash window below is a repairable state
rather than a corrupt one.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_engine import compaction, recovery
from sv_engine.db import Database
from sv_engine.index import VectorIndex

DIM = 4


def _store(tmp_path, videos=("a", "b", "c"), per_video=2):
    """A store with `per_video` frames for each named video."""
    index = VectorIndex(dim=DIM)
    database = Database(tmp_path / "sv.sqlite")
    (tmp_path / "index").mkdir(exist_ok=True)

    counter = 0
    for name in videos:
        vectors = np.zeros((per_video, DIM), dtype=np.float32)
        for row in range(per_video):
            # Distinct, normalized, and recoverable so we can prove which
            # vector survived where.
            vectors[row][counter % DIM] = 1.0
            counter += 1
        ids = index.add(vectors)
        database.upsert_video(name, f"{name}.mp4", f"/v/{name}.mp4", 10.0)
        database.add_frames(
            name,
            [
                {
                    "timestamp_sec": float(i),
                    "thumbnail_path": f"/t/{name}{i}.jpg",
                    "reason": "baseline",
                    "vector_index_id": ids[i],
                }
                for i in range(per_video)
            ],
        )
    index.save(tmp_path / "index")
    return index, database


def _mapping(database) -> dict[str, list[int]]:
    rows = database.conn.execute(
        "SELECT video_id, vector_index_id FROM frames ORDER BY vector_index_id"
    ).fetchall()
    out: dict[str, list[int]] = {}
    for row in rows:
        out.setdefault(row["video_id"], []).append(row["vector_index_id"])
    return out


# ---- the happy path ----------------------------------------------------


def test_dropping_a_video_removes_its_frames_and_vectors(tmp_path):
    index, database = _store(tmp_path)
    compaction.drop_video("b", index, database, index_dir=tmp_path / "index")

    assert database.get_video("b") is None
    assert len(index) == 4
    assert set(_mapping(database)) == {"a", "c"}
    database.close()


def test_surviving_ids_are_renumbered_contiguously_from_zero(tmp_path):
    """The invariant the whole system rests on: ids are positions in the
    index, so after a rebuild they must be 0..n-1 with no gaps."""
    index, database = _store(tmp_path)
    compaction.drop_video("b", index, database, index_dir=tmp_path / "index")

    ids = sorted(i for group in _mapping(database).values() for i in group)
    assert ids == list(range(len(index)))
    database.close()


def test_every_surviving_frame_still_points_at_its_own_vector(tmp_path):
    """Renumbering is only correct if each row follows its *own* vector. A
    frame that keeps its id but inherits a neighbour's vector is exactly the
    corruption this exists to prevent, and it is invisible without this."""
    index, database = _store(tmp_path)
    before = {
        row["id"]: index.index.reconstruct(row["vector_index_id"]).tolist()
        for row in database.conn.execute("SELECT id, vector_index_id FROM frames")
        if row["id"] is not None
    }

    compaction.drop_video("b", index, database, index_dir=tmp_path / "index")

    for row in database.conn.execute("SELECT id, vector_index_id FROM frames"):
        assert index.index.reconstruct(row["vector_index_id"]).tolist() == before[
            row["id"]
        ], f"frame {row['id']} now points at a different vector"
    database.close()


def test_the_compacted_index_survives_a_reload(tmp_path):
    index, database = _store(tmp_path)
    compaction.drop_video("b", index, database, index_dir=tmp_path / "index")

    reloaded = VectorIndex.load(tmp_path / "index")
    assert len(reloaded) == 4
    database.check_consistency(len(reloaded))
    database.close()


def test_dropping_the_only_video_leaves_an_empty_store(tmp_path):
    index, database = _store(tmp_path, videos=("solo",))
    compaction.drop_video("solo", index, database, index_dir=tmp_path / "index")

    assert len(index) == 0
    assert len(VectorIndex.load(tmp_path / "index")) == 0
    database.close()


def test_dropping_the_last_video_does_not_disturb_the_others(tmp_path):
    index, database = _store(tmp_path)
    compaction.drop_video("c", index, database, index_dir=tmp_path / "index")

    assert _mapping(database) == {"a": [0, 1], "b": [2, 3]}
    database.close()


def test_an_unknown_video_is_refused(tmp_path):
    index, database = _store(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        compaction.drop_video("nope", index, database, index_dir=tmp_path / "index")
    database.close()


# ---- crash windows -----------------------------------------------------


def test_a_crash_before_the_transaction_leaves_the_store_untouched(tmp_path):
    """Staged index written, nothing committed. The old index is still the
    live one, so recovery only has to sweep the orphan away."""
    index, database = _store(tmp_path)
    before = _mapping(database)
    staged = compaction.staged_path(tmp_path / "index")
    staged.write_bytes(b"partial garbage")

    report = recovery.recover(index, database, index_dir=tmp_path / "index")

    assert not staged.exists()
    assert _mapping(database) == before
    assert len(VectorIndex.load(tmp_path / "index")) == 6
    assert report is not None
    database.close()


def test_a_crash_after_the_transaction_finishes_the_swap(tmp_path):
    """Rows committed with new ids and the swap recorded, but the file was
    never moved into place. Recovery must complete it -- the database is
    already describing the compacted index."""
    index, database = _store(tmp_path)
    compaction.drop_video(
        "b", index, database, index_dir=tmp_path / "index", _stop_before_swap=True
    )
    # The live index file is still the old six-vector one...
    assert len(VectorIndex.load(tmp_path / "index")) == 6
    # ...while the database already says four.
    assert database.frame_count() == 4

    recovery.recover(index, database, index_dir=tmp_path / "index")

    assert len(VectorIndex.load(tmp_path / "index")) == 4
    database.check_consistency(4)
    database.close()


def test_a_crash_after_the_swap_just_clears_the_marker(tmp_path):
    """The replace happened but the marker was never cleared. Repair must be
    idempotent rather than trying to swap a file that is already gone."""
    index, database = _store(tmp_path)
    compaction.drop_video("b", index, database, index_dir=tmp_path / "index")
    database.set_meta(compaction.PENDING_SWAP_KEY, "1")  # simulate the crash

    recovery.recover(index, database, index_dir=tmp_path / "index")

    assert database.get_meta(compaction.PENDING_SWAP_KEY) is None
    assert len(VectorIndex.load(tmp_path / "index")) == 4
    database.check_consistency(4)
    database.close()


def test_repair_is_safe_to_run_twice(tmp_path):
    index, database = _store(tmp_path)
    compaction.drop_video(
        "b", index, database, index_dir=tmp_path / "index", _stop_before_swap=True
    )
    recovery.recover(index, database, index_dir=tmp_path / "index")
    recovery.recover(index, database, index_dir=tmp_path / "index")

    assert len(VectorIndex.load(tmp_path / "index")) == 4
    database.check_consistency(4)
    database.close()


# ---- the key-value store the marker lives in ---------------------------


def test_meta_round_trips(tmp_path):
    database = Database(tmp_path / "m.sqlite")
    assert database.get_meta("missing") is None
    database.set_meta("k", "v")
    assert database.get_meta("k") == "v"
    database.set_meta("k", "v2")
    assert database.get_meta("k") == "v2"
    database.delete_meta("k")
    assert database.get_meta("k") is None
    database.close()


def test_compaction_refuses_to_guess_where_the_index_lives(tmp_path, monkeypatch):
    """Regression, and a scar. `drop_video` used to default index_dir to
    config.INDEX_DIR; a test that omitted it compacted the real index. A
    destructive file operation must not reach a global path by default."""
    import inspect

    from sv_engine import compaction as module

    assert (
        inspect.signature(module.drop_video).parameters["index_dir"].default
        is inspect.Parameter.empty
    ), "index_dir must stay required"

"""VectorIndex tests use synthetic vectors rather than CLIP.

The index now stores vectors only; the mapping back to videos lives in SQLite.
What must hold here is that positions are assigned contiguously and that search
returns them in score order -- everything downstream joins on those positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_engine.index import VectorIndex


def _unit(*rows: list[float]) -> np.ndarray:
    arr = np.array(rows, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


@pytest.fixture
def populated() -> VectorIndex:
    index = VectorIndex(dim=3)
    index.add(_unit([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    return index


def test_add_assigns_contiguous_positions():
    index = VectorIndex(dim=3)
    assert index.add(_unit([1, 0, 0], [0, 1, 0])) == [0, 1]
    # A second batch must continue where the first stopped, never restart.
    assert index.add(_unit([0, 0, 1])) == [2]
    assert len(index) == 3


def test_add_empty_is_a_noop():
    index = VectorIndex(dim=3)
    assert index.add(np.empty((0, 3), dtype=np.float32)) == []
    assert len(index) == 0


def test_search_returns_the_matching_position(populated):
    hits = populated.search(_unit([0, 1, 0])[0], top_k=3)
    assert hits[0].vector_index_id == 1
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_ordered_descending(populated):
    hits = populated.search(_unit([0.9, 0.4, 0.1])[0], top_k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].vector_index_id == 0


def test_top_k_larger_than_corpus_is_clamped(populated):
    """FAISS pads short result sets with -1; those must not become hits."""
    hits = populated.search(_unit([1, 0, 0])[0], top_k=50)
    assert len(hits) == 3
    assert all(h.vector_index_id >= 0 for h in hits)


def test_top_k_zero_is_rejected(populated):
    with pytest.raises(ValueError):
        populated.search(_unit([1, 0, 0])[0], top_k=0)


def test_search_on_empty_index_returns_nothing():
    assert VectorIndex(dim=3).search(_unit([1, 0, 0])[0], top_k=5) == []


def test_wrong_dimension_is_rejected_on_add():
    with pytest.raises(ValueError, match="dim"):
        VectorIndex(dim=3).add(np.zeros((1, 5), dtype=np.float32))


def test_wrong_dimension_is_rejected_on_search(populated):
    with pytest.raises(ValueError, match="dim"):
        populated.search(np.zeros(5, dtype=np.float32), top_k=1)


def test_one_dimensional_input_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        VectorIndex(dim=3).add(np.zeros(3, dtype=np.float32))


def test_roundtrip_preserves_positions_and_scores(populated, tmp_path):
    populated.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)

    assert len(loaded) == len(populated)
    assert loaded.dim == populated.dim
    for query in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        before = populated.search(_unit(query)[0], top_k=1)[0]
        after = loaded.search(_unit(query)[0], top_k=1)[0]
        assert before.vector_index_id == after.vector_index_id
        assert before.score == pytest.approx(after.score, abs=1e-6)


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path)


def test_load_or_create_returns_empty_index_when_absent(tmp_path):
    assert len(VectorIndex.load_or_create(dim=3, directory=tmp_path)) == 0


def test_load_or_create_rejects_a_dimension_change(populated, tmp_path):
    """Swapping CLIP checkpoints changes the vector width; reusing the old
    index would compare incomparable spaces."""
    populated.save(tmp_path)
    with pytest.raises(ValueError, match="rebuild"):
        VectorIndex.load_or_create(dim=512, directory=tmp_path)


def test_concurrent_adds_assign_unique_positions():
    """The race the internal lock exists to prevent.

    ``add`` reads the current length, then appends. Without a lock two threads
    can read the same length and both claim the same position, producing
    duplicate vector ids -- the desync that makes search return the wrong
    timestamp with no error anywhere.
    """
    import threading

    index = VectorIndex(dim=3)
    assigned: list[int] = []
    guard = threading.Lock()

    def worker() -> None:
        for _ in range(20):
            ids = index.add(_unit([1.0, 0.0, 0.0]))
            with guard:
                assigned.extend(ids)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(assigned) == 160
    assert len(set(assigned)) == 160, "duplicate positions were handed out"
    assert sorted(assigned) == list(range(160))
    assert len(index) == 160


def test_search_during_concurrent_adds_does_not_error():
    """Searching while another thread writes must not crash or return junk."""
    import threading

    index = VectorIndex(dim=3)
    index.add(_unit([1.0, 0.0, 0.0]))
    errors: list[Exception] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            for _ in range(100):
                index.add(_unit([0.0, 1.0, 0.0]))
        except Exception as exc:  # noqa: BLE001 - recorded and re-raised below
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            while not stop.is_set():
                for hit in index.search(_unit([1.0, 0.0, 0.0])[0], top_k=3):
                    assert hit.vector_index_id >= 0
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent access raised: {errors}"
    assert len(index) == 101


# ---- truncation (M4 crash recovery) --------------------------------------


def test_truncate_drops_the_tail_and_keeps_the_rest(populated):
    """Recovery's only safe repair. Positions below the cut must not move --
    every vector_index_id in the database depends on that."""
    populated.truncate(2)

    assert len(populated) == 2
    hits = populated.search(_unit([0, 1, 0])[0], top_k=2)
    assert hits[0].vector_index_id == 1
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_truncate_to_zero_empties_the_index(populated):
    populated.truncate(0)
    assert len(populated) == 0
    assert populated.search(_unit([1, 0, 0])[0], top_k=1) == []


def test_truncate_to_current_size_is_a_noop(populated):
    populated.truncate(3)
    assert len(populated) == 3


def test_truncate_cannot_grow_the_index(populated):
    with pytest.raises(ValueError, match="grow"):
        populated.truncate(5)


def test_truncate_rejects_a_negative_size(populated):
    with pytest.raises(ValueError):
        populated.truncate(-1)


def test_new_positions_continue_after_a_truncation(populated):
    """After a repair the next ingest must not reuse a discarded position as
    if it were fresh -- it may, but only because nothing references it."""
    populated.truncate(1)
    assert populated.add(_unit([0, 1, 0])) == [1]
    assert len(populated) == 2


def test_save_leaves_no_partial_file_behind(populated, tmp_path):
    """The write is atomic: a crash mid-save must leave the previous index
    intact rather than a truncated file that will not load."""
    populated.save(tmp_path)
    assert list(tmp_path.glob("*.tmp*")) == []
    assert len(VectorIndex.load(tmp_path)) == 3


def test_appending_serializes_writers():
    """The append lock is what makes 'orphan vectors are always the tail' true.

    It is deliberately coarser than the FAISS lock -- a writer holds it across
    add, persist and commit -- but it blocks only other *writers*. Searches are
    never held up by it.
    """
    import threading

    index = VectorIndex(dim=3)
    order: list[str] = []
    first_inside = threading.Event()
    release = threading.Event()

    def slow_writer() -> None:
        with index.appending():
            order.append("a-in")
            first_inside.set()
            release.wait(timeout=2)
            order.append("a-out")

    def second_writer() -> None:
        first_inside.wait(timeout=2)
        release.set()
        with index.appending():
            order.append("b-in")

    threads = [threading.Thread(target=slow_writer), threading.Thread(target=second_writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert order == ["a-in", "a-out", "b-in"]


def test_search_is_not_blocked_by_an_open_append(populated):
    """CLAUDE.md's rule 2, restated as a test: holding the writer lock across a
    long ingest must not freeze search."""
    import threading

    done = threading.Event()

    def searcher() -> None:
        populated.search(_unit([1, 0, 0])[0], top_k=1)
        done.set()

    with populated.appending():
        thread = threading.Thread(target=searcher)
        thread.start()
        assert done.wait(timeout=2), "search blocked while a writer held the lock"
        thread.join()

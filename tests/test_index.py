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

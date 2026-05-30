"""Index tests use synthetic vectors rather than CLIP.

The mapping from a FAISS hit back to the right video and timestamp is the
sharpest failure mode in the system, and it is worth testing without paying for
a model load on every run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sv_engine.index import FrameIndex, FrameRecord


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array([vec], dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def _record(video_id: str, ts: float) -> FrameRecord:
    return FrameRecord(
        video_id=video_id,
        filename=f"{video_id}.mp4",
        timestamp_sec=ts,
        thumbnail_path=f"/thumbs/{video_id}_{ts}.jpg",
        reason="baseline",
    )


@pytest.fixture
def populated() -> FrameIndex:
    index = FrameIndex(dim=3)
    index.add(_unit([1, 0, 0]), [_record("vid_a", 1.0)])
    index.add(_unit([0, 1, 0]), [_record("vid_b", 2.0)])
    index.add(_unit([0, 0, 1]), [_record("vid_c", 3.0)])
    return index


def test_search_returns_the_matching_record(populated):
    hits = populated.search(_unit([0, 1, 0])[0], top_k=3)

    assert hits[0].record.video_id == "vid_b"
    assert hits[0].record.timestamp_sec == 2.0
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_ordered_descending(populated):
    hits = populated.search(_unit([0.9, 0.4, 0.1])[0], top_k=3)
    scores = [h.score for h in hits]

    assert scores == sorted(scores, reverse=True)
    assert hits[0].record.video_id == "vid_a"


def test_top_k_larger_than_corpus_is_clamped(populated):
    """FAISS pads short result sets with -1; those must not become records."""
    hits = populated.search(_unit([1, 0, 0])[0], top_k=50)
    assert len(hits) == 3


def test_top_k_zero_is_rejected(populated):
    with pytest.raises(ValueError):
        populated.search(_unit([1, 0, 0])[0], top_k=0)


def test_search_on_empty_index_returns_nothing():
    assert FrameIndex(dim=3).search(_unit([1, 0, 0])[0], top_k=5) == []


def test_vector_record_count_mismatch_is_rejected():
    index = FrameIndex(dim=3)
    with pytest.raises(ValueError, match="mismatch"):
        index.add(_unit([1, 0, 0]), [_record("a", 0.0), _record("b", 1.0)])


def test_wrong_dimension_is_rejected():
    index = FrameIndex(dim=3)
    with pytest.raises(ValueError, match="dim"):
        index.add(np.zeros((1, 5), dtype=np.float32), [_record("a", 0.0)])


def test_roundtrip_preserves_the_vector_to_record_mapping(populated, tmp_path):
    populated.save(tmp_path)
    loaded = FrameIndex.load(tmp_path)

    assert len(loaded) == len(populated)
    for query in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        before = populated.search(_unit(query)[0], top_k=1)[0]
        after = loaded.search(_unit(query)[0], top_k=1)[0]
        assert before.record == after.record


def test_load_detects_desynced_metadata(populated, tmp_path):
    """A metadata file with fewer rows than the index must fail loudly --
    silently returning the wrong timestamp is the outcome to avoid."""
    populated.save(tmp_path)
    meta = tmp_path / "frames.json"
    payload = json.loads(meta.read_text())
    payload["records"].pop()
    meta.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="out of sync"):
        FrameIndex.load(tmp_path)


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FrameIndex.load(tmp_path)


def test_load_or_create_returns_empty_index_when_absent(tmp_path):
    index = FrameIndex.load_or_create(dim=3, directory=tmp_path)
    assert len(index) == 0


def test_load_or_create_rejects_a_dimension_change(populated, tmp_path):
    """Swapping CLIP checkpoints changes the vector width; reusing the old
    index would compare incomparable spaces."""
    populated.save(tmp_path)
    with pytest.raises(ValueError, match="rebuild"):
        FrameIndex.load_or_create(dim=512, directory=tmp_path)


def test_video_ids_are_deduplicated():
    index = FrameIndex(dim=3)
    index.add(_unit([1, 0, 0]), [_record("vid_a", 1.0)])
    index.add(_unit([0, 1, 0]), [_record("vid_a", 2.0)])
    assert index.video_ids() == {"vid_a"}

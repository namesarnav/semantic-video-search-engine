"""Search tests with a stub embedder, so no CLIP checkpoint is loaded.

What matters here is the join: FAISS returns positions, and those must map back
to the right video and timestamp, in FAISS's ranking order.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_engine.db import Database
from sv_engine.index import VectorIndex
from sv_engine.search import SearchResult, collapse_near_duplicates, search


class StubEmbedder:
    """Maps a handful of known queries to fixed 3-D vectors."""

    dim = 3
    _VECTORS = {
        "bridge": [1.0, 0.0, 0.0],
        "mountain": [0.0, 1.0, 0.0],
        "sunset": [0.0, 0.0, 1.0],
    }

    def encode_text(self, queries):
        arr = np.array([self._VECTORS[q] for q in queries], dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


@pytest.fixture
def corpus(tmp_path):
    """Two videos, three frames each, with orthogonal vectors per subject."""
    database = Database(tmp_path / "test.sqlite")
    index = VectorIndex(dim=3)

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # bridge_a @ 1.0
            [0.9, 0.1, 0.0],  # bridge_a @ 2.0  (near-duplicate)
            [0.0, 1.0, 0.0],  # bridge_a @ 8.0
            [0.0, 0.9, 0.1],  # mountain_b @ 1.0
            [0.0, 0.0, 1.0],  # mountain_b @ 5.0
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = index.add(vectors)

    database.upsert_video("vid_a", "a.mp4", "/a.mp4", 10.0)
    database.upsert_video("vid_b", "b.mp4", "/b.mp4", 10.0)
    database.add_frames(
        "vid_a",
        [
            {"timestamp_sec": 1.0, "thumbnail_path": "/t/0.jpg", "reason": "baseline", "vector_index_id": ids[0]},
            {"timestamp_sec": 2.0, "thumbnail_path": "/t/1.jpg", "reason": "baseline", "vector_index_id": ids[1]},
            {"timestamp_sec": 8.0, "thumbnail_path": "/t/2.jpg", "reason": "scene_cut", "vector_index_id": ids[2]},
        ],
    )
    database.add_frames(
        "vid_b",
        [
            {"timestamp_sec": 1.0, "thumbnail_path": "/t/3.jpg", "reason": "baseline", "vector_index_id": ids[3]},
            {"timestamp_sec": 5.0, "thumbnail_path": "/t/4.jpg", "reason": "baseline", "vector_index_id": ids[4]},
        ],
    )
    yield index, database
    database.close()


def test_search_returns_the_right_video_and_timestamp(corpus):
    index, database = corpus
    results = search("bridge", index, database, StubEmbedder(), top_k=1)

    assert results[0].video_id == "vid_a"
    assert results[0].timestamp_sec == 1.0
    assert results[0].thumbnail_path == "/t/0.jpg"


def test_search_spans_multiple_videos(corpus):
    """FR4: one query searches every ingested video, not one at a time."""
    index, database = corpus
    results = search("sunset", index, database, StubEmbedder(), top_k=5)

    assert {r.video_id for r in results} == {"vid_a", "vid_b"}
    assert results[0].video_id == "vid_b"
    assert results[0].timestamp_sec == 5.0


def test_results_preserve_faiss_ranking(corpus):
    index, database = corpus
    results = search("bridge", index, database, StubEmbedder(), top_k=5)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reason_is_carried_through(corpus):
    index, database = corpus
    results = search("mountain", index, database, StubEmbedder(), top_k=1)
    assert results[0].reason == "scene_cut"


def test_top_k_limits_results(corpus):
    index, database = corpus
    assert len(search("bridge", index, database, StubEmbedder(), top_k=2)) == 2


def test_empty_query_is_rejected(corpus):
    index, database = corpus
    with pytest.raises(ValueError, match="empty"):
        search("   ", index, database, StubEmbedder(), top_k=5)


def test_zero_top_k_is_rejected(corpus):
    index, database = corpus
    with pytest.raises(ValueError, match="positive"):
        search("bridge", index, database, StubEmbedder(), top_k=0)


def test_search_on_empty_index_returns_nothing(tmp_path):
    database = Database(tmp_path / "empty.sqlite")
    results = search("bridge", VectorIndex(dim=3), database, StubEmbedder(), top_k=5)
    assert results == []
    database.close()


def test_missing_frame_row_is_surfaced_not_swallowed(corpus):
    """A vector with no row means the two stores diverged. Returning fewer
    results would hide it; the query must fail loudly instead."""
    index, database = corpus
    database.conn.execute("DELETE FROM frames WHERE vector_index_id = 0")
    database.conn.commit()

    with pytest.raises(ValueError, match="out of sync"):
        search("bridge", index, database, StubEmbedder(), top_k=5)


def test_collapse_merges_near_duplicates(corpus):
    """A long static shot should not fill the result list with the same moment."""
    index, database = corpus
    wide = search("bridge", index, database, StubEmbedder(), top_k=5)
    collapsed = search(
        "bridge", index, database, StubEmbedder(), top_k=5, collapse_window_sec=3.0
    )

    # 1.0s and 2.0s in vid_a are within the window; only the better survives.
    assert [(r.video_id, r.timestamp_sec) for r in wide][:2] == [
        ("vid_a", 1.0),
        ("vid_a", 2.0),
    ]
    assert ("vid_a", 2.0) not in [(r.video_id, r.timestamp_sec) for r in collapsed]
    assert ("vid_a", 1.0) in [(r.video_id, r.timestamp_sec) for r in collapsed]


def _result(video_id: str, ts: float, score: float) -> SearchResult:
    return SearchResult(
        score=score,
        video_id=video_id,
        filename=f"{video_id}.mp4",
        timestamp_sec=ts,
        thumbnail_path="/t.jpg",
        reason="baseline",
        frame_id=0,
    )


def test_collapse_keeps_the_highest_scoring_of_a_cluster():
    results = [_result("a", 5.0, 0.9), _result("a", 5.5, 0.8), _result("a", 9.0, 0.7)]
    kept = collapse_near_duplicates(results, window_sec=2.0)

    assert [r.timestamp_sec for r in kept] == [5.0, 9.0]


def test_collapse_does_not_merge_across_videos():
    """Same timestamp in two different videos is two distinct moments."""
    results = [_result("a", 5.0, 0.9), _result("b", 5.0, 0.8)]
    kept = collapse_near_duplicates(results, window_sec=10.0)

    assert len(kept) == 2

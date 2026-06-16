"""Query path: text -> CLIP text embedding -> FAISS -> join metadata.

The text encoder puts a query into the same space as the frame embeddings,
which is what makes comparing words to pictures meaningful at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from .db import Database, FrameRow
from .embedder import ClipEmbedder, get_embedder
from .index import VectorIndex


@dataclass(frozen=True)
class SearchResult:
    score: float
    video_id: str
    filename: str
    timestamp_sec: float
    thumbnail_path: str
    reason: str
    frame_id: int


def search(
    query: str,
    index: VectorIndex,
    database: Database,
    embedder: ClipEmbedder | None = None,
    top_k: int = 10,
    collapse_window_sec: float | None = None,
) -> list[SearchResult]:
    """Rank frames across every ingested video by similarity to ``query``.

    ``collapse_window_sec`` optionally merges hits from the same video that
    fall within that many seconds of each other, keeping the highest-scoring
    one. Without it a long static shot floods the results with near-identical
    entries. Off by default -- see §4.3; it is a product decision, not plumbing.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    embedder = embedder or get_embedder()
    vector = embedder.encode_text([query])[0]

    # Over-fetch when collapsing, since near-duplicates get discarded and we
    # still want top_k distinct moments back.
    fetch_k = top_k * 5 if collapse_window_sec else top_k
    hits = index.search(vector, top_k=fetch_k)
    if not hits:
        return []

    rows: dict[int, FrameRow] = database.frames_by_vector_ids(
        h.vector_index_id for h in hits
    )

    results: list[SearchResult] = []
    for hit in hits:  # FAISS order is the ranking; preserve it
        row = rows.get(hit.vector_index_id)
        if row is None:
            # A vector with no matching row means index and database have
            # diverged. Surface it rather than quietly returning fewer results.
            raise ValueError(
                f"vector {hit.vector_index_id} has no frame row; "
                "index and database are out of sync. Re-ingest with --rebuild."
            )
        results.append(
            SearchResult(
                score=hit.score,
                video_id=row.video_id,
                filename=row.filename,
                timestamp_sec=row.timestamp_sec,
                thumbnail_path=row.thumbnail_path,
                reason=row.reason,
                frame_id=row.id,
            )
        )

    if collapse_window_sec:
        results = collapse_near_duplicates(results, collapse_window_sec)

    return results[:top_k]


def collapse_near_duplicates(
    results: list[SearchResult], window_sec: float
) -> list[SearchResult]:
    """Drop lower-scoring hits within ``window_sec`` of a kept hit, per video.

    Input must already be sorted best-first, so the first hit seen in a
    neighbourhood is the one worth keeping.
    """
    kept: list[SearchResult] = []
    for candidate in results:
        if any(
            k.video_id == candidate.video_id
            and abs(k.timestamp_sec - candidate.timestamp_sec) < window_sec
            for k in kept
        ):
            continue
        kept.append(candidate)
    return kept

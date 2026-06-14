"""FAISS vector index.

This module stores vectors and nothing else. The mapping from a vector back to
a video and timestamp lives in SQLite (see db.py), joined on the vector's
position in this index.

A vector's position is assigned implicitly by FAISS: the first vector added is
0, the next is 1, and so on. Nothing here may ever reorder or remove a vector,
because every id after it would shift and every stored ``vector_index_id`` in
the database would then point at the wrong frame. Removal is done by rebuilding
both together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from . import config

_INDEX_FILENAME = "frames.faiss"


@dataclass(frozen=True)
class VectorHit:
    """A raw FAISS result: a position and its similarity score."""

    vector_index_id: int
    score: float


class VectorIndex:
    """A flat inner-product index over L2-normalized vectors.

    Flat means exact, exhaustive search. At this corpus size that is the right
    call -- an approximate index (IVF/HNSW) trades recall for speed the system
    does not yet need. Switch only when a measured p95 says so.

    Inner product equals cosine similarity *only because* vectors are
    normalized by the embedder before they arrive here.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray) -> list[int]:
        """Append vectors and return the positions they were assigned."""
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        if len(vectors) == 0:
            return []

        first = len(self)
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return list(range(first, first + len(vectors)))

    def next_vector_id(self) -> int:
        return len(self)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[VectorHit]:
        """Search with one normalized query vector, shaped (dim,) or (1, dim)."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(self) == 0:
            return []

        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if query.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {query.shape[1]}")

        scores, ids = self.index.search(query, min(top_k, len(self)))
        # FAISS pads with -1 when it has fewer results than requested.
        return [
            VectorHit(vector_index_id=int(i), score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]

    # ---- persistence --------------------------------------------------

    @staticmethod
    def _path(directory: Path | str) -> Path:
        return Path(directory) / _INDEX_FILENAME

    def save(self, directory: Path | str = config.INDEX_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._path(directory)))

    @classmethod
    def load(cls, directory: Path | str = config.INDEX_DIR) -> "VectorIndex":
        path = cls._path(directory)
        if not path.exists():
            raise FileNotFoundError(f"no index found at {path}")
        raw = faiss.read_index(str(path))
        instance = cls(dim=raw.d)
        instance.index = raw
        return instance

    @classmethod
    def load_or_create(
        cls, dim: int, directory: Path | str = config.INDEX_DIR
    ) -> "VectorIndex":
        try:
            existing = cls.load(directory)
        except FileNotFoundError:
            return cls(dim=dim)
        if existing.dim != dim:
            raise ValueError(
                f"existing index has dim {existing.dim}, embedder produces {dim}. "
                "The checkpoint changed -- rebuild the index."
            )
        return existing

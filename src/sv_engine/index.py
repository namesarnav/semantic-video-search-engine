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

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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
        # FAISS is not safe for a concurrent add and search. The lock lives
        # here rather than in callers so it cannot be forgotten, and it is held
        # only for the microseconds of the FAISS call itself -- never around
        # sampling or embedding, which would freeze search for the whole of a
        # 20-second ingest.
        self._lock = threading.Lock()
        # A second, deliberately coarser lock, held by a writer across its
        # whole add -> persist -> commit unit (see ``appending``). It blocks
        # other *writers* only; searches take ``_lock`` and are never held up
        # by it, so the rule above still holds.
        self._writer_lock = threading.RLock()

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

        with self._lock:
            first = len(self)
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            return list(range(first, first + len(vectors)))

    @contextmanager
    def appending(self) -> Iterator["VectorIndex"]:
        """Claim the index for one writer, for the whole add-and-commit unit.

        This is what makes crash recovery possible. Vectors are only ever
        appended, so an ingest that dies part-way leaves its vectors at the end
        of the index -- but only if no other ingest appended in the meantime.
        Serializing writers keeps "the orphans are the tail" true, and a tail
        is the one thing ``truncate`` can drop without shifting a single
        surviving ``vector_index_id``.

        Hold it around add/persist/commit only -- never around sampling or
        embedding, which would make one slow ingest block every other.
        """
        with self._writer_lock:
            yield self

    def truncate(self, size: int) -> None:
        """Drop every vector from ``size`` onwards.

        The only safe way to remove vectors from a flat index: positions below
        the cut keep their ids, so the ``vector_index_id`` stored on every
        surviving frame row still points at the same vector. Removing from the
        middle would shift ids and silently mis-answer every later query, which
        is why there is no such method.
        """
        if size < 0:
            raise ValueError(f"size must not be negative, got {size}")
        with self._lock:
            current = int(self.index.ntotal)
            if size > current:
                raise ValueError(
                    f"truncate cannot grow the index: {current} vectors, asked for {size}"
                )
            if size == current:
                return
            kept = (
                self.index.reconstruct_n(0, size)
                if size
                else np.empty((0, self.dim), dtype=np.float32)
            )
            fresh = faiss.IndexFlatIP(self.dim)
            if size:
                fresh.add(np.ascontiguousarray(kept, dtype=np.float32))
            self.index = fresh

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

        with self._lock:
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
        """Write the index out atomically.

        Staged under a temporary name and renamed into place, because
        ``os.replace`` is atomic within a filesystem. A crash mid-write then
        leaves the *previous* index intact rather than a truncated file that
        will not load -- recovery can repair a stale index, but not an
        unreadable one.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        final = self._path(directory)
        staging = final.with_name(f"{final.name}.{os.getpid()}.tmp")
        with self._lock:
            try:
                faiss.write_index(self.index, str(staging))
                os.replace(staging, final)
            finally:
                staging.unlink(missing_ok=True)

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

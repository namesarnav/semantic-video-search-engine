"""FAISS index plus its sidecar metadata.

FAISS stores vectors and nothing else. Everything needed to turn a hit back into
a human-meaningful result -- which video, which timestamp, which thumbnail --
lives alongside it, keyed by the vector's position in the index.

Keeping those two in sync is the sharpest failure mode in this system: a
mismatch does not crash, it silently returns the wrong timestamp. So the index
and its metadata are always written together, and loading validates that their
lengths agree.

M1 uses a JSON sidecar. M2 replaces it with SQLite; the FrameRecord shape is
already the target row.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from . import config

_INDEX_FILENAME = "frames.faiss"
_META_FILENAME = "frames.json"


@dataclass(frozen=True)
class FrameRecord:
    """Metadata for one indexed frame. Mirrors the future `frames` table."""

    video_id: str
    filename: str
    timestamp_sec: float
    thumbnail_path: str
    reason: str


@dataclass(frozen=True)
class SearchHit:
    score: float
    record: FrameRecord


class FrameIndex:
    """A flat inner-product FAISS index over L2-normalized vectors.

    Flat means exact search. At this corpus size that is the right call -- an
    approximate index (IVF/HNSW) trades recall for speed the system does not yet
    need. Switch only when a measured p95 says so.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.records: list[FrameRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, vectors: np.ndarray, records: list[FrameRecord]) -> None:
        if len(vectors) != len(records):
            raise ValueError(
                f"vector/record count mismatch: {len(vectors)} vs {len(records)}"
            )
        if len(vectors) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.records.extend(records)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[SearchHit]:
        """Search with a single (dim,) or (1, dim) normalized query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(self) == 0:
            return []

        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        scores, ids = self.index.search(query, min(top_k, len(self)))

        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], ids[0]):
            # FAISS pads with -1 when it has fewer results than requested.
            if idx < 0:
                continue
            hits.append(SearchHit(score=float(score), record=self.records[idx]))
        return hits

    def video_ids(self) -> set[str]:
        return {r.video_id for r in self.records}

    def save(self, directory: Path | str = config.INDEX_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / _INDEX_FILENAME))
        payload = {
            "dim": self.dim,
            "records": [asdict(r) for r in self.records],
        }
        (directory / _META_FILENAME).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, directory: Path | str = config.INDEX_DIR) -> "FrameIndex":
        directory = Path(directory)
        index_path = directory / _INDEX_FILENAME
        meta_path = directory / _META_FILENAME
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"no index found in {directory}")

        payload = json.loads(meta_path.read_text())
        instance = cls(dim=payload["dim"])
        instance.index = faiss.read_index(str(index_path))
        instance.records = [FrameRecord(**r) for r in payload["records"]]

        if instance.index.ntotal != len(instance.records):
            raise ValueError(
                "index/metadata are out of sync: "
                f"{instance.index.ntotal} vectors vs {len(instance.records)} records"
            )
        return instance

    @classmethod
    def load_or_create(cls, dim: int, directory: Path | str = config.INDEX_DIR) -> "FrameIndex":
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

# Semantic Video Search Engine

**Disclaimer**: AI is used only for debugging and writing comments. I (Arnav) am responsible for the core logic, architecture of the project, all the components and tech stack. No AI was used to make core decisions on this project. 

---

Point it at a folder of videos, describe a moment in plain language — *"a red car
driving at night"*, *"a person opens a laptop"* — and get back ranked timestamps
across every video you have ingested, each one playable at the moment it matched.

It searches **visual content**. It is deliberately not transcript search, not OCR,
and not caption search.

```bash
docker compose up --build          # API + UI on http://localhost:8000
```

## How it works

Two paths share one embedding space, and that shared space is the whole trick:

**Ingest** — video → scene-aware frame sampling → CLIP *image* encoder → vectors
into FAISS, one row per frame in SQLite, thumbnail on disk.

**Query** — text → CLIP *text* encoder → cosine similarity in FAISS → top-K vector
ids → joined back to SQLite for the video, timestamp and thumbnail.

FAISS holds vectors and nothing else; SQLite is the source of truth for everything
else. They are joined on `frames.vector_index_id`, the frame's *position* in the
index. Any code that rebuilds or mutates the index has to keep that mapping
consistent or results point at the wrong video — the single sharpest failure mode
in the system, and the reason index rebuilds and row updates are always one
atomic unit of work.

| Component | Choice | Why |
|---|---|---|
| Embedding | CLIP ViT-B/32 (`open_clip`) | joint image/text space |
| Vector index | FAISS, flat | exact search; measured as fast enough |
| Metadata | SQLite | zero-ops, genuinely fine at this scale |
| API | FastAPI | background ingestion, status tracking |
| Frontend | React + Tailwind | search, library, results as videos |
| Packaging | Docker, single container | one command to run |

## Results

Quality is measured, not eyeballed. A hand-labelled eval set scores Recall@K, and
every design decision below was settled by A/B against it rather than by argument.

Corpus: **205 videos, 4415 frames, 70 minutes.**

| eval set | R@1 | R@5 | R@10 |
|---|---|---|---|
| hand-labelled (12 queries) | 66.7% | 75.0% | 83.3% |
| corpus (187, weak supervision) | 38.0% | 52.4% | 58.8% |

Search latency **p50 11ms / p95 12ms**, against a 500ms p95 target — so the flat
index has ~40× headroom and IVF/HNSW stays unjustified.

The two sets measure different things and the second is not a quality claim; see
[`eval/README.md`](eval/README.md).

### What the measurements settled

- **Scene-aware sampling beats fixed-interval**, +14.3 points at R@5. It could not
  be measured on ordinary footage — four of five original videos were single
  continuous shots, so both arms tied. A cut-dense corpus was built specifically
  to make the question answerable. *A corpus that cannot exhibit the phenomenon
  cannot measure it.*
- **Near-duplicate collapsing helps; the safe window is 3–5s.** Beyond ~10s it
  starts deleting the correct moment inside the right video.
- **ViT-L/14 beats ViT-B/32 by ~7 points at R@5** — 26 queries gained, 13 lost.
  Not adopted as the default: the vector dimension changes with the checkpoint,
  so switching invalidates every existing index.
- **Latency is not a problem**, so no approximate index.

## Commands

Python is managed by [uv](https://docs.astral.sh/uv/), pinned to 3.12.

```bash
uv sync
uv run python scripts/fix_openmp.py    # macOS: faiss and torch both ship libomp

uv run python -m sv_engine.cli index data/videos       # ingest a file or folder
uv run python -m sv_engine.cli search "a red car" -k 10
uv run python -m sv_engine.cli videos                  # status per video
uv run python -m sv_engine.cli eval                    # Recall@K
uv run python -m sv_engine.cli serve --port 8000       # API + UI, docs at /docs
```

```bash
uv run pytest -m "not slow"       # ~220 tests, ~2s, no CLIP loaded
npm --prefix web test             # UI tests
```

The UI needs building once before `serve` will serve it:

```bash
npm --prefix web install && npm --prefix web run build
```

## Design notes worth knowing

**Videos are keyed by content hash, not filename.** Re-ingesting the same file is
a no-op or a clean overwrite, never a duplicate set of frames.

**Status is persisted, not in-memory.** `kill -9` runs no `except` block, so what
survives on disk has to be repairable from disk alone. The ingest write path is
ordered so a crash always leaves the same repairable shape — surplus vectors at
the tail of the index — which recovery truncates without shifting a single
surviving id.

**Removing a video compacts the index.** A flat index cannot delete a vector
without shifting every id after it, so the index is rebuilt and the mapping
rewritten together. That has no safe write order, so the operation records its
intent in the database before swapping the file, and recovery finishes or
discards it at startup.

**Every handler that can reach CLIP is a plain `def`, never `async def`.** FastAPI
runs `async def` on the event loop and plain `def` in a worker thread; CLIP
inference never yields, so `async def` would stall the whole server for the length
of an ingest.

**The client holds no API base URL.** Search results return relative URLs, the dev
server proxies them, and in production FastAPI serves the built files itself — so
there is no setting that can be wrong in two environments.

## Layout

```
src/sv_engine/     sampler → embedder → index, glued by ingest, queried via search
                   db (metadata), recovery (repair), compaction (safe removal)
                   cli.py and api.py are two front ends over the same core
web/               React client; knows nothing but the HTTP API
eval/              hand-labelled ground truth + methodology
scripts/           corpus builders and A/B runners
```

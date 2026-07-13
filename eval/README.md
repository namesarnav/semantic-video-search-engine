# Eval set

Hand-labelled ground truth for `sv-engine eval`. This is the project's primary
"does it work" signal — the one number here that is measured rather than
eyeballed. `labels.json` is **source and is committed**; a metric whose labels
are not version-controlled cannot be re-derived, so it is not a metric.

```bash
uv run python -m sv_engine.cli eval                      # score the current store
uv run python -m sv_engine.cli eval --json out.json      # ...and save the report
uv run python -m sv_engine.cli eval -k 1 3 5 --tolerance 0.5
```

## What Recall@K means here

Recall over **queries**, not over relevant items. Each label names the moments
that correctly answer it; a query counts as found if any of them lands in the
top K, and Recall@K is the fraction of queries that did. Finding two acceptable
moments is worth no more than finding one — the question is "did the person get
their answer", which is what known-item search actually is.

## Writing a label

```json
{
  "query": "the sun setting behind a dark mountain",
  "note": "why this range and not the whole clip",
  "targets": [
    {"video": "16504192_3840_2160_30fps.mp4", "start_sec": 0.0, "end_sec": 5.0},
    {"video": "multishot_4cuts_720p.mp4",     "start_sec": 20.7, "end_sec": 25.7}
  ]
}
```

- **`video` is a filename**, not a `videos.id`. The id is a content hash: it
  changes if the file is re-encoded, and nobody can write one from memory.
- **`targets` is a list because a moment can genuinely recur.**
  `multishot_4cuts_720p.mp4` is a concatenation of the other four clips, so
  most footage in this corpus has two correct answers. Naming only one would
  score a perfect retrieval as a miss and cap recall for reasons that have
  nothing to do with retrieval.
- **Prefer ranges narrower than the whole clip** where the footage supports it.
  A whole-clip target only tests "did it pick the right video"; a narrow one
  tests the timestamp too, which is the harder and more useful claim.
- **Keep labels that fail.** "a nearly black frame after dark" and "a winding
  road curving to the left" are both expected to be hard. Deleting labels
  because they score badly is how an eval set stops measuring anything.

Loading is deliberately strict — unknown keys, backwards ranges, and empty
query sets are all errors. Every one of those would otherwise surface as a
*lower score*, indistinguishable from the retriever getting worse.

A label naming a video that is not searchable in the current corpus is an
error, not a miss, for the same reason.

## Tolerance

Each labelled range is widened by `--tolerance` seconds on both sides (default
1.0). Sampling is ~1 frame/sec, so the nearest sampled frame can sit up to a
baseline interval from the moment a human read off the clock. Scoring strictly
would charge the retriever for the sampler's grid. Lower it only alongside a
denser `--baseline-fps`.

## The sampling A/B

`eval` reads the store as it stands and never re-ingests, so the A/B is two
runs over two builds of the same corpus:

```bash
uv run python -m sv_engine.cli index data/videos --rebuild
uv run python -m sv_engine.cli eval --json eval/reports/scene-aware.json

uv run python -m sv_engine.cli index data/videos --rebuild --fixed-interval
uv run python -m sv_engine.cli eval --json eval/reports/fixed-interval.json
```

Every report records which arm produced it (`sampling.arm`, inferred from
whether any frame has `reason = scene_cut`), so the two files cannot be
silently mixed up. `--rebuild` is required: it drops index, database and
thumbnails together, and a store half-built by each arm measures neither.

Reports are outputs, not source, and are gitignored.

## Caveat on the current corpus

Five videos, 12 queries. That is enough to catch a broken retriever and not
nearly enough to resolve a few points of difference between two sampling
strategies — one query is 8.3 percentage points. Treat small gaps as noise
until there is more footage in `data/videos`.

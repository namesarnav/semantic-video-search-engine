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

## The sampling A/B — settled

**Scene-aware sampling wins: Recall@5 100% vs 85.7% for fixed-interval, +14.3
points on the queries CLIP can answer at all.**

This could not be measured on the real corpus, and that is worth understanding
before trusting the number. Scene-aware sampling contributes exactly 3 frames
out of 87 there: only `multishot_4cuts_720p.mp4` has any detected cuts, and the
other four videos are single continuous shots. Both arms score identically, and
that tie says nothing about the design.

So the A/B runs on a purpose-built corpus:

```bash
uv run python scripts/make_cut_dense_corpus.py   # once; regenerates the video
uv run python scripts/run_sampling_ab.py
```

`cut_dense.mp4` is 16 shots cut together — 8 sub-second, 8 sustained — built
from the four source clips crossed with four visual treatments so every shot is
uniquely addressable by one query. Boundaries are known by construction, which
is why it scores at **tolerance 0**: the ranges are exact and abut, so the
default 1s tolerance would let a frame from the neighbouring shot count as a
hit. The runner ingests into scratch stores under `data/ab/`, so **the main
store is never touched**.

Result:

| arm | frames | shot coverage | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| scene-aware | 43 | 100.0% | 81.2% | 93.8% | 100.0% |
| fixed-interval | 28 | 81.2% | 68.8% | 81.2% | 81.2% |
| dense (control) | 277 | 100.0% | 68.8% | 87.5% | 87.5% |

Two queries changed outcome, both sub-second shots (0.63s and 0.57s), found
only by scene-aware. Fixed-interval put no frame inside them at all.

**Why coverage is reported alongside recall.** Recall alone cannot separate
"scene-aware captured shots baseline missed and retrieval improved" from
"captured them and retrieval did not improve anyway" — the second is a finding
about CLIP, not about sampling. Here the two agree: coverage rises 81.2% →
100%, and exactly the shots that gained coverage are the ones that became
findable. That agreement is the actual evidence.

**Why there is a dense control arm.** Some queries fail because CLIP cannot see
"sepia", not because of sampling. Filtering those out using either test arm's
own successes would bias the comparison toward it, so the filter comes from a
10 fps arm that captures every shot and is neutral between the two. 14 of 16
queries are answerable by that standard; both filtered and unfiltered numbers
are printed.

Reports are outputs, not source, and are gitignored. The video is regenerated
by the script; the labels and shot manifest are committed.

## Caveats

**The real corpus is small.** Five videos, 12 queries; one query is 8.3
percentage points. Enough to catch a broken retriever, not enough to resolve a
few points of difference.

**The cut-dense corpus is synthetic.** It tests one specific design claim —
that sparse sampling misses short distinct moments — directly and honestly, and
is not a substitute for real-world recall. `labels.json` remains the headline
number. 16 queries means one query is 6.25 points, so only a large,
one-directional difference is signal; the +14.3 point gap here is two queries
wide and is corroborated by the independent coverage measure, which is why it
is reported as a result rather than as noise.

**The four treatments are a device**, not a claim that people search by colour
grade. They exist so each shot has exactly one correct answer.

**A drift bug worth remembering.** The first version of the builder rounded
shot durations to 1/100s while the renderer wrote whole frames, so the declared
boundaries drifted from the rendered ones and the ground truth pointed at the
neighbouring shot's content. It produced a coherent-looking but entirely false
result: 100% coverage with *worse* recall. Boundaries are now derived from
cumulative frame counts, and `test_cut_dense_corpus.py` pins it. Ground truth
that is generated still has to be verified against the artefact it describes.

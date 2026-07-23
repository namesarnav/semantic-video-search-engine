"""Recall@K over a hand-labelled eval set -- the project's "does it work" signal.

Every other quality claim on this project is eyeballed. This is the one that
is not, and it is what settles the open decisions in CLAUDE.md: scene-aware
sampling vs `--fixed-interval`, ViT-B/32 vs ViT-L/14, whether collapsing helps
or hides. Run it, record the number, then argue.

**Recall@K here is over queries, not over relevant items.** Each label names
the moment(s) that answer it, so a query either finds one inside the top K or
it does not, and Recall@K is the fraction of queries that did. Finding two
acceptable moments is not worth more than finding one; what is being measured
is "did the person get their answer". That is known-item retrieval,
which is the shape of this product: a person hunting one moment they remember.
It is deliberately *not* the mean-average-precision framing -- there is no
notion of "all relevant frames" to be complete against, and inventing one
would mean labelling every frame of every video.

The module is `evaluation` rather than `eval` only to keep `eval` unshadowed;
the CLI command is `sv-engine eval`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .db import Database
from .embedder import ClipEmbedder
from .index import VectorIndex
from .search import SearchResult, search

# Reported at every K the targets care about: @1 is "did it nail it", @5 is
# the stated correctness bar, @10 is the headroom.
DEFAULT_KS = (1, 5, 10)

# A labelled moment is a human's reading of the clock; the nearest *sampled*
# frame can sit up to one baseline interval away from it. Scoring strictly
# would charge the retriever for the sampler's grid, so the labelled range is
# widened by this much on both sides. One second = one baseline interval at
# the default 1 fps. Lower it only alongside a denser `--baseline-fps`.
DEFAULT_TOLERANCE_SEC = 1.0

_QUERY_FIELDS = ("query", "targets")
_QUERY_OPTIONAL = ("note",)
_TARGET_FIELDS = ("video", "start_sec", "end_sec")


class LabelError(Exception):
    """A malformed eval set, or one that does not match the corpus.

    Separate from ValueError because these are all "fix your labels file"
    problems, and the CLI reports them as such rather than as a crash.
    """


@dataclass(frozen=True)
class Target:
    """One acceptable answer: a moment in a particular video.

    ``video`` is a *filename*, not a `videos.id`. The id is a content hash --
    it changes if the file is re-encoded, and no human can write one from
    memory. The filename is what the labeller can actually see.
    """

    video: str
    start_sec: float
    end_sec: float

    def matches(self, result: SearchResult, tolerance_sec: float) -> bool:
        return (
            result.filename == self.video
            and self.start_sec - tolerance_sec
            <= result.timestamp_sec
            <= self.end_sec + tolerance_sec
        )


@dataclass(frozen=True)
class Label:
    """One hand-labelled query and every moment that correctly answers it.

    Targets are a *list* because the same footage legitimately appears in more
    than one video -- this corpus has a compilation clip containing all four
    of the standalone ones. Forcing a single target there would score a
    perfect retrieval as a miss and cap recall for reasons that have nothing
    to do with the retriever. A query is satisfied by whichever target the
    engine found highest, not by the one listed first.
    """

    query: str
    targets: tuple[Target, ...]
    note: str = ""

    @property
    def videos(self) -> set[str]:
        return {t.video for t in self.targets}

    def matches(self, result: SearchResult, tolerance_sec: float) -> bool:
        return any(t.matches(result, tolerance_sec) for t in self.targets)


@dataclass(frozen=True)
class QueryOutcome:
    label: Label
    rank: int | None  # 1-based rank of the first correct hit; None if absent
    elapsed_ms: float
    top_hit: tuple[str, float] | None  # (filename, timestamp) actually returned

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass(frozen=True)
class EvalReport:
    outcomes: list[QueryOutcome]
    ks: tuple[int, ...]
    tolerance_sec: float
    top_k: int
    collapse_window_sec: float | None
    corpus_videos: int
    corpus_frames: int
    # Frames per sampling reason, so the report records which arm built the
    # store it measured. No scene_cut frames means --fixed-interval.
    sampling: dict = field(default_factory=dict)
    label_path: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def sampling_arm(self) -> str:
        return "scene-aware" if self.sampling.get("scene_cut") else "fixed-interval"

    def recall_at(self, k: int) -> float:
        """Fraction of queries whose labelled moment landed in the top ``k``."""
        if not self.outcomes:
            return 0.0
        found = sum(1 for o in self.outcomes if o.rank is not None and o.rank <= k)
        return found / len(self.outcomes)

    @property
    def misses(self) -> list[QueryOutcome]:
        """Queries that never found their moment at any depth searched.

        The most useful part of a run: these are the labels to look at.
        """
        return [o for o in self.outcomes if o.rank is None]

    @property
    def latency_p50(self) -> float:
        return _percentile([o.elapsed_ms for o in self.outcomes], 50)

    @property
    def latency_p95(self) -> float:
        return _percentile([o.elapsed_ms for o in self.outcomes], 95)

    def to_dict(self) -> dict:
        return {
            "queries": len(self.outcomes),
            "recall": {str(k): self.recall_at(k) for k in self.ks},
            "tolerance_sec": self.tolerance_sec,
            "top_k": self.top_k,
            "collapse_window_sec": self.collapse_window_sec,
            "corpus": {"videos": self.corpus_videos, "frames": self.corpus_frames},
            "sampling": {"arm": self.sampling_arm, "frames_by_reason": self.sampling},
            "latency_ms": {"p50": self.latency_p50, "p95": self.latency_p95},
            "label_path": self.label_path,
            **self.meta,
            "outcomes": [
                {
                    "query": o.label.query,
                    "targets": [
                        {
                            "video": t.video,
                            "start_sec": t.start_sec,
                            "end_sec": t.end_sec,
                        }
                        for t in o.label.targets
                    ],
                    "rank": o.rank,
                    "elapsed_ms": round(o.elapsed_ms, 2),
                    "top_hit": (
                        {"video": o.top_hit[0], "timestamp_sec": o.top_hit[1]}
                        if o.top_hit
                        else None
                    ),
                }
                for o in self.outcomes
            ],
        }

    def describe(self) -> str:
        lines = [
            f"{len(self.outcomes)} queries over "
            f"{self.corpus_frames} frames / {self.corpus_videos} videos "
            f"[{self.sampling_arm}] "
            f"(tolerance ±{self.tolerance_sec:g}s, top_k={self.top_k}"
            + (
                f", collapse={self.collapse_window_sec:g}s)"
                if self.collapse_window_sec
                else ")"
            ),
            "",
        ]
        for k in self.ks:
            hits = sum(1 for o in self.outcomes if o.rank is not None and o.rank <= k)
            lines.append(
                f"  Recall@{k:<3} {self.recall_at(k):6.1%}   ({hits}/{len(self.outcomes)})"
            )
        lines += [
            "",
            f"  latency      p50 {self.latency_p50:.0f}ms   p95 {self.latency_p95:.0f}ms",
        ]
        if self.misses:
            lines += ["", f"missed ({len(self.misses)}):"]
            for o in self.misses:
                got = (
                    f"got {o.top_hit[0]} @ {o.top_hit[1]:.1f}s"
                    if o.top_hit
                    else "no results"
                )
                first = o.label.targets[0]
                extra = (
                    f" (+{len(o.label.targets) - 1} more)"
                    if len(o.label.targets) > 1
                    else ""
                )
                lines.append(
                    f"  \"{o.label.query}\"  "
                    f"want {first.video} @ {first.start_sec:g}-{first.end_sec:g}s{extra}  "
                    f"({got})"
                )
        return "\n".join(lines)


def load_labels(path: Path | str) -> list[Label]:
    """Parse and validate an eval set.

    Validation is strict on purpose. Every mistake a labels file can contain --
    a typo'd key, a backwards range, an empty set -- otherwise shows up as a
    *lower recall score*, which is indistinguishable from the retriever getting
    worse. A metric you cannot trust to be wrong for the right reason is not a
    metric.
    """
    path = Path(path)
    if not path.is_file():
        raise LabelError(f"eval set not found: {path}")

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise LabelError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or "queries" not in payload:
        raise LabelError(f"{path} must be an object with a 'queries' list")

    raw = payload["queries"]
    if not isinstance(raw, list) or not raw:
        raise LabelError(f"{path} contains no queries; nothing to measure")

    labels: list[Label] = []
    for position, entry in enumerate(raw, start=1):
        where = f"query {position}"
        if not isinstance(entry, dict):
            raise LabelError(f"{where}: expected an object, got {type(entry).__name__}")

        _reject_unknown(entry, _QUERY_FIELDS + _QUERY_OPTIONAL, where)
        missing = [f for f in _QUERY_FIELDS if f not in entry]
        if missing:
            raise LabelError(f"{where}: missing required field(s) {missing}")

        query = str(entry["query"]).strip()
        if not query:
            raise LabelError(f"{where}: query is empty")

        raw_targets = entry["targets"]
        if not isinstance(raw_targets, list) or not raw_targets:
            raise LabelError(f"{where}: targets must be a non-empty list")

        targets = tuple(
            _parse_target(t, f"{where} target {i}")
            for i, t in enumerate(raw_targets, start=1)
        )
        labels.append(
            Label(query=query, targets=targets, note=str(entry.get("note", "")))
        )
    return labels


def _reject_unknown(entry: dict, allowed: Sequence[str], where: str) -> None:
    """Catch `start` for `start_sec` and friends.

    An ignored typo shows up as a *lower recall score*, which is
    indistinguishable from the retriever getting worse.
    """
    unknown = set(entry) - set(allowed)
    if unknown:
        raise LabelError(
            f"{where}: unknown field(s) {sorted(unknown)}; expected {list(allowed)}"
        )


def _parse_target(entry: object, where: str) -> Target:
    if not isinstance(entry, dict):
        raise LabelError(f"{where}: expected an object, got {type(entry).__name__}")
    _reject_unknown(entry, _TARGET_FIELDS, where)
    missing = [f for f in _TARGET_FIELDS if f not in entry]
    if missing:
        raise LabelError(f"{where}: missing required field(s) {missing}")

    video = str(entry["video"]).strip()
    if not video:
        raise LabelError(f"{where}: video is empty")
    try:
        start = float(entry["start_sec"])
        end = float(entry["end_sec"])
    except (TypeError, ValueError) as exc:
        raise LabelError(f"{where}: start_sec/end_sec must be numbers") from exc
    if start < 0:
        raise LabelError(f"{where}: start_sec must not be negative")
    if end < start:
        raise LabelError(f"{where}: end_sec ({end:g}) is before start_sec ({start:g})")
    return Target(video=video, start_sec=start, end_sec=end)


def first_correct_rank(
    label: Label, results: Sequence[SearchResult], tolerance_sec: float
) -> int | None:
    """1-based rank of the first result inside the labelled moment, else None."""
    for rank, result in enumerate(results, start=1):
        if label.matches(result, tolerance_sec):
            return rank
    return None


def evaluate(
    labels: Sequence[Label],
    index: VectorIndex,
    database: Database,
    embedder: ClipEmbedder | None = None,
    top_k: int | None = None,
    ks: Iterable[int] = DEFAULT_KS,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    collapse_window_sec: float | None = None,
    label_path: Path | str | None = None,
    meta: dict | None = None,
) -> EvalReport:
    """Run every label against the live index and score the results.

    ``top_k`` defaults to the largest K being reported -- fetching deeper than
    that would change nothing and only cost latency.
    """
    ks = tuple(sorted(set(ks)))
    if not ks or min(ks) < 1:
        raise ValueError("ks must be positive integers")
    if top_k is None:
        top_k = max(ks)
    if top_k < max(ks):
        raise ValueError(
            f"top_k ({top_k}) is smaller than the largest K reported ({max(ks)}); "
            "Recall@K past the fetch depth would always read as a miss"
        )
    if not labels:
        raise LabelError("no labels to evaluate")

    _check_labels_against_corpus(labels, database)

    outcomes: list[QueryOutcome] = []
    for label in labels:
        started = time.perf_counter()
        results = search(
            label.query,
            index,
            database,
            embedder,
            top_k=top_k,
            collapse_window_sec=collapse_window_sec,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        outcomes.append(
            QueryOutcome(
                label=label,
                rank=first_correct_rank(label, results, tolerance_sec),
                elapsed_ms=elapsed_ms,
                top_hit=(
                    (results[0].filename, results[0].timestamp_sec) if results else None
                ),
            )
        )

    return EvalReport(
        outcomes=outcomes,
        ks=ks,
        tolerance_sec=tolerance_sec,
        top_k=top_k,
        collapse_window_sec=collapse_window_sec,
        corpus_videos=len(database.searchable_filenames()),
        corpus_frames=len(index),
        sampling=database.frame_reason_counts(),
        label_path=str(label_path) if label_path else None,
        meta=meta or {},
    )


def _check_labels_against_corpus(labels: Sequence[Label], database: Database) -> None:
    """Refuse to score labels that name a video the corpus cannot return.

    A typo'd or un-ingested filename scores exactly zero, which is
    indistinguishable from a retriever that has stopped working. This is the
    single most likely way to be quietly lied to by your own eval set.
    """
    available = database.searchable_filenames()
    named = {video for label in labels for video in label.videos}
    unknown = sorted(named - available)
    if unknown:
        raise LabelError(
            f"label(s) name video(s) not searchable in this corpus: {unknown}. "
            "Ingest them, or fix the filename -- scoring them would report a "
            "retrieval failure that is really a labelling one. "
            f"Available: {sorted(available) or '(nothing ingested)'}"
        )


@dataclass(frozen=True)
class CoverageReport:
    """Which ground-truth shots the sampler actually put a frame inside."""

    covered: tuple[int, ...]
    missed: tuple[int, ...]
    total: int

    @property
    def fraction(self) -> float:
        return len(self.covered) / self.total if self.total else 0.0

    def describe(self) -> str:
        return (
            f"{len(self.covered)}/{self.total} shots covered "
            f"({self.fraction:.1%})"
            + (f", missed {list(self.missed)}" if self.missed else "")
        )


def shot_coverage(
    sampled_timestamps: Iterable[float], shots: Sequence[tuple[float, float]]
) -> CoverageReport:
    """Fraction of shots containing at least one sampled frame.

    This measures the *sampler*, not the retriever -- no CLIP, no index, no
    queries. It exists because Recall@K alone cannot distinguish two very
    different A/B outcomes: "scene-aware captured shots that baseline missed,
    and retrieval improved" from "scene-aware captured them and retrieval did
    not improve anyway". The second is a finding about CLIP rather than about
    sampling, and only coverage separates them.

    Shots are half-open ``[start, end)``. They abut, so counting a sample that
    lands exactly on a boundary for both neighbours would inflate coverage at
    every cut -- which is precisely where the interesting samples are.
    """
    timestamps = sorted(sampled_timestamps)
    covered = tuple(
        i
        for i, (start, end) in enumerate(shots)
        if any(start <= t < end for t in timestamps)
    )
    missed = tuple(i for i in range(len(shots)) if i not in set(covered))
    return CoverageReport(covered=covered, missed=missed, total=len(shots))


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile.

    No interpolation: on an eval set of a few dozen queries, interpolating
    between two samples invents precision the sample size does not support.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(pct / 100 * len(ordered) + 0.5))))
    return ordered[rank - 1]


@dataclass(frozen=True)
class Comparison:
    """A paired comparison of two eval reports over the same queries.

    Deltas alone hide churn: a +2 point result made of nine wins and seven
    losses says something very different about a change from one made of two
    wins and no losses. Both are kept.
    """

    ks: tuple[int, ...]
    ranks: dict[str, dict[str, int | None]]  # side -> query -> rank
    labels: tuple[str, str] = ("baseline", "candidate")

    def _hits(self, side: str, k: int) -> set[str]:
        return {
            q for q, rank in self.ranks[side].items() if rank is not None and rank <= k
        }

    def recall(self, side: str, k: int) -> float:
        total = len(self.ranks[side])
        return len(self._hits(side, k)) / total if total else 0.0

    def delta(self, k: int) -> float:
        return self.recall("candidate", k) - self.recall("baseline", k)

    def gained(self, k: int) -> list[str]:
        """Queries the candidate finds within k and the baseline does not."""
        return sorted(self._hits("candidate", k) - self._hits("baseline", k))

    def lost(self, k: int) -> list[str]:
        return sorted(self._hits("baseline", k) - self._hits("candidate", k))

    def describe(self, show: int = 6) -> str:
        base, cand = self.labels
        width = max(len(base), len(cand), 9)
        lines = [
            f"{'':<{width}}  " + "".join(f"{'R@' + str(k):>9}" for k in self.ks),
            f"{base:<{width}}  "
            + "".join(f"{self.recall('baseline', k):>9.1%}" for k in self.ks),
            f"{cand:<{width}}  "
            + "".join(f"{self.recall('candidate', k):>9.1%}" for k in self.ks),
            f"{'delta':<{width}}  "
            + "".join(f"{self.delta(k):>+9.1%}" for k in self.ks),
        ]
        for k in self.ks:
            gained, lost = self.gained(k), self.lost(k)
            if not gained and not lost:
                continue
            lines += ["", f"R@{k}: +{len(gained)} gained, -{len(lost)} lost"]
            for query in gained[:show]:
                lines.append(f'  + "{query}"')
            for query in lost[:show]:
                lines.append(f'  - "{query}"')
            if len(gained) > show or len(lost) > show:
                lines.append(f"  ... ({len(gained) + len(lost)} total)")
        return "\n".join(lines)


def compare_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    ks: Iterable[int] = DEFAULT_KS,
    labels: tuple[str, str] = ("baseline", "candidate"),
) -> Comparison:
    """Pair two `EvalReport.to_dict()` payloads by query text.

    Refuses mismatched query sets. Intersecting them would quietly report a
    delta over whichever queries happened to overlap, which is not the
    comparison anyone intended to run.
    """
    sides = {}
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        sides[name] = {o["query"]: o.get("rank") for o in payload.get("outcomes", [])}

    if set(sides["baseline"]) != set(sides["candidate"]):
        only_a = sorted(set(sides["baseline"]) - set(sides["candidate"]))[:3]
        only_b = sorted(set(sides["candidate"]) - set(sides["baseline"]))[:3]
        raise ValueError(
            "the two reports cover different queries; they were scored against "
            f"different label sets. Only in {labels[0]}: {only_a}. "
            f"Only in {labels[1]}: {only_b}."
        )

    return Comparison(ks=tuple(sorted(set(ks))), ranks=sides, labels=labels)

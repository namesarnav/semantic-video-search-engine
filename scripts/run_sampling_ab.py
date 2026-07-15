#!/usr/bin/env python
"""Run the sampling A/B: scene-aware vs fixed-interval, on the cut-dense corpus.

    uv run python scripts/make_cut_dense_corpus.py    # once
    uv run python scripts/run_sampling_ab.py

Each arm ingests into its own scratch store under data/ab/, so **the main
store is never touched** -- no --rebuild, nothing to restore afterwards.

Two things are reported per arm, and both are needed:

* **shot coverage** -- did the sampler put a frame inside each shot at all.
  Pure sampler behaviour, no CLIP involved.
* **Recall@K** -- did that translate into the moment being findable.

Recall alone cannot separate "scene-aware captured shots baseline missed and
retrieval improved" from "captured them and retrieval did not improve anyway".
The second is a finding about CLIP, not about sampling.

Scored at **tolerance 0**: the shot boundaries are exact by construction and
abut, so the default 1s tolerance would widen a 0.7s shot's window to 2.7s and
let a frame from the neighbouring shot count as a hit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sv_engine import config  # noqa: E402
from sv_engine.db import Database  # noqa: E402
from sv_engine.embedder import get_embedder  # noqa: E402
from sv_engine.evaluation import evaluate, load_labels, shot_coverage  # noqa: E402
from sv_engine.index import VectorIndex  # noqa: E402
from sv_engine.ingest import ingest_video  # noqa: E402
from sv_engine.sampler import sample_video  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VIDEO = REPO / "data" / "eval-cutdense" / "cut_dense.mp4"
LABELS = REPO / "eval" / "labels-cutdense.json"
MANIFEST = REPO / "eval" / "cutdense-shots.json"
WORK = REPO / "data" / "ab"
REPORTS = REPO / "eval" / "reports"

# The dense arm exists only to decide which labels are answerable at all. Some
# queries will fail because CLIP cannot see "sepia", not because of sampling;
# filtering those with either *test* arm's own successes would bias the
# comparison toward it, so the filter has to come from an arm that captures
# every shot and is neutral between the two.
DENSE_FPS = 10.0

ARMS = {
    "scene-aware": {"scene_threshold": config.SCENE_THRESHOLD, "baseline_fps": 1.0},
    "fixed-interval": {"scene_threshold": None, "baseline_fps": 1.0},
    "dense": {"scene_threshold": None, "baseline_fps": DENSE_FPS},
}


def run_arm(name: str, kwargs: dict, labels, shots, embedder, ks) -> dict:
    work = WORK / name
    if work.exists():
        shutil.rmtree(work)
    (work / "index").mkdir(parents=True)
    (work / "thumbnails").mkdir(parents=True)

    timestamps = [f.timestamp_sec for f in sample_video(VIDEO, **kwargs)]
    coverage = shot_coverage(timestamps, shots)

    index = VectorIndex(dim=embedder.dim)
    with Database(work / "sv.sqlite") as database:
        ingest_video(
            VIDEO,
            index,
            database,
            embedder,
            index_dir=work / "index",
            thumbnail_dir=work / "thumbnails",
            **kwargs,
        )
        report = evaluate(
            labels,
            index,
            database,
            embedder,
            ks=ks,
            tolerance_sec=0.0,
            label_path=LABELS,
            meta={"arm": name, "coverage": coverage.fraction},
        )
    return {"name": name, "coverage": coverage, "report": report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", "--ks", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args(argv)

    if not VIDEO.is_file() or not LABELS.is_file():
        print(
            "error: corpus missing. Run:\n"
            "  uv run python scripts/make_cut_dense_corpus.py",
            file=sys.stderr,
        )
        return 1

    labels = load_labels(LABELS)
    shots = [
        (s["start_sec"], s["end_sec"])
        for s in json.loads(MANIFEST.read_text())["shots"]
    ]
    embedder = get_embedder()
    print(f"device={embedder.device} model={embedder.model_name}/{embedder.pretrained}")
    print(f"corpus: {VIDEO.name}, {len(shots)} shots, {len(labels)} queries\n")

    results = {
        name: run_arm(name, kwargs, labels, shots, embedder, tuple(args.ks))
        for name, kwargs in ARMS.items()
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        (REPORTS / f"cutdense-{name}.json").write_text(
            json.dumps(result["report"].to_dict(), indent=2) + "\n"
        )

    _print_table(results, args.ks)
    _print_paired(results, labels, args.ks)
    print(f"\nreports written to {REPORTS.relative_to(REPO)}/")
    return 0


def _print_table(results: dict, ks: list[int]) -> None:
    header = f"{'arm':<16}{'frames':>7}{'coverage':>10}" + "".join(
        f"{'R@' + str(k):>8}" for k in ks
    )
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        report = result["report"]
        row = (
            f"{name:<16}{report.corpus_frames:>7}"
            f"{result['coverage'].fraction:>9.1%} "
            + "".join(f"{report.recall_at(k):>8.1%}" for k in ks)
        )
        print(row)
        if result["coverage"].missed:
            print(f"{'':<16}missed shots {list(result['coverage'].missed)}")


def _print_paired(results: dict, labels, ks: list[int]) -> None:
    """The comparison that actually answers the question.

    Restricted to labels the dense arm can find, because a query CLIP cannot
    answer at any sampling density is measuring the embedder, not the sampler.
    Unfiltered numbers are printed above; both belong in the writeup.
    """
    scene = results["scene-aware"]["report"]
    fixed = results["fixed-interval"]["report"]
    dense = results["dense"]["report"]

    answerable = [
        i for i, o in enumerate(dense.outcomes) if o.rank is not None and o.rank <= 5
    ]
    print(
        f"\nanswerable at all (found by the dense control @5): "
        f"{len(answerable)}/{len(labels)}"
    )
    if not answerable:
        print("  nothing to compare -- CLIP cannot identify these shots at all")
        return

    for k in ks:
        s = sum(
            1
            for i in answerable
            if scene.outcomes[i].rank is not None and scene.outcomes[i].rank <= k
        ) / len(answerable)
        f = sum(
            1
            for i in answerable
            if fixed.outcomes[i].rank is not None and fixed.outcomes[i].rank <= k
        ) / len(answerable)
        print(f"  R@{k:<3} scene-aware {s:6.1%}   fixed-interval {f:6.1%}   "
              f"delta {s - f:+.1%}")

    changed = [
        i
        for i in answerable
        if (scene.outcomes[i].rank is None) != (fixed.outcomes[i].rank is None)
    ]
    if changed:
        print("\nqueries that changed:")
        for i in changed:
            label = labels[i]
            target = label.targets[0]
            winner = "scene-aware" if scene.outcomes[i].rank is not None else "fixed"
            print(
                f'  "{label.query}"\n'
                f"      {target.start_sec:g}-{target.end_sec:g}s "
                f"({target.end_sec - target.start_sec:.2f}s)  found only by {winner}"
            )
    else:
        print("\nno query changed outcome between the arms.")


if __name__ == "__main__":
    raise SystemExit(main())

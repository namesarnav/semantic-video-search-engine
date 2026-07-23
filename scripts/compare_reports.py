#!/usr/bin/env python
"""Diff two eval reports over the same queries.

    uv run python scripts/compare_reports.py baseline.json candidate.json \
        --labels vit-b-32 vit-l-14

Reports are written by `sv-engine eval --json`. Both must have been scored
against the same label set; comparing across sets is refused rather than
silently intersected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sv_engine.evaluation import compare_reports  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("-k", "--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--labels", nargs=2, default=["baseline", "candidate"], metavar="NAME"
    )
    args = parser.parse_args(argv)

    payloads = []
    for path in (args.baseline, args.candidate):
        if not Path(path).is_file():
            print(f"error: report not found: {path}", file=sys.stderr)
            return 1
        payloads.append(json.loads(Path(path).read_text()))

    try:
        result = compare_reports(
            *payloads, ks=args.ks, labels=(args.labels[0], args.labels[1])
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for payload, name in zip(payloads, args.labels):
        corpus = payload.get("corpus", {})
        print(
            f"{name}: {payload.get('model', '?')} · "
            f"{corpus.get('frames', '?')} frames · "
            f"p50 {payload.get('latency_ms', {}).get('p50', 0):.0f}ms"
        )
    print()
    print(result.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

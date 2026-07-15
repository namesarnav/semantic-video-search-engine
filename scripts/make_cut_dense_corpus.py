#!/usr/bin/env python
"""Generate the cut-dense A/B corpus and its ground truth.

    uv run python scripts/make_cut_dense_corpus.py

Writes the video into gitignored data/, and the labels and shot manifest into
eval/ where they are committed -- the video is regenerable from this script,
the ground truth is what has to be reviewable in a diff.

See sv_engine/cutdense.py for why this corpus exists and how it is laid out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sv_engine import config  # noqa: E402
from sv_engine.cutdense import (  # noqa: E402
    SUBJECTS,
    build_schedule,
    labels_payload,
    manifest_payload,
    render,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "data" / "eval-cutdense" / "cut_dense.mp4"
LABELS = REPO / "eval" / "labels-cutdense.json"
MANIFEST = REPO / "eval" / "cutdense-shots.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--source-dir",
        default=str(config.VIDEO_DIR),
        help="where the four standalone source clips live",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    source_dir = Path(args.source_dir)
    sources = {name: source_dir / meta["file"] for name, meta in SUBJECTS.items()}
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        print("error: missing source clip(s):", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    schedule = build_schedule()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    render(schedule, sources, out, fps=args.fps)

    LABELS.write_text(
        json.dumps(labels_payload(schedule, out.name), indent=2) + "\n"
    )
    MANIFEST.write_text(json.dumps(manifest_payload(schedule), indent=2) + "\n")

    brief = sum(1 for s in schedule if s.brief)
    print(f"wrote {out}  ({schedule[-1].end_sec:g}s, {len(schedule)} shots)")
    print(f"  {brief} brief (<1s), {len(schedule) - brief} sustained")
    print(f"wrote {LABELS.relative_to(REPO)}")
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    print("\nnext: uv run python scripts/run_sampling_ab.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Turn a fetched corpus manifest into an eval set.

    uv run python scripts/make_corpus_labels.py --manifest corpus/manifest.json

Writes eval/labels-corpus.json. See sv_engine/manifest_labels.py for why these
labels are weak supervision and what that means for how they may be used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sv_engine.manifest_labels import (  # noqa: E402
    ManifestError,
    labels_from_manifest,
    load_manifest,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "eval" / "labels-corpus.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO / "corpus" / "manifest.json"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    try:
        payload = labels_from_manifest(load_manifest(args.manifest))
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    multi = sum(1 for q in payload["queries"] if len(q["targets"]) > 1)
    print(f"wrote {out.relative_to(REPO)}")
    print(f"  {len(payload['queries'])} queries ({multi} with several targets)")
    print(f"  skipped: {payload['_skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M1 command line: index a folder of videos, then query it.

    uv run python -m sv_engine.cli index data/videos
    uv run python -m sv_engine.cli search "a red car at night"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config
from .embedder import get_embedder
from .index import FrameIndex
from .ingest import ingest_video

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _collect_videos(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def cmd_index(args: argparse.Namespace) -> int:
    target = Path(args.target)
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 1

    videos = _collect_videos(target)
    if not videos:
        print(f"error: no video files found in {target}", file=sys.stderr)
        return 1

    config.ensure_dirs()
    embedder = get_embedder()
    print(f"device={embedder.device} model={embedder.model_name}/{embedder.pretrained}")

    index = (
        FrameIndex(dim=embedder.dim)
        if args.rebuild
        else FrameIndex.load_or_create(dim=embedder.dim)
    )

    started = time.perf_counter()
    total_frames = 0
    for video in videos:
        video_started = time.perf_counter()
        result = ingest_video(
            video,
            index,
            embedder,
            scene_threshold=None if args.fixed_interval else config.SCENE_THRESHOLD,
            baseline_fps=args.baseline_fps,
            skip_if_present=not args.rebuild,
        )
        if result.skipped:
            print(f"  skip  {result.filename}  (already indexed as {result.video_id})")
            continue
        total_frames += result.frames_indexed
        print(
            f"  ok    {result.filename}  "
            f"{result.frames_indexed} frames  "
            f"{result.duration_sec:.1f}s  "
            f"[{time.perf_counter() - video_started:.1f}s]"
        )

    index.save()
    print(
        f"\nindexed {total_frames} new frames "
        f"({len(index)} total, {len(index.video_ids())} videos) "
        f"in {time.perf_counter() - started:.1f}s"
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    try:
        index = FrameIndex.load()
    except FileNotFoundError:
        print("error: no index found -- run `index` first", file=sys.stderr)
        return 1

    embedder = get_embedder()
    started = time.perf_counter()
    vector = embedder.encode_text([args.query])[0]
    hits = index.search(vector, top_k=args.top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000

    if not hits:
        print("no results")
        return 0

    print(f'\n"{args.query}"  ({elapsed_ms:.0f}ms, {len(index)} frames searched)\n')
    for rank, hit in enumerate(hits, start=1):
        r = hit.record
        print(
            f"{rank:>2}. {hit.score:.4f}  {r.filename}  "
            f"@ {r.timestamp_sec:6.2f}s  [{r.reason}]"
        )
        print(f"      {r.thumbnail_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sv-engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="ingest a video file or folder")
    p_index.add_argument("target", help="video file or directory of videos")
    p_index.add_argument(
        "--rebuild", action="store_true", help="discard the existing index first"
    )
    p_index.add_argument(
        "--fixed-interval",
        action="store_true",
        help="disable scene-change sampling (control arm for recall comparison)",
    )
    p_index.add_argument("--baseline-fps", type=float, default=config.BASELINE_FPS)
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="query the index")
    p_search.add_argument("query", help="natural-language query")
    p_search.add_argument("-k", "--top-k", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

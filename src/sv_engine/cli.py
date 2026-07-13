"""Command line: index a folder of videos, inspect them, serve the API, query.

    uv run python -m sv_engine.cli index data/videos
    uv run python -m sv_engine.cli videos
    uv run python -m sv_engine.cli recover
    uv run python -m sv_engine.cli serve --port 8000
    uv run python -m sv_engine.cli search "a red car at night"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import config, db, evaluation, recovery
from .db import Database
from .embedder import get_embedder
from .index import VectorIndex
from .ingest import ingest_video
from .search import search

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _collect_videos(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def _reset_store() -> None:
    """Drop index, database and thumbnails together.

    All three or none: a rebuilt index with a stale database is exactly the
    desync that produces confident wrong answers.
    """
    for path in (config.DB_PATH, config.INDEX_DIR / "frames.faiss"):
        path.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(config.DB_PATH) + suffix).unlink(missing_ok=True)
    if config.THUMBNAIL_DIR.exists():
        shutil.rmtree(config.THUMBNAIL_DIR)


def cmd_index(args: argparse.Namespace) -> int:
    target = Path(args.target)
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 1

    videos = _collect_videos(target)
    if not videos:
        print(f"error: no video files found in {target}", file=sys.stderr)
        return 1

    if args.rebuild:
        _reset_store()
    config.ensure_dirs()

    embedder = get_embedder()
    print(f"device={embedder.device} model={embedder.model_name}/{embedder.pretrained}")

    index = VectorIndex.load_or_create(dim=embedder.dim, directory=config.INDEX_DIR)
    with Database() as database:
        if not args.rebuild:
            # Repair whatever the last run left behind before adding to it.
            # --rebuild skips this: the store was just deleted.
            report = recovery.recover(index, database, index_dir=config.INDEX_DIR)
            if not report.clean:
                print(report.describe())

        started = time.perf_counter()
        total_frames = 0
        failures = 0

        for video in videos:
            video_started = time.perf_counter()
            try:
                result = ingest_video(
                    video,
                    index,
                    database,
                    embedder,
                    scene_threshold=None if args.fixed_interval else config.SCENE_THRESHOLD,
                    baseline_fps=args.baseline_fps,
                    force=args.rebuild or args.force,
                    index_dir=config.INDEX_DIR,
                )
            except Exception as exc:  # noqa: BLE001 - keep going, report at end
                failures += 1
                print(f"  FAIL  {video.name}  {exc}", file=sys.stderr)
                continue

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

        index.save(config.INDEX_DIR)
        database.check_consistency(len(index))
        print(
            f"\nindexed {total_frames} new frames "
            f"({len(index)} total, {len(database.list_videos(status=db.DONE))} videos) "
            f"in {time.perf_counter() - started:.1f}s"
        )
        if failures:
            print(f"{failures} video(s) failed -- see `videos` for status", file=sys.stderr)
            return 1
    return 0


def cmd_videos(args: argparse.Namespace) -> int:
    with Database() as database:
        rows = database.list_videos(status=args.status)
        if not rows:
            print("no videos ingested")
            return 0
        print(f"{'STATUS':<11}{'ID':<18}{'FRAMES':>7}  {'DURATION':>9}  FILENAME")
        for row in rows:
            frames = len(
                [
                    f
                    for f in database.conn.execute(
                        "SELECT 1 FROM frames WHERE video_id = ?", (row.id,)
                    ).fetchall()
                ]
            )
            print(
                f"{row.status:<11}{row.id:<18}{frames:>7}  "
                f"{row.duration_sec:>8.1f}s  {row.filename}"
            )
            if row.error:
                print(f"{'':<11}error: {row.error}")
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    """Repair a store left inconsistent by a crash.

    Runs automatically before `index` and at API startup; exposed as a command
    so a store can be inspected and repaired without ingesting anything.
    """
    with Database() as database:
        try:
            index = VectorIndex.load(config.INDEX_DIR)
        except FileNotFoundError:
            if database.frame_count():
                print(
                    f"error: {database.frame_count()} frame rows but no index file. "
                    "Every vector is gone, which no repair can undo -- re-ingest "
                    "with --rebuild.",
                    file=sys.stderr,
                )
                return 1
            # Nothing indexed yet: a crash during the very first ingest.
            swept = recovery.sweep_interrupted(database)
            print(
                f"recovered: {len(swept)} interrupted video(s) marked failed"
                if swept
                else "store is consistent; nothing to recover"
            )
            return 0

        report = recovery.recover(index, database, index_dir=config.INDEX_DIR)
        print(report.describe())
        for video_id in report.swept:
            print(f"  failed  {video_id}  (interrupted -- re-ingest to retry)")
        for video_id in report.dropped_videos:
            print(f"  failed  {video_id}  (vectors lost -- re-ingest to retry)")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    try:
        index = VectorIndex.load(config.INDEX_DIR)
    except FileNotFoundError:
        print("error: no index found -- run `index` first", file=sys.stderr)
        return 1

    with Database() as database:
        database.check_consistency(len(index))
        embedder = get_embedder()
        started = time.perf_counter()
        try:
            results = search(
                args.query,
                index,
                database,
                embedder,
                top_k=args.top_k,
                collapse_window_sec=args.collapse,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        elapsed_ms = (time.perf_counter() - started) * 1000

        if not results:
            print("no results")
            return 0

        print(f'\n"{args.query}"  ({elapsed_ms:.0f}ms, {len(index)} frames searched)\n')
        for rank, r in enumerate(results, start=1):
            print(
                f"{rank:>2}. {r.score:.4f}  {r.filename}  "
                f"@ {r.timestamp_sec:6.2f}s  [{r.reason}]"
            )
            print(f"      {r.thumbnail_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Score the corpus against the hand-labelled eval set.

    Reads the store as it stands -- it never re-ingests. The sampling A/B is
    therefore two runs: build the store one way, `eval --json a.json`, rebuild
    the other way, `eval --json b.json`, compare. Each report records which arm
    produced it, so the two files cannot be silently mixed up.
    """
    try:
        labels = evaluation.load_labels(args.labels)
    except evaluation.LabelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        index = VectorIndex.load(config.INDEX_DIR)
    except FileNotFoundError:
        print("error: no index found -- run `index` first", file=sys.stderr)
        return 1

    with Database() as database:
        database.check_consistency(len(index))
        embedder = get_embedder()
        try:
            report = evaluation.evaluate(
                labels,
                index,
                database,
                embedder,
                top_k=args.top_k,
                ks=args.ks,
                tolerance_sec=args.tolerance,
                collapse_window_sec=args.collapse,
                label_path=args.labels,
                meta={"model": f"{embedder.model_name}/{embedder.pretrained}",
                      "device": embedder.device},
            )
        except evaluation.LabelError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print()
        print(report.describe())
        print()

        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
            print(f"wrote {out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the HTTP API. A thin wrapper over uvicorn so the incantation lives
    in one place; the app itself is built in api.py."""
    import uvicorn

    print(f"serving on http://{args.host}:{args.port}  (docs at /docs)")
    uvicorn.run(
        "sv_engine.api:create_default_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sv-engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="ingest a video file or folder")
    p_index.add_argument("target", help="video file or directory of videos")
    p_index.add_argument(
        "--rebuild",
        action="store_true",
        help="discard index, database and thumbnails, then re-ingest everything",
    )
    p_index.add_argument(
        "--force", action="store_true", help="re-ingest videos already marked done"
    )
    p_index.add_argument(
        "--fixed-interval",
        action="store_true",
        help="disable scene-change sampling (control arm for recall comparison)",
    )
    p_index.add_argument("--baseline-fps", type=float, default=config.BASELINE_FPS)
    p_index.set_defaults(func=cmd_index)

    p_videos = sub.add_parser("videos", help="list ingested videos and their status")
    p_videos.add_argument(
        "--status", choices=sorted(db.VALID_STATUSES), help="filter by status"
    )
    p_videos.set_defaults(func=cmd_videos)

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument(
        "--reload", action="store_true", help="restart on code changes (development)"
    )
    p_serve.set_defaults(func=cmd_serve)

    p_recover = sub.add_parser(
        "recover", help="repair a store left inconsistent by a crash"
    )
    p_recover.set_defaults(func=cmd_recover)

    p_eval = sub.add_parser("eval", help="score Recall@K against the labelled eval set")
    p_eval.add_argument(
        "--labels",
        default=str(config.EVAL_LABELS_PATH),
        help="labels JSON (default: eval/labels.json)",
    )
    p_eval.add_argument(
        "-k",
        "--ks",
        type=int,
        nargs="+",
        default=list(evaluation.DEFAULT_KS),
        metavar="K",
        help="report Recall at these depths (default: 1 5 10)",
    )
    p_eval.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="fetch depth (default: the largest K reported)",
    )
    p_eval.add_argument(
        "--tolerance",
        type=float,
        default=evaluation.DEFAULT_TOLERANCE_SEC,
        metavar="SEC",
        help="widen each labelled range by SEC on both sides, absorbing the "
        "sampling grid (default: %(default)s = one baseline interval)",
    )
    p_eval.add_argument(
        "--collapse",
        type=float,
        metavar="SEC",
        help="merge near-duplicate hits before scoring (A/B the collapse window)",
    )
    p_eval.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    p_eval.set_defaults(func=cmd_eval)

    p_search = sub.add_parser("search", help="query the index")
    p_search.add_argument("query", help="natural-language query")
    p_search.add_argument("-k", "--top-k", type=int, default=10)
    p_search.add_argument(
        "--collapse",
        type=float,
        metavar="SEC",
        help="merge hits from the same video within SEC seconds of each other",
    )
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

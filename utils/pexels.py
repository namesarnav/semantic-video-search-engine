"""
python fetch_pexels_videos.py --out ./corpus --target 150
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests

PEXELS_API_URL = "https://api.pexels.com/videos/search"
PER_PAGE = 15  

DEFAULT_QUERIES = [
    "person cooking",
    "person eating",
    "city traffic at night",
    "ocean waves",
    "mountain hiking",
    "dog playing",
    "cat sleeping",
    "office meeting",
    "person typing on laptop",
    "rain on window",
    "sunset over water",
    "crowd walking street",
    "car driving highway",
    "children playing park",
    "coffee shop interior",
    "forest trail",
    "concert crowd",
    "basketball game",
    "person running",
    "bicycle riding",
]


def fetch_query(api_key: str, query: str, per_page: int, page: int) -> dict:
    resp = requests.get(
        PEXELS_API_URL,
        headers={"Authorization": api_key},
        params={"query": query, "per_page": per_page, "page": page},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def pick_smallest_file(video: dict) -> dict | None:
    """Pick the smallest usable video file (enough for CLIP embedding, saves bandwidth)."""
    files = video.get("video_files", [])
    if not files:
        return None
    # Prefer explicit "sd"/"tiny" quality tags; fall back to smallest by resolution.
    preferred = [f for f in files if f.get("quality") in ("sd", "tiny")]
    candidates = preferred if preferred else files
    return min(candidates, key=lambda f: (f.get("width") or 99999))


def download_file(url: str, dest: Path, chunk_size: int = 1 << 16) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("./corpus"),
                         help="Output directory for videos + manifest.json")
    parser.add_argument("--target", type=int, default=150,
                         help="Total number of videos to fetch (100-200 recommended)")
    parser.add_argument("--queries", type=str, default=None,
                         help="Comma-separated query list to override the defaults")
    parser.add_argument("--per-query-max", type=int, default=20,
                         help="Cap on videos pulled from any single query term, "
                              "to keep the corpus from being dominated by one topic")
    args = parser.parse_args()

    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise SystemExit(
            "PEXELS_API_KEY not set. Get a free key at pexels.com/api and:\n"
            "  export PEXELS_API_KEY=your_key_here"
        )

    queries = (
        [q.strip() for q in args.queries.split(",")]
        if args.queries
        else DEFAULT_QUERIES
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    already_have = {int(k) for k in manifest.keys()}

    print(f"Target: {args.target} videos across {len(queries)} queries "
          f"(already have {len(already_have)})")

    downloaded_this_run = 0
    query_idx = 0

    while len(manifest) < args.target and query_idx < len(queries):
        query = queries[query_idx]
        per_query_count = sum(1 for v in manifest.values() if v["query"] == query)
        page = 1

        while (
            per_query_count < args.per_query_max
            and len(manifest) < args.target
        ):
            try:
                data = fetch_query(api_key, query, PER_PAGE, page)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print("Rate limited — sleeping 60s")
                    time.sleep(60)
                    continue
                print(f"  [{query} p{page}] request failed: {e}")
                break

            videos = data.get("videos", [])
            if not videos:
                break  # no more results for this query

            for video in videos:
                if per_query_count >= args.per_query_max or len(manifest) >= args.target:
                    break

                vid_id = video["id"]
                if vid_id in already_have:
                    continue

                file_info = pick_smallest_file(video)
                if not file_info:
                    continue

                dest = args.out / f"{vid_id}.mp4"
                try:
                    download_file(file_info["link"], dest)
                except requests.RequestException as e:
                    print(f"  failed to download {vid_id}: {e}")
                    continue

                manifest[str(vid_id)] = {
                    "query": query,
                    "pexels_id": vid_id,
                    "width": file_info.get("width"),
                    "height": file_info.get("height"),
                    "duration_sec": video.get("duration"),
                    "url": video.get("url"),
                    "file": dest.name,
                }
                already_have.add(vid_id)
                per_query_count += 1
                downloaded_this_run += 1

                # Persist incrementally so a crash/Ctrl+C doesn't lose progress.
                manifest_path.write_text(json.dumps(manifest, indent=2))

                print(f"  [{query}] {len(manifest)}/{args.target} -> {dest.name} "
                      f"({video.get('duration')}s)")

            page += 1
            time.sleep(0.5)  # stay well under the 200 req/hr free-tier limit

        query_idx += 1

    print(f"\nDone. Downloaded {downloaded_this_run} new videos this run.")
    print(f"Corpus total: {len(manifest)} videos in {args.out}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
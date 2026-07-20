"""Serving the source video itself (`GET /videos/{id}/file`).

Until now the API exposed frames -- thumbnails -- but never the video they
came from, so a client could show evidence of a match and not the thing that
matched. Playing a result at its timestamp needs the file, and seeking needs
byte ranges: without a 206 the browser must download the whole video before it
can jump anywhere, which on a 4K clip is the difference between instant and
unusable.

Stub embedder throughout; no CLIP is loaded.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sv_engine import config
from sv_engine.api import AppState, create_app
from sv_engine.db import Database
from sv_engine.index import VectorIndex
from test_api import StubEmbedder


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "THUMBNAIL_DIR", tmp_path / "thumbs")
    state = AppState(
        embedder=StubEmbedder(),
        index=VectorIndex(dim=StubEmbedder.dim),
        db_path=tmp_path / "api.sqlite",
        index_dir=tmp_path / "index",
        thumbnail_dir=tmp_path / "thumbs",
        upload_dir=tmp_path / "uploads",
    )
    Database(state.db_path).close()
    with TestClient(create_app(state)) as client:
        yield client, state


@pytest.fixture
def ingested(api, synthetic_video):
    """One ingested video, so there is a row with a real path on disk."""
    client, state = api
    video = synthetic_video(duration_sec=2.0)
    response = client.post("/videos", json={"path": str(video)})
    assert response.status_code == 202
    return client, state, response.json()["video_id"], video


def test_the_video_file_is_served(ingested):
    client, _, video_id, _ = ingested
    response = client.get(f"/videos/{video_id}/file")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 0


def test_the_response_advertises_range_support(ingested):
    """Without accept-ranges the browser will not attempt a seek at all."""
    client, _, video_id, _ = ingested
    response = client.get(f"/videos/{video_id}/file")

    assert response.headers.get("accept-ranges") == "bytes"


def test_a_range_request_returns_206_with_only_that_slice(ingested):
    """The whole point: seeking to a timestamp must not download the file."""
    client, _, video_id, path = ingested
    whole = path.read_bytes()

    response = client.get(f"/videos/{video_id}/file", headers={"Range": "bytes=0-99"})

    assert response.status_code == 206
    assert response.content == whole[:100]
    assert response.headers["content-range"] == f"bytes 0-99/{len(whole)}"
    assert response.headers["content-length"] == "100"


def test_an_open_ended_range_runs_to_the_end(ingested):
    client, _, video_id, path = ingested
    whole = path.read_bytes()
    start = len(whole) - 50

    response = client.get(
        f"/videos/{video_id}/file", headers={"Range": f"bytes={start}-"}
    )

    assert response.status_code == 206
    assert response.content == whole[start:]


def test_a_range_past_the_end_is_clamped_not_an_error(ingested):
    """Browsers routinely ask for more than exists while probing."""
    client, _, video_id, path = ingested
    whole = path.read_bytes()

    response = client.get(
        f"/videos/{video_id}/file", headers={"Range": f"bytes=0-{len(whole) + 5000}"}
    )

    assert response.status_code == 206
    assert response.content == whole


def test_an_unsatisfiable_range_is_rejected(ingested):
    client, _, video_id, path = ingested
    start = len(path.read_bytes()) + 10

    response = client.get(
        f"/videos/{video_id}/file", headers={"Range": f"bytes={start}-"}
    )

    assert response.status_code == 416


def test_a_malformed_range_falls_back_to_the_whole_file(ingested):
    """A header we do not understand must not 500; serving everything is the
    correct, spec-sanctioned fallback."""
    client, _, video_id, path = ingested

    response = client.get(
        f"/videos/{video_id}/file", headers={"Range": "kilobytes=1-2"}
    )

    assert response.status_code == 200
    assert response.content == path.read_bytes()


def test_an_unknown_video_is_404(api):
    client, _ = api
    assert client.get("/videos/nope/file").status_code == 404


def test_a_row_whose_file_has_been_deleted_is_404_not_500(ingested):
    """The database is the source of truth for what exists, but the file can
    still be moved out from under it."""
    client, _, video_id, path = ingested
    path.unlink()

    response = client.get(f"/videos/{video_id}/file")
    assert response.status_code == 404
    assert "disk" in response.json()["detail"]


def test_search_results_carry_a_video_url(api, synthetic_video):
    """The client is never handed a filesystem path -- same rule as
    thumbnail_url. It should not have to build this URL itself either."""
    client, _ = api
    video = synthetic_video(duration_sec=2.0)
    client.post("/videos", json={"path": str(video)})

    results = client.post("/search", json={"query": "anything", "top_k": 3}).json()
    assert results["results"], "expected the ingest to have produced frames"

    for hit in results["results"]:
        assert hit["video_url"] == f"/videos/{hit['video_id']}/file"
        assert not hit["video_url"].startswith("/Users")


def test_the_video_listing_carries_a_video_url(ingested):
    client, _, video_id, _ = ingested
    videos = client.get("/videos").json()["videos"]

    assert videos[0]["video_url"] == f"/videos/{video_id}/file"

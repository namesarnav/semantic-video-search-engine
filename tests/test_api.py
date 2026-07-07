"""M3 API tests.

These use a stub embedder so no CLIP checkpoint is loaded -- what is under test
is the HTTP layer: status codes, response shapes, background execution and the
job lifecycle. Retrieval quality is tested elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sv_engine import config, db
from sv_engine.api import AppState, create_app
from sv_engine.db import Database
from sv_engine.index import VectorIndex


class StubEmbedder:
    """Deterministic stand-in for ClipEmbedder.

    Image vectors vary with frame brightness and text vectors with the query
    string, so different inputs produce different vectors -- enough for
    ranking to be meaningful without a real model.
    """

    dim = 4
    device = "cpu"
    model_name = "stub"
    pretrained = "stub"

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def encode_images(self, frames, batch_size: int = 32) -> np.ndarray:
        rows = []
        for frame in frames:
            brightness = float(np.mean(frame)) / 255.0
            rows.append([brightness, 1.0 - brightness, 0.5, 0.25])
        return self._normalize(np.array(rows, dtype=np.float32))

    def encode_text(self, queries) -> np.ndarray:
        rows = []
        for query in queries:
            seed = sum(ord(c) for c in query) % 100 / 100.0
            rows.append([seed, 1.0 - seed, 0.5, 0.25])
        return self._normalize(np.array(rows, dtype=np.float32))


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A TestClient backed by scratch storage and a stub embedder."""
    monkeypatch.setattr(config, "THUMBNAIL_DIR", tmp_path / "thumbs")

    state = AppState(
        embedder=StubEmbedder(),
        index=VectorIndex(dim=StubEmbedder.dim),
        db_path=tmp_path / "api.sqlite",
        index_dir=tmp_path / "index",
        thumbnail_dir=tmp_path / "thumbs",
        # Never the real data/videos: an upload test must not write into the
        # actual corpus.
        upload_dir=tmp_path / "uploads",
    )
    # Create the schema up front so read endpoints work before any ingest.
    Database(state.db_path).close()

    with TestClient(create_app(state)) as client:
        yield client, state


# ---- POST /videos ---------------------------------------------------------


def test_ingest_by_path_is_accepted_not_completed(api, synthetic_video):
    """202 with status queued: the job is taken, not finished."""
    client, _ = api
    video = synthetic_video()

    response = client.post("/videos", json={"path": str(video)})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == db.QUEUED
    assert body["filename"] == video.name
    assert len(body["video_id"]) == 16


def test_background_task_completes_the_ingest(api, synthetic_video):
    """The work really happens after the response, and status reflects it."""
    client, state = api
    video = synthetic_video()

    video_id = client.post("/videos", json={"path": str(video)}).json()["video_id"]
    status = client.get(f"/videos/{video_id}/status").json()

    assert status["status"] == db.DONE
    assert status["frame_count"] > 0
    assert len(state.index) == status["frame_count"]


def test_ingest_nonexistent_path_is_404(api):
    client, _ = api
    response = client.post("/videos", json={"path": "/nope/missing.mp4"})
    assert response.status_code == 404


def test_ingest_directory_is_400(api, tmp_path):
    """A folder is not a video; reject it rather than failing later in a worker."""
    client, _ = api
    response = client.post("/videos", json={"path": str(tmp_path)})
    assert response.status_code == 400


def test_ingest_empty_path_is_422(api):
    """Validation is Pydantic's job and should never reach the worker."""
    client, _ = api
    assert client.post("/videos", json={"path": ""}).status_code == 422


def test_ingest_missing_body_is_422(api):
    client, _ = api
    assert client.post("/videos", json={}).status_code == 422


def test_reingesting_the_same_video_does_not_duplicate(api, synthetic_video):
    """FR5 over HTTP: same content, one video, no extra frames."""
    client, state = api
    video = synthetic_video()

    first = client.post("/videos", json={"path": str(video)}).json()
    frames_after_first = len(state.index)
    second = client.post("/videos", json={"path": str(video)}).json()

    assert first["video_id"] == second["video_id"]
    assert len(state.index) == frames_after_first
    assert len(client.get("/videos").json()["videos"]) == 1


def test_failed_ingest_is_reported_as_failed(api, tmp_path):
    """FR6 over HTTP: a broken file surfaces as failed with an error, and the
    request itself still succeeds -- the failure is job state, not a 500."""
    client, _ = api
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    video_id = client.post("/videos", json={"path": str(broken)}).json()["video_id"]
    status = client.get(f"/videos/{video_id}/status").json()

    assert status["status"] == db.FAILED
    assert status["error"]


def test_upload_ingests_the_posted_file(api, synthetic_video):
    """FR1's upload half: bytes over the wire, not a server-side path."""
    client, state = api
    video = synthetic_video()

    response = client.post(
        "/videos/upload",
        files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
    )

    assert response.status_code == 202
    video_id = response.json()["video_id"]
    assert client.get(f"/videos/{video_id}/status").json()["status"] == db.DONE
    assert len(state.index) > 0


def test_upload_rejects_unsupported_extension(api):
    client, _ = api
    response = client.post(
        "/videos/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


# ---- GET /videos ----------------------------------------------------------


def test_listing_is_empty_before_any_ingest(api):
    client, _ = api
    assert client.get("/videos").json()["videos"] == []


def test_listing_reports_each_video(api, synthetic_video):
    client, _ = api
    client.post("/videos", json={"path": str(synthetic_video("a.mp4"))})
    client.post("/videos", json={"path": str(synthetic_video("b.mp4", cut_at_sec=0.0))})

    videos = client.get("/videos").json()["videos"]

    assert len(videos) == 2
    assert {v["filename"] for v in videos} == {"a.mp4", "b.mp4"}
    assert all(v["status"] == db.DONE for v in videos)
    assert all(v["frame_count"] > 0 for v in videos)


def test_listing_can_filter_by_status(api, synthetic_video, tmp_path):
    client, _ = api
    client.post("/videos", json={"path": str(synthetic_video("good.mp4"))})
    broken = tmp_path / "bad.mp4"
    broken.write_bytes(b"nope")
    client.post("/videos", json={"path": str(broken)})

    done = client.get("/videos", params={"status": db.DONE}).json()["videos"]
    failed = client.get("/videos", params={"status": db.FAILED}).json()["videos"]

    assert [v["filename"] for v in done] == ["good.mp4"]
    assert [v["filename"] for v in failed] == ["bad.mp4"]


def test_listing_rejects_an_invalid_status(api):
    client, _ = api
    assert client.get("/videos", params={"status": "banana"}).status_code == 422


# ---- GET /videos/{id}/status ---------------------------------------------


def test_status_of_unknown_video_is_404(api):
    client, _ = api
    assert client.get("/videos/deadbeef/status").status_code == 404


# ---- POST /search ---------------------------------------------------------


def test_search_returns_ranked_results(api, synthetic_video):
    client, _ = api
    client.post("/videos", json={"path": str(synthetic_video())})

    body = client.post("/search", json={"query": "anything", "top_k": 3}).json()

    assert body["count"] == len(body["results"]) <= 3
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["took_ms"] >= 0


def test_search_results_carry_the_fields_the_ui_needs(api, synthetic_video):
    """FR3: every result must identify the video, the moment and a preview."""
    client, _ = api
    client.post("/videos", json={"path": str(synthetic_video())})

    result = client.post("/search", json={"query": "anything"}).json()["results"][0]

    assert set(result) >= {
        "score",
        "video_id",
        "filename",
        "timestamp_sec",
        "thumbnail_url",
        "reason",
        "frame_id",
    }
    # A URL the browser can fetch -- never a path on the server's disk.
    assert result["thumbnail_url"] == f"/thumbnails/{result['frame_id']}"
    assert "/" in result["thumbnail_url"]
    assert not result["thumbnail_url"].startswith("/Volumes")


def test_search_before_any_ingest_returns_nothing(api):
    client, _ = api
    body = client.post("/search", json={"query": "anything"}).json()
    assert body["results"] == []
    assert body["count"] == 0


def test_search_empty_query_is_422(api):
    client, _ = api
    assert client.post("/search", json={"query": "   "}).status_code == 422


def test_search_zero_top_k_is_422(api):
    client, _ = api
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422


def test_search_negative_top_k_is_422(api):
    client, _ = api
    assert client.post("/search", json={"query": "x", "top_k": -5}).status_code == 422


def test_search_collapse_reduces_near_duplicates(api, synthetic_video):
    """A long static shot must not fill the results with the same moment."""
    client, _ = api
    client.post("/videos", json={"path": str(synthetic_video(duration_sec=8.0))})

    wide = client.post("/search", json={"query": "x", "top_k": 10}).json()
    collapsed = client.post(
        "/search", json={"query": "x", "top_k": 10, "collapse_window_sec": 5.0}
    ).json()

    assert collapsed["count"] < wide["count"]


# ---- GET /thumbnails/{frame_id} ------------------------------------------


def test_thumbnail_is_served_as_an_image(api, synthetic_video):
    client, _ = api
    client.post("/videos", json={"path": str(synthetic_video())})
    frame_id = client.post("/search", json={"query": "x"}).json()["results"][0]["frame_id"]

    response = client.get(f"/thumbnails/{frame_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # JPEG files start with these two bytes.
    assert response.content[:2] == b"\xff\xd8"


def test_thumbnail_of_unknown_frame_is_404(api):
    client, _ = api
    assert client.get("/thumbnails/999999").status_code == 404


# ---- health ---------------------------------------------------------------


def test_health_reports_corpus_size(api, synthetic_video):
    client, _ = api
    client.post("/videos", json={"path": str(synthetic_video())})

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["videos"] == 1
    assert body["frames"] > 0
    assert body["device"] == "cpu"

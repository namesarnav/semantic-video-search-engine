"""Serving the built UI from the API (M5).

`sv-engine serve` should hand you a working page, not a page that needs a
second process. These tests pin the two things that can silently go wrong: the
static mount swallowing API routes, and an unbuilt UI failing in a way that
does not say what to do about it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sv_engine.api import AppState, create_app
from sv_engine.db import Database
from sv_engine.index import VectorIndex

from test_api import StubEmbedder


@pytest.fixture
def built_ui(tmp_path):
    """A stand-in for `npm run build` output."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Semantic Video Search</title>")
    (dist / "assets" / "index-abc123.js").write_text("console.log('app')")
    return dist


def _client(tmp_path, web_dir):
    state = AppState(
        embedder=StubEmbedder(),
        index=VectorIndex(dim=StubEmbedder.dim),
        db_path=tmp_path / "api.sqlite",
        index_dir=tmp_path / "index",
        thumbnail_dir=tmp_path / "thumbs",
        upload_dir=tmp_path / "uploads",
        web_dir=web_dir,
    )
    Database(state.db_path).close()
    return TestClient(create_app(state))


def test_the_built_page_is_served_at_the_root(tmp_path, built_ui):
    with _client(tmp_path, built_ui) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Semantic Video Search" in response.text


def test_hashed_assets_are_served(tmp_path, built_ui):
    with _client(tmp_path, built_ui) as client:
        response = client.get("/assets/index-abc123.js")
    assert response.status_code == 200
    assert "console.log" in response.text


def test_the_mount_does_not_shadow_the_api(tmp_path, built_ui):
    """The sharpest failure here: a catch-all at / that eats /health and
    /search, so the page loads and every request from it 404s."""
    with _client(tmp_path, built_ui) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/videos").json() == {"videos": []}
        assert client.post("/search", json={"query": "anything"}).status_code == 200


def test_without_a_build_the_root_says_how_to_build_it(tmp_path):
    with _client(tmp_path, tmp_path / "does-not-exist") as client:
        response = client.get("/")
    assert response.status_code == 404
    # Not just "Not Found": the fix has to be in the message.
    assert "npm" in response.json()["detail"]


def test_the_api_works_without_a_build(tmp_path):
    """The UI is optional. A headless deployment must not be broken by it."""
    with _client(tmp_path, tmp_path / "does-not-exist") as client:
        assert client.get("/health").status_code == 200

"""Startup recovery through the two front ends (M4).

Recovery is only useful if something actually runs it. These pin the two entry
points a restart goes through: the ``recover`` command, and the API's startup.
No CLIP is loaded -- the repair works on the index and the database alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sv_engine import cli, config, db
from sv_engine.api import AppState, create_app
from sv_engine.db import Database
from sv_engine.index import VectorIndex

DIM = 4


class StubEmbedder:
    dim = DIM
    device = "cpu"
    model_name = "stub"
    pretrained = "stub"

    def encode_images(self, frames, batch_size: int = 32):
        arr = np.ones((len(frames), DIM), dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def encode_text(self, queries):
        arr = np.ones((len(queries), DIM), dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the CLI's default paths at scratch storage."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "sv.sqlite")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "THUMBNAIL_DIR", tmp_path / "thumbs")
    return tmp_path


def _crashed_store(tmp_path) -> None:
    """One healthy video, one killed mid-ingest: still ``processing``, with two
    of its vectors already persisted and no rows committed."""
    index = VectorIndex(dim=DIM)
    database = Database(tmp_path / "sv.sqlite")

    vectors = np.eye(DIM, dtype=np.float32)[:3]
    ids = index.add(vectors)
    database.upsert_video("good", "good.mp4", "/v/good.mp4", 5.0)
    database.add_frames(
        "good",
        [
            {
                "timestamp_sec": float(i),
                "thumbnail_path": f"/t/{i}.jpg",
                "reason": "baseline",
                "vector_index_id": vid,
            }
            for i, vid in enumerate(ids)
        ],
        status=db.DONE,
    )

    index.add(np.eye(DIM, dtype=np.float32)[:2])  # persisted, never committed
    database.upsert_video("dead", "dead.mp4", "/v/dead.mp4", status=db.PROCESSING)

    index.save(tmp_path / "index")
    database.close()


def test_recover_command_repairs_a_crashed_store(store, capsys):
    _crashed_store(store)

    assert cli.main(["recover"]) == 0

    out = capsys.readouterr().out
    assert "recovered" in out
    with Database(config.DB_PATH) as database:
        assert database.get_video("dead").status == db.FAILED
        assert database.get_video("good").status == db.DONE
        index = VectorIndex.load(config.INDEX_DIR)
        # The orphaned vectors are gone from disk, not just from memory.
        assert len(index) == 3
        database.check_consistency(len(index))


def test_recover_command_on_a_healthy_store_says_so(store, capsys):
    Database(config.DB_PATH).close()
    VectorIndex(dim=DIM).save(config.INDEX_DIR)

    assert cli.main(["recover"]) == 0
    assert "nothing to recover" in capsys.readouterr().out


def test_recover_command_reports_a_missing_index(store, capsys):
    """Rows but no index file at all: not a tail, so not repairable here."""
    with Database(config.DB_PATH) as database:
        database.upsert_video("a", "a.mp4", "/v/a.mp4")
        database.add_frames(
            "a",
            [
                {
                    "timestamp_sec": 0.0,
                    "thumbnail_path": "/t/0.jpg",
                    "reason": "baseline",
                    "vector_index_id": 0,
                }
            ],
            status=db.DONE,
        )

    assert cli.main(["recover"]) == 1
    assert "--rebuild" in capsys.readouterr().err


def test_recover_command_sweeps_before_the_first_index_exists(store, capsys):
    """A crash during the very first ingest leaves a status row and no index."""
    with Database(config.DB_PATH) as database:
        database.upsert_video("dead", "dead.mp4", "/v/dead.mp4", status=db.PROCESSING)

    assert cli.main(["recover"]) == 0
    with Database(config.DB_PATH) as database:
        assert database.get_video("dead").status == db.FAILED


def test_api_startup_recovers_a_crashed_store(store):
    _crashed_store(store)

    state = AppState(
        embedder=StubEmbedder(),
        index=VectorIndex.load(store / "index"),
        db_path=store / "sv.sqlite",
        index_dir=store / "index",
        thumbnail_dir=store / "thumbs",
        upload_dir=store / "uploads",
    )

    with TestClient(create_app(state)) as client:
        # Startup must leave the server searchable rather than raising
        # "out of sync" on the first query.
        assert client.post("/search", json={"query": "anything"}).status_code == 200
        body = client.get("/videos").json()["videos"]
        assert {v["id"]: v["status"] for v in body} == {
            "good": db.DONE,
            "dead": db.FAILED,
        }
    assert len(state.index) == 3

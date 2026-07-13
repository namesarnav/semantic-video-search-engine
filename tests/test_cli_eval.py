"""The `eval` command end to end, with a stub embedder so no CLIP is loaded.

The command is the only way anyone actually runs the metric, and the failures
that matter are the boring ones: a labels file that does not parse, or that
names a video nobody ingested, must print a fixable message and exit non-zero
rather than raising a traceback or -- worse -- reporting a low score.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sv_engine import cli, config
from sv_engine.db import Database
from sv_engine.index import VectorIndex

DIM = 4


class StubEmbedder:
    dim = DIM
    device = "cpu"
    model_name = "stub"
    pretrained = "stub"

    def encode_text(self, queries):
        arr = np.ones((len(queries), DIM), dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A two-frame corpus wired to scratch paths, with CLIP stubbed out."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "sv.sqlite")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(cli, "get_embedder", lambda: StubEmbedder())

    index = VectorIndex(dim=DIM)
    ids = index.add(np.eye(DIM, dtype=np.float32)[:2])
    (tmp_path / "index").mkdir()
    index.save(tmp_path / "index")

    database = Database(tmp_path / "sv.sqlite")
    database.upsert_video("vid", "clip.mp4", "/v/clip.mp4", 9.0)
    database.add_frames(
        "vid",
        [
            {
                "timestamp_sec": float(i),
                "thumbnail_path": f"/t/{i}.jpg",
                "reason": "baseline",
                "vector_index_id": ids[i],
            }
            for i in range(2)
        ],
    )
    database.close()
    return tmp_path


def _labels(tmp_path, video="clip.mp4", name="labels.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query": "anything",
                        "targets": [
                            {"video": video, "start_sec": 0.0, "end_sec": 1.0}
                        ],
                    }
                ]
            }
        )
    )
    return str(path)


def test_eval_scores_the_corpus_and_writes_json(store, capsys):
    out = store / "report.json"
    code = cli.main(["eval", "--labels", _labels(store), "--json", str(out)])

    assert code == 0
    assert "Recall@1" in capsys.readouterr().out
    payload = json.loads(out.read_text())
    assert payload["queries"] == 1
    assert payload["recall"]["1"] == 1.0
    assert payload["model"] == "stub/stub"


def test_a_label_naming_an_unknown_video_fails_loudly(store, capsys):
    """The whole point: this must not quietly report Recall@1 = 0%."""
    code = cli.main(["eval", "--labels", _labels(store, video="typo.mp4")])

    assert code == 1
    err = capsys.readouterr().err
    assert "typo.mp4" in err
    assert "Recall" not in err


def test_a_missing_labels_file_is_reported(store, capsys):
    code = cli.main(["eval", "--labels", str(store / "nope.json")])

    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_eval_without_an_index_says_to_index_first(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "sv.sqlite")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "empty")

    code = cli.main(["eval", "--labels", _labels(tmp_path)])

    assert code == 1
    assert "no index" in capsys.readouterr().err

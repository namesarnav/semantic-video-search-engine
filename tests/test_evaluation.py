"""Eval-harness tests, run against the stub corpus so no CLIP is loaded.

The harness is the project's primary "does it work" signal, so its own failure
modes matter: a mislabelled filename must not be indistinguishable from a
retrieval miss, and the sampling grid must not be scored as a retrieval error.
"""

from __future__ import annotations

import json

import pytest

from sv_engine.evaluation import (
    Label,
    LabelError,
    Target,
    evaluate,
    first_correct_rank,
    load_labels,
)
from test_search import StubEmbedder, corpus  # noqa: F401 - fixture import


def _label(query="bridge", video="a.mp4", start=1.0, end=1.0, **kw) -> Label:
    return Label(query=query, targets=(Target(video, start, end),), **kw)


def _write(tmp_path, payload) -> str:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload))
    return path


# ---- loading -----------------------------------------------------------


def _entry(query="q", video="a.mp4", start=1.0, end=2.0, **kw):
    return {"query": query, "targets": [{"video": video, "start_sec": start, "end_sec": end}], **kw}


def test_load_labels_reads_a_valid_file(tmp_path):
    path = _write(
        tmp_path,
        {
            "queries": [
                {
                    "query": "a red car at night",
                    "targets": [{"video": "a.mp4", "start_sec": 3.0, "end_sec": 6.0}],
                    "note": "the car enters frame",
                }
            ]
        },
    )
    labels = load_labels(path)

    assert len(labels) == 1
    assert labels[0].query == "a red car at night"
    assert labels[0].targets == (Target("a.mp4", 3.0, 6.0),)
    assert labels[0].note == "the car enters frame"


def test_a_label_can_carry_several_acceptable_targets(tmp_path):
    """The same footage appears in more than one video in a real corpus. Both
    are correct answers, and scoring one of them as a miss would cap recall
    for reasons that have nothing to do with retrieval."""
    path = _write(
        tmp_path,
        {
            "queries": [
                {
                    "query": "the bridge",
                    "targets": [
                        {"video": "a.mp4", "start_sec": 0, "end_sec": 9},
                        {"video": "b.mp4", "start_sec": 20, "end_sec": 29},
                    ],
                }
            ]
        },
    )
    label = load_labels(path)[0]

    assert len(label.targets) == 2
    assert label.videos == {"a.mp4", "b.mp4"}


def test_note_is_optional(tmp_path):
    assert load_labels(_write(tmp_path, {"queries": [_entry()]}))[0].note == ""


def test_a_missing_field_names_the_offending_label(tmp_path):
    path = _write(
        tmp_path,
        {"queries": [{"query": "q", "targets": [{"video": "a.mp4", "start_sec": 1}]}]},
    )
    with pytest.raises(LabelError, match="end_sec"):
        load_labels(path)


def test_an_unknown_field_is_rejected(tmp_path):
    """Catches `start` for `start_sec`, which would otherwise score as a miss."""
    path = _write(tmp_path, {"queries": [_entry(strat=0)]})
    with pytest.raises(LabelError, match="strat"):
        load_labels(path)


def test_an_unknown_field_inside_a_target_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        {"queries": [{"query": "q", "targets": [
            {"video": "a.mp4", "start_sec": 1, "end_sec": 2, "at": 3}]}]},
    )
    with pytest.raises(LabelError, match="at"):
        load_labels(path)


def test_a_backwards_range_is_rejected(tmp_path):
    path = _write(tmp_path, {"queries": [_entry(start=9, end=2)]})
    with pytest.raises(LabelError, match="end_sec"):
        load_labels(path)


def test_an_empty_query_is_rejected(tmp_path):
    path = _write(tmp_path, {"queries": [_entry(query="  ")]})
    with pytest.raises(LabelError, match="query"):
        load_labels(path)


def test_a_query_with_no_targets_is_rejected(tmp_path):
    path = _write(tmp_path, {"queries": [{"query": "q", "targets": []}]})
    with pytest.raises(LabelError, match="targets"):
        load_labels(path)


def test_an_empty_label_set_is_rejected(tmp_path):
    """Recall over zero queries is 100% of nothing -- refuse to report it."""
    path = _write(tmp_path, {"queries": []})
    with pytest.raises(LabelError, match="no queries"):
        load_labels(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(LabelError, match="not found"):
        load_labels(tmp_path / "nope.json")


# ---- scoring one query -------------------------------------------------


def test_rank_is_one_based(corpus):
    index, database = corpus
    from sv_engine.search import search

    results = search("bridge", index, database, StubEmbedder(), top_k=5)
    # bridge ranks a@1.0 first.
    assert first_correct_rank(_label(start=1.0, end=1.0), results, 0.0) == 1
    assert first_correct_rank(_label(start=2.0, end=2.0), results, 0.0) == 2


def test_no_correct_hit_returns_none(corpus):
    index, database = corpus
    from sv_engine.search import search

    results = search("bridge", index, database, StubEmbedder(), top_k=5)
    assert first_correct_rank(_label(start=99.0, end=99.0), results, 0.0) is None


def test_the_right_timestamp_in_the_wrong_video_is_not_a_hit(corpus):
    """Only a.mp4 has a frame at 8.0s. A label pointing that moment at b.mp4
    must miss, even though a frame at 8.0s is sitting in the results."""
    index, database = corpus
    from sv_engine.search import search

    results = search("bridge", index, database, StubEmbedder(), top_k=5)
    assert any(r.timestamp_sec == 8.0 for r in results)

    assert first_correct_rank(_label(video="a.mp4", start=8.0, end=8.0), results, 0.0)
    assert first_correct_rank(_label(video="b.mp4", start=8.0, end=8.0), results, 0.0) is None


def test_tolerance_widens_the_labelled_range(corpus):
    """The nearest sampled frame can sit up to a sampling interval away from
    the moment a human labelled. That is the sampler's grid, not a miss."""
    index, database = corpus
    from sv_engine.search import search

    results = search("bridge", index, database, StubEmbedder(), top_k=5)
    # Nothing is sampled in 2.5-2.6s; a@2.0 (rank 2) is the nearest frame, and
    # a@1.0 (rank 1) stays outside even the widened window.
    label = _label(start=2.5, end=2.6)

    assert first_correct_rank(label, results, 0.0) is None
    assert first_correct_rank(label, results, 1.0) == 2


def test_any_target_counts_and_the_best_ranked_one_wins(corpus):
    """A label with two acceptable moments is satisfied by whichever the
    engine found first -- not by the one listed first."""
    index, database = corpus
    from sv_engine.search import search

    results = search("bridge", index, database, StubEmbedder(), top_k=5)
    label = Label(
        query="bridge",
        targets=(Target("a.mp4", 8.0, 8.0), Target("a.mp4", 1.0, 1.0)),
    )
    # a@8.0 is listed first but ranks last; a@1.0 ranks first.
    assert first_correct_rank(label, results, 0.0) == 1


# ---- the report --------------------------------------------------------


def test_recall_counts_a_query_only_within_k(corpus):
    index, database = corpus
    # mountain ranks a@8.0 first, then b@1.0 second.
    report = evaluate(
        [_label(query="mountain", video="b.mp4", start=1.0, end=1.0)],
        index,
        database,
        StubEmbedder(),
        ks=(1, 5),
        tolerance_sec=0.0,
    )

    assert report.recall_at(1) == 0.0
    assert report.recall_at(5) == 1.0


def test_recall_is_a_fraction_of_queries(corpus):
    index, database = corpus
    report = evaluate(
        [
            _label(query="bridge", video="a.mp4", start=1.0, end=1.0),  # rank 1
            _label(query="sunset", video="a.mp4", start=1.0, end=1.0),  # never
        ],
        index,
        database,
        StubEmbedder(),
        ks=(1,),
        tolerance_sec=0.0,
    )

    assert report.recall_at(1) == 0.5


def test_every_query_gets_an_outcome_even_when_it_misses(corpus):
    index, database = corpus
    report = evaluate(
        [_label(query="sunset", video="a.mp4", start=99.0, end=99.0)],
        index,
        database,
        StubEmbedder(),
        tolerance_sec=0.0,
    )

    assert len(report.outcomes) == 1
    assert report.outcomes[0].rank is None
    assert report.misses == report.outcomes


def test_latency_is_recorded_per_query(corpus):
    index, database = corpus
    report = evaluate(
        [_label(query="bridge"), _label(query="mountain", video="a.mp4", start=8.0, end=8.0)],
        index,
        database,
        StubEmbedder(),
    )

    assert all(o.elapsed_ms >= 0 for o in report.outcomes)
    assert report.latency_p50 >= 0
    assert report.latency_p95 >= report.latency_p50


def test_a_label_naming_an_unknown_video_is_an_error_not_a_miss(corpus):
    """A typo'd filename scores zero and looks exactly like a broken retriever.
    Fail loudly instead."""
    index, database = corpus
    with pytest.raises(LabelError, match="c.mp4"):
        evaluate([_label(video="c.mp4")], index, database, StubEmbedder())


def test_top_k_must_cover_the_largest_k(corpus):
    index, database = corpus
    with pytest.raises(ValueError, match="top_k"):
        evaluate([_label()], index, database, StubEmbedder(), top_k=5, ks=(1, 10))


def test_report_serialises_to_json(corpus):
    """Two runs are compared by diffing their reports, so the shape has to
    survive a round trip through a file."""
    index, database = corpus
    report = evaluate(
        [_label(query="bridge", video="a.mp4", start=1.0, end=1.0)],
        index,
        database,
        StubEmbedder(),
        ks=(1, 5),
        tolerance_sec=0.0,
    )
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["recall"]["1"] == 1.0
    assert payload["queries"] == 1
    assert payload["tolerance_sec"] == 0.0
    assert payload["outcomes"][0]["query"] == "bridge"
    assert payload["outcomes"][0]["rank"] == 1
    assert payload["outcomes"][0]["targets"][0]["video"] == "a.mp4"


def test_report_records_which_sampling_arm_built_the_corpus(corpus):
    """Two A/B reports are otherwise two numbers with no record of what they
    compare. The stub corpus has one scene_cut frame, so it reads as
    scene-aware; a store with none was built with --fixed-interval."""
    index, database = corpus
    report = evaluate([_label()], index, database, StubEmbedder(), tolerance_sec=0.0)
    assert report.sampling_arm == "scene-aware"
    assert report.to_dict()["sampling"]["frames_by_reason"]["scene_cut"] == 1

    database.conn.execute("UPDATE frames SET reason = 'baseline'")
    database.conn.commit()
    fixed = evaluate([_label()], index, database, StubEmbedder(), tolerance_sec=0.0)
    assert fixed.sampling_arm == "fixed-interval"


def test_describe_reports_every_k(corpus):
    index, database = corpus
    report = evaluate(
        [_label(query="bridge", video="a.mp4", start=1.0, end=1.0)],
        index,
        database,
        StubEmbedder(),
        ks=(1, 5, 10),
        tolerance_sec=0.0,
    )
    text = report.describe()

    assert "Recall@1" in text and "Recall@5" in text and "Recall@10" in text

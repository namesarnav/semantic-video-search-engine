"""Comparing two eval reports.

A/Bs on this project are paired: the same queries, two configurations. The
number that matters is not "which run scored higher" but *which queries
changed and in which direction* -- a two-point gap made of nine wins and seven
losses is a different animal from one made of two wins, even though the
headline delta is the same.
"""

from __future__ import annotations

import pytest

from sv_engine.evaluation import compare_reports


def report(outcomes: list[tuple[str, int | None]], ks=(1, 5)) -> dict:
    """Minimal report payload: (query, rank) pairs."""
    return {
        "queries": len(outcomes),
        "recall": {str(k): 0.0 for k in ks},
        "outcomes": [{"query": q, "rank": r} for q, r in outcomes],
        "model": "stub",
    }


def test_recall_is_recomputed_per_k_for_both_sides():
    result = compare_reports(
        report([("a", 1), ("b", None)]),
        report([("a", 1), ("b", 3)]),
        ks=(1, 5),
    )

    assert result.recall("baseline", 5) == 0.5
    assert result.recall("candidate", 5) == 1.0
    assert result.delta(5) == 0.5


def test_wins_and_losses_are_counted_separately():
    """A net of zero can hide a lot of churn."""
    result = compare_reports(
        report([("a", 1), ("b", None), ("c", 2)]),
        report([("a", None), ("b", 1), ("c", 2)]),
        ks=(5,),
    )

    assert result.gained(5) == ["b"]
    assert result.lost(5) == ["a"]
    assert result.delta(5) == 0.0


def test_queries_are_matched_by_text_not_by_position():
    """Two runs need not emit outcomes in the same order."""
    result = compare_reports(
        report([("a", 1), ("b", None)]),
        report([("b", 1), ("a", 1)]),
        ks=(5,),
    )

    assert result.gained(5) == ["b"]
    assert result.lost(5) == []


def test_a_rank_outside_k_counts_as_a_miss_at_that_k():
    result = compare_reports(
        report([("a", 9)]), report([("a", 2)]), ks=(1, 5)
    )

    assert result.recall("baseline", 5) == 0.0
    assert result.recall("candidate", 5) == 1.0
    assert result.gained(5) == ["a"]
    # Neither is inside the top 1, so nothing moved there.
    assert result.gained(1) == []


def test_comparing_different_query_sets_is_refused():
    """Silently intersecting them would report a delta over whichever queries
    happened to overlap, which is not the comparison anyone asked for."""
    with pytest.raises(ValueError, match="different quer"):
        compare_reports(report([("a", 1)]), report([("b", 1)]), ks=(5,))


def test_describe_names_the_queries_that_changed():
    result = compare_reports(
        report([("a", 1), ("keeps up", None)]),
        report([("a", 1), ("keeps up", 2)]),
        ks=(5,),
    )
    text = result.describe()

    assert "keeps up" in text
    assert "R@5" in text

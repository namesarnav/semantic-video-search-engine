"""Deriving an eval set from a fetched corpus manifest.

These labels are **weak supervision**, not hand-authored ground truth: the
query text comes from the uploader's own description of the clip, never from
looking at the frames. That is a real methodological difference from
`eval/labels.json`, and the point of these tests is that the difference stays
visible rather than being laundered into something that looks hand-checked.
"""

from __future__ import annotations

import json

import pytest

from sv_engine.manifest_labels import (
    ManifestError,
    labels_from_manifest,
    load_manifest,
    slug_of,
)


def entry(**over):
    base = {
        "query": "person cooking",
        "pexels_id": 1,
        "width": 426,
        "height": 240,
        "duration_sec": 20,
        "url": "https://www.pexels.com/video/person-measuring-flour-6287159/",
        "file": "6287159.mp4",
    }
    base.update(over)
    return base


def write(tmp_path, payload):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


# ---- slugs -------------------------------------------------------------


def test_slug_is_the_human_written_part_of_the_url():
    assert (
        slug_of("https://www.pexels.com/video/woman-cracking-eggs-8503205/")
        == "woman cracking eggs"
    )


def test_a_trailing_id_is_not_part_of_the_description():
    assert slug_of("https://www.pexels.com/video/stray-tabby-cat-123/") == (
        "stray tabby cat"
    )


def test_an_unrecognised_url_has_no_slug():
    assert slug_of("https://example.com/whatever") is None


# ---- building labels ---------------------------------------------------


def test_one_label_per_video(tmp_path):
    path = write(
        tmp_path,
        {
            "1": entry(url="https://www.pexels.com/video/a-man-riding-a-bike-1/", file="1.mp4"),
            "2": entry(url="https://www.pexels.com/video/a-cat-on-a-red-sofa-2/", file="2.mp4"),
        },
    )
    payload = labels_from_manifest(load_manifest(path))

    assert len(payload["queries"]) == 2
    assert {q["query"] for q in payload["queries"]} == {
        "a man riding a bike",
        "a cat on a red sofa",
    }


def test_the_target_is_the_whole_video(tmp_path):
    """We know which video the description belongs to, not which second of it.
    Claiming a timestamp we never checked would be inventing precision."""
    path = write(
        tmp_path,
        {"1": entry(url="https://www.pexels.com/video/a-man-riding-a-bike-1/", duration_sec=17, file="1.mp4")},
    )
    target = labels_from_manifest(load_manifest(path))["queries"][0]["targets"][0]

    assert target["video"] == "1.mp4"
    assert target["start_sec"] == 0.0
    assert target["end_sec"] == 17


def test_videos_sharing_a_description_become_one_multi_target_label(tmp_path):
    """Two clips both called "cat grooming" are both correct answers. Emitting
    two labels would mean each scores the other as a miss."""
    path = write(
        tmp_path,
        {
            "1": entry(url="https://www.pexels.com/video/cat-grooming-itself-1/", file="1.mp4"),
            "2": entry(url="https://www.pexels.com/video/cat-grooming-itself-2/", file="2.mp4"),
        },
    )
    payload = labels_from_manifest(load_manifest(path))

    assert len(payload["queries"]) == 1
    assert {t["video"] for t in payload["queries"][0]["targets"]} == {"1.mp4", "2.mp4"}


def test_a_description_that_merely_repeats_its_category_is_dropped(tmp_path):
    """"cat sleeping" is the search term all twenty cat clips were fetched
    with, so it identifies a category, not an item. Scoring it as known-item
    retrieval would count nineteen correct videos as failures."""
    path = write(
        tmp_path,
        {
            "1": entry(query="cat sleeping", url="https://www.pexels.com/video/cat-sleeping-1/", file="1.mp4"),
            "2": entry(query="cat sleeping", url="https://www.pexels.com/video/tabby-asleep-on-a-radiator-2/", file="2.mp4"),
        },
    )
    payload = labels_from_manifest(load_manifest(path))

    assert [q["query"] for q in payload["queries"]] == [
        "tabby asleep on a radiator"
    ]


def test_very_short_descriptions_are_dropped(tmp_path):
    """A one- or two-word slug is a topic, not a description of one clip."""
    path = write(
        tmp_path,
        {
            "1": entry(url="https://www.pexels.com/video/regentropfen-1/", file="1.mp4"),
            "2": entry(url="https://www.pexels.com/video/a-woman-typing-on-a-laptop-2/", file="2.mp4"),
        },
    )
    payload = labels_from_manifest(load_manifest(path))

    assert [q["query"] for q in payload["queries"]] == ["a woman typing on a laptop"]


def test_entries_without_a_usable_slug_are_skipped_not_fatal(tmp_path):
    path = write(
        tmp_path,
        {
            "1": entry(url="https://example.com/nope", file="1.mp4"),
            "2": entry(url="https://www.pexels.com/video/a-dog-in-the-snow-2/", file="2.mp4"),
        },
    )
    assert len(labels_from_manifest(load_manifest(path))["queries"]) == 1


def test_the_output_records_that_it_is_generated_and_weakly_supervised(tmp_path):
    """This set must never be mistaken for the hand-checked one."""
    path = write(
        tmp_path,
        {"1": entry(url="https://www.pexels.com/video/a-man-riding-a-bike-1/", file="1.mp4")},
    )
    readme = labels_from_manifest(load_manifest(path))["_readme"].lower()

    assert "generated" in readme
    assert "weak" in readme or "not hand" in readme


def test_output_loads_through_the_strict_label_loader(tmp_path):
    """The generator must produce the same shape a human would hand-write."""
    from sv_engine.evaluation import load_labels

    path = write(
        tmp_path,
        {
            "1": entry(url="https://www.pexels.com/video/a-man-riding-a-bike-1/", file="1.mp4"),
            "2": entry(url="https://www.pexels.com/video/a-cat-on-a-red-sofa-2/", file="2.mp4"),
        },
    )
    out = tmp_path / "labels.json"
    out.write_text(json.dumps(labels_from_manifest(load_manifest(path))))

    labels = load_labels(out)
    assert len(labels) == 2


def test_a_manifest_with_nothing_usable_is_an_error(tmp_path):
    """Silently emitting zero queries would produce an eval set that reports
    100% of nothing."""
    path = write(tmp_path, {"1": entry(url="https://example.com/nope", file="1.mp4")})
    with pytest.raises(ManifestError, match="no usable"):
        labels_from_manifest(load_manifest(path))


def test_a_missing_manifest_says_so(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.json")

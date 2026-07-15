"""The cut-dense A/B corpus: its schedule, its ground truth, and coverage.

This corpus exists to make the sampling A/B resolvable. The real corpus has
three scene-cut frames out of 87, so both arms score identically there and the
comparison says nothing. Everything here is checked without loading CLIP.
"""

from __future__ import annotations

import cv2
import pytest

from sv_engine import config
from sv_engine.cutdense import (
    SUBJECTS,
    TREATMENTS,
    Shot,
    build_schedule,
    labels_payload,
    manifest_payload,
    render,
)
from sv_engine.evaluation import load_labels, shot_coverage
from sv_engine.sampler import sample_video


# ---- coverage ----------------------------------------------------------


def test_coverage_is_full_when_every_shot_has_a_sample():
    shots = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    report = shot_coverage([0.5, 1.5, 2.5], shots)

    assert report.fraction == 1.0
    assert report.missed == ()


def test_coverage_names_the_shots_that_were_missed():
    """The point of the measure: which shots fell through the sampling grid."""
    shots = [(0.0, 1.0), (1.0, 1.5), (1.5, 3.0)]
    report = shot_coverage([0.5, 2.0], shots)

    assert report.missed == (1,)
    assert report.covered == (0, 2)
    assert report.fraction == pytest.approx(2 / 3)


def test_a_shot_is_half_open_so_a_boundary_sample_belongs_to_one_shot():
    """Shots abut, so a sample landing exactly on a boundary must not be
    counted for both -- that would inflate coverage at every cut."""
    shots = [(0.0, 1.0), (1.0, 2.0)]
    report = shot_coverage([1.0], shots)

    assert report.covered == (1,)
    assert report.missed == (0,)


def test_coverage_of_nothing_is_zero_not_a_crash():
    assert shot_coverage([], [(0.0, 1.0)]).fraction == 0.0
    assert shot_coverage([0.5], []).fraction == 0.0


# ---- the schedule ------------------------------------------------------


def test_every_subject_treatment_pair_appears_exactly_once():
    """Uniqueness is what lets a label name one shot. If a pair repeated, its
    query would have two correct answers and both arms would trivially hit."""
    schedule = build_schedule()
    pairs = [(s.subject, s.treatment) for s in schedule]

    assert len(pairs) == len(SUBJECTS) * len(TREATMENTS)
    assert len(set(pairs)) == len(pairs)


def test_adjacent_shots_never_share_a_subject():
    """A cut is detected from a histogram change. Cutting from a subject to
    itself is the weakest possible cut, and would test the detector's floor
    rather than the sampling strategy."""
    schedule = build_schedule()
    assert all(a.subject != b.subject for a, b in zip(schedule, schedule[1:]))


def test_shots_abut_exactly_with_no_gaps():
    schedule = build_schedule()
    assert schedule[0].start_sec == 0.0
    assert all(a.end_sec == b.start_sec for a, b in zip(schedule, schedule[1:]))


def test_every_boundary_lands_on_a_whole_frame():
    """Regression. The schedule used to round durations to 0.01s while render
    wrote round(duration * fps) whole frames, so the real boundaries drifted
    from the declared ones cumulatively. Ground truth then pointed at the
    *neighbouring* shot's content: coverage counted a frame the query could
    never match, and the A/B would have read as "scene-aware covers more shots
    but retrieves none of them"."""
    fps = 30
    schedule = build_schedule(fps=fps)
    for shot in schedule:
        assert shot.start_sec * fps == pytest.approx(round(shot.start_sec * fps))
        assert shot.end_sec * fps == pytest.approx(round(shot.end_sec * fps))
        assert shot.frame_count == round(shot.duration_sec * fps)


def test_brief_shots_are_shorter_than_the_baseline_interval():
    """These are the discriminating cases: a 1 fps tick often misses them."""
    brief = [s for s in build_schedule() if s.brief]

    assert brief, "the corpus is pointless without sub-interval shots"
    assert all(s.duration_sec < 1.0 for s in brief)


def test_no_shot_is_shorter_than_the_cut_throttle():
    """`sample_video` throttles cut-against-cut by MIN_SAMPLE_GAP_SEC. A shot
    shorter than that would have its own opening cut suppressed, so the
    scene-aware arm would miss it too and the A/B would measure the throttle
    rather than the strategy."""
    assert all(
        s.duration_sec > config.MIN_SAMPLE_GAP_SEC for s in build_schedule()
    )


def test_a_schedule_violating_the_throttle_is_rejected():
    with pytest.raises(ValueError, match="MIN_SAMPLE_GAP|throttle"):
        build_schedule(brief_range=(0.1, 0.2))


def test_the_schedule_is_deterministic():
    """The corpus is regenerated from this script, so two runs must agree or
    the committed labels stop describing the video."""
    assert build_schedule() == build_schedule()
    assert build_schedule(seed=1) != build_schedule(seed=2)


# ---- generated ground truth --------------------------------------------


def test_labels_have_one_exact_target_each(tmp_path):
    """No tolerance baked into the range: the boundaries are exact by
    construction, and widening them here would overlap adjacent shots."""
    schedule = build_schedule()
    path = tmp_path / "labels.json"
    import json

    path.write_text(json.dumps(labels_payload(schedule, "cut_dense.mp4")))
    labels = load_labels(path)  # the strict loader is the validator

    assert len(labels) == len(schedule)
    for label, shot in zip(labels, schedule):
        assert len(label.targets) == 1
        assert label.targets[0].video == "cut_dense.mp4"
        assert label.targets[0].start_sec == shot.start_sec
        assert label.targets[0].end_sec == shot.end_sec


def test_every_query_is_distinct():
    payload = labels_payload(build_schedule(), "cut_dense.mp4")
    queries = [q["query"] for q in payload["queries"]]
    assert len(set(queries)) == len(queries)


def test_manifest_carries_the_boundaries_coverage_needs():
    manifest = manifest_payload(build_schedule())
    shots = [(s["start_sec"], s["end_sec"]) for s in manifest["shots"]]

    assert shots[0][0] == 0.0
    assert all(a[1] == b[0] for a, b in zip(shots, shots[1:]))


# ---- rendering ---------------------------------------------------------


@pytest.fixture
def tiny_sources(synthetic_video):
    """One small, flat-coloured clip per subject, long enough to draw from."""
    return {
        subject: synthetic_video(
            path_name=f"{subject}.mp4", duration_sec=6.0, cut_at_sec=99.0
        )
        for subject in SUBJECTS
    }


def test_render_refuses_a_schedule_built_for_a_different_fps(tmp_path, tiny_sources):
    """Frame counts come from the schedule, so a mismatch would silently
    reintroduce the drift this corpus was fixed for."""
    schedule = build_schedule(fps=30, brief=1, sustained=1)
    with pytest.raises(ValueError, match="fps"):
        render(schedule, tiny_sources, tmp_path / "o.mp4", fps=15, size=(160, 90))


def test_render_produces_the_declared_duration(tmp_path, tiny_sources):
    schedule = build_schedule(fps=15, brief=2, sustained=2)
    out = tmp_path / "cut_dense.mp4"
    render(schedule, tiny_sources, out, fps=15, size=(160, 90))

    capture = cv2.VideoCapture(str(out))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()

    assert out.is_file()
    assert frames / fps == pytest.approx(schedule[-1].end_sec, abs=0.2)


def test_declared_boundaries_are_the_rendered_ones(tmp_path, tiny_sources):
    """Two claims, one render: the cuts the corpus declares are detectable at
    all, and they land where the manifest says to the frame.

    If the first fails the A/B measures the detector rather than the strategy.
    If the second fails the ground truth points at the neighbouring shot's
    content -- the drift bug this corpus was fixed for."""
    schedule = build_schedule(fps=15, brief=2, sustained=2)
    out = tmp_path / "cut_dense.mp4"
    render(schedule, tiny_sources, out, fps=15, size=(160, 90))

    cuts = [
        f.timestamp_sec
        for f in sample_video(out, baseline_fps=1.0)
        if f.reason == "scene_cut"
    ]
    for boundary in [s.start_sec for s in schedule[1:]]:
        assert any(abs(c - boundary) <= 1.5 / 15 for c in cuts), (
            f"no cut detected at {boundary}s; got {cuts}"
        )


def test_a_missing_source_clip_is_reported_not_silently_skipped(tmp_path):
    schedule = build_schedule(fps=15, brief=1, sustained=1)
    with pytest.raises(FileNotFoundError):
        render(schedule, {}, tmp_path / "out.mp4", fps=15, size=(160, 90))

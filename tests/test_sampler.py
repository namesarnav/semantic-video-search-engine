from __future__ import annotations

import pytest

from sv_engine import sampler


def test_fixed_interval_sampling_hits_expected_timestamps(synthetic_video):
    """1 fps over a 6s clip should land one sample per second, starting at 0."""
    path = synthetic_video()
    frames = list(
        sampler.sample_video(path, baseline_fps=1.0, scene_threshold=None)
    )

    assert [f.reason for f in frames] == ["baseline"] * len(frames)
    assert len(frames) == 6
    assert [round(f.timestamp_sec) for f in frames] == [0, 1, 2, 3, 4, 5]


def test_baseline_rate_scales_with_fps(synthetic_video):
    path = synthetic_video()
    dense = list(sampler.sample_video(path, baseline_fps=2.0, scene_threshold=None))
    sparse = list(sampler.sample_video(path, baseline_fps=0.5, scene_threshold=None))

    assert len(dense) > len(sparse)
    assert len(dense) == 12
    assert len(sparse) == 3


def test_scene_cut_adds_a_sample_between_baseline_ticks(synthetic_video):
    """A cut at 2.5s falls between 1 fps ticks, so it is only captured by
    scene-awareness -- which is the entire justification for the strategy."""
    path = synthetic_video(cut_at_sec=2.5)

    fixed = list(sampler.sample_video(path, baseline_fps=1.0, scene_threshold=None))
    aware = list(sampler.sample_video(path, baseline_fps=1.0, scene_threshold=0.35))

    cuts = [f for f in aware if f.reason == "scene_cut"]
    assert len(cuts) == 1
    assert cuts[0].timestamp_sec == pytest.approx(2.5, abs=0.15)
    assert len(aware) == len(fixed) + 1


def test_min_gap_suppresses_a_burst_of_rapid_cuts(synthetic_video):
    """Cuts in quick succession collapse to one sample.

    This is min_gap's actual job: a fast-cutting or strobing sequence should
    not emit a sample per cut.
    """
    path = synthetic_video(cut_at_sec=[2.2, 2.4, 2.6])

    permissive = list(
        sampler.sample_video(path, baseline_fps=1.0, scene_threshold=0.35, min_gap_sec=0.05)
    )
    throttled = list(
        sampler.sample_video(path, baseline_fps=1.0, scene_threshold=0.35, min_gap_sec=0.5)
    )

    assert len([f for f in permissive if f.reason == "scene_cut"]) == 3
    assert len([f for f in throttled if f.reason == "scene_cut"]) == 1


def test_cut_shortly_after_a_baseline_tick_is_still_captured(synthetic_video):
    """Regression: min_gap must throttle cut-against-cut, never cut-against-baseline.

    A cut 0.1s after a baseline sample is the *least* duplicative frame there
    is -- the picture just changed. Suppressing it leaves the opening of every
    new scene unrepresented until the next tick.
    """
    path = synthetic_video(cut_at_sec=3.1)

    frames = list(
        sampler.sample_video(path, baseline_fps=1.0, scene_threshold=0.35, min_gap_sec=0.9)
    )
    cuts = [f for f in frames if f.reason == "scene_cut"]

    assert len(cuts) == 1
    assert cuts[0].timestamp_sec == pytest.approx(3.1, abs=0.15)


def test_every_cut_is_labelled_when_it_falls_between_baseline_ticks(synthetic_video):
    """Cuts at varied offsets from the 1s grid must all be flagged."""
    expected = [2.05, 5.5, 8.45]
    path = synthetic_video(duration_sec=12.0, cut_at_sec=expected)

    frames = list(
        sampler.sample_video(path, baseline_fps=1.0, scene_threshold=0.35, min_gap_sec=0.4)
    )
    cuts = sorted(f.timestamp_sec for f in frames if f.reason == "scene_cut")

    assert len(cuts) == 3
    for found, want in zip(cuts, expected):
        assert found == pytest.approx(want, abs=0.15)


def test_new_scene_is_sampled_promptly(synthetic_video):
    """The property that actually matters for retrieval: every scene change is
    followed by a sample almost immediately, so no scene goes unrepresented.

    Asserted on sample *coverage* rather than on the ``scene_cut`` label,
    because a cut coinciding with a baseline tick is sampled but labelled
    ``baseline`` -- indexed either way, which is what retrieval cares about.
    """
    cuts = [2.05, 4.0, 6.45, 8.95]
    path = synthetic_video(duration_sec=12.0, cut_at_sec=cuts)

    frames = list(
        sampler.sample_video(path, baseline_fps=1.0, scene_threshold=0.35, min_gap_sec=0.4)
    )
    times = [f.timestamp_sec for f in frames]

    for cut in cuts:
        following = [t for t in times if t >= cut - 0.05]
        assert following, f"no sample at all after the cut at {cut}s"
        assert following[0] - cut < 0.2, (
            f"scene starting at {cut}s was not sampled until {following[0]}s"
        )


def test_scene_distance_is_zero_for_identical_frames(synthetic_video):
    path = synthetic_video()
    frames = [f.image for f in sampler.sample_video(path, baseline_fps=10.0, scene_threshold=None)]

    assert sampler.scene_distance(frames[0], frames[0]) == pytest.approx(0.0, abs=1e-6)
    # First and last frames straddle the cut, so they must differ substantially.
    assert sampler.scene_distance(frames[0], frames[-1]) > 0.5


def test_unopenable_file_raises(tmp_path):
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_bytes(b"definitely not a video")

    with pytest.raises(ValueError):
        list(sampler.sample_video(bogus))


def test_duration_is_reported(synthetic_video):
    path = synthetic_video(duration_sec=6.0)
    assert sampler.video_duration_sec(path) == pytest.approx(6.0, abs=0.2)

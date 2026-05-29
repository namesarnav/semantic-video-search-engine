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


def test_min_gap_suppresses_burst_of_cuts(synthetic_video):
    """With a large min gap, a cut landing just after a baseline tick is
    dropped rather than emitted as a near-duplicate."""
    path = synthetic_video(cut_at_sec=3.1)

    permissive = list(
        sampler.sample_video(path, baseline_fps=1.0, scene_threshold=0.35, min_gap_sec=0.05)
    )
    strict = list(
        sampler.sample_video(path, baseline_fps=1.0, scene_threshold=0.35, min_gap_sec=0.9)
    )

    assert any(f.reason == "scene_cut" for f in permissive)
    assert not any(f.reason == "scene_cut" for f in strict)


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

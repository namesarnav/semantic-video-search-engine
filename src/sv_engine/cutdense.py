"""Builds a cut-dense video whose shot boundaries are known by construction.

The real corpus cannot resolve the sampling A/B. Scene-aware sampling
contributes three frames out of 87 there -- only the compilation clip has any
detected cuts, and the other four videos are single continuous shots. Both arms
score identically, which says nothing about the design decision.

The claim under test is that *sparse sampling misses short, visually distinct
moments*. So this builds footage made of exactly that: short shots, cut
together, with the boundaries recorded rather than eyeballed. Ground truth by
construction is stronger than hand-labelling here -- there is no clock-reading
error to absorb, which is why the A/B runs at tolerance 0.

**Shots must be uniquely addressable.** Four source subjects alone would mean
"the golden gate bridge" matches every bridge shot, so both arms hit trivially
and nothing is measured. Crossing four subjects with four visual treatments
gives sixteen classes that each appear exactly once, so a label can name one
shot and only that shot. The treatments are a device for addressability, not a
claim that people search by colour grade -- and because the A/B is a paired
comparison over identical queries, a treatment CLIP cannot see costs power
rather than validity.

This lives in the package rather than in `scripts/` only so it can be imported
and tested; `scripts/make_cut_dense_corpus.py` is a thin wrapper over it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from . import config

# Source clip per subject, and the noun phrase that describes it. The
# filenames are the four standalone videos in data/videos.
SUBJECTS: dict[str, dict[str, str]] = {
    "bridge": {
        "file": "15342677_3840_2160_30fps.mp4",
        "phrase": "a red suspension bridge over water",
    },
    "ridge": {
        "file": "15549988_3840_2160_30fps.mp4",
        "phrase": "a snow covered mountain ridge",
    },
    "sunset": {
        "file": "16504192_3840_2160_30fps.mp4",
        "phrase": "a sunset over a dark mountain",
    },
    "motorcycle": {
        "file": "16615911_3840_2160_59fps.mp4",
        "phrase": "riding a motorcycle down a forest road",
    },
}


def _identity(frame: np.ndarray) -> np.ndarray:
    return frame


def _grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _sepia(frame: np.ndarray) -> np.ndarray:
    # Standard sepia matrix, applied in RGB order then written back as BGR.
    kernel = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    return cv2.transform(frame, kernel).clip(0, 255).astype(np.uint8)


def _blue(frame: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32)
    out[:, :, 0] *= 1.6  # B
    out[:, :, 2] *= 0.5  # R
    return out.clip(0, 255).astype(np.uint8)


# Each treatment shifts the HSV histogram as well as the description, so a cut
# between two treatments of the same subject would still be detectable -- but
# the schedule avoids that case anyway (see build_schedule).
TREATMENTS: dict[str, dict] = {
    "colour": {"phrase": "a colour photo of", "fn": _identity},
    "bw": {"phrase": "a black and white photo of", "fn": _grayscale},
    "sepia": {"phrase": "a sepia toned photo of", "fn": _sepia},
    "blue": {"phrase": "a blue tinted photo of", "fn": _blue},
}

# Brief shots are the discriminating cases. A 1 fps baseline tick lands inside
# a shot of duration d starting at t only when ceil(t) < t + d, so sub-interval
# shots are missed outright a large fraction of the time.
BRIEF_RANGE = (0.5, 0.9)
# Sustained shots always contain a tick. They are the control: the two arms
# should agree here, which is what makes a gap on the brief shots meaningful.
SUSTAINED_RANGE = (2.0, 3.5)

DEFAULT_SEED = 20260713
DEFAULT_FPS = 30


@dataclass(frozen=True)
class Shot:
    index: int
    subject: str
    treatment: str
    start_sec: float
    end_sec: float
    brief: bool
    # Whole frames, and the authority on this shot's length. Boundaries are
    # derived from cumulative frame counts rather than from seconds, so the
    # declared timestamps *are* the rendered ones -- see build_schedule.
    frame_count: int
    fps: int

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def query(self) -> str:
        return f"{TREATMENTS[self.treatment]['phrase']} {SUBJECTS[self.subject]['phrase']}"


def build_schedule(
    *,
    fps: int = DEFAULT_FPS,
    seed: int = DEFAULT_SEED,
    brief: int | None = None,
    sustained: int | None = None,
    brief_range: tuple[float, float] = BRIEF_RANGE,
    sustained_range: tuple[float, float] = SUSTAINED_RANGE,
) -> list[Shot]:
    """Lay out the shots. Deterministic for a given seed.

    Every duration is a whole number of frames, and boundaries are computed
    from cumulative frame counts. This is not tidiness: an earlier version
    rounded durations to 1/100s while the renderer wrote whole frames, so the
    real boundaries drifted from the declared ones and the ground truth ended
    up pointing at the *neighbouring* shot's content. Coverage then counted
    frames the query could never match, and the A/B read as "scene-aware
    covers more shots and retrieves none of them" -- a conclusion entirely
    manufactured by the misalignment.

    Determinism matters because the labels are committed while the video is
    regenerated from this script: if two runs disagreed, the committed ground
    truth would stop describing the footage it is scored against.

    Subjects walk cyclically while treatments advance, which covers every
    (subject, treatment) pair exactly once *and* guarantees adjacent shots
    differ in subject. Cutting from a subject to itself is the weakest possible
    cut, and would test the detector's floor rather than the sampling strategy.
    """
    subjects = list(SUBJECTS)
    treatments = list(TREATMENTS)
    total = len(subjects) * len(treatments)

    pairs = [
        (subjects[i % len(subjects)], treatments[(i // len(subjects) + i) % len(treatments)])
        for i in range(total)
    ]
    if len(set(pairs)) != total:  # pragma: no cover - guards the walk above
        raise ValueError("the subject/treatment walk did not cover every pair once")

    n_brief = len(pairs) // 2 if brief is None else brief
    n_sustained = len(pairs) - n_brief if sustained is None else sustained
    pairs = pairs[: n_brief + n_sustained]

    rng = random.Random(seed)
    flags = [True] * n_brief + [False] * n_sustained
    rng.shuffle(flags)

    shots: list[Shot] = []
    cursor_frames = 0
    for index, ((subject, treatment), is_brief) in enumerate(zip(pairs, flags)):
        low, high = brief_range if is_brief else sustained_range
        frame_count = max(1, round(rng.uniform(low, high) * fps))
        duration = frame_count / fps
        if duration <= config.MIN_SAMPLE_GAP_SEC:
            raise ValueError(
                f"shot {index} is {duration:g}s, at or under MIN_SAMPLE_GAP_SEC "
                f"({config.MIN_SAMPLE_GAP_SEC:g}s). sample_video throttles "
                "cut-against-cut, so its opening cut would be suppressed and "
                "the scene-aware arm would miss it too -- the A/B would be "
                "measuring the throttle, not the strategy."
            )
        shots.append(
            Shot(
                index=index,
                subject=subject,
                treatment=treatment,
                start_sec=cursor_frames / fps,
                end_sec=(cursor_frames + frame_count) / fps,
                brief=is_brief,
                frame_count=frame_count,
                fps=fps,
            )
        )
        cursor_frames += frame_count
    return shots


def labels_payload(schedule: Sequence[Shot], video_filename: str) -> dict:
    """Ground truth in the format `load_labels` validates.

    One target per label, with the shot's exact boundaries. No padding: the
    boundaries are exact by construction, and widening them would overlap the
    neighbouring shot, so a frame from the wrong shot would score as a hit.
    Run this eval set at tolerance 0.
    """
    return {
        "_readme": (
            "GENERATED by scripts/make_cut_dense_corpus.py -- do not hand-edit. "
            "Score with --tolerance 0; the ranges are exact and abut."
        ),
        "queries": [
            {
                "query": shot.query,
                "note": (
                    f"shot {shot.index}: {shot.treatment} {shot.subject}, "
                    f"{shot.frame_count} frames "
                    f"({'brief' if shot.brief else 'sustained'})"
                ),
                "targets": [
                    {
                        "video": video_filename,
                        "start_sec": shot.start_sec,
                        "end_sec": shot.end_sec,
                    }
                ],
            }
            for shot in schedule
        ],
    }


def manifest_payload(schedule: Sequence[Shot]) -> dict:
    """Shot boundaries for `shot_coverage`, which works on the sampler alone."""
    return {
        "_readme": "GENERATED by scripts/make_cut_dense_corpus.py -- do not hand-edit.",
        "shots": [
            {
                "index": s.index,
                "subject": s.subject,
                "treatment": s.treatment,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "brief": s.brief,
            }
            for s in schedule
        ],
    }


def render(
    schedule: Sequence[Shot],
    sources: Mapping[str, Path | str],
    out_path: Path | str,
    *,
    fps: int = 30,
    size: tuple[int, int] = (854, 480),
) -> Path:
    """Write the cut-dense video.

    Sources are 4K; full resolution buys nothing for a retrieval test and makes
    every ingest of this corpus slower, so frames are downscaled.

    Each shot reads from a different offset in its source clip so repeated
    subjects are not frame-identical, and loops the source if it runs short.
    """
    out_path = Path(out_path)
    mismatched = sorted({s.fps for s in schedule} - {fps})
    if mismatched:
        raise ValueError(
            f"schedule was built for fps {mismatched} but render was asked for "
            f"{fps}. Frame counts come from the schedule, so this would put the "
            "declared boundaries back out of step with the rendered ones."
        )
    missing = sorted({s.subject for s in schedule} - set(sources))
    if missing:
        raise FileNotFoundError(f"no source clip for subject(s): {missing}")
    for subject, path in sources.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"source clip for {subject} not found: {path}")

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    try:
        for shot in schedule:
            treat: Callable[[np.ndarray], np.ndarray] = TREATMENTS[shot.treatment]["fn"]
            for frame in _read_frames(
                sources[shot.subject], shot.frame_count, offset=shot.index
            ):
                writer.write(treat(cv2.resize(frame, size)))
    finally:
        writer.release()
    return out_path


def _read_frames(
    path: Path | str, count: int, *, offset: int = 0
) -> Iterable[np.ndarray]:
    """Yield ``count`` frames, looping the clip if it is too short."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open source clip: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        capture.set(cv2.CAP_PROP_POS_FRAMES, (offset * 7) % total)
        last: np.ndarray | None = None
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                if last is None:
                    raise ValueError(f"source clip yielded no frames: {path}")
                frame = last
            last = frame
            yield frame
    finally:
        capture.release()

"""Frame sampling.

The core ingestion trade-off: sampling densely (every frame) produces mostly
near-duplicate embeddings and wastes storage and search time; sampling sparsely
(every 10s) misses short, visually distinct moments.

The strategy here is scene-change-aware: take a fixed baseline rate, and insert
*extra* samples where the picture actually changes. Both strategies are exposed
so they can be A/B'd against Recall@K rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np

from . import config

SampleReason = Literal["baseline", "scene_cut"]

# Frames are downscaled to this width before histogram comparison. Scene cuts
# are a global property of the picture, so comparing 4K pixels buys nothing and
# costs a lot.
_DIFF_WIDTH = 160


@dataclass(frozen=True)
class SampledFrame:
    """One sampled frame, still in memory as BGR pixels."""

    timestamp_sec: float
    frame_index: int
    image: np.ndarray
    reason: SampleReason


def _histogram(frame: np.ndarray) -> np.ndarray:
    """Normalized HSV histogram of a downscaled frame."""
    height = max(1, int(frame.shape[0] * (_DIFF_WIDTH / frame.shape[1])))
    small = cv2.resize(frame, (_DIFF_WIDTH, height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def scene_distance(prev: np.ndarray, curr: np.ndarray) -> float:
    """Bhattacharyya distance between two frame histograms, in [0, 1].

    0 means identical; larger means the picture changed. Bhattacharyya is used
    over correlation because it is already bounded, so the threshold is
    interpretable without per-video calibration.
    """
    return float(cv2.compareHist(_histogram(prev), _histogram(curr), cv2.HISTCMP_BHATTACHARYYA))


def sample_video(
    path: Path | str,
    *,
    baseline_fps: float = config.BASELINE_FPS,
    scene_threshold: float | None = config.SCENE_THRESHOLD,
    min_gap_sec: float = config.MIN_SAMPLE_GAP_SEC,
) -> Iterator[SampledFrame]:
    """Yield sampled frames from a video, in timestamp order.

    Set ``scene_threshold=None`` for pure fixed-interval sampling -- that is the
    control arm when measuring whether scene-awareness actually improves recall.

    ``min_gap_sec`` stops a fast-cutting or shaky sequence from emitting a burst
    of near-identical scene-cut samples.
    """
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or not np.isfinite(fps):
            raise ValueError(f"video reports unusable fps ({fps!r}): {path}")

        baseline_interval = 1.0 / baseline_fps if baseline_fps > 0 else float("inf")

        prev_frame: np.ndarray | None = None
        last_emit_ts = -float("inf")
        next_baseline_ts = 0.0
        frame_index = -1

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            timestamp = frame_index / fps

            reason: SampleReason | None = None
            if timestamp >= next_baseline_ts:
                reason = "baseline"
            elif scene_threshold is not None and prev_frame is not None:
                if scene_distance(prev_frame, frame) >= scene_threshold:
                    reason = "scene_cut"

            # Only the previous *compared* frame matters for cut detection, so
            # this must update every iteration, not only when we emit.
            prev_frame = frame

            if reason is None:
                continue
            if timestamp - last_emit_ts < min_gap_sec:
                continue

            last_emit_ts = timestamp
            if reason == "baseline":
                # Advance past the current timestamp rather than by a single
                # step, so a low baseline_fps on a long video cannot drift.
                while next_baseline_ts <= timestamp:
                    next_baseline_ts += baseline_interval

            yield SampledFrame(
                timestamp_sec=timestamp,
                frame_index=frame_index,
                image=frame.copy(),
                reason=reason,
            )
    finally:
        capture.release()


def video_duration_sec(path: Path | str) -> float:
    """Duration in seconds, or 0.0 if the container does not report it."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and frames and fps > 0 and frames > 0:
            return float(frames / fps)
        return 0.0
    finally:
        capture.release()

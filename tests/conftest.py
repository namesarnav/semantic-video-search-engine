from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path):
    """A 6s/10fps video with exactly one hard scene cut at t=3.0s.

    Two flat, very different colours means the cut is unambiguous, so a test
    asserting on cut detection is testing the sampler's logic rather than the
    sensitivity of a threshold to noisy real footage.
    """

    def _make(
        path_name: str = "clip.mp4",
        fps: int = 10,
        duration_sec: float = 6.0,
        cut_at_sec: float = 3.0,
        size: tuple[int, int] = (128, 72),
    ) -> Path:
        path = tmp_path / path_name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        assert writer.isOpened(), "could not open VideoWriter"

        total = int(fps * duration_sec)
        for i in range(total):
            colour = (200, 30, 30) if i / fps < cut_at_sec else (30, 30, 200)
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            frame[:, :] = colour
            writer.write(frame)
        writer.release()
        return path

    return _make

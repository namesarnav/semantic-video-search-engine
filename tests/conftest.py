from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


# Flat, well-separated colours. Each cut switches to the next one, so every
# scene change is unambiguous and a test failure means the sampler's logic is
# wrong rather than that a threshold was borderline on noisy footage.
_PALETTE = [
    (200, 30, 30),
    (30, 30, 200),
    (30, 200, 30),
    (200, 200, 30),
    (200, 30, 200),
]


@pytest.fixture
def synthetic_video(tmp_path: Path):
    """Build a video with hard scene cuts at known timestamps."""

    def _make(
        path_name: str = "clip.mp4",
        fps: int = 10,
        duration_sec: float = 6.0,
        cut_at_sec: float | list[float] = 3.0,
        size: tuple[int, int] = (128, 72),
    ) -> Path:
        cuts = [cut_at_sec] if isinstance(cut_at_sec, (int, float)) else list(cut_at_sec)
        cuts = sorted(cuts)

        path = tmp_path / path_name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        assert writer.isOpened(), "could not open VideoWriter"

        total = int(fps * duration_sec)
        for i in range(total):
            timestamp = i / fps
            # Scene index = how many cuts we have passed.
            scene = sum(1 for c in cuts if timestamp >= c)
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            frame[:, :] = _PALETTE[scene % len(_PALETTE)]
            writer.write(frame)
        writer.release()
        return path

    return _make

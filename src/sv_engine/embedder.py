"""CLIP embedding.

Images and text land in the same vector space -- that shared space is the whole
reason CLIP is the right model for this system. A text query embedded here is
directly comparable to a frame embedded here.

All vectors are L2-normalized on the way out, which makes inner product on the
FAISS side equivalent to cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from . import config


class ClipEmbedder:
    """Wraps an open_clip checkpoint and its preprocessing."""

    def __init__(
        self,
        model_name: str = config.CLIP_MODEL,
        pretrained: str = config.CLIP_PRETRAINED,
        device: str | None = None,
    ) -> None:
        self.device = device or config.resolve_device()
        self.model_name = model_name
        self.pretrained = pretrained

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            probe = self.model.encode_text(self.tokenizer(["probe"]).to(self.device))
        self.dim = int(probe.shape[-1])

    @staticmethod
    def _to_pil(frame: np.ndarray) -> Image.Image:
        """OpenCV hands back BGR; CLIP's preprocessing expects RGB."""
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _normalize(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True)
        return tensor.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_images(
        self, frames: Sequence[np.ndarray], batch_size: int = config.EMBED_BATCH_SIZE
    ) -> np.ndarray:
        """Embed BGR frames. Returns (len(frames), dim) float32, normalized."""
        if not frames:
            return np.empty((0, self.dim), dtype=np.float32)

        out: list[np.ndarray] = []
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            batch = torch.stack([self.preprocess(self._to_pil(f)) for f in chunk])
            features = self.model.encode_image(batch.to(self.device))
            out.append(self._normalize(features))
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_text(self, queries: Sequence[str]) -> np.ndarray:
        """Embed text queries. Returns (len(queries), dim) float32, normalized."""
        if not queries:
            return np.empty((0, self.dim), dtype=np.float32)
        tokens = self.tokenizer(list(queries)).to(self.device)
        return self._normalize(self.model.encode_text(tokens))


@lru_cache(maxsize=1)
def get_embedder() -> ClipEmbedder:
    """Process-wide embedder. Loading the checkpoint is slow; do it once."""
    return ClipEmbedder()

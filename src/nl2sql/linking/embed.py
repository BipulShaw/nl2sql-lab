"""Sentence embeddings for schema linking: bge-small-en-v1.5 (a 33M-parameter bi-encoder), pinned, on the CPU.

A bi-encoder embeds the question and every schema doc separately, so the docs of a database are embedded
once and each question costs one forward pass; similarity is then a dot product of unit vectors (cosine)."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# bge v1.5 was trained to embed short retrieval queries with this prefix and the passages without one
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """One L2-normalized row per text."""
        ...


class BgeEmbedder:
    def __init__(
        self, model_id: str = EMBED_MODEL, revision: str = EMBED_REVISION, device: str = "cpu"
    ) -> None:
        from sentence_transformers import SentenceTransformer  # the ml extra; CI tests use a fake embedder

        self.model = SentenceTransformer(model_id, revision=revision, device=device)
        self.id = f"{model_id}@{revision}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)

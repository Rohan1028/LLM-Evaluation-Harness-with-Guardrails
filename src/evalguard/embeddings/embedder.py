from __future__ import annotations

import hashlib
from typing import List, Protocol, Sequence

import numpy as np

from ..logging import get_logger

LOGGER = get_logger(__name__)


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class _FallbackEmbedder:
    """Deterministic hashing-based embedding for offline tests."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = token_hash % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec) or 1.0
        return (vec / norm).tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_fallback: bool = False) -> None:
        self.model_name = model_name
        self._use_fallback = use_fallback
        self._fallback = _FallbackEmbedder()
        self._model = None
        if not use_fallback:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(model_name)
                LOGGER.info("Loaded sentence-transformer model '%s'", model_name)
            except Exception as exc:
                LOGGER.warning("Falling back to hashing embedder: %s", exc)
                self._use_fallback = True

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if self._use_fallback or self._model is None:
            return self._fallback.embed_documents(texts)
        embeddings = self._model.encode(list(texts), normalize_embeddings=True)  # type: ignore
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        if self._use_fallback or self._model is None:
            return self._fallback.embed_query(text)
        embedding = self._model.encode(text, normalize_embeddings=True)  # type: ignore
        return embedding.tolist()


def build_embedder(
    model_name: str = "all-MiniLM-L6-v2", prefer_fallback: bool = False
) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name=model_name, use_fallback=prefer_fallback)

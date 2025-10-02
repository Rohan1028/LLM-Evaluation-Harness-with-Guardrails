from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence, cast

import numpy as np

from ..logging import get_logger
from ..utils import cosine_similarity

LOGGER = get_logger(__name__)


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: int
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorStore(Protocol):
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        documents: Sequence[str],
    ) -> None: ...

    def query(self, embedding: Sequence[float], k: int) -> List[DocumentChunk]: ...


class _InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        documents: Sequence[str],
    ) -> None:
        for idx, emb, meta, doc in zip(ids, embeddings, metadatas, documents, strict=False):
            self._entries.append(
                {
                    "id": idx,
                    "embedding": np.array(emb, dtype=np.float32),
                    "metadata": meta,
                    "document": doc,
                }
            )

    def query(self, embedding: Sequence[float], k: int) -> List[DocumentChunk]:
        vector = np.array(embedding, dtype=np.float32)
        vector_list = vector.tolist()
        ranked = sorted(
            (
                (
                    cosine_similarity(vector_list, entry["embedding"].tolist()),
                    entry["metadata"],
                    entry["document"],
                )
                for entry in self._entries
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            DocumentChunk(
                doc_id=meta["doc_id"],
                chunk_id=meta["chunk_id"],
                text=document,
                score=float(score),
                metadata=meta,
            )
            for score, meta, document in ranked[:k]
        ]


class ChromaVectorStore:
    def __init__(self, collection_name: str, persist_directory: str | None = None) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._use_fallback = False
        try:
            import chromadb

            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.Client()
            self._collection: Any = self._client.get_or_create_collection(collection_name)
        except Exception as exc:  # pragma: no cover - optional
            LOGGER.warning("Chroma unavailable (%s); using in-memory store", exc)
            self._use_fallback = True
            self._collection = _InMemoryVectorStore()

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        documents: Sequence[str],
    ) -> None:
        if self._use_fallback:
            fallback = cast(_InMemoryVectorStore, self._collection)
            fallback.add(ids, embeddings, metadatas, documents)
            return
        self._collection.add(
            ids=list(ids),
            embeddings=list(embeddings),
            metadatas=list(metadatas),
            documents=list(documents),
        )

    def query(self, embedding: Sequence[float], k: int) -> List[DocumentChunk]:
        if self._use_fallback:
            fallback = cast(_InMemoryVectorStore, self._collection)
            return fallback.query(embedding, k)

        result: Any = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=k,
        )
        documents = cast(List[str], result.get("documents", [[]])[0])
        metadatas = cast(List[Dict[str, Any]], result.get("metadatas", [[]])[0])
        scores = cast(List[float], result.get("distances", [[]])[0])
        chunks: List[DocumentChunk] = []
        for doc, meta, score in zip(documents, metadatas, scores, strict=False):
            chunks.append(
                DocumentChunk(
                    doc_id=meta.get("doc_id", meta.get("source", "unknown")),
                    chunk_id=int(meta.get("chunk_id", 0)),
                    text=doc,
                    score=float(score),
                    metadata=meta,
                )
            )
        return chunks

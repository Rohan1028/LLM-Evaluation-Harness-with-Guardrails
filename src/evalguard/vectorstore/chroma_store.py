from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, cast

import numpy as np

from ..logging import get_logger
from ..utils import cosine_similarity, ensure_dir

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
            api_key = os.getenv("CHROMA_API_KEY")
            tenant = os.getenv("CHROMA_TENANT")
            database = os.getenv("CHROMA_DATABASE")
            host = os.getenv("CHROMA_HOST")
            if api_key and tenant and database:
                kwargs: Dict[str, Any] = {
                    "api_key": api_key,
                    "tenant": tenant,
                    "database": database,
                }
                if host:
                    kwargs["host"] = host
                self._client = chromadb.CloudClient(**kwargs)
                collection_name = os.getenv("CHROMA_COLLECTION", collection_name)
                self.collection_name = collection_name
            elif persist_directory:
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

    def count(self) -> int:
        if self._use_fallback:
            fallback = cast(_InMemoryVectorStore, self._collection)
            return len(fallback._entries)
        return int(self._collection.count())

    def export(self, path: Path) -> Path:
        path = path.resolve()
        ensure_dir(path.parent)
        if self._use_fallback:
            fallback = cast(_InMemoryVectorStore, self._collection)
            payload = fallback._entries
        else:
            payload = self._collection.get(include=["metadatas", "documents", "ids"])
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Exported collection '%s' to %s", self.collection_name, path)
        return path

    def compact(self) -> Dict[str, Any]:
        if self._use_fallback:
            LOGGER.info("Fallback vector store in use; nothing to compact.")
            return {"mode": "memory", "status": "noop"}
        rows = self._collection.get(include=["metadatas", "documents", "ids", "embeddings"])
        ids = rows.get("ids", [])
        if not ids:
            LOGGER.info("Collection '%s' empty; skipping compaction.", self.collection_name)
            return {"mode": "client", "status": "skipped"}
        embeddings = rows.get("embeddings")
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.create_collection(self.collection_name)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=rows.get("metadatas"),
            documents=rows.get("documents"),
        )
        LOGGER.info("Compacted collection '%s' (%d vectors)", self.collection_name, len(ids))
        return {"mode": "client", "status": "compacted", "count": len(ids)}

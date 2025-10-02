from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm

from ..config import RAGConfig
from ..embeddings import Embedder
from ..logging import get_logger
from ..utils import ensure_dir, normalize_text
from .chroma_store import ChromaVectorStore

LOGGER = get_logger(__name__)


@dataclass
class IngestStats:
    documents: int
    chunks: int


def _read_documents(corpus_dir: Path) -> List[tuple[str, str]]:
    documents: List[tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        documents.append((doc_id, text))
    return documents


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_text(text)
    sentences = re.split(r"(?<=[.?!])\s+", text)
    chunks: List[str] = []
    chunk: List[str] = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and chunk:
            chunks.append(" ".join(chunk).strip())
            chunk = chunk[-overlap:] if overlap else []
            current_len = sum(len(s) for s in chunk)
        chunk.append(sentence)
        current_len += len(sentence)
    if chunk:
        chunks.append(" ".join(chunk).strip())
    return [c for c in chunks if c]


def ingest_corpus(
    corpus_dir: Path,
    collection_name: str,
    embedder: Embedder,
    vector_store: ChromaVectorStore,
    rag_config: RAGConfig,
) -> IngestStats:
    ensure_dir(Path(vector_store.persist_directory or "."))
    documents = _read_documents(corpus_dir)
    if not documents:
        raise FileNotFoundError(f"No markdown/text documents found in {corpus_dir}")

    chunk_size = rag_config.chunk_size
    overlap = rag_config.chunk_overlap
    chunk_records: List[str] = []
    metadatas = []
    ids = []

    for doc_id, text in tqdm(documents, desc="Chunking corpus"):
        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{doc_id}-{idx}")
            chunk_records.append(chunk)
            metadatas.append({"doc_id": doc_id, "chunk_id": idx, "source": doc_id})

    LOGGER.info("Embedding %d chunks for collection '%s'", len(chunk_records), collection_name)
    embeddings = embedder.embed_documents(chunk_records)
    vector_store.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunk_records)

    stats = IngestStats(documents=len(documents), chunks=len(chunk_records))
    LOGGER.info("Ingested %d documents into collection '%s'", stats.documents, collection_name)
    return stats


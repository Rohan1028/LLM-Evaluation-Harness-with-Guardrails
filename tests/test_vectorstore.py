from pathlib import Path

from evalguard.config import RAGConfig
from evalguard.embeddings import build_embedder
from evalguard.vectorstore import ChromaVectorStore, ingest_corpus


def test_ingest_and_query(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc.md").write_text("Sample text about SentinelIQ guardrails.", encoding="utf-8")

    vector_store = ChromaVectorStore(
        collection_name="test", persist_directory=str(tmp_path / "persist")
    )
    embedder = build_embedder(prefer_fallback=True)
    ingest_corpus(
        corpus_dir=corpus_dir,
        collection_name="test",
        embedder=embedder,
        vector_store=vector_store,
        rag_config=RAGConfig(),
    )
    result = vector_store.query(embedder.embed_query("What is SentinelIQ?"), k=1)
    assert result
    assert result[0].doc_id == "doc"

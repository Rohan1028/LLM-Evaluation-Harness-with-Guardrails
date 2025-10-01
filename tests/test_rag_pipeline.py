from evalguard.config import Settings
from evalguard.embeddings import build_embedder
from evalguard.pipelines import RAGPipeline
from evalguard.providers import create_provider
from evalguard.vectorstore import ChromaVectorStore, ingest_corpus


def test_pipeline_with_mock_provider(tmp_path):
    settings = Settings.default()
    settings.rag.collection = "test"
    settings.guardrails.max_retries = 1

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc.md").write_text("SentinelIQ helps security teams stay compliant. Citations required.", encoding="utf-8")

    vector_store = ChromaVectorStore(collection_name="test", persist_directory=str(tmp_path / "persist"))
    embedder = build_embedder(prefer_fallback=True)
    ingest_corpus(corpus_dir=corpus_dir, collection_name="test", embedder=embedder, vector_store=vector_store, rag_config=settings.rag)

    provider = create_provider(settings.providers["mock"])
    pipeline = RAGPipeline(provider=provider, embedder=embedder, vector_store=vector_store, settings=settings)
    result = pipeline.run("How does SentinelIQ help?", ground_truth="It supports compliance for security teams.")
    assert result.answer
    assert result.guardrail.citations
    assert result.guardrail.passed

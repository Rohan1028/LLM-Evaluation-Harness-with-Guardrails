from evalguard.pipelines.guardrails import Citation, extract_citations, validate_citations


def test_extract_and_validate_citations() -> None:
    text = "Answer referencing [welcome:0] and [product:1]."
    citations = extract_citations(text)
    assert len(citations) == 2
    contexts = [
        {"doc_id": "welcome", "chunk_id": 0, "text": "Welcome text"},
        {"doc_id": "product", "chunk_id": 1, "text": "Product text"},
    ]
    assert validate_citations(citations, contexts)


def test_invalid_citations() -> None:
    citations = [Citation(doc_id="welcome", chunk_id=99)]
    contexts = [{"doc_id": "welcome", "chunk_id": 0, "text": "Welcome text"}]
    assert not validate_citations(citations, contexts)

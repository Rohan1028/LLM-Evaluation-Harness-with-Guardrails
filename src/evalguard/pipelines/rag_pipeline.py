from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..config import GuardrailConfig, RAGConfig, Settings
from ..embeddings import Embedder
from ..logging import get_logger
from ..providers import Provider
from ..vectorstore import ChromaVectorStore, DocumentChunk
from .guardrails import GuardrailResult, GuardrailRunner

LOGGER = get_logger(__name__)


@dataclass
class RetrievedContext:
    doc_id: str
    chunk_id: int
    text: str
    score: float

    @classmethod
    def from_chunk(cls, chunk: DocumentChunk) -> "RetrievedContext":
        return cls(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=chunk.score,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
        }


@dataclass
class PipelineMetadata:
    model: str
    provider: str
    question: str
    retry_count: int
    guardrail_passed: bool
    citations: List[Dict[str, Any]]
    toxicity: float


@dataclass
class PipelineRunResult:
    question: str
    answer: str
    ground_truth: Optional[str]
    model: str
    provider: str
    contexts: List[RetrievedContext]
    guardrail: GuardrailResult
    metadata: PipelineMetadata

    def to_record(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "model": self.model,
            "provider": self.provider,
            "contexts": [ctx.to_dict() for ctx in self.contexts],
            "citations": [citation.as_tuple() for citation in self.guardrail.citations],
            "toxicity": self.guardrail.toxicity,
            "guardrail_passed": self.guardrail.passed,
            "retry_count": self.metadata.retry_count,
        }


class RAGPipeline:
    def __init__(
        self,
        provider: Provider,
        embedder: Embedder,
        vector_store: ChromaVectorStore,
        settings: Settings,
    ) -> None:
        self.provider = provider
        self.embedder = embedder
        self.vector_store = vector_store
        self.rag_config: RAGConfig = settings.rag
        self.guardrail_runner = GuardrailRunner(
            settings.guardrails
            if isinstance(settings.guardrails, GuardrailConfig)
            else GuardrailConfig()
        )

    def run(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineRunResult:
        contexts = self._retrieve(question)
        attempts = 0
        max_attempts = self.guardrail_runner.config.max_retries + 1
        answer = ""
        guardrail_result: Optional[GuardrailResult] = None
        prompt = self._build_prompt(question, contexts, metadata or {})

        while attempts < max_attempts:
            attempts += 1
            LOGGER.debug("Generating answer attempt %d/%d", attempts, max_attempts)
            answer = self.provider.generate(prompt)
            guardrail_result = self.guardrail_runner.enforce(
                answer=answer,
                contexts=[ctx.to_dict() for ctx in contexts],
                question=question,
                provider=self.provider,
            )
            if guardrail_result.needs_retry:
                LOGGER.info("Fact-check retry triggered for question '%s'", question)
                prompt = guardrail_result.retry_prompt or self._build_prompt(
                    question, contexts, metadata or {}
                )
                continue
            if guardrail_result.passed:
                break
            else:
                prompt = self._build_prompt(question, contexts, metadata or {}, corrective=True)

        guardrail_result = guardrail_result or GuardrailResult(
            answer=answer,
            citations=[],
            toxicity=0.0,
            violations=["unknown"],
            needs_retry=False,
            retry_prompt=None,
        )
        final_answer = guardrail_result.answer

        meta = PipelineMetadata(
            model=self.provider.model,
            provider=self.provider.name,
            question=question,
            retry_count=attempts - 1,
            guardrail_passed=guardrail_result.passed,
            citations=[
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id} for c in guardrail_result.citations
            ],
            toxicity=guardrail_result.toxicity,
        )

        return PipelineRunResult(
            question=question,
            answer=final_answer,
            ground_truth=ground_truth,
            model=self.provider.model,
            provider=self.provider.name,
            contexts=contexts,
            guardrail=guardrail_result,
            metadata=meta,
        )

    def _retrieve(self, question: str) -> List[RetrievedContext]:
        embedding = self.embedder.embed_query(question)
        chunks = self.vector_store.query(embedding, k=self.rag_config.retriever_top_k)
        return [RetrievedContext.from_chunk(chunk) for chunk in chunks]

    def _build_prompt(
        self,
        question: str,
        contexts: Sequence[RetrievedContext],
        metadata: Dict[str, Any],
        corrective: bool = False,
    ) -> str:
        instructions = (
            "Provide a concise answer using only the retrieved context. "
            "Include bracketed citations referencing [doc_id:chunk_id] for every claim."
        )
        if corrective:
            instructions += " The previous answer violated guardrails; repair it now."

        if metadata.get("doc_hint"):
            instructions += (
                f" Emphasize evidence from documents related to '{metadata['doc_hint']}'."
            )

        prompt = f"{instructions}\n\nQuestion: {question}\nContexts:\n"
        for ctx in contexts:
            prompt += f"- [{ctx.doc_id}:{ctx.chunk_id}] {ctx.text}\n"
        prompt += "\nAnswer:"
        return prompt

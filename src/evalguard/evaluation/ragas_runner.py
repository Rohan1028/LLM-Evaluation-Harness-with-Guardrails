from __future__ import annotations

import statistics
from typing import Dict, List

from ..logging import get_logger
from ..pipelines import PipelineRunResult
from ..utils import normalize_text
from . import EvaluationReport

LOGGER = get_logger(__name__)


class RagasRunner:
    """Compute RAG-specific metrics with graceful fallback if ragas is unavailable."""

    def __init__(self) -> None:
        self._ragas = None
        try:
            import ragas  # type: ignore

            self._ragas = ragas
            LOGGER.info("Ragas available; advanced metrics enabled")
        except Exception as exc:  # pragma: no cover - optional
            LOGGER.debug("Ragas not available: %s. Using heuristic evaluator.", exc)

    def evaluate(self, results: List[PipelineRunResult]) -> EvaluationReport:
        per_example: List[Dict[str, float]] = []
        for result in results:
            relevancy = self._answer_relevancy(result)
            faithfulness = self._faithfulness(result)
            ctx_precision = self._context_precision(result)
            ctx_recall = self._context_recall(result)
            per_example.append(
                {
                    "question": result.question,
                    "model": result.model,
                    "answer_relevancy": relevancy,
                    "faithfulness": faithfulness,
                    "context_precision": ctx_precision,
                    "context_recall": ctx_recall,
                }
            )
        aggregate = {
            "answer_relevancy_mean": statistics.fmean(
                item["answer_relevancy"] for item in per_example
            ),
            "faithfulness_mean": statistics.fmean(item["faithfulness"] for item in per_example),
            "context_precision_mean": statistics.fmean(
                item["context_precision"] for item in per_example
            ),
            "context_recall_mean": statistics.fmean(item["context_recall"] for item in per_example),
        }
        return EvaluationReport(name="ragas", per_example=per_example, aggregate=aggregate)

    def _answer_relevancy(self, result: PipelineRunResult) -> float:
        if not result.ground_truth:
            return 0.5
        answer = normalize_text(result.answer)
        truth = normalize_text(result.ground_truth)
        overlap = len(set(answer.split()) & set(truth.split()))
        return min(1.0, overlap / (len(truth.split()) or 1))

    def _faithfulness(self, result: PipelineRunResult) -> float:
        if not result.guardrail.citations:
            return 0.0
        return min(1.0, len(result.guardrail.citations) / max(1, len(result.contexts)))

    def _context_precision(self, result: PipelineRunResult) -> float:
        if not result.contexts:
            return 0.0
        cited_ids = {citation.doc_id for citation in result.guardrail.citations}
        relevant = sum(1 for ctx in result.contexts if ctx.doc_id in cited_ids)
        return relevant / len(result.contexts)

    def _context_recall(self, result: PipelineRunResult) -> float:
        total_citations = len(result.guardrail.citations)
        if not total_citations:
            return 0.0
        unique = len({c.as_tuple() for c in result.guardrail.citations})
        return unique / total_citations

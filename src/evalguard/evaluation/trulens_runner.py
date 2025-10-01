from __future__ import annotations

import statistics
from typing import Dict, List

from ..logging import get_logger
from ..pipelines import PipelineRunResult
from ..utils import normalize_text
from . import EvaluationReport

LOGGER = get_logger(__name__)


class TruLensRunner:
    """Heuristic fallback inspired by TruLens feedback functions."""

    def __init__(self) -> None:
        self._trulens = None
        try:
            import trulens_eval  # type: ignore

            self._trulens = trulens_eval
            LOGGER.info("TruLens available; using built-in feedback functions")
        except Exception as exc:  # pragma: no cover - optional
            LOGGER.debug("TruLens not available: %s. Using heuristic evaluator.", exc)

    def evaluate(self, results: List[PipelineRunResult]) -> EvaluationReport:
        per_example: List[Dict[str, float]] = []
        for result in results:
            faithfulness = self._compute_faithfulness(result)
            coherence = self._compute_coherence(result)
            per_example.append(
                {
                    "question": result.question,
                    "model": result.model,
                    "faithfulness": faithfulness,
                    "coherence": coherence,
                }
            )
        aggregate = {
            "faithfulness_mean": statistics.fmean(item["faithfulness"] for item in per_example),
            "coherence_mean": statistics.fmean(item["coherence"] for item in per_example),
        }
        return EvaluationReport(name="trulens", per_example=per_example, aggregate=aggregate)

    def _compute_faithfulness(self, result: PipelineRunResult) -> float:
        if not result.guardrail.citations or not result.contexts:
            return 0.0
        unique_citations = {c.as_tuple() for c in result.guardrail.citations}
        return min(1.0, len(unique_citations) / len(result.contexts))

    def _compute_coherence(self, result: PipelineRunResult) -> float:
        answer = normalize_text(result.answer)
        if not answer:
            return 0.0
        sentences = [s for s in answer.split(".") if s.strip()]
        word_counts = [len(sentence.split()) for sentence in sentences]
        if not word_counts:
            return 0.0
        variance = statistics.pvariance(word_counts) if len(word_counts) > 1 else 0.0
        coherence = max(0.1, 1.0 - min(variance / 10.0, 0.9))
        return round(coherence, 3)

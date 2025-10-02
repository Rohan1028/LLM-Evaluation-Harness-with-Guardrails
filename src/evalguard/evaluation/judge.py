from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..logging import get_logger
from ..providers import Provider

LOGGER = get_logger(__name__)


@dataclass
class JudgeResult:
    score: float
    reasoning: str


class LLMJudge:
    """Optional LLM-as-judge for fluency/coherence scoring."""

    def __init__(self, provider: Optional[Provider]) -> None:
        self.provider = provider

    def score(self, question: str, answer: str) -> JudgeResult:
        if not self.provider:
            return JudgeResult(
                score=0.75, reasoning="LLM judge unavailable; using heuristic score."
            )
        prompt = (
            "Rate the fluency and coherence (0-1) of the assistant's answer.\n"
            f"Question: {question}\nAnswer: {answer}\n"
            'Respond with JSON: {"score": <float>, "reasoning": "..."}'
        )
        try:
            response = self.provider.generate(prompt)
            if response.strip().startswith("{"):
                import json

                data = json.loads(response)
                return JudgeResult(
                    score=float(data.get("score", 0.7)), reasoning=data.get("reasoning", "")
                )
            return JudgeResult(score=0.7, reasoning=response)
        except Exception as exc:  # pragma: no cover - provider
            LOGGER.debug("Judge scoring failed: %s", exc)
            return JudgeResult(score=0.7, reasoning="Judge call failed; defaulting.")

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

from ..config import GuardrailConfig
from ..logging import get_logger
from ..providers.base import Provider
from ..utils import optional_import

LOGGER = get_logger(__name__)

CITATION_PATTERN = re.compile(r"\[(?P<doc>[^:\]]+):(?P<chunk>\d+)\]")

_ToxicityPipeline = Callable[[str], Sequence[Mapping[str, Any]]]

try:  # pragma: no cover - optional dependency
    from jsonschema import Draft7Validator, ValidationError
except Exception:  # pragma: no cover - optional dependency
    Draft7Validator = None  # type: ignore[assignment]
    ValidationError = Exception


@dataclass
class Citation:
    doc_id: str
    chunk_id: int

    def as_tuple(self) -> Tuple[str, int]:
        return self.doc_id, self.chunk_id


@dataclass
class GuardrailResult:
    answer: str
    citations: List[Citation]
    toxicity: float
    violations: List[str]
    needs_retry: bool
    retry_prompt: Optional[str]

    @property
    def passed(self) -> bool:
        return not self.violations and not self.needs_retry


class GuardrailRunner:
    def __init__(self, config: GuardrailConfig) -> None:
        self.config = config
        self._toxicity_pipeline: Optional[_ToxicityPipeline] = self._load_toxicity_pipeline()
        self._guardrails: Optional[Any] = self._load_guardrails_spec()
        self._schema_validator: Optional[Draft7Validator] = self._load_schema_validator()

    def enforce(
        self,
        answer: str,
        contexts: Sequence[Mapping[str, Any]],
        question: str,
        provider: Provider,
    ) -> GuardrailResult:
        citations = extract_citations(answer)
        violations: List[str] = []
        mapped = {(ctx["doc_id"], ctx["chunk_id"]) for ctx in contexts}
        missing = [c for c in citations if (c.doc_id, c.chunk_id) not in mapped]
        if missing:
            violations.append("invalid_citation")
        if len(citations) < self.config.min_citations:
            violations.append("insufficient_citations")

        toxicity = self._score_toxicity(answer)
        if toxicity > self.config.toxicity_threshold:
            violations.append("toxicity_threshold_exceeded")
            answer = "[REDACTED] Response withheld due to safety policy."

        needs_retry = False
        retry_prompt: Optional[str] = None
        if self.config.factcheck_required and not violations:
            faithful = self._verify_faithfulness(question, answer, contexts, provider)
            if not faithful:
                needs_retry = True
                retry_prompt = self._build_retry_prompt(question, contexts)

        if self._guardrails:
            try:
                answer = self._guardrails_parse(answer)
            except Exception as exc:  # pragma: no cover - optional
                LOGGER.debug("Guardrails.ai parse failed: %s", exc)

        result = GuardrailResult(
            answer=answer,
            citations=citations,
            toxicity=toxicity,
            violations=violations,
            needs_retry=needs_retry,
            retry_prompt=retry_prompt,
        )
        schema_error = self._validate_schema(result)
        if schema_error:
            result.violations.append(schema_error)
        return result

    def _score_toxicity(self, text: str) -> float:
        if not text.strip():
            return 0.0
        if self._toxicity_pipeline:
            try:
                outputs = self._toxicity_pipeline(text)
                if outputs:
                    score = outputs[0].get("score", 0.0)
                    return float(score)
            except Exception as exc:  # pragma: no cover - optional
                LOGGER.debug("Toxicity pipeline failed: %s", exc)
        lower = text.lower()
        hits = sum(lower.count(word) for word in self.config.banned_keywords)
        return min(1.0, hits / 5.0)

    def _verify_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: Sequence[Mapping[str, Any]],
        provider: Provider,
    ) -> bool:
        prompt = (
            "You are a fact-checking assistant. Determine if the candidate answer is fully supported "
            "by the provided contexts. Respond with 'yes' or 'no' only.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Contexts:\n"
        )
        for ctx in contexts:
            prompt += f"- [{ctx['doc_id']}:{ctx['chunk_id']}] {ctx['text']}\n"
        prompt += "Is the answer faithful?"
        try:
            verdict = provider.generate(prompt, system="Respond with 'yes' or 'no'.")
            return verdict.text.strip().lower().startswith("y")
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Fact-check call failed: %s", exc)
            return True

    def _build_retry_prompt(self, question: str, contexts: Sequence[Mapping[str, Any]]) -> str:
        prompt = (
            "The previous answer may contain unsupported information. "
            "Review the contexts and produce a corrected answer citing specific chunks.\n"
            f"Question: {question}\nContexts:\n"
        )
        for ctx in contexts:
            prompt += f"- [{ctx['doc_id']}:{ctx['chunk_id']}] {ctx['text']}\n"
        prompt += "Provide an updated answer with bracketed citations."
        return prompt

    def _load_toxicity_pipeline(self) -> Optional[_ToxicityPipeline]:
        transformers = optional_import("transformers")
        if not transformers:  # pragma: no cover - optional dependency
            return None
        try:
            pipeline = transformers.pipeline("text-classification", model="unitary/toxic-bert")
            LOGGER.info("Loaded transformers toxicity pipeline")
            return cast(_ToxicityPipeline, pipeline)
        except Exception as exc:  # pragma: no cover - heavy dependency
            LOGGER.warning("Failed to load toxicity model: %s", exc)
            return None

    def _load_guardrails_spec(self) -> Optional[Any]:
        if not self.config.rail_spec_enabled:
            return None
        guardrails_module = optional_import("guardrails")
        if not guardrails_module:  # pragma: no cover - optional
            LOGGER.warning("guardrails-ai package not installed; skipping schema enforcement")
            return None
        schema: Mapping[str, Any] = {
            "rail_spec": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_id": {"type": "string"},
                                "chunk_id": {"type": "integer"},
                            },
                        },
                    },
                },
                "required": ["answer", "citations"],
            }
        }
        if self.config.rail_spec_path and self.config.rail_spec_path.exists():
            try:
                schema = json.loads(self.config.rail_spec_path.read_text())
            except Exception as exc:  # pragma: no cover - file parsing
                LOGGER.warning("Failed to load rail spec: %s", exc)
        guard_cls = getattr(guardrails_module, "Guard", None)
        if guard_cls is None:  # pragma: no cover - optional
            LOGGER.warning("Guard class missing from guardrails module")
            return None
        return guard_cls.from_pydantic(output_class=dict, prompt=None, spec=schema)

    def _guardrails_parse(self, answer: str) -> str:
        guard = self._guardrails
        if not guard:
            return answer
        parsed = guard.parse(answer)
        if isinstance(parsed, dict) and "answer" in parsed:
            return cast(str, parsed["answer"])
        return answer

    def _load_schema_validator(self) -> Optional[Draft7Validator]:
        if not self.config.rail_spec_enabled or not self.config.rail_spec_path:
            return None
        if Draft7Validator is None:
            LOGGER.warning("jsonschema not installed; structured validation disabled")
            return None
        if not self.config.rail_spec_path.exists():
            LOGGER.warning("Guardrail schema path %s missing", self.config.rail_spec_path)
            return None
        try:
            schema = json.loads(self.config.rail_spec_path.read_text(encoding="utf-8"))
            return Draft7Validator(schema)
        except Exception as exc:  # pragma: no cover - schema parsing
            LOGGER.warning("Failed to load schema %s: %s", self.config.rail_spec_path, exc)
            return None

    def _validate_schema(self, result: GuardrailResult) -> Optional[str]:
        if not self._schema_validator:
            return None
        payload = {
            "answer": result.answer,
            "citations": [
                {"doc_id": citation.doc_id, "chunk_id": citation.chunk_id}
                for citation in result.citations
            ],
            "toxicity": result.toxicity,
        }
        try:
            self._schema_validator.validate(payload)
            return None
        except ValidationError as exc:
            LOGGER.debug("Structured schema validation failed: %s", exc)
            return "schema_validation_failed"


def extract_citations(text: str) -> List[Citation]:
    citations: List[Citation] = []
    for match in CITATION_PATTERN.finditer(text):
        citations.append(Citation(doc_id=match.group("doc"), chunk_id=int(match.group("chunk"))))
    return citations


def validate_citations(
    citations: Iterable[Citation],
    contexts: Sequence[Mapping[str, Any]],
) -> bool:
    mapped = {(ctx["doc_id"], ctx["chunk_id"]) for ctx in contexts}
    return all((citation.doc_id, citation.chunk_id) in mapped for citation in citations)

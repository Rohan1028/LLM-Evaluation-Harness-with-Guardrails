from __future__ import annotations

from pathlib import Path

from evalguard.config import GuardrailConfig, ProviderConfig
from evalguard.pipelines.guardrails import GuardrailRunner
from evalguard.providers.base import MockProvider


def _build_runner(**overrides: object) -> GuardrailRunner:
    config = GuardrailConfig(
        min_citations=0,
        factcheck_required=False,
        rail_spec_enabled=True,
        rail_spec_path=Path("data/rails/answer_schema.json"),
        **overrides,
    )
    return GuardrailRunner(config)


def _mock_provider() -> MockProvider:
    provider_cfg = ProviderConfig(provider="mock", model="deterministic")
    return MockProvider(provider_cfg)


def _contexts() -> list[dict[str, object]]:
    return [{"doc_id": "00_welcome", "chunk_id": 0, "text": "Welcome text."}]


def test_schema_validation_passes_for_realistic_output() -> None:
    runner = _build_runner()
    answer = (
        "SentinelIQ is a unified intelligence platform for regulated industries. [00_welcome:0]"
    )
    result = runner.enforce(
        answer=answer,
        contexts=_contexts(),
        question="What is SentinelIQ?",
        provider=_mock_provider(),
    )
    assert result.violations == []
    assert result.passed


def test_schema_validation_flags_missing_citations() -> None:
    runner = _build_runner()
    result = runner.enforce(
        answer="This response omitted citations entirely.",
        contexts=_contexts(),
        question="What is SentinelIQ?",
        provider=_mock_provider(),
    )
    assert "schema_validation_failed" in result.violations
    assert not result.passed

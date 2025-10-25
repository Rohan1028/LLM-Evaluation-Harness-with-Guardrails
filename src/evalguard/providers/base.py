from __future__ import annotations

import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, TypeVar

from ..config import CostConfig, ProviderConfig, RateLimitConfig
from ..logging import get_logger
from .rate_limit import RateLimiter
from .telemetry import (
    ProviderCallDetails,
    ProviderCallTelemetry,
    ProviderResponse,
)

LOGGER = get_logger(__name__)


class ProviderError(RuntimeError):
    """Raised when provider interaction fails."""


class Provider(ABC):
    name: str

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._rate_config: RateLimitConfig = config.rate_limit or RateLimitConfig()
        self._cost_config: CostConfig = config.cost or CostConfig()
        self._rate_limiter = RateLimiter(self._rate_config.requests_per_minute)

    @abstractmethod
    def _call_model(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> ProviderCallDetails:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> ProviderResponse:
        attempt = 0
        max_attempts = max(self._rate_config.max_retries, 0) + 1
        last_error: Optional[Exception] = None
        while attempt < max_attempts:
            queued_seconds = self._rate_limiter.acquire()
            start = time.perf_counter()
            try:
                details = self._call_model(prompt, system=system, stop=stop)
                latency_ms = (time.perf_counter() - start) * 1000.0
                return self._build_response(
                    details=details,
                    latency_ms=latency_ms,
                    retries=attempt,
                    queued_ms=queued_seconds * 1000.0,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                attempt += 1
                LOGGER.debug(
                    "Provider %s attempt %d/%d failed: %s",
                    self.name,
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt >= max_attempts:
                    self._build_response(
                        details=ProviderCallDetails(text="", metadata={"failed_prompt": bool(prompt)}),
                        latency_ms=(time.perf_counter() - start) * 1000.0,
                        retries=attempt - 1,
                        queued_ms=queued_seconds * 1000.0,
                        status="error",
                        error=str(exc),
                    )
                    raise ProviderError(f"{self.name} generate failed: {exc}") from last_error
                delay = self._compute_backoff(attempt)
                time.sleep(delay)
        raise ProviderError(f"{self.name} generate failed: {last_error}")

    def validate(self) -> None:
        """Optional hook to validate provider credentials."""
        return None

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @property
    def model(self) -> str:
        return self.config.model

    def _compute_backoff(self, retry_index: int) -> float:
        base = max(self._rate_config.initial_delay, 0.1)
        multiplier = max(self._rate_config.backoff_multiplier, 1.0)
        jitter = max(self._rate_config.jitter, 0.0)
        delay = base * (multiplier ** max(retry_index - 1, 0))
        return delay + random.uniform(0, jitter)

    def _build_response(
        self,
        details: ProviderCallDetails,
        latency_ms: float,
        retries: int,
        queued_ms: float,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> ProviderResponse:
        prompt_tokens = details.prompt_tokens or 0
        completion_tokens = details.completion_tokens or 0
        total_tokens = prompt_tokens + completion_tokens
        telemetry = ProviderCallTelemetry(
            provider=self.name,
            model=self.model,
            status=status,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=self._cost_config.estimate(prompt_tokens, completion_tokens),
            retries=retries,
            queued_ms=queued_ms,
            error=error,
            metadata=details.metadata,
        )
        return ProviderResponse(text=details.text, telemetry=telemetry)

    def __repr__(self) -> str:  # pragma: no cover - introspection convenience
        return f"{self.__class__.__name__}(model={self.config.model})"


ProviderType = TypeVar("ProviderType", bound="Provider")
PROVIDER_REGISTRY: Dict[str, type[Provider]] = {}


def register_provider(name: str) -> Callable[[type[ProviderType]], type[ProviderType]]:
    def decorator(cls: type[ProviderType]) -> type[ProviderType]:
        PROVIDER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def create_provider(config: ProviderConfig) -> Provider:
    if config.provider not in PROVIDER_REGISTRY:
        raise ProviderError(f"Provider '{config.provider}' is not registered")
    provider_cls = PROVIDER_REGISTRY[config.provider]
    provider = provider_cls(config)
    provider.validate()
    return provider


def parse_provider_spec(spec: str) -> Tuple[str, str]:
    if ":" in spec:
        provider, model = spec.split(":", 1)
    else:
        provider, model = spec, ""
    return provider.strip(), model.strip() or PROVIDER_REGISTRY.get(provider, Provider).__name__


@dataclass
class MockContext:
    citation: str
    text: str


@register_provider("mock")
class MockProvider(Provider):
    """Deterministic provider that follows prompts and guardrails for tests/demo."""

    citation_pattern = re.compile(r"\[(?P<doc>[^:\]]+):(?P<chunk>\d+)\]")

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._last_prompt = ""

    def _call_model(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> ProviderCallDetails:
        self._last_prompt = prompt
        lower_prompt = prompt.lower()
        if (
            "answer with 'yes' or 'no'" in lower_prompt
            or "respond with 'yes' or 'no'" in lower_prompt
        ):
            return ProviderCallDetails(text="yes", prompt_tokens=len(prompt.split()), completion_tokens=1)
        contexts = self._extract_contexts(prompt)
        if not contexts:
            return ProviderCallDetails(
                text="I do not have enough context to answer safely.",
                prompt_tokens=len(prompt.split()),
                completion_tokens=12,
            )

        sentences = []
        for context in contexts[:2]:
            snippet = context.text.split(".")[0].strip()
            sentences.append(f"{snippet} [{context.citation}]")
        answer = " ".join(sentences)
        if "include audit log" in lower_prompt:
            answer += " Guardrail audit log maintained. [welcome:0]"
        if "sensitive" in lower_prompt:
            answer += " I must defer to policy for sensitive topics. [security:0]"
        return ProviderCallDetails(
            text=answer,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(answer.split()),
        )

    def _extract_contexts(self, prompt: str) -> list[MockContext]:
        matches = self.citation_pattern.findall(prompt)
        contexts: list[MockContext] = []
        for match in matches:
            citation = f"{match[0]}:{match[1]}"
            marker = f"[{citation}]"
            before, _, after = prompt.partition(marker)
            snippet = after.splitlines()[0].strip()
            contexts.append(MockContext(citation=citation, text=snippet or "Context missing"))
        return contexts

    def validate(self) -> None:
        LOGGER.debug("MockProvider ready with deterministic outputs.")

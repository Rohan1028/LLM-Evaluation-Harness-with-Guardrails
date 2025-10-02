from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from ..config import ProviderConfig
from ..logging import get_logger

LOGGER = get_logger(__name__)


class ProviderError(RuntimeError):
    """Raised when provider interaction fails."""


class Provider(ABC):
    name: str

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        raise NotImplementedError

    def validate(self) -> None:\n        """Optional hook to validate provider credentials."""\n        return None

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @property
    def model(self) -> str:
        return self.config.model

    def __repr__(self) -> str:  # pragma: no cover - introspection convenience
        return f"{self.__class__.__name__}(model={self.config.model})"


PROVIDER_REGISTRY: Dict[str, type[Provider]] = {}


def register_provider(name: str):
    def decorator(cls: type[Provider]) -> type[Provider]:
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

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        self._last_prompt = prompt
        lower_prompt = prompt.lower()
        if "answer with 'yes' or 'no'" in lower_prompt or "respond with 'yes' or 'no'" in lower_prompt:
            return "yes"
        contexts = self._extract_contexts(prompt)
        if not contexts:
            return "I do not have enough context to answer safely."

        sentences = []
        for context in contexts[:2]:
            snippet = context.text.split(".")[0].strip()
            sentences.append(f"{snippet} [{context.citation}]")
        answer = " ".join(sentences)
        if "include audit log" in lower_prompt:
            answer += " Guardrail audit log maintained. [welcome:0]"
        if "sensitive" in lower_prompt:
            answer += " I must defer to policy for sensitive topics. [security:0]"
        return answer

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


from __future__ import annotations

from typing import Iterable, Optional

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider

LOGGER = get_logger(__name__)


@register_provider("anthropic")
class AnthropicProvider(Provider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ProviderError("anthropic package is required for AnthropicProvider") from exc
        api_key = config.metadata.get("api_key")
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        try:
            response = self._client.messages.create(  # type: ignore[attr-defined]
                model=self.model,
                system=system or "You are a helpful assistant that cites sources.",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.config.max_tokens or 512,
                stop_sequences=list(stop) if stop else None,
            )
            content = response.content[0]
            return content.text if hasattr(content, "text") else str(content)
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(f"Anthropic generate failed: {exc}") from exc

    def validate(self) -> None:
        api_key = getattr(self._client, "api_key", None)
        if not api_key:
            raise ProviderError("Anthropic API key missing (ANTHROPIC_API_KEY)")

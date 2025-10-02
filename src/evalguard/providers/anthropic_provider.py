from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, cast

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider

LOGGER = get_logger(__name__)

if TYPE_CHECKING:
    from anthropic import Anthropic as AnthropicClient
else:
    AnthropicClient = Any


@register_provider("anthropic")
class AnthropicProvider(Provider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            module = importlib.import_module("anthropic")
        except ImportError as exc:  # pragma: no cover - optional
            raise ProviderError("anthropic package is required for AnthropicProvider") from exc
        if not hasattr(module, "Anthropic"):
            raise ProviderError("anthropic package is required for AnthropicProvider")
        anthropic_cls = module.Anthropic
        api_key = config.metadata.get("api_key")
        self._client = cast(AnthropicClient, anthropic_cls(api_key=api_key))

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        try:
            kwargs: Dict[str, Any] = {}
            if stop:
                kwargs["stop_sequences"] = list(stop)
            if not hasattr(self._client, "messages"):
                raise ProviderError("Anthropic messages interface unavailable")
            messages = self._client.messages
            response: Any = messages.create(
                model=self.model,
                system=system or "You are a helpful assistant that cites sources.",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.config.max_tokens or 512,
                **kwargs,
            )
            content = response.content[0]
            if hasattr(content, "text"):
                return cast(str, content.text)
            return str(content)
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(f"Anthropic generate failed: {exc}") from exc

    def validate(self) -> None:
        api_key = getattr(self._client, "api_key", None)
        if not api_key:
            raise ProviderError("Anthropic API key missing (ANTHROPIC_API_KEY)")

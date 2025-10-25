from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, cast

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider
from .telemetry import ProviderCallDetails

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

    def _call_model(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> ProviderCallDetails:
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
            text = cast(str, getattr(content, "text", content))
            usage = getattr(response, "usage", None)
            usage_dict = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
            prompt_tokens = (
                getattr(usage, "input_tokens", None)
                or (usage_dict.get("input_tokens") if isinstance(usage_dict, dict) else None)
                or 0
            )
            completion_tokens = (
                getattr(usage, "output_tokens", None)
                or (usage_dict.get("output_tokens") if isinstance(usage_dict, dict) else None)
                or 0
            )
            return ProviderCallDetails(
                text=text,
                prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(completion_tokens or 0),
                metadata={"request_id": getattr(response, "id", None)},
            )
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(f"Anthropic generate failed: {exc}") from exc

    def validate(self) -> None:
        api_key = getattr(self._client, "api_key", None)
        if not api_key:
            raise ProviderError("Anthropic API key missing (ANTHROPIC_API_KEY)")

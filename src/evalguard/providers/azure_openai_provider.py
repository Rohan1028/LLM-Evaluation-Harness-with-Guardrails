from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, cast

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider

LOGGER = get_logger(__name__)

if TYPE_CHECKING:
    from openai import AzureOpenAI as AzureOpenAIClient
else:
    AzureOpenAIClient = Any


@register_provider("azure-openai")
class AzureOpenAIProvider(Provider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            module = importlib.import_module("openai")
        except ImportError as exc:  # pragma: no cover - optional
            raise ProviderError("openai>=1.0.0 is required for Azure OpenAI") from exc
        if not hasattr(module, "AzureOpenAI"):
            raise ProviderError("openai>=1.0.0 is required for Azure OpenAI")
        azure_cls = module.AzureOpenAI
        api_key = config.metadata.get("api_key") or None
        endpoint = config.metadata.get("endpoint") or None
        if not endpoint:
            raise ProviderError("Azure OpenAI endpoint missing (AZURE_OPENAI_ENDPOINT)")
        self._client = cast(AzureOpenAIClient, azure_cls(api_key=api_key, azure_endpoint=endpoint))
        self._deployment = config.metadata.get("deployment_name", config.model)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        try:
            if not hasattr(self._client, "responses"):
                raise ProviderError("Responses interface unavailable for Azure OpenAI client")
            responses = self._client.responses
            create = cast(Callable[..., Any], responses.create)
            response: Any = create(
                model=self._deployment,
                input=prompt,
                system=system or "",
                temperature=self.temperature,
                max_output_tokens=self.config.max_tokens,
                stop=stop,
            )
            if hasattr(response, "output_text"):
                return cast(str, response.output_text)
            return ""
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(f"Azure OpenAI generate failed: {exc}") from exc

    def validate(self) -> None:
        if not getattr(self._client, "api_key", None):
            raise ProviderError("Azure OpenAI API key missing (AZURE_OPENAI_API_KEY)")

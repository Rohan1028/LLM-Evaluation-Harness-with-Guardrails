from __future__ import annotations

import os
from typing import Iterable, Optional

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider

LOGGER = get_logger(__name__)


@register_provider("azure-openai")
class AzureOpenAIProvider(Provider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            from openai import AzureOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ProviderError("openai>=1.0.0 is required for Azure OpenAI") from exc
        api_key = config.metadata.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = config.metadata.get("endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ProviderError("Azure OpenAI endpoint missing (AZURE_OPENAI_ENDPOINT)")
        self._client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint)
        self._deployment = config.metadata.get("deployment_name", config.model)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        try:
            response = self._client.responses.create(  # type: ignore[attr-defined]
                model=self._deployment,
                input=prompt,
                system=system or "",
                temperature=self.temperature,
                max_output_tokens=self.config.max_tokens,
                stop=stop,
            )
            return getattr(response, "output_text", "")
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(f"Azure OpenAI generate failed: {exc}") from exc

    def validate(self) -> None:
        if not getattr(self._client, "api_key", None):
            raise ProviderError("Azure OpenAI API key missing (AZURE_OPENAI_API_KEY)")

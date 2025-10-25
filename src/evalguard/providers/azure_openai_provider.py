from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, cast

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider
from .telemetry import ProviderCallDetails

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
        api_key = config.metadata.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = config.metadata.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ProviderError("Azure OpenAI endpoint missing (AZURE_OPENAI_ENDPOINT)")
        self._client = cast(AzureOpenAIClient, azure_cls(api_key=api_key, azure_endpoint=endpoint))
        self._deployment = (
            config.metadata.get("deployment_name")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or config.model
        )

    def _call_model(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> ProviderCallDetails:
        try:
            if hasattr(self._client, "chat") and hasattr(self._client.chat, "completions"):
                chat_client = self._client.chat
                create_chat = cast(Callable[..., Any], chat_client.completions.create)
                chat_response: Any = create_chat(
                    model=self._deployment,
                    messages=[
                        {"role": "system", "content": system or "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.config.max_tokens,
                    stop=stop,
                )
                message = chat_response.choices[0].message
                if isinstance(message, dict):
                    output_text = cast(str, message.get("content", ""))
                else:
                    output_text = cast(str, getattr(message, "content", ""))
                prompt_tokens, completion_tokens = _extract_usage(chat_response)
                return ProviderCallDetails(
                    text=output_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    metadata={"request_id": getattr(chat_response, "id", None)},
                )
            raise ProviderError("Azure OpenAI client missing chat completions interface")
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(f"Azure OpenAI generate failed: {exc}") from exc

    def validate(self) -> None:
        if not getattr(self._client, "api_key", None):
            raise ProviderError("Azure OpenAI API key missing (AZURE_OPENAI_API_KEY)")


def _extract_usage(payload: Any) -> tuple[int, int]:
    usage = getattr(payload, "usage", None)
    if usage is None and isinstance(payload, dict):
        usage = payload.get("usage")
    if not usage:
        return 0, 0
    usage_dict = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
    prompt_tokens = (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or (usage_dict.get("prompt_tokens") if isinstance(usage_dict, dict) else None)
        or (usage_dict.get("input_tokens") if isinstance(usage_dict, dict) else None)
        or 0
    )
    completion_tokens = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or (usage_dict.get("completion_tokens") if isinstance(usage_dict, dict) else None)
        or (usage_dict.get("output_tokens") if isinstance(usage_dict, dict) else None)
        or 0
    )
    return int(prompt_tokens or 0), int(completion_tokens or 0)

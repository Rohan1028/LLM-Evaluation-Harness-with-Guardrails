from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, cast

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider
from .telemetry import ProviderCallDetails

LOGGER = get_logger(__name__)

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = Any


@register_provider("openai")
class OpenAIProvider(Provider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            module = importlib.import_module("openai")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError("openai package is required for OpenAIProvider") from exc
        if not hasattr(module, "OpenAI"):
            raise ProviderError("openai package is required for OpenAIProvider")
        openai_cls = module.OpenAI
        api_key = config.metadata.get("api_key")
        self._client = cast(OpenAIClient, openai_cls(api_key=api_key))
        self._supports_responses = False

    def _call_model(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> ProviderCallDetails:
        try:
            if self._supports_responses and hasattr(self._client, "responses"):
                responses = self._client.responses
                create = cast(Callable[..., Any], responses.create)
                input_payload: Any
                if system:
                    input_payload = [
                        {"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    ]
                else:
                    input_payload = prompt
                response: Any = create(
                    model=self.model,
                    input=input_payload,
                    temperature=self.temperature,
                    max_output_tokens=self.config.max_tokens,
                    stop=stop,
                )
                output_text = ""
                if hasattr(response, "output") and response.output:
                    segments: list[str] = []
                    for item in response.output:
                        content = item.get("content", [{}])[0].get("text", "")
                        segments.append(cast(str, content))
                    output_text = "".join(segments)
                if hasattr(response, "output_text"):
                    output_text = cast(str, response.output_text)
                prompt_tokens, completion_tokens = _extract_usage(response)
                return ProviderCallDetails(
                    text=output_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    metadata={"request_id": getattr(response, "id", None)},
                )
            if not hasattr(self._client, "chat"):
                raise ProviderError("Chat completions unavailable for this client")
            chat_client = self._client.chat
            if not hasattr(chat_client, "completions"):
                raise ProviderError("Chat completions unavailable for this client")
            create_chat = cast(Callable[..., Any], chat_client.completions.create)
            chat: Any = create_chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.config.max_tokens,
                stop=stop,
            )
            message = chat.choices[0].message
            if isinstance(message, dict):
                output_text = cast(str, message.get("content", ""))
            elif hasattr(message, "content"):
                output_text = cast(str, message.content)
            else:
                output_text = str(message)
            prompt_tokens, completion_tokens = _extract_usage(chat)
            return ProviderCallDetails(
                text=output_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={"request_id": getattr(chat, "id", None)},
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise ProviderError(f"OpenAI generate failed: {exc}") from exc

    def validate(self) -> None:
        api_key = getattr(self._client, "api_key", None)
        if not api_key:
            raise ProviderError("OpenAI API key missing (OPENAI_API_KEY)")


def _extract_usage(payload: Any) -> tuple[int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    usage = getattr(payload, "usage", None)
    if usage is None and isinstance(payload, dict):
        usage = payload.get("usage")
    if usage:
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

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, cast

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider

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
        self._supports_responses = hasattr(self._client, "responses")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        try:
            if self._supports_responses and hasattr(self._client, "responses"):
                responses = self._client.responses
                create = cast(Callable[..., Any], responses.create)
                response: Any = create(
                    model=self.model,
                    input=prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.config.max_tokens,
                    system=system or "",
                    stop=stop,
                )
                if hasattr(response, "output") and response.output:
                    segments: list[str] = []
                    for item in response.output:
                        content = item.get("content", [{}])[0].get("text", "")
                        segments.append(cast(str, content))
                    return "".join(segments)
                if hasattr(response, "output_text"):
                    return cast(str, response.output_text)
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
                return cast(str, message.get("content", ""))
            if hasattr(message, "content"):
                return cast(str, message.content)
            return str(message)
        except Exception as exc:  # pragma: no cover - network dependent
            raise ProviderError(f"OpenAI generate failed: {exc}") from exc

    def validate(self) -> None:
        api_key = getattr(self._client, "api_key", None)
        if not api_key:
            raise ProviderError("OpenAI API key missing (OPENAI_API_KEY)")

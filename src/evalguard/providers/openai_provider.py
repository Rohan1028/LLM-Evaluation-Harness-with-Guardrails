from __future__ import annotations

from typing import Iterable, Optional

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider

LOGGER = get_logger(__name__)


@register_provider("openai")
class OpenAIProvider(Provider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError("openai package is required for OpenAIProvider") from exc
        api_key = config.metadata.get("api_key")
        self._client = OpenAI(api_key=api_key)
        self._supports_responses = hasattr(self._client, "responses")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        try:
            if self._supports_responses:
                response = self._client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.config.max_tokens,
                    system=system or "",
                    stop=stop,
                )
                if hasattr(response, "output") and response.output:
                    return "".join(item.get("content", [{}])[0].get("text", "") for item in response.output)
                return getattr(response, "output_text", "")
            chat = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.config.max_tokens,
                stop=stop,
            )
            return chat.choices[0].message["content"]
        except Exception as exc:  # pragma: no cover - network dependent
            raise ProviderError(f"OpenAI generate failed: {exc}") from exc

    def validate(self) -> None:
        api_key = self._client.api_key  # type: ignore[attr-defined]
        if not api_key:
            raise ProviderError("OpenAI API key missing (OPENAI_API_KEY)")

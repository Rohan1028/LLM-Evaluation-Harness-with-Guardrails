from __future__ import annotations

from typing import Iterable, Optional

import httpx

from ..config import ProviderConfig
from ..logging import get_logger
from .base import Provider, ProviderError, register_provider

LOGGER = get_logger(__name__)


@register_provider("ollama")
class OllamaProvider(Provider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._base_url = config.metadata.get("api_base") or "http://localhost:11434"

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt if system is None else f"{system}\n\n{prompt}",
            "options": {"temperature": self.temperature},
            "stream": False,
        }
        try:
            response = httpx.post(f"{self._base_url}/api/generate", json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            text = data.get("response") or data.get("output") or ""
            if stop:
                for token in stop:
                    text = text.split(token)[0]
            return text
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(f"Ollama generate failed: {exc}") from exc

    def validate(self) -> None:
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            resp.raise_for_status()
        except Exception as exc:
            LOGGER.warning("Ollama ping failed: %s", exc)

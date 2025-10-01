from __future__ import annotations

from typing import Dict, Tuple

from ..config import ProviderConfig
from .base import (
    PROVIDER_REGISTRY,
    Provider,
    ProviderError,
    create_provider,
    parse_provider_spec,
)
from . import openai_provider  # noqa: F401
from . import anthropic_provider  # noqa: F401
from . import azure_openai_provider  # noqa: F401
from . import ollama_provider  # noqa: F401

__all__ = [
    "Provider",
    "ProviderConfig",
    "ProviderError",
    "create_provider",
    "parse_provider_spec",
    "PROVIDER_REGISTRY",
]


def build_provider_from_spec(
    spec: str,
    available_configs: Dict[str, ProviderConfig],
) -> Tuple[str, Provider]:
    provider_name, model_name = parse_provider_spec(spec)
    if provider_name not in available_configs:
        raise ProviderError(f"No configuration found for provider '{provider_name}'")
    config = available_configs[provider_name].with_model(model_name)
    provider = create_provider(config)
    return provider_name, provider

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ProviderCallTelemetry:
    provider: str
    model: str
    status: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    retries: int
    queued_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "retries": self.retries,
            "queued_ms": round(self.queued_ms, 3),
            "error": self.error,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def attach(self, **extra: Any) -> "ProviderCallTelemetry":
        merged = {**self.metadata, **extra}
        return ProviderCallTelemetry(
            provider=self.provider,
            model=self.model,
            status=self.status,
            latency_ms=self.latency_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cost_usd=self.cost_usd,
            retries=self.retries,
            queued_ms=self.queued_ms,
            error=self.error,
            timestamp=self.timestamp,
            metadata=merged,
        )


@dataclass
class ProviderCallDetails:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    text: str
    telemetry: ProviderCallTelemetry

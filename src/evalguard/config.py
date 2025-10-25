from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_NAME = "config.yaml"


class GlobalConfig(BaseModel):
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    persist_dir: Path = Path(".vectorstore")
    reports_dir: Path = Path("reports")
    random_seed: int = 42
    device: str = "cpu"

    @model_validator(mode="after")
    def resolve_paths(self) -> "GlobalConfig":
        root = self.project_root.resolve()
        self.data_dir = (root / self.data_dir).resolve()
        self.persist_dir = (root / self.persist_dir).resolve()
        self.reports_dir = (root / self.reports_dir).resolve()
        return self


class RateLimitConfig(BaseModel):
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    burst: Optional[int] = None
    concurrent_requests: int = 1
    max_retries: int = 2
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    jitter: float = 0.25


class CostConfig(BaseModel):
    prompt_per_1k: float = 0.0
    completion_per_1k: float = 0.0

    def estimate(self, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1000.0) * self.prompt_per_1k
        completion_cost = (completion_tokens / 1000.0) * self.completion_per_1k
        return round(prompt_cost + completion_cost, 6)


class TelemetryConfig(BaseModel):
    enabled: bool = True
    redact_prompts: bool = True
    redact_responses: bool = False
    redact_context: bool = True
    persist_requests_jsonl: bool = True
    persist_requests_csv: bool = True


class ProviderConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 60.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rate_limit: Optional[RateLimitConfig] = None
    cost: Optional[CostConfig] = None

    def with_model(self, model: str) -> "ProviderConfig":
        data = self.model_dump()
        data["model"] = model
        return ProviderConfig(**data)


class RAGConfig(BaseModel):
    retriever_top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 120
    collection: str = "demo"


class GuardrailConfig(BaseModel):
    min_citations: int = 2
    toxicity_threshold: float = 0.2
    factcheck_required: bool = True
    max_retries: int = 1
    rail_spec_enabled: bool = False
    rail_spec_path: Optional[Path] = Path("data/rails/answer_schema.json")
    toxicity_labels: tuple[str, ...] = ("toxic", "insult", "hate", "threat")
    banned_keywords: tuple[str, ...] = (
        "attack",
        "exploit",
        "malware",
        "weapon",
        "bomb",
        "hate",
        "incite",
    )

    @model_validator(mode="after")
    def resolve_spec(self) -> "GuardrailConfig":
        if self.rail_spec_path:
            self.rail_spec_path = self.rail_spec_path.expanduser().resolve()
        return self


class PolicyThresholds(BaseModel):
    faithfulness: float = 0.75
    answer_relevancy: float = 0.12
    context_precision: float = 0.55
    context_recall: float = 0.95
    coherence: float = 0.90
    toxicity: float = 0.20

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class Settings(BaseModel):
    global_config: GlobalConfig = Field(default_factory=GlobalConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    policies: PolicyThresholds = Field(default_factory=PolicyThresholds)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)

    @property
    def project_root(self) -> Path:
        return self.global_config.project_root

    @property
    def data_dir(self) -> Path:
        return self.global_config.data_dir

    @property
    def persist_dir(self) -> Path:
        return self.global_config.persist_dir

    @property
    def reports_dir(self) -> Path:
        return self.global_config.reports_dir

    @property
    def random_seed(self) -> int:
        return self.global_config.random_seed

    def get_provider_config(
        self, name: str, model_override: Optional[str] = None
    ) -> ProviderConfig:
        if name not in self.providers:
            raise KeyError(f"Provider '{name}' not found in configuration")
        config = self.providers[name]
        if model_override:
            return config.with_model(model_override)
        return config

    @classmethod
    def default(cls) -> "Settings":
        return cls(
            providers={
                "mock": ProviderConfig(provider="mock", model="deterministic", temperature=0.1),
                "openai": ProviderConfig(
                    provider="openai",
                    model="gpt-4o-mini",
                    temperature=0.1,
                    rate_limit=RateLimitConfig(
                        requests_per_minute=1000,
                        tokens_per_minute=450_000,
                        max_retries=3,
                    ),
                    cost=CostConfig(prompt_per_1k=0.03, completion_per_1k=0.06),
                ),
                "anthropic": ProviderConfig(
                    provider="anthropic",
                    model="claude-3-5-sonnet-20240620",
                    rate_limit=RateLimitConfig(
                        requests_per_minute=100,
                        tokens_per_minute=20_000,
                        max_retries=3,
                    ),
                    cost=CostConfig(prompt_per_1k=0.003, completion_per_1k=0.015),
                ),
                "azure-openai": ProviderConfig(
                    provider="azure-openai",
                    model="gpt-4o",
                    rate_limit=RateLimitConfig(requests_per_minute=30, max_retries=3),
                    cost=CostConfig(prompt_per_1k=0.005, completion_per_1k=0.015),
                ),
                "ollama": ProviderConfig(
                    provider="ollama",
                    model="llama3",
                    rate_limit=RateLimitConfig(requests_per_minute=120, max_retries=1),
                    cost=CostConfig(prompt_per_1k=0.0, completion_per_1k=0.0),
                ),
            }
        )


DEFAULT_SETTINGS = Settings.default()


def load_settings(config_path: Optional[Path] = None) -> Settings:
    if config_path:
        path = config_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        merged = _merge_dicts(DEFAULT_SETTINGS.model_dump(mode="json"), data)
        return _apply_env_overrides(Settings(**merged))

    for candidate in (
        Path(os.environ.get("EVALGUARD_CONFIG", "")),
        Path.cwd() / DEFAULT_CONFIG_NAME,
        DEFAULT_SETTINGS.project_root / DEFAULT_CONFIG_NAME,
    ):
        if not candidate or candidate.is_dir() or not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        merged = _merge_dicts(DEFAULT_SETTINGS.model_dump(mode="json"), data)
        return _apply_env_overrides(Settings(**merged))

    return _apply_env_overrides(Settings(**DEFAULT_SETTINGS.model_dump(mode="json")))


def export_default_config(path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(DEFAULT_SETTINGS.model_dump(mode="json"), fh, sort_keys=False)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(settings: Settings) -> Settings:
    telemetry_updates: Dict[str, Any] = {}
    telemetry_env = {
        "enabled": os.getenv("TELEMETRY_ENABLED"),
        "redact_prompts": os.getenv("TELEMETRY_REDACT_PROMPTS"),
        "redact_responses": os.getenv("TELEMETRY_REDACT_RESPONSES"),
        "redact_context": os.getenv("TELEMETRY_REDACT_CONTEXT"),
    }
    for field_name, env_value in telemetry_env.items():
        if env_value is None:
            continue
        telemetry_updates[field_name] = _env_to_bool(env_value)
    if telemetry_updates:
        data = settings.telemetry.model_dump()
        data.update(telemetry_updates)
        settings.telemetry = TelemetryConfig(**data)

    updated_providers: Dict[str, ProviderConfig] = {}
    for name, provider in settings.providers.items():
        prefix = name.replace("-", "_").upper()
        rpm_env = os.getenv(f"{prefix}_MAX_TPS")
        prompt_env = os.getenv(f"{prefix}_COST_INPUT_PER_1K")
        completion_env = os.getenv(f"{prefix}_COST_OUTPUT_PER_1K")
        cost_updates = {}
        if prompt_env is not None:
            cost_updates["prompt_per_1k"] = float(prompt_env)
        if completion_env is not None:
            cost_updates["completion_per_1k"] = float(completion_env)
        rate_updates = {}
        if rpm_env is not None:
            rate_updates["requests_per_minute"] = int(rpm_env)
        rate_limit = provider.rate_limit or RateLimitConfig()
        cost_config = provider.cost or CostConfig()
        if rate_updates:
            rl_data = rate_limit.model_dump()
            rl_data.update(rate_updates)
            rate_limit = RateLimitConfig(**rl_data)
        if cost_updates:
            cost_data = cost_config.model_dump()
            cost_data.update(cost_updates)
            cost_config = CostConfig(**cost_data)
        provider_data = provider.model_dump()
        provider_data["rate_limit"] = rate_limit.model_dump()
        provider_data["cost"] = cost_config.model_dump()
        updated_providers[name] = ProviderConfig(**provider_data)
    settings.providers = updated_providers
    return settings


def _env_to_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}

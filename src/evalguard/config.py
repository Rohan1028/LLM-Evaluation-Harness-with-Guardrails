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
    rail_spec_path: Optional[Path] = None
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
    answer_relevancy: float = 0.70
    context_precision: float = 0.65
    context_recall: float = 0.60
    coherence: float = 0.70
    toxicity: float = 0.20

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class Settings(BaseModel):
    global_config: GlobalConfig = Field(default_factory=GlobalConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    policies: PolicyThresholds = Field(default_factory=PolicyThresholds)
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
                "openai": ProviderConfig(provider="openai", model="gpt-4o-mini", temperature=0.1),
                "anthropic": ProviderConfig(
                    provider="anthropic", model="claude-3-5-sonnet-20240620"
                ),
                "azure-openai": ProviderConfig(provider="azure-openai", model="gpt-4o"),
                "ollama": ProviderConfig(provider="ollama", model="llama3"),
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
        return Settings(**merged)

    for candidate in (
        Path(os.environ.get("EVALGUARD_CONFIG", "")),
        Path.cwd() / DEFAULT_CONFIG_NAME,
        DEFAULT_SETTINGS.project_root / DEFAULT_CONFIG_NAME,
    ):
        if candidate and candidate.exists():
            with candidate.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            merged = _merge_dicts(DEFAULT_SETTINGS.model_dump(mode="json"), data)
            return Settings(**merged)

    return DEFAULT_SETTINGS


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

"""Configuration profiles for plug-and-play integration with external models.

Profiles are dataclasses that describe how to connect to a model provider,
how to generate embeddings, and what token budgets / retrieval parameters
to use. Load from YAML/JSON or instantiate directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from mnemosyne.exceptions import ConfigurationError
from mnemosyne.schemas import ModelProvider


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding sub-system."""

    provider: str = "local"  # local | openai | ollama | vllm
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    dimensions: int = 384
    batch_size: int = 32
    normalize: bool = True


@dataclass
class RetrievalConfig:
    """Configuration for dual-path retrieval and fusion."""

    mode: str = "hybrid"  # vector | graph | hybrid
    top_k: int = 10
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    min_importance: float = 0.1
    rrf_k: int = 60  # reciprocal-rank fusion constant
    graph_hop_depth: int = 2
    tag_filter: Optional[List[str]] = None


@dataclass
class ConsolidationConfig:
    """Configuration for the autonomous consolidation pipeline."""

    enabled: bool = True
    interval_seconds: int = 300
    max_blocks_per_run: int = 200
    importance_threshold: float = 0.3
    age_threshold_seconds: int = 86400  # 1 day
    compression_ratio: float = 0.4
    prune_low_importance: bool = True
    extract_entities: bool = True
    worker_threads: int = 2


@dataclass
class TokenBudgetConfig:
    """Configuration for dynamic token-budget calculation."""

    total_context_window: int = 8192
    system_reserved: int = 512
    user_input_reserve: int = 1024
    retrieval_ratio: float = 0.35
    overhead_ratio: float = 0.05
    max_blocks: int = 20


@dataclass
class ModelProfile:
    """A plug-and-play profile for a specific external model provider."""

    name: str
    provider: ModelProvider
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_seconds: int = 60
    extra: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    @classmethod
    def openai(
        cls,
        name: str = "gpt-4o",
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        **kwargs,
    ) -> "ModelProfile":
        return cls(
            name=name,
            provider=ModelProvider.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=model,
            **kwargs,
        )

    @classmethod
    def anthropic(
        cls,
        name: str = "claude-3-opus",
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        **kwargs,
    ) -> "ModelProfile":
        return cls(
            name=name,
            provider=ModelProvider.ANTHROPIC,
            base_url="https://api.anthropic.com",
            api_key=api_key,
            model=model,
            **kwargs,
        )

    @classmethod
    def ollama(
        cls,
        name: str = "ollama-local",
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        **kwargs,
    ) -> "ModelProfile":
        return cls(
            name=name,
            provider=ModelProvider.OLLAMA,
            base_url=base_url,
            model=model,
            **kwargs,
        )

    @classmethod
    def vllm(
        cls,
        name: str = "vllm-local",
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        **kwargs,
    ) -> "ModelProfile":
        return cls(
            name=name,
            provider=ModelProvider.VLLM,
            base_url=base_url,
            model=model,
            **kwargs,
        )


@dataclass
class MnemosyneConfig:
    """Top-level configuration aggregating all sub-configs.

    Example::

        cfg = MnemosyneConfig(
            embedding=EmbeddingConfig(provider="ollama", model="nomic-embed-text"),
            profiles={
                "default": ModelProfile.openai(api_key="sk-..."),
            },
        )
    """

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    profiles: Dict[str, ModelProfile] = field(default_factory=dict)
    default_profile: Optional[str] = None
    default_tenant: str = "default"
    storage_path: str = "./mnemosyne_data"
    db_echo: bool = False

    def register_profile(self, profile: ModelProfile) -> None:
        self.profiles[profile.name] = profile
        if self.default_profile is None:
            self.default_profile = profile.name

    def get_profile(self, name: Optional[str] = None) -> ModelProfile:
        name = name or self.default_profile
        if name is None or name not in self.profiles:
            raise ConfigurationError(
                f"No model profile named '{name}' registered. "
                f"Available: {list(self.profiles.keys())}",
                code="profile_not_found",
                details={"available": list(self.profiles.keys())},
            )
        return self.profiles[name]


def load_config_from_dict(data: Dict[str, Any]) -> MnemosyneConfig:
    """Build a MnemosyneConfig from a plain dict (e.g. parsed YAML/JSON)."""
    embedding = EmbeddingConfig(**data.get("embedding", {}))
    retrieval = RetrievalConfig(**data.get("retrieval", {}))
    consolidation = ConsolidationConfig(**data.get("consolidation", {}))
    token_budget = TokenBudgetConfig(**data.get("token_budget", {}))
    profiles = {}
    for name, pdata in data.get("profiles", {}).items():
        provider = pdata.get("provider", "openai")
        profiles[name] = ModelProfile(
            name=name,
            provider=ModelProvider(provider),
            base_url=pdata.get("base_url"),
            api_key=pdata.get("api_key"),
            model=pdata.get("model", ""),
            temperature=pdata.get("temperature", 0.7),
            max_tokens=pdata.get("max_tokens", 2048),
            timeout_seconds=pdata.get("timeout_seconds", 60),
            extra=pdata.get("extra", {}),
        )
    return MnemosyneConfig(
        embedding=embedding,
        retrieval=retrieval,
        consolidation=consolidation,
        token_budget=token_budget,
        profiles=profiles,
        default_profile=data.get("default_profile"),
        storage_path=data.get("storage_path", "./mnemosyne_data"),
        db_echo=data.get("db_echo", False),
    )


__all__ = [
    "EmbeddingConfig",
    "RetrievalConfig",
    "ConsolidationConfig",
    "TokenBudgetConfig",
    "ModelProfile",
    "MnemosyneConfig",
    "load_config_from_dict",
]

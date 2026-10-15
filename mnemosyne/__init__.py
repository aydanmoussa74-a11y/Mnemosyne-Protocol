"""Mnemosyne Protocol — Hyper-scaling, model-agnostic agent memory engine.

A hybrid vector & knowledge graph memory infrastructure with autonomous
consolidation, dynamic context injection, and universal model-agnostic hooks.

Version: 1.0.0
"""

__version__ = "1.0.0"

from mnemosyne.core import MnemosyneEngine
from mnemosyne.config import MnemosyneConfig, ModelProfile
from mnemosyne.schemas import (
    MemoryBlock,
    MemoryEntity,
    MemoryRelation,
    MemoryEdge,
    ConsolidationJob,
    TokenBudget,
    InjectionResult,
    RetrievalResult,
    MemorySearchQuery,
)
from mnemosyne.exceptions import MnemosyneError

__all__ = [
    "__version__",
    "MnemosyneEngine",
    "MnemosyneConfig",
    "ModelProfile",
    "MemoryBlock",
    "MemoryEntity",
    "MemoryRelation",
    "MemoryEdge",
    "ConsolidationJob",
    "TokenBudget",
    "InjectionResult",
    "RetrievalResult",
    "MemorySearchQuery",
    "MnemosyneError",
]

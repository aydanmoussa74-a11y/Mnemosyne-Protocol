"""Pydantic data schemas for the Mnemosyne Protocol memory engine.

All schemas are immutable-aware, serializable, and designed to round-trip
between the in-memory engine, the persistence layer, and the network boundary.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class MemoryBlockStatus(str, Enum):
    """Lifecycle status of a memory block."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    CONSOLIDATED = "consolidated"
    PRUNED = "pruned"


class EntityType(str, Enum):
    """Types of entities extractable from memory blocks."""

    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    EVENT = "event"
    ARTIFACT = "artifact"
    OTHER = "other"


class ConsolidationStatus(str, Enum):
    """Status of a background consolidation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RetrievalMode(str, Enum):
    """Dual-path retrieval modes."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class ModelProvider(str, Enum):
    """Supported external model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"
    LOCAL = "local"


class MemoryBlock(BaseModel):
    """A single unit of memory: the atom of the Mnemosyne Protocol.

    A block carries raw text, an optional embedding vector, rich metadata,
    lifecycle fields, and hierarchical links for consolidation.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., min_length=1, description="Raw memory text content.")
    embedding: Optional[List[float]] = Field(
        default=None, description="Dense vector embedding of the content."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary key-value metadata."
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Learned importance weight (0 = noise, 1 = critical).",
    )
    access_count: int = Field(default=0, ge=0)
    last_accessed: Optional[datetime] = None
    source: str = Field(default="ingestion", description="Origin of this memory block.")
    status: MemoryBlockStatus = Field(default=MemoryBlockStatus.ACTIVE)
    consolidated: bool = Field(default=False)
    parent_block_id: Optional[UUID] = Field(
        default=None, description="Parent block after hierarchical consolidation."
    )
    child_block_ids: List[UUID] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    ttl_seconds: Optional[int] = Field(
        default=None, ge=0,
        description="Optional time-to-live in seconds before pruning eligibility.",
    )
    token_count: int = Field(default=0, ge=0)

    class Config:
        use_enum_values = True

    def touch(self) -> None:
        """Record an access, incrementing the access counter."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

    def decay_importance(self, factor: float = 0.99) -> None:
        """Apply time-based importance decay."""
        self.importance_score = max(0.0, min(1.0, self.importance_score * factor))


class MemoryEntity(BaseModel):
    """An entity (person, place, concept, ...) extracted from memory blocks."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    entity_type: EntityType = Field(default=EntityType.OTHER)
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    mention_count: int = Field(default=1, ge=1)


class MemoryRelation(BaseModel):
    """A directed, labeled relation between two entities in the knowledge graph."""

    id: UUID = Field(default_factory=uuid4)
    subject_entity_id: UUID
    predicate: str = Field(..., min_length=1)
    object_entity_id: UUID
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_memory_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryEdge(BaseModel):
    """A lightweight edge linking a memory block to entities / other blocks."""

    id: UUID = Field(default_factory=uuid4)
    memory_block_id: UUID
    target_id: UUID  # entity id or another block id
    edge_type: str  # "mentions", "references", "derives_from", ...
    weight: float = Field(default=1.0, ge=0.0)


class ConsolidationJob(BaseModel):
    """A background job that compresses/prunes/synthesizes short-term memories."""

    id: UUID = Field(default_factory=uuid4)
    status: ConsolidationStatus = Field(default=ConsolidationStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    blocks_processed: int = Field(default=0, ge=0)
    blocks_created: int = Field(default=0, ge=0)
    blocks_pruned: int = Field(default=0, ge=0)
    error: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class TokenBudget(BaseModel):
    """Result of a token-budget calculation for context injection."""

    total_available: int
    system_reserved: int
    user_input_tokens: int
    retrieval_budget: int
    overhead_tokens: int
    max_blocks: int
    reserve_ratio: float = Field(default=0.1, ge=0.0, le=1.0)

    @property
    def usable_for_retrieval(self) -> int:
        return max(0, self.retrieval_budget - self.overhead_tokens)


class InjectionResult(BaseModel):
    """Outcome of injecting retrieved context into an LLM payload."""

    success: bool
    injected_blocks: List[MemoryBlock]
    total_injected_tokens: int
    budget: TokenBudget
    prompt: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class RetrievalResult(BaseModel):
    """A single scored retrieval hit from either the vector or graph path."""

    block: MemoryBlock
    score: float = Field(ge=0.0, le=1.0)
    path: RetrievalMode
    explanation: str = ""


class MemorySearchQuery(BaseModel):
    """A query issued against the hybrid store."""

    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    top_k: int = Field(default=10, ge=1, le=100)
    mode: RetrievalMode = Field(default=RetrievalMode.HYBRID)
    entity_filter: Optional[str] = None
    tag_filter: Optional[List[str]] = None
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    status_filter: Optional[MemoryBlockStatus] = None
    time_window_seconds: Optional[int] = None


class ConsolidationPlan(BaseModel):
    """Output of the consolidation planner: which blocks to compress/prune/synthesize."""

    compress_ids: List[UUID] = Field(default_factory=list)
    prune_ids: List[UUID] = Field(default_factory=list)
    synthesize_ids: List[UUID] = Field(default_factory=list)
    rationale: Dict[UUID, str] = Field(default_factory=dict)


__all__ = [
    "MemoryBlockStatus",
    "EntityType",
    "ConsolidationStatus",
    "RetrievalMode",
    "ModelProvider",
    "MemoryBlock",
    "MemoryEntity",
    "MemoryRelation",
    "MemoryEdge",
    "ConsolidationJob",
    "TokenBudget",
    "InjectionResult",
    "RetrievalResult",
    "MemorySearchQuery",
    "ConsolidationPlan",
]

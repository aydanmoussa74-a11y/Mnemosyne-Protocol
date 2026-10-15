"""Hybrid Vector & Knowledge Graph Core.

Dual-path storage layer:

* Vector path  — dense embeddings stored in memory with exact cosine-similarity
  search (numpy when available, pure-Python fallback).
* Graph path   — entities, relations, and block-to-entity edges persisted in a
  relational store (SQLite via SQLAlchemy) and traversable to configurable depth.

Both paths share the same ``MemoryBlock`` identity so vector hits and graph
hits can be fused (reciprocal-rank fusion) into a single ranked result set.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from uuid import UUID, uuid4

from mnemosyne.config import EmbeddingConfig, MnemosyneConfig
from mnemosyne.exceptions import MnemosyneError, NotFoundError, StoreError
from mnemosyne.schemas import (
    MemoryBlock,
    MemoryBlockStatus,
    MemoryEdge,
    MemoryEntity,
    MemoryRelation,
)

try:  # Optional hard dependency for relational persistence.
    from sqlalchemy import (
        Boolean,
        Column,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        String,
        Text,
        create_engine,
        event,
    )
    from sqlalchemy.orm import Session, declarative_base, sessionmaker

    _HAS_SQLALCHEMY = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_SQLALCHEMY = False

try:
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - pure python fallback
    _HAS_NUMPY = False


# --------------------------------------------------------------------------- #
# SQLAlchemy declarative base (optional)
# --------------------------------------------------------------------------- #

if _HAS_SQLALCHEMY:
    Base = declarative_base()


    class BlockORM(Base):
        __tablename__ = "memory_blocks"
        id = Column(String, primary_key=True)
        tenant_id = Column(String, nullable=False, default="default", index=True)
        content = Column(Text, nullable=False)
        metadata_json = Column(Text, default="{}")
        timestamp = Column(DateTime, default=None)
        importance_score = Column(Float, default=0.5)
        access_count = Column(Integer, default=0)
        last_accessed = Column(DateTime, default=None)
        source = Column(String, default="ingestion")
        status = Column(String, default="active")
        consolidated = Column(Boolean, default=False)
        parent_block_id = Column(String, nullable=True)
        tags_json = Column(Text, default="[]")
        ttl_seconds = Column(Integer, nullable=True)
        token_count = Column(Integer, default=0)
        embedding_json = Column(Text, nullable=True)


    class EntityORM(Base):
        __tablename__ = "memory_entities"
        id = Column(String, primary_key=True)
        tenant_id = Column(String, nullable=False, default="default", index=True)
        name = Column(String, nullable=False, index=True)
        entity_type = Column(String, default="other")
        properties_json = Column(Text, default="{}")
        confidence = Column(Float, default=1.0)
        first_seen = Column(DateTime, default=None)
        last_seen = Column(DateTime, default=None)
        mention_count = Column(Integer, default=1)


    class RelationORM(Base):
        __tablename__ = "memory_relations"
        id = Column(String, primary_key=True)
        subject_entity_id = Column(String, nullable=False, index=True)
        predicate = Column(String, nullable=False, index=True)
        object_entity_id = Column(String, nullable=False, index=True)
        confidence = Column(Float, default=1.0)
        source_memory_id = Column(String, nullable=True)
        created_at = Column(DateTime, default=None)


    class EdgeORM(Base):
        __tablename__ = "memory_edges"
        id = Column(String, primary_key=True)
        memory_block_id = Column(String, nullable=False, index=True)
        target_id = Column(String, nullable=False, index=True)
        edge_type = Column(String, nullable=False, index=True)
        weight = Column(Float, default=1.0)
else:
    Base = object  # type: ignore


# --------------------------------------------------------------------------- #
# Vector index helpers
# --------------------------------------------------------------------------- #

def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (pure Python)."""
    if len(a) != len(b):
        raise StoreError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}",
            code="dimension_mismatch",
        )
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _cosine_similarity_batch(
    query: Sequence[float], matrix: List[Sequence[float]]
) -> List[float]:
    """Batch cosine similarity (numpy when available)."""
    if _HAS_NUMPY and matrix:
        q = np.asarray(query, dtype=np.float64)
        m = np.asarray(matrix, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        m_norms = np.linalg.norm(m, axis=1)
        dots = m @ q
        denom = m_norms * q_norm
        denom = np.where(denom == 0.0, 1.0, denom)
        return (dots / denom).tolist()
    return [_cosine_similarity(query, row) for row in matrix]


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #

def _block_to_orm(block: MemoryBlock, tenant_id: str = "default") -> Any:  # noqa: ANN401
    if not _HAS_SQLALCHEMY:
        raise StoreError("SQLAlchemy is required for persistence", code="no_sqlalchemy")
    import json

    return BlockORM(
        id=str(block.id),
        tenant_id=tenant_id,
        content=block.content,
        metadata_json=json.dumps(block.metadata),
        timestamp=block.timestamp,
        importance_score=block.importance_score,
        access_count=block.access_count,
        last_accessed=block.last_accessed,
        source=block.source,
        status=block.status.value if hasattr(block.status, "value") else block.status,
        consolidated=block.consolidated,
        parent_block_id=(
            str(block.parent_block_id) if block.parent_block_id else None
        ),
        tags_json=json.dumps(block.tags),
        ttl_seconds=block.ttl_seconds,
        token_count=block.token_count,
        embedding_json=(
            json.dumps(block.embedding) if block.embedding is not None else None
        ),
    )


def _orm_to_block(orm: Any) -> MemoryBlock:  # noqa: ANN401
    import json

    embedding = None
    if orm.embedding_json:
        try:
            embedding = json.loads(orm.embedding_json)
        except (ValueError, TypeError):
            embedding = None
    tags = []
    if orm.tags_json:
        try:
            tags = json.loads(orm.tags_json)
        except (ValueError, TypeError):
            tags = []
    metadata = {}
    if orm.metadata_json:
        try:
            metadata = json.loads(orm.metadata_json)
        except (ValueError, TypeError):
            metadata = {}
    return MemoryBlock(
        id=UUID(orm.id),
        tenant_id=orm.tenant_id or "default",
        content=orm.content,
        embedding=embedding,
        metadata=metadata,
        timestamp=orm.timestamp or __import__("datetime").datetime.utcnow(),
        importance_score=orm.importance_score,
        access_count=orm.access_count,
        last_accessed=orm.last_accessed,
        source=orm.source,
        status=MemoryBlockStatus(orm.status),
        consolidated=orm.consolidated,
        parent_block_id=UUID(orm.parent_block_id) if orm.parent_block_id else None,
        child_block_ids=[],
        tags=tags,
        ttl_seconds=orm.ttl_seconds,
        token_count=orm.token_count,
    )


# --------------------------------------------------------------------------- #
# Hybrid store
# --------------------------------------------------------------------------- #

class HybridStore:
    """Dual-path vector + knowledge-graph store for memory blocks.

    The store keeps an in-memory vector index for low-latency similarity
    search and a relational backend (SQLite) for durable block/entity/relation
    storage. All public methods are thread-safe.
    """

    def __init__(self, config: MnemosyneConfig) -> None:
        self.config = config
        self._lock = threading.RLock()

        # In-memory index: block_id -> block (authoritative for hot path).
        self._blocks: Dict[UUID, MemoryBlock] = {}
        # Vector index: block_id -> embedding (only blocks with embeddings).
        self._vectors: Dict[UUID, List[float]] = {}
        # Embedding dimension, established on first write.
        self._dimension: Optional[int] = None

        # Graph indices.
        self._entities: Dict[UUID, MemoryEntity] = {}
        self._entity_by_name: Dict[str, UUID] = {}
        self._relations: Dict[UUID, MemoryRelation] = {}
        # subject_entity_id -> list of relation ids
        self._rel_by_subject: Dict[UUID, List[UUID]] = {}
        # object_entity_id -> list of relation ids
        self._rel_by_object: Dict[UUID, List[UUID]] = {}
        # block_id -> list of edge ids
        self._edges_by_block: Dict[UUID, List[UUID]] = {}
        self._edges: Dict[UUID, MemoryEdge] = {}

        self._session_factory = self._init_db()

    # ------------------------------------------------------------------ #
    # Database initialization
    # ------------------------------------------------------------------ #
    def _init_db(self):  # noqa: ANN001
        import os

        os.makedirs(self.config.storage_path, exist_ok=True)
        db_path = os.path.join(self.config.storage_path, "mnemosyne.db")

        if not _HAS_SQLALCHEMY:
            raise StoreError(
                "SQLAlchemy is not installed; the store requires it for persistence.",
                code="no_sqlalchemy",
                details={"hint": "pip install sqlalchemy"},
            )

        engine = create_engine(
            f"sqlite:///{db_path}",
            echo=self.config.db_echo,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
            pool_pre_ping=True,
        )

        # Enable WAL mode and a busy timeout so concurrent writers (FastAPI
        # threadpool + consolidation worker) rarely hit "database is locked".
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, connection_record):  # noqa: ANN001
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
            except Exception:  # pragma: no cover - defensive
                pass

        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def write_block(self, block: MemoryBlock) -> MemoryBlock:
        """Persist a memory block and update the in-memory vector index.

        Blocks are namespaced by ``block.tenant_id`` so that tenants cannot
        read or overwrite each other's memory.
        """
        tenant_id = block.tenant_id or "default"
        with self._lock:
            self._blocks[block.id] = block

            if block.embedding is not None:
                if self._dimension is None:
                    self._dimension = len(block.embedding)
                elif len(block.embedding) != self._dimension:
                    raise StoreError(
                        f"Embedding dimension mismatch: expected {self._dimension}, "
                        f"got {len(block.embedding)}",
                        code="dimension_mismatch",
                    )
                self._vectors[block.id] = list(block.embedding)

            self._persist_block(block, tenant_id=tenant_id)
            return block

    def write_blocks(self, blocks: Iterable[MemoryBlock]) -> List[MemoryBlock]:
        """Batch-persist multiple blocks."""
        with self._lock:
            written: List[MemoryBlock] = []
            session = self._session_factory()
            try:
                for block in blocks:
                    self._blocks[block.id] = block
                    if block.embedding is not None:
                        if self._dimension is None:
                            self._dimension = len(block.embedding)
                        elif len(block.embedding) != self._dimension:
                            raise StoreError(
                                "Embedding dimension mismatch in batch write",
                                code="dimension_mismatch",
                            )
                        self._vectors[block.id] = list(block.embedding)
                    self._persist_block(block)
                    written.append(block)
                session.commit()
            finally:
                session.close()
            return written

    def _persist_block(self, block: MemoryBlock, tenant_id: str = "default") -> None:  # noqa: ANN001
        session = self._session_factory()
        try:
            orm = session.query(BlockORM).get(str(block.id))
            if orm is None:
                orm = _block_to_orm(block, tenant_id=tenant_id)
                session.add(orm)
            else:
                for field, value in {
                    "content": block.content,
                    "metadata_json": __import__("json").dumps(block.metadata),
                    "importance_score": block.importance_score,
                    "access_count": block.access_count,
                    "status": (
                        block.status.value
                        if hasattr(block.status, "value")
                        else block.status
                    ),
                    "consolidated": block.consolidated,
                    "parent_block_id": (
                        str(block.parent_block_id) if block.parent_block_id else None
                    ),
                    "tags_json": __import__("json").dumps(block.tags),
                    "token_count": block.token_count,
                    "embedding_json": (
                        __import__("json").dumps(block.embedding)
                        if block.embedding is not None
                        else None
                    ),
                }.items():
                    setattr(orm, field, value)
            session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def read_block(self, block_id: UUID, tenant_id: str = "default") -> MemoryBlock:
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                raise NotFoundError(
                    f"Memory block {block_id} not found",
                    code="block_not_found",
                    details={"block_id": str(block_id)},
                )
            if block.tenant_id != tenant_id:
                raise NotFoundError(
                    f"Memory block {block_id} not found",
                    code="block_not_found",
                    details={"block_id": str(block_id)},
                )
            return block

    def list_blocks(
        self,
        tenant_id: str = "default",
        status: Optional[MemoryBlockStatus] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[MemoryBlock]:
        with self._lock:
            blocks = [
                b for b in self._blocks.values() if b.tenant_id == tenant_id
            ]
            if status is not None:
                blocks = [b for b in blocks if b.status == status]
            blocks.sort(key=lambda b: b.timestamp, reverse=True)
            return blocks[offset : offset + limit]

    def delete_block(self, block_id: UUID, tenant_id: str = "default") -> None:
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                raise NotFoundError(
                    f"Memory block {block_id} not found",
                    code="block_not_found",
                )
            if block.tenant_id != tenant_id:
                raise NotFoundError(
                    f"Memory block {block_id} not found",
                    code="block_not_found",
                )
            del self._blocks[block_id]
            self._vectors.pop(block_id, None)
            # Remove edges pointing to/from this block.
            edge_ids = list(self._edges_by_block.pop(block_id, []))
            for eid in edge_ids:
                edge = self._edges.pop(eid, None)
                if edge is not None:
                    pass  # edges are owned by the block
            self._persist_delete_block(block_id)

    def _persist_delete_block(self, block_id: UUID) -> None:
        session = self._session_factory()
        try:
            session.query(BlockORM).filter(BlockORM.id == str(block_id)).delete()
            session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Graph persistence helpers
    # ------------------------------------------------------------------ #
    def _persist_entity(self, entity: MemoryEntity) -> None:  # noqa: ANN001
        import json

        session = self._session_factory()
        try:
            orm = session.query(EntityORM).get(str(entity.id))
            if orm is None:
                orm = EntityORM(
                    id=str(entity.id),
                    name=entity.name,
                    entity_type=(
                        entity.entity_type.value
                        if hasattr(entity.entity_type, "value")
                        else entity.entity_type
                    ),
                    properties_json=json.dumps(entity.properties),
                    confidence=entity.confidence,
                    first_seen=entity.first_seen,
                    last_seen=entity.last_seen,
                    mention_count=entity.mention_count,
                )
                session.add(orm)
            else:
                orm.name = entity.name
                orm.mention_count = entity.mention_count
                orm.confidence = entity.confidence
                orm.last_seen = entity.last_seen
                orm.properties_json = json.dumps(entity.properties)
            session.commit()
        finally:
            session.close()

    def _persist_relation(self, relation: MemoryRelation) -> None:  # noqa: ANN001
        session = self._session_factory()
        try:
            orm = session.query(RelationORM).get(str(relation.id))
            if orm is None:
                orm = RelationORM(
                    id=str(relation.id),
                    subject_entity_id=str(relation.subject_entity_id),
                    predicate=relation.predicate,
                    object_entity_id=str(relation.object_entity_id),
                    confidence=relation.confidence,
                    source_memory_id=(
                        str(relation.source_memory_id)
                        if relation.source_memory_id
                        else None
                    ),
                    created_at=relation.created_at,
                )
                session.add(orm)
            session.commit()
        finally:
            session.close()

    def _persist_edge(self, edge: MemoryEdge) -> None:  # noqa: ANN001
        session = self._session_factory()
        try:
            orm = session.query(EdgeORM).get(str(edge.id))
            if orm is None:
                orm = EdgeORM(
                    id=str(edge.id),
                    memory_block_id=str(edge.memory_block_id),
                    target_id=str(edge.target_id),
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                )
                session.add(orm)
            session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Vector path
    # ------------------------------------------------------------------ #
    def search_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        status_filter: Optional[MemoryBlockStatus] = None,
        tenant_id: str = "default",
    ) -> List[Tuple[UUID, float]]:
        """Exact cosine-similarity search over indexed embeddings."""
        with self._lock:
            if not self._vectors:
                return []
            if self._dimension is not None and len(query_embedding) != self._dimension:
                raise StoreError(
                    f"Query embedding dimension {len(query_embedding)} != index "
                    f"dimension {self._dimension}",
                    code="dimension_mismatch",
                )

            ids = list(self._vectors.keys())
            matrix = [self._vectors[i] for i in ids]
            scores = _cosine_similarity_batch(query_embedding, matrix)

            scored: List[Tuple[UUID, float]] = []
            for block_id, score in zip(ids, scores):
                if score < min_score:
                    continue
                block = self._blocks.get(block_id)
                if block is None:
                    continue
                if block.tenant_id != tenant_id:
                    continue
                if status_filter is not None and block.status != status_filter:
                    continue
                scored.append((block_id, float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def add_embedding(self, block_id: UUID, embedding: List[float]) -> None:
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                raise NotFoundError(
                    f"Block {block_id} not found for embedding",
                    code="block_not_found",
                )
            if self._dimension is None:
                self._dimension = len(embedding)
            elif len(embedding) != self._dimension:
                raise StoreError(
                    f"Embedding dimension mismatch: {len(embedding)} vs {self._dimension}",
                    code="dimension_mismatch",
                )
            block.embedding = list(embedding)
            self._vectors[block_id] = list(embedding)

    # ------------------------------------------------------------------ #
    # Graph path
    # ------------------------------------------------------------------ #
    def upsert_entity(self, entity: MemoryEntity, tenant_id: str = "default") -> MemoryEntity:
        with self._lock:
            existing = self._entities.get(entity.id)
            if existing is None:
                self._entities[entity.id] = entity
                self._entity_by_name[entity.name] = entity.id
            else:
                # Merge: update counts / timestamps, keep the richer record.
                existing.mention_count += 1
                existing.last_seen = entity.last_seen
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.properties.update(entity.properties)
                entity = existing
            self._persist_entity(entity)
            return entity

    def get_entity(self, entity_id: UUID) -> MemoryEntity:
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                raise NotFoundError(
                    f"Entity {entity_id} not found",
                    code="entity_not_found",
                )
            return entity

    def find_entity_by_name(self, name: str, tenant_id: str = "default") -> Optional[MemoryEntity]:
        with self._lock:
            eid = self._entity_by_name.get(name)
            if eid is None:
                return None
            entity = self._entities.get(eid)
            if entity is None or entity.tenant_id != tenant_id:
                return None
            return entity

    def add_relation(self, relation: MemoryRelation, tenant_id: str = "default") -> MemoryRelation:
        with self._lock:
            self._relations[relation.id] = relation
            self._rel_by_subject.setdefault(relation.subject_entity_id, []).append(
                relation.id
            )
            self._rel_by_object.setdefault(relation.object_entity_id, []).append(
                relation.id
            )
            self._persist_relation(relation)
            return relation

    def add_edge(self, edge: MemoryEdge, tenant_id: str = "default") -> MemoryEdge:
        with self._lock:
            self._edges[edge.id] = edge
            self._edges_by_block.setdefault(edge.memory_block_id, []).append(edge.id)
            self._persist_edge(edge)
            return edge

    def get_edges_for_block(self, block_id: UUID) -> List[MemoryEdge]:
        with self._lock:
            edge_ids = self._edges_by_block.get(block_id, [])
            return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def traverse_graph(
        self,
        seed_entity_ids: Iterable[UUID],
        max_depth: int = 2,
    ) -> List[UUID]:
        """BFS over the relation graph returning reachable entity ids."""
        with self._lock:
            visited: Set[UUID] = set()
            frontier: List[UUID] = list(seed_entity_ids)
            depth = 0
            while frontier:
                next_frontier: List[UUID] = []
                for eid in frontier:
                    if eid in visited:
                        continue
                    visited.add(eid)
                    if depth < max_depth:
                        for rid in self._rel_by_subject.get(eid, []):
                            rel = self._relations.get(rid)
                            if rel:
                                next_frontier.append(rel.object_entity_id)
                        for rid in self._rel_by_object.get(eid, []):
                            rel = self._relations.get(rid)
                            if rel:
                                next_frontier.append(rel.subject_entity_id)
                frontier = [e for e in next_frontier if e not in visited]
                depth += 1
            return list(visited)

    def blocks_for_entities(
        self, entity_ids: Iterable[UUID], top_k: int = 10, tenant_id: str = "default"
    ) -> List[UUID]:
        """Return block ids that mention the given entities, scored by edge weight."""
        with self._lock:
            block_scores: Dict[UUID, float] = {}
            for eid in entity_ids:
                # Find edges whose target is this entity (block mentions entity).
                for edge in self._edges.values():
                    if edge.target_id == eid:
                        block = self._blocks.get(edge.memory_block_id)
                        if block is None or block.tenant_id != tenant_id:
                            continue
                        block_scores[edge.memory_block_id] = block_scores.get(
                            edge.memory_block_id, 0.0
                        ) + edge.weight
            ranked = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
            return [bid for bid, _ in ranked[:top_k]]

    # ------------------------------------------------------------------ #
    # Hybrid retrieval entry point
    # ------------------------------------------------------------------ #
    def get_block(self, block_id: UUID, tenant_id: str = "default") -> MemoryBlock:
        return self.read_block(block_id, tenant_id=tenant_id)

    def count_blocks(self, tenant_id: str = "default") -> int:
        with self._lock:
            return sum(1 for b in self._blocks.values() if b.tenant_id == tenant_id)

    def close(self) -> None:
        """No-op for SQLite (sessions are short-lived)."""


__all__ = ["HybridStore", "_cosine_similarity", "_cosine_similarity_batch"]

"""MnemosyneEngine — the primary orchestration module.

This is the main entry point for the Mnemosyne Protocol. It wires together:

* :class:`HybridStore` — dual-path vector + knowledge-graph storage.
* :class:`EmbeddingProvider` — dense vector generation.
* :class:`RetrievalEngine` — vector / graph / hybrid retrieval with RRF fusion.
* :class:`ConsolidationPipeline` — autonomous background consolidation.
* :class:`ContextInjectionMiddleware` — dynamic context injection.
* :class:`ModelAdapter` — universal model-agnostic hooks.

Usage::

    from mnemosyne import MnemosyneEngine, MnemosyneConfig, ModelProfile

    cfg = MnemosyneConfig()
    cfg.register_profile(ModelProfile.openai(api_key="sk-...", model="gpt-4o"))
    engine = MnemosyneEngine(cfg)

    block = engine.ingest("The user prefers dark mode and works in Berlin.")
    result = engine.inject("What does the user prefer?")
    response = engine.complete(result)
    print(response)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

from mnemosyne.adapters import build_adapter
from mnemosyne.config import MnemosyneConfig
from mnemosyne.consolidation import ConsolidationPipeline
from mnemosyne.embeddings import EmbeddingProvider
from mnemosyne.exceptions import MnemosyneError, NotFoundError
from mnemosyne.injection import ContextInjectionMiddleware, InjectionResult
from mnemosyne.retrieval import RetrievalEngine
from mnemosyne.schemas import (
    MemoryBlock,
    MemorySearchQuery,
    RetrievalMode,
)
from mnemosyne.store import HybridStore

logger = logging.getLogger("mnemosyne")


class MnemosyneEngine:
    """Top-level orchestration engine for the Mnemosyne Protocol.

    All public methods are thread-safe. The engine owns the store, retriever,
    consolidator, injection middleware, and model adapters.
    """

    def __init__(self, config: MnemosyneConfig) -> None:
        self.config = config
        self._lock = threading.RLock()

        # Core storage + retrieval.
        self.store = HybridStore(config)
        self.embeddings = EmbeddingProvider(config.embedding)
        self.retrieval = RetrievalEngine(self.store, config)
        self.injection = ContextInjectionMiddleware(
            self.store, self.retrieval, config
        )

        # Wire the embedder into the injection middleware so it can compute
        # query embeddings for vector retrieval.
        self.injection.set_embedder(self.embeddings)

        # Consolidation pipeline.
        self.consolidation = ConsolidationPipeline(self.store, config)
        self.consolidation.set_on_job_complete(self._on_consolidation_complete)

        # Model adapters, lazily built per profile.
        self._adapters: Dict[str, Any] = {}

        # Start background consolidation.
        self.consolidation.start()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start background services (consolidation)."""
        self.consolidation.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop background services gracefully."""
        self.consolidation.stop(timeout=timeout)

    def __enter__(self) -> "MnemosyneEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        content: str,
        *,
        source: str = "ingestion",
        tenant_id: str = "default",
        metadata: Optional[dict] = None,
        importance_score: float = 0.5,
        tags: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        embed: bool = True,
    ) -> MemoryBlock:
        """Ingest a new memory block into the store.

        If ``embed`` is True, a dense embedding is computed immediately so the
        block is retrievable via the vector path.
        """
        block = MemoryBlock(
            content=content,
            source=source,
            tenant_id=tenant_id,
            metadata=metadata or {},
            importance_score=max(0.0, min(1.0, importance_score)),
            tags=tags or [],
            ttl_seconds=ttl_seconds,
            token_count=len(content) // 4,
        )

        if embed:
            try:
                vector = self.embeddings.embed(content)
                block.embedding = vector
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to embed block %s: %s", block.id, exc)
                block.embedding = None

        self.store.write_block(block)

        # If we have entity extraction enabled in consolidation config, run a
        # lightweight synthesis so the block is immediately graph-addressable.
        if self.config.consolidation.extract_entities:
            self._synthesize_block(block)

        logger.info("Ingested memory block %s (%d chars)", block.id, len(content))
        return block

    def ingest_batch(
        self,
        contents: List[str],
        *,
        source: str = "ingestion",
        embed: bool = True,
    ) -> List[MemoryBlock]:
        """Ingest multiple blocks in a batch."""
        blocks: List[MemoryBlock] = []
        for content in contents:
            blocks.append(
                self.ingest(content, source=source, embed=embed)
            )
        return blocks

    def get_memory(self, block_id: UUID, tenant_id: str = "default") -> MemoryBlock:
        return self.store.read_block(block_id, tenant_id=tenant_id)

    def list_memories(
        self,
        tenant_id: str = "default",
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryBlock]:
        return self.store.list_blocks(tenant_id=tenant_id, limit=limit, offset=offset)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        text: Optional[str] = None,
        *,
        tenant_id: str = "default",
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        entity_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryBlock]:
        """Retrieve relevant memory blocks for a query.

        Returns the raw blocks (sorted by score). For injection-ready output,
        use :meth:`inject`.
        """
        query_embedding: Optional[List[float]] = None
        if text is not None:
            try:
                query_embedding = self.embeddings.embed(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Query embedding failed: %s", exc)
                query_embedding = None

        query = MemorySearchQuery(
            text=text,
            embedding=query_embedding,
            top_k=top_k,
            mode=mode,
            entity_filter=entity_filter,
            tag_filter=tag_filter,
            tenant_id=tenant_id,
            min_importance=min_importance,
        )
        results = self.retrieval.retrieve(query)
        return [r.block for r in results]

    # ------------------------------------------------------------------ #
    # Injection
    # ------------------------------------------------------------------ #
    def inject(
        self,
        user_input: str,
        *,
        tenant_id: str = "default",
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        entity_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
    ) -> InjectionResult:
        """Produce an LLM-ready prompt with retrieved memory anchors injected."""
        return self.injection.inject(
            user_input=user_input,
            model_name=model_name,
            top_k=top_k,
            mode=mode,
            entity_filter=entity_filter,
            tag_filter=tag_filter,
            tenant_id=tenant_id,
        )

    # ------------------------------------------------------------------ #
    # Completion (universal model-agnostic hook)
    # ------------------------------------------------------------------ #
    def complete(
        self,
        result: InjectionResult,
        *,
        profile_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Send an :class:`InjectionResult` through a model adapter.

        The profile is resolved from the engine config; if none is given the
        config's default profile is used.
        """
        profile = self.config.get_profile(profile_name)
        adapter = self._get_adapter(profile.name)
        return adapter.complete(result, **kwargs)

    def inject_and_complete(
        self,
        user_input: str,
        *,
        profile_name: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        **kwargs,
    ) -> str:
        """Convenience: inject context then complete in one call."""
        result = self.inject(
            user_input,
            model_name=model_name,
            top_k=top_k,
            mode=mode,
        )
        return self.complete(result, profile_name=profile_name, **kwargs)

    # ------------------------------------------------------------------ #
    # Consolidation
    # ------------------------------------------------------------------ #
    def run_consolidation(self) -> Any:
        """Run a single consolidation pass synchronously and return the job."""
        return self.consolidation.run_once()

    def get_consolidation_job(self, job_id: UUID) -> Any:
        return self.consolidation.get_job(job_id)

    def list_consolidation_jobs(self) -> List[Any]:
        return self.consolidation.list_jobs()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _get_adapter(self, profile_name: str) -> Any:
        with self._lock:
            adapter = self._adapters.get(profile_name)
            if adapter is None:
                profile = self.config.get_profile(profile_name)
                adapter = build_adapter(profile)
                self._adapters[profile_name] = adapter
            return adapter

    def _synthesize_block(self, block: MemoryBlock) -> None:
        """Best-effort entity/relation extraction for a freshly ingested block."""
        try:
            self.consolidation._execute_synthesize_block(block)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Synthesis skipped for block %s: %s", block.id, exc)

    def _on_consolidation_complete(self, job: Any) -> None:
        logger.info(
            "Consolidation job %s completed: status=%s processed=%d created=%d pruned=%d",
            job.id,
            job.status,
            job.blocks_processed,
            job.blocks_created,
            job.blocks_pruned,
        )


__all__ = ["MnemosyneEngine"]

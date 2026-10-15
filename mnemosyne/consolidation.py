"""Autonomous Memory Consolidation Pipeline.

Background worker threads that compress, prune, and synthesize short-term
chat logs into long-term semantic knowledge blocks without losing granular
detail.
"""

from __future__ import annotations

import heapq
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

from mnemosyne.config import ConsolidationConfig, MnemosyneConfig
from mnemosyne.exceptions import ConsolidationError
from mnemosyne.schemas import (
    ConsolidationJob,
    ConsolidationPlan,
    MemoryBlock,
    MemoryBlockStatus,
    MemoryEntity,
    MemoryRelation,
)
from mnemosyne.store import HybridStore


class ConsolidationPipeline:
    """Plans and executes consolidation runs over the hybrid store.

    Strategies:
      * ``compress``  — merge several low-importance related blocks into one
        summarized block, keeping the originals as children.
      * ``prune``     — mark expired / very low-importance blocks as pruned.
      * ``synthesize``— extract entities & relations from a block into the
        knowledge graph.
    """

    def __init__(self, store: HybridStore, config: MnemosyneConfig) -> None:
        self.store = store
        self.config = config
        self.consolidation: ConsolidationConfig = config.consolidation

        self._lock = threading.RLock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._jobs: Dict[UUID, ConsolidationJob] = {}
        self._on_job_complete: Optional[Callable[[ConsolidationJob], None]] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the background consolidation loop (if enabled)."""
        if not self.consolidation.enabled:
            return
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._loop, daemon=True, name="mnemosyne-consolidation"
            )
            self._worker_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the background loop to stop and wait briefly."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)

    def set_on_job_complete(
        self, callback: Optional[Callable[[ConsolidationJob], None]]
    ) -> None:
        self._on_job_complete = callback

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self.consolidation.interval_seconds)
            if self._stop_event.is_set():
                break
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                # Never let a consolidation failure kill the worker thread.
                import logging

                logging.getLogger("mnemosyne").exception(
                    "Consolidation run failed: %s", exc
                )

    # ------------------------------------------------------------------ #
    # On-demand execution
    # ------------------------------------------------------------------ #
    def run_once(self) -> ConsolidationJob:
        """Execute a single consolidation pass synchronously."""
        job = ConsolidationJob(config=self._config_snapshot())
        with self._lock:
            self._jobs[job.id] = job
        try:
            job.status = self._jobs[job.id].status.__class__.RUNNING  # type: ignore[attr-defined]
            from mnemosyne.schemas import ConsolidationStatus

            job.status = ConsolidationStatus.RUNNING
            job.started_at = datetime.utcnow()

            plan = self._plan()

            # Prune first.
            pruned = self._execute_prune(plan.prune_ids)
            job.blocks_pruned = pruned

            # Compress next.
            compressed_ids = self._execute_compress(plan.compress_ids)
            job.blocks_processed += len(compressed_ids)

            # Synthesize entities/relations from surviving blocks.
            synthesized = self._execute_synthesize()
            job.blocks_created += synthesized

            job.completed_at = datetime.utcnow()
            job.status = ConsolidationStatus.COMPLETED
        except Exception as exc:  # noqa: BLE001
            job.completed_at = datetime.utcnow()
            job.status = ConsolidationStatus.FAILED
            job.error = str(exc)
            raise ConsolidationError(
                f"Consolidation job {job.id} failed: {exc}",
                code="consolidation_failed",
                details={"job_id": str(job.id)},
            ) from exc
        finally:
            with self._lock:
                self._jobs[job.id] = job
            if self._on_job_complete is not None:
                try:
                    self._on_job_complete(job)
                except Exception:  # noqa: BLE001
                    pass
        return job

    def get_job(self, job_id: UUID) -> ConsolidationJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                from mnemosyne.exceptions import NotFoundError

                raise NotFoundError(
                    f"Consolidation job {job_id} not found",
                    code="job_not_found",
                )
            return job

    def list_jobs(self) -> List[ConsolidationJob]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True,
            )

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    def _config_snapshot(self) -> dict:
        return {
            "importance_threshold": self.consolidation.importance_threshold,
            "age_threshold_seconds": self.consolidation.age_threshold_seconds,
            "compression_ratio": self.consolidation.compression_ratio,
        }

    def _plan(self) -> ConsolidationPlan:
        """Decide which blocks to compress, prune, or synthesize."""
        now = datetime.utcnow()
        age_cutoff = now - timedelta(
            seconds=self.consolidation.age_threshold_seconds
        )

        blocks = self.store.list_blocks(limit=self.consolidation.max_blocks_per_run)

        compress_ids: List[UUID] = []
        prune_ids: List[UUID] = []
        synthesize_ids: List[UUID] = []
        rationale: Dict[UUID, str] = {}

        # Group blocks by source tag for compression candidates.
        by_source: Dict[str, List[MemoryBlock]] = {}
        for block in blocks:
            by_source.setdefault(block.source, []).append(block)

        for block in blocks:
            age_seconds = (now - block.timestamp).total_seconds()

            # Prune: expired TTL or very low importance & old.
            if block.ttl_seconds is not None and age_seconds > block.ttl_seconds:
                prune_ids.append(block.id)
                rationale[block.id] = "ttl_expired"
                continue
            if (
                self.consolidation.prune_low_importance
                and block.importance_score < self.consolidation.importance_threshold
                and age_seconds > self.consolidation.age_threshold_seconds
            ):
                prune_ids.append(block.id)
                rationale[block.id] = "low_importance_and_stale"
                continue

            # Compress: clusters of low-importance blocks from the same source.
            if (
                block.importance_score < self.consolidation.importance_threshold
                and block.status == MemoryBlockStatus.ACTIVE
            ):
                compress_ids.append(block.id)
                rationale[block.id] = "low_importance_compress_candidate"

            # Synthesize: active, non-consolidated blocks with content.
            if (
                not block.consolidated
                and block.status == MemoryBlockStatus.ACTIVE
                and block.content.strip()
            ):
                synthesize_ids.append(block.id)
                rationale[block.id] = "entity_relation_extraction"

        return ConsolidationPlan(
            compress_ids=compress_ids[: self.consolidation.max_blocks_per_run],
            prune_ids=prune_ids,
            synthesize_ids=synthesize_ids[: self.consolidation.max_blocks_per_run],
            rationale=rationale,
        )

    # ------------------------------------------------------------------ #
    # Execution primitives
    # ------------------------------------------------------------------ #
    def _execute_prune(self, ids: List[UUID]) -> int:
        count = 0
        for bid in ids:
            try:
                block = self.store.read_block(bid)
                block.status = MemoryBlockStatus.PRUNED
                self.store.write_block(block)
                count += 1
            except Exception:  # noqa: BLE001
                continue
        return count

    def _execute_compress(self, ids: List[UUID]) -> List[UUID]:
        """Merge blocks sharing a source into a single compressed block."""
        if not ids:
            return []

        by_source: Dict[str, List[MemoryBlock]] = {}
        for bid in ids:
            try:
                block = self.store.read_block(bid)
            except Exception:  # noqa: BLE001
                continue
            by_source.setdefault(block.source, []).append(block)

        created: List[UUID] = []
        for source, group in by_source.items():
            if len(group) < 2:
                # Still mark the singleton as consolidated so it isn't recompressed.
                try:
                    b = group[0]
                    b.consolidated = True
                    self.store.write_block(b)
                    created.append(b.id)
                except Exception:  # noqa: BLE001
                    continue
                continue

            group.sort(key=lambda b: b.importance_score, reverse=True)
            primary = group[0]
            children = [b.id for b in group[1:]]

            # Build summary content preserving the most important details.
            parts = [f"[Consolidated from {len(group)} blocks, source={source}]"]
            for b in group:
                parts.append(f"- {b.content.strip()}")
            summary = "\n".join(parts)

            importance = max(b.importance_score for b in group)
            merged = MemoryBlock(
                content=summary,
                source=f"consolidated:{source}",
                importance_score=min(1.0, importance + 0.1),
                tags=list(primary.tags) + ["consolidated"],
                token_count=len(summary) // 4,
                parent_block_id=None,
            )
            merged.child_block_ids = children
            # Point children at the merged block.
            for child_id in children:
                try:
                    child = self.store.read_block(child_id)
                    child.parent_block_id = merged.id
                    child.status = MemoryBlockStatus.CONSOLIDATED
                    self.store.write_block(child)
                except Exception:  # noqa: BLE001
                    continue

            try:
                self.store.write_block(merged)
                created.append(merged.id)
            except Exception:  # noqa: BLE001
                continue
        return created

    def _execute_synthesize(self) -> int:
        """Extract entities and relations from blocks into the knowledge graph."""
        count = 0
        blocks = self.store.list_blocks(
            limit=self.consolidation.max_blocks_per_run
        )
        for block in blocks:
            if block.consolidated or block.status != MemoryBlockStatus.ACTIVE:
                continue
            entities = self._extract_entities(block.content)
            entity_ids: List[UUID] = []
            for ent in entities:
                try:
                    created = self.store.upsert_entity(ent)
                    entity_ids.append(created.id)
                    count += 1
                except Exception:  # noqa: BLE001
                    continue
            # Link block -> entities with edges.
            from mnemosyne.schemas import MemoryEdge

            for eid in entity_ids:
                try:
                    self.store.add_edge(
                        MemoryEdge(
                            memory_block_id=block.id,
                            target_id=eid,
                            edge_type="mentions",
                            weight=1.0,
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
            # Mark block consolidated.
            try:
                block.consolidated = True
                self.store.write_block(block)
            except Exception:  # noqa: BLE001
                continue
        return count

    def _execute_synthesize_block(self, block: MemoryBlock) -> int:
        """Extract entities and relations for a single block into the graph."""
        from mnemosyne.schemas import MemoryEdge

        entities = self._extract_entities(block.content)
        entity_ids: List[UUID] = []
        for ent in entities:
            try:
                created = self.store.upsert_entity(ent)
                entity_ids.append(created.id)
            except Exception:  # noqa: BLE001
                continue
        for eid in entity_ids:
            try:
                self.store.add_edge(
                    MemoryEdge(
                        memory_block_id=block.id,
                        target_id=eid,
                        edge_type="mentions",
                        weight=1.0,
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        try:
            block.consolidated = True
            self.store.write_block(block)
        except Exception:  # noqa: BLE001
            pass
        return len(entity_ids)

    def _extract_entities(self, text: str) -> List[MemoryEntity]:
        """Heuristic entity extraction (capitalized noun phrases).

        A lightweight regex/token-based extractor so consolidation works
        without an NLP dependency. Replace with an NER model for production.
        """
        import re

        entities: List[MemoryEntity] = []
        # Capitalized multi-word phrases.
        for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
            name = m.group(0)
            if len(name) < 2:
                continue
            entities.append(
                MemoryEntity(
                    name=name,
                    entity_type=self._guess_type(name),
                    confidence=0.7,
                    mention_count=1,
                )
            )
        # Deduplicate by name.
        seen: Set[str] = set()
        unique: List[MemoryEntity] = []
        for ent in entities:
            if ent.name in seen:
                continue
            seen.add(ent.name)
            unique.append(ent)
        return unique[:20]

    def _guess_type(self, name: str):
        from mnemosyne.schemas import EntityType

        lower = name.lower()
        org_keywords = ("inc", "corp", "llc", "ltd", "company", "university", "lab")
        if any(k in lower for k in org_keywords):
            return EntityType.ORGANIZATION
        place_keywords = ("street", "avenue", "city", "country", "park", "river")
        if any(k in lower for k in place_keywords):
            return EntityType.PLACE
        # Single capitalized word (proper name) -> person by default.
        if " " not in name and name[0].isupper():
            return EntityType.PERSON
        return EntityType.CONCEPT


__all__ = ["ConsolidationPipeline"]

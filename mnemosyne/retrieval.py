"""Hybrid retrieval engine: vector + graph + reciprocal-rank fusion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from mnemosyne.config import MnemosyneConfig, RetrievalConfig
from mnemosyne.exceptions import RetrievalError
from mnemosyne.schemas import (
    MemoryBlock,
    MemoryBlockStatus,
    MemorySearchQuery,
    RetrievalMode,
    RetrievalResult,
)
from mnemosyne.store import HybridStore


def _reciprocal_rank_fusion(
    ranked_lists: List[List[UUID]], k: int = 60
) -> List[Tuple[UUID, float]]:
    """Fuse multiple ranked result lists via Reciprocal Rank Fusion (RRF).

    RRF score for an item = sum(1 / (k + rank_i)) across all lists where it
    appears. Items ranked highly in multiple lists rise to the top.
    """
    scores: Dict[UUID, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class RetrievalEngine:
    """Dual-path retrieval over the hybrid store.

    Modes:
      * ``vector``   — cosine similarity over embeddings only.
      * ``graph``    — entity-mention expansion + graph traversal.
      * ``hybrid``   — RRF fusion of vector and graph results.
    """

    def __init__(self, store: HybridStore, config: MnemosyneConfig) -> None:
        self.store = store
        self.config = config
        self.retrieval: RetrievalConfig = config.retrieval

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def retrieve(self, query: MemorySearchQuery) -> List[RetrievalResult]:
        """Execute a hybrid search and return ranked results."""
        try:
            if query.mode == RetrievalMode.VECTOR:
                return self._retrieve_vector(query)
            if query.mode == RetrievalMode.GRAPH:
                return self._retrieve_graph(query)
            return self._retrieve_hybrid(query)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(
                f"Retrieval failed: {exc}",
                code="retrieval_failed",
                details={"query": query.text},
            ) from exc

    # ------------------------------------------------------------------ #
    # Vector path
    # ------------------------------------------------------------------ #
    def _retrieve_vector(self, query: MemorySearchQuery) -> List[RetrievalResult]:
        if query.embedding is None:
            raise RetrievalError(
                "Vector retrieval requires a query embedding",
                code="missing_embedding",
            )
        hits = self.store.search_vectors(
            query_embedding=query.embedding,
            top_k=query.top_k,
            min_score=self.retrieval.min_importance,
            status_filter=query.status_filter,
            tenant_id=query.tenant_id,
        )
        return self._to_results(hits, RetrievalMode.VECTOR, query.top_k)

    # ------------------------------------------------------------------ #
    # Graph path
    # ------------------------------------------------------------------ #
    def _retrieve_graph(self, query: MemorySearchQuery) -> List[RetrievalResult]:
        seed_entity_ids: List[UUID] = []

        # Resolve named entities mentioned in the query text.
        if query.text:
            for token in set(query.text.split()):
                entity = self.store.find_entity_by_name(
                    token.strip(".,!?:;\"'"), tenant_id=query.tenant_id
                )
                if entity is not None:
                    seed_entity_ids.append(entity.id)

        # Explicit entity filter.
        if query.entity_filter:
            entity = self.store.find_entity_by_name(
                query.entity_filter, tenant_id=query.tenant_id
            )
            if entity is not None:
                seed_entity_ids.append(entity.id)

        # Expand through graph hops (tenant-scoped).
        expanded = self.store.traverse_graph(
            seed_entity_ids, max_depth=self.retrieval.graph_hop_depth
        )

        # Blocks mentioning any of the expanded entities, tenant-scoped.
        block_ids = self.store.blocks_for_entities(
            expanded, top_k=query.top_k * 5, tenant_id=query.tenant_id
        )

        # Apply tag filter if provided.
        tag_filter = set(query.tag_filter or [])

        # Fetch blocks and score by graph centrality (edge-weight sum).
        results: List[RetrievalResult] = []
        seen: set = set()
        for rank, block_id in enumerate(block_ids):
            if block_id in seen:
                continue
            seen.add(block_id)
            try:
                block = self.store.read_block(block_id, tenant_id=query.tenant_id)
            except Exception:
                continue
            if tag_filter and not set(block.tags or []).intersection(tag_filter):
                continue
            if query.status_filter is not None and block.status != query.status_filter:
                continue
            if block.importance_score < query.min_importance:
                continue
            # Graph score: decay by rank position.
            score = 1.0 / (rank + 1)
            results.append(
                RetrievalResult(
                    block=block,
                    score=min(1.0, score),
                    path=RetrievalMode.GRAPH,
                    explanation="graph traversal via entity mentions",
                )
            )
            if len(results) >= query.top_k:
                break
        return results

    # ------------------------------------------------------------------ #
    # Hybrid path (RRF fusion)
    # ------------------------------------------------------------------ #
    def _retrieve_hybrid(self, query: MemorySearchQuery) -> List[RetrievalResult]:
        vector_hits: List[Tuple[UUID, float]] = []
        graph_block_ids: List[UUID] = []

        if query.embedding is not None:
            vector_hits = self.store.search_vectors(
                query_embedding=query.embedding,
                top_k=query.top_k * 2,
                min_score=self.retrieval.min_importance,
                status_filter=query.status_filter,
                tenant_id=query.tenant_id,
            )

        graph_block_ids = self._graph_block_ids(query)

        # Build ranked lists for RRF.
        vector_ranked = [bid for bid, _ in vector_hits]
        graph_ranked = graph_block_ids
        fused = _reciprocal_rank_fusion(
            [vector_ranked, graph_ranked], k=self.retrieval.rrf_k
        )

        # Apply tag filter if provided.
        tag_filter = set(query.tag_filter or [])

        # Fetch blocks and attach fused scores.
        score_by_id: Dict[UUID, float] = dict(fused)
        vector_score_by_id: Dict[UUID, float] = dict(vector_hits)

        results: List[RetrievalResult] = []
        seen: set = set()
        for block_id, fused_score in fused:
            if block_id in seen:
                continue
            seen.add(block_id)
            try:
                block = self.store.read_block(block_id, tenant_id=query.tenant_id)
            except Exception:
                continue
            if tag_filter and not set(block.tags or []).intersection(tag_filter):
                continue
            if query.status_filter is not None and block.status != query.status_filter:
                continue
            if block.importance_score < query.min_importance:
                continue

            # Blend fused RRF score with raw vector similarity.
            vec_score = vector_score_by_id.get(block_id, 0.0)
            w_v = self.retrieval.vector_weight
            w_g = self.retrieval.graph_weight
            blended = w_v * vec_score + w_g * fused_score

            explanation = []
            if vec_score > 0:
                explanation.append("vector")
            if block_id in set(graph_ranked):
                explanation.append("graph")
            results.append(
                RetrievalResult(
                    block=block,
                    score=min(1.0, blended),
                    path=RetrievalMode.HYBRID,
                    explanation=" + ".join(explanation) or "fused",
                )
            )
            if len(results) >= query.top_k:
                break
        return results

    def _graph_block_ids(self, query: MemorySearchQuery) -> List[UUID]:
        seed_entity_ids: List[UUID] = []
        if query.text:
            for token in set(query.text.split()):
                entity = self.store.find_entity_by_name(
                    token.strip(".,!?:;\"'"), tenant_id=query.tenant_id
                )
                if entity is not None:
                    seed_entity_ids.append(entity.id)
        if query.entity_filter:
            entity = self.store.find_entity_by_name(
                query.entity_filter, tenant_id=query.tenant_id
            )
            if entity is not None:
                seed_entity_ids.append(entity.id)
        expanded = self.store.traverse_graph(
            seed_entity_ids, max_depth=self.retrieval.graph_hop_depth
        )
        return self.store.blocks_for_entities(
            expanded, top_k=query.top_k * 5, tenant_id=query.tenant_id
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _to_results(
        self, hits: List[Tuple[UUID, float]], path: RetrievalMode, top_k: int
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for block_id, score in hits[:top_k]:
            try:
                block = self.store.read_block(block_id)
            except Exception:
                continue
            results.append(
                RetrievalResult(
                    block=block,
                    score=min(1.0, max(0.0, score)),
                    path=path,
                    explanation=f"{path.value} similarity",
                )
            )
        return results


__all__ = ["RetrievalEngine", "_reciprocal_rank_fusion"]

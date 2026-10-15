"""Tests for Pydantic data schemas."""

import pytest
from mnemosyne.schemas import (
    MemoryBlock,
    MemoryBlockStatus,
    MemoryEntity,
    MemoryRelation,
    RetrievalMode,
    TokenBudget,
    MemorySearchQuery,
)


def test_memory_block_defaults():
    block = MemoryBlock(content="hello world")
    assert block.status == MemoryBlockStatus.ACTIVE
    assert block.importance_score == 0.5
    assert block.access_count == 0
    assert block.content == "hello world"


def test_memory_block_touch():
    block = MemoryBlock(content="x")
    block.touch()
    assert block.access_count == 1
    assert block.last_accessed is not None


def test_token_budget_usable():
    budget = TokenBudget(
        total_available=8192,
        system_reserved=512,
        user_input_tokens=100,
        retrieval_budget=2000,
        overhead_tokens=100,
        max_blocks=10,
    )
    assert budget.usable_for_retrieval == 1900


def test_retrieval_mode_enum():
    assert RetrievalMode.HYBRID == "hybrid"
    assert RetrievalMode.VECTOR == "vector"
    assert RetrievalMode.GRAPH == "graph"


def test_search_query_defaults():
    q = MemorySearchQuery(text="hello")
    assert q.top_k == 10
    assert q.mode == RetrievalMode.HYBRID
    assert q.min_importance == 0.0


def test_entity_and_relation():
    e = MemoryEntity(name="Acme Corp", entity_type="organization")
    assert e.mention_count == 1
    r = MemoryRelation(
        subject_entity_id=e.id,
        predicate="located_in",
        object_entity_id=e.id,
    )
    assert r.confidence == 1.0

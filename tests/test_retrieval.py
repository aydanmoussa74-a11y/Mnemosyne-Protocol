"""Tests for the RetrievalEngine."""

import os
import tempfile

import pytest

from mnemosyne.config import MnemosyneConfig
from mnemosyne.schemas import (
    MemoryBlock,
    MemorySearchQuery,
    RetrievalMode,
)
from mnemosyne.store import HybridStore
from mnemosyne.retrieval import RetrievalEngine


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    cfg = MnemosyneConfig(storage_path=os.path.join(tmp, "data"))
    store = HybridStore(cfg)
    ret = RetrievalEngine(store, cfg)

    # Seed blocks with embeddings.
    blocks = [
        MemoryBlock(content="the user likes dark mode", source="test", embedding=[1.0, 0.0, 0.0]),
        MemoryBlock(content="the user works in Berlin", source="test", embedding=[0.0, 1.0, 0.0]),
        MemoryBlock(content="unrelated quantum physics text", source="test", embedding=[0.0, 0.0, 1.0]),
    ]
    store.write_blocks(blocks)
    return ret


def test_vector_retrieval(engine):
    results = engine._retrieve_vector(
        MemorySearchQuery(text="dark mode", embedding=[1.0, 0.0, 0.0], top_k=5)
    )
    assert len(results) >= 1
    assert "dark mode" in results[0].block.content


def test_graph_retrieval_no_entities(engine):
    results = engine._retrieve_graph(
        MemorySearchQuery(text="nothing matches", top_k=5)
    )
    # No entities in graph yet; should return empty.
    assert results == []


def test_hybrid_retrieval(engine):
    results = engine.retrieve(
        MemorySearchQuery(text="dark mode", embedding=[1.0, 0.0, 0.0], top_k=5, mode=RetrievalMode.HYBRID)
    )
    assert len(results) >= 1

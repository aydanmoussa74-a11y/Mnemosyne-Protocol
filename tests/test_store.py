"""Tests for the HybridStore (vector + graph core)."""

import os
import tempfile
import uuid

import pytest

from mnemosyne.config import MnemosyneConfig
from mnemosyne.schemas import MemoryBlock, MemoryEntity, MemoryRelation, MemoryEdge
from mnemosyne.store import HybridStore


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp()
    cfg = MnemosyneConfig(storage_path=os.path.join(tmp, "data"))
    s = HybridStore(cfg)
    yield s


def test_write_and_read_block(store):
    block = MemoryBlock(content="the quick brown fox", source="test")
    store.write_block(block)
    fetched = store.read_block(block.id)
    assert fetched.content == "the quick brown fox"
    assert fetched.id == block.id


def test_block_not_found(store):
    with pytest.raises(Exception):
        store.read_block(uuid.uuid4())


def test_list_blocks(store):
    for i in range(5):
        store.write_block(MemoryBlock(content=f"block {i}", source="test"))
    blocks = store.list_blocks(limit=10)
    assert len(blocks) == 5


def test_delete_block(store):
    block = MemoryBlock(content="temporary", source="test")
    store.write_block(block)
    store.delete_block(block.id)
    with pytest.raises(Exception):
        store.read_block(block.id)


def test_entity_and_relation_graph(store):
    e1 = MemoryEntity(name="Alice", entity_type="person")
    e2 = MemoryEntity(name="Bob", entity_type="person")
    store.upsert_entity(e1)
    store.upsert_entity(e2)

    found = store.find_entity_by_name("Alice")
    assert found is not None
    assert found.name == "Alice"

    rel = MemoryRelation(
        subject_entity_id=e1.id,
        predicate="knows",
        object_entity_id=e2.id,
    )
    store.add_relation(rel)

    # Edge linking block to entity.
    block = MemoryBlock(content="Alice knows Bob", source="test")
    store.write_block(block)
    store.add_edge(
        MemoryEdge(
            memory_block_id=block.id,
            target_id=e1.id,
            edge_type="mentions",
            weight=1.0,
        )
    )
    edges = store.get_edges_for_block(block.id)
    assert len(edges) == 1


def test_traversal(store):
    e1 = MemoryEntity(name="A", entity_type="concept")
    e2 = MemoryEntity(name="B", entity_type="concept")
    e3 = MemoryEntity(name="C", entity_type="concept")
    store.upsert_entity(e1)
    store.upsert_entity(e2)
    store.upsert_entity(e3)

    store.add_relation(
        MemoryRelation(subject_entity_id=e1.id, predicate="leads_to", object_entity_id=e2.id)
    )
    store.add_relation(
        MemoryRelation(subject_entity_id=e2.id, predicate="leads_to", object_entity_id=e3.id)
    )

    reached = store.traverse_graph([e1.id], max_depth=2)
    assert e2.id in reached
    assert e3.id in reached


def test_vector_search_dimension_mismatch(store):
    block = MemoryBlock(content="vec block", source="test")
    store.write_block(block)
    # First embedding establishes the dimension.
    store.add_embedding(block.id, [0.1, 0.2])
    # A different-dimension embedding must raise.
    with pytest.raises(Exception):
        store.add_embedding(block.id, [0.1, 0.2, 0.3])
    hits = store.search_vectors([0.1, 0.2], top_k=5)
    assert len(hits) == 1
    assert hits[0][0] == block.id

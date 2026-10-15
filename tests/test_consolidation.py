"""Tests for the ConsolidationPipeline."""

import os
import tempfile

import pytest

from mnemosyne.config import MnemosyneConfig
from mnemosyne.consolidation import ConsolidationPipeline
from mnemosyne.schemas import MemoryBlock, MemoryBlockStatus
from mnemosyne.store import HybridStore


@pytest.fixture
def pipeline():
    tmp = tempfile.mkdtemp()
    cfg = MnemosyneConfig(storage_path=os.path.join(tmp, "data"))
    store = HybridStore(cfg)
    pipe = ConsolidationPipeline(store, cfg)
    return pipe


def test_run_once_empty(pipeline):
    job = pipeline.run_once()
    assert job.status.value == "completed"
    assert job.blocks_processed == 0


def test_prune_low_importance_stale(pipeline):
    block = MemoryBlock(
        content="stale low importance memory",
        source="test",
        importance_score=0.05,
    )
    pipeline.store.write_block(block)

    # Manually set an old timestamp to simulate staleness.
    import datetime

    block.timestamp = datetime.datetime.utcnow() - datetime.timedelta(days=10)
    pipeline.store.write_block(block)

    job = pipeline.run_once()
    assert job.blocks_pruned >= 1
    fetched = pipeline.store.read_block(block.id)
    assert fetched.status == MemoryBlockStatus.PRUNED


def test_synthesize_extracts_entities(pipeline):
    block = MemoryBlock(
        content="Alice works at Acme Corp in Berlin.",
        source="test",
        importance_score=0.8,
    )
    pipeline.store.write_block(block)

    created = pipeline._execute_synthesize_block(block)
    assert created >= 1

    # Entity should be findable by name.
    ent = pipeline.store.find_entity_by_name("Alice")
    assert ent is not None
    assert ent.entity_type == "person"

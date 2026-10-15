"""Tests for the top-level MnemosyneEngine orchestration."""

import os
import tempfile

import pytest

from mnemosyne import MnemosyneEngine, MnemosyneConfig
from mnemosyne.config import ModelProfile


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    cfg = MnemosyneConfig(storage_path=os.path.join(tmp, "data"))
    cfg.register_profile(ModelProfile.openai(name="default", api_key="sk-test", model="gpt-4o"))
    eng = MnemosyneEngine(cfg)
    yield eng
    eng.stop()


def test_ingest_and_retrieve(engine):
    block = engine.ingest("The user prefers dark mode.", source="test", importance_score=0.9)
    blocks = engine.retrieve("dark mode", top_k=5)
    assert len(blocks) >= 1
    assert any(b.id == block.id for b in blocks)


def test_inject_no_model(engine):
    result = engine.inject("Hello, what do you know?")
    assert result.success
    assert "Hello, what do you know?" in result.prompt


def test_complete_adapter_resolution(engine):
    # Adapter should resolve even without a real API call.
    profile = engine.config.get_profile("default")
    adapter = engine._get_adapter(profile.name)
    assert adapter is not None


def test_context_manager():
    tmp = tempfile.mkdtemp()
    cfg = MnemosyneConfig(storage_path=os.path.join(tmp, "data"))
    cfg.register_profile(ModelProfile.openai(api_key="sk-test", model="gpt-4o"))
    with MnemosyneEngine(cfg) as eng:
        block = eng.ingest("context manager test", source="test")
        assert block.id is not None

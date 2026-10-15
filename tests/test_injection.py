"""Tests for the ContextInjectionMiddleware."""

import os
import tempfile

import pytest

from mnemosyne.config import MnemosyneConfig
from mnemosyne.injection import ContextInjectionMiddleware
from mnemosyne.retrieval import RetrievalEngine
from mnemosyne.schemas import RetrievalMode
from mnemosyne.store import HybridStore
from mnemosyne.token_budget import TokenCounter, calculate_budget


@pytest.fixture
def middleware():
    tmp = tempfile.mkdtemp()
    cfg = MnemosyneConfig(storage_path=os.path.join(tmp, "data"))
    store = HybridStore(cfg)
    ret = RetrievalEngine(store, cfg)
    mw = ContextInjectionMiddleware(store, ret, cfg)
    return mw


def test_budget_calculation():
    cfg = MnemosyneConfig().token_budget
    counter = TokenCounter()
    budget = calculate_budget(cfg, "hello world", counter=counter)
    assert budget.total_available == 8192
    assert budget.user_input_tokens > 0
    assert budget.max_blocks >= 1


def test_inject_returns_prompt(middleware):
    result = middleware.inject("What is the user's preference?")
    assert result.success
    assert result.prompt
    assert "What is the user's preference?" in result.prompt


def test_inject_with_no_blocks(middleware):
    # Fresh store: no blocks, should return raw input.
    result = middleware.inject("hello there")
    assert result.success
    assert result.prompt == "hello there"
    assert result.total_injected_tokens == 0


def test_token_counter_heuristic():
    counter = TokenCounter()
    assert counter.count("") == 0
    assert counter.count("hello") >= 1

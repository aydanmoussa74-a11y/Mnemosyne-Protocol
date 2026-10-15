"""Tests for the Mnemosyne CLI chat client (mnemosyne/chat_client.py).

The interactive REPL itself cannot be driven through pytest, so these tests
exercise the ``MnemosyneChatClient`` unit methods (profile switching, context
summaries, command handling, turn processing with a fake adapter) and the
argument parser / entry point wiring.
"""

import argparse
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mnemosyne import MnemosyneEngine, MnemosyneConfig
from mnemosyne.adapters import ModelAdapter
from mnemosyne.chat_client import MnemosyneChatClient, main
from mnemosyne.config import ModelProfile
from mnemosyne.exceptions import MnemosyneError
from mnemosyne.injection import InjectionResult
from mnemosyne.schemas import RetrievalMode


class FakeAdapter(ModelAdapter):
    """Adapter that returns a canned response without calling any model."""

    def __init__(self, profile, response: str = "canned reply"):
        super().__init__(profile)
        self._response = response
        self.calls = 0

    def complete(self, result, **kwargs):
        self.calls += 1
        self.last_result = result
        self.last_kwargs = kwargs
        return self._response


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    cfg = MnemosyneConfig(storage_path=os.path.join(tmp, "data"))
    cfg.register_profile(ModelProfile.openai(name="default", api_key="sk-test", model="gpt-4o"))
    eng = MnemosyneEngine(cfg)
    yield eng
    eng.stop()


def _make_client(engine, response="canned reply", profile_name="default"):
    client = MnemosyneChatClient.__new__(MnemosyneChatClient)
    client.engine = engine
    client.profile_name = profile_name or engine.config.default_profile
    client.top_k = 8
    client.mode = RetrievalMode.HYBRID
    client.temperature = 0.7
    client.max_tokens = 2048
    client.auto_ingest = True
    client._turn_count = 0
    client.adapter = FakeAdapter(engine.config.get_profile(profile_name or engine.config.default_profile), response=response)
    return client


# --------------------------------------------------------------------------- #
# Client construction
# --------------------------------------------------------------------------- #
def test_client_defaults(engine):
    client = MnemosyneChatClient(engine=engine)
    assert client.profile_name == engine.config.default_profile
    assert client.top_k == 8
    assert client.mode == RetrievalMode.HYBRID
    assert client.auto_ingest is True
    assert client.adapter is not None


def test_client_custom(engine):
    client = MnemosyneChatClient(
        engine=engine,
        profile_name="default",
        top_k=5,
        mode=RetrievalMode.VECTOR,
        temperature=0.2,
        max_tokens=512,
        auto_ingest=False,
    )
    assert client.top_k == 5
    assert client.mode == RetrievalMode.VECTOR
    assert client.temperature == 0.2
    assert client.max_tokens == 512
    assert client.auto_ingest is False


# --------------------------------------------------------------------------- #
# Profile switching
# --------------------------------------------------------------------------- #
def test_switch_profile(engine):
    client = _make_client(engine)
    assert client.profile_name == "default"
    client.switch_profile("default")
    assert client.profile_name == "default"


# --------------------------------------------------------------------------- #
# Turn processing (ingest + inject + complete + ingest)
# --------------------------------------------------------------------------- #
def test_process_turn_ingests_and_responds(engine):
    client = _make_client(engine, response="The assistant remembers.")
    client._process_turn("Do you remember my preference?")

    # User turn ingested.
    blocks = engine.store.list_blocks()
    user_blocks = [b for b in blocks if b.source == "chat" and b.content.startswith("[user]")]
    assert len(user_blocks) >= 1
    assert "Do you remember my preference?" in user_blocks[-1].content

    # Assistant turn ingested.
    assistant_blocks = [b for b in blocks if b.source == "chat" and b.content.startswith("[assistant]")]
    assert len(assistant_blocks) >= 1
    assert "The assistant remembers." in assistant_blocks[-1].content

    # Adapter was called with an InjectionResult.
    assert client.adapter.calls == 1
    assert isinstance(client.adapter.last_result, InjectionResult)
    assert client.adapter.last_result.prompt


def test_process_turn_no_auto_ingest(engine):
    client = _make_client(engine, response="reply")
    client.auto_ingest = False
    client._process_turn("hello")
    blocks = engine.store.list_blocks(limit=100)
    chat_blocks = [b for b in blocks if b.source == "chat"]
    assert len(chat_blocks) == 0


# --------------------------------------------------------------------------- #
# Context summary rendering
# --------------------------------------------------------------------------- #
def test_show_context_summary(engine):
    client = _make_client(engine)
    # Build a real injection result.
    injection = engine.inject("test query", top_k=3)
    # Should not raise.
    client._show_context_summary(injection)


# --------------------------------------------------------------------------- #
# Command handling
# --------------------------------------------------------------------------- #
def test_handle_command_quit(engine):
    client = _make_client(engine)
    with pytest.raises(SystemExit):
        client._handle_command("/quit")


def test_handle_command_clear(engine, capsys):
    client = _make_client(engine)
    client._handle_command("/clear")  # should not raise


def test_handle_command_help(engine, capsys):
    client = _make_client(engine)
    client._handle_command("/help")  # should not raise


def test_handle_command_profile_no_arg(engine, capsys):
    client = _make_client(engine)
    client._handle_command("/profile")  # shows profiles, should not raise


def test_handle_command_unknown(engine):
    client = _make_client(engine)
    client._handle_command("/frobnicate")  # should not raise


# --------------------------------------------------------------------------- #
# Stats / history rendering
# --------------------------------------------------------------------------- #
def test_show_stats(engine):
    client = _make_client(engine)
    client._show_stats()  # should not raise


def test_show_history_empty(engine):
    client = _make_client(engine)
    client._show_history()  # no chat blocks yet, should not raise


def test_show_history_with_blocks(engine):
    engine.ingest("[user] hello", source="chat")
    engine.ingest("[assistant] hi there", source="chat")
    client = _make_client(engine)
    client._show_history(limit=5)  # should not raise


# --------------------------------------------------------------------------- #
# Argument parser / main entry point
# --------------------------------------------------------------------------- #
def test_parser_defaults():
    from mnemosyne.chat_client import _build_parser
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.storage_path == "./mnemosyne_data"
    assert args.profile is None
    assert args.top_k == 8
    assert args.mode == "hybrid"
    assert args.temperature == 0.7
    assert args.max_tokens == 2048
    assert args.no_auto_ingest is False


def test_parser_custom():
    from mnemosyne.chat_client import _build_parser
    parser = _build_parser()
    args = parser.parse_args(
        ["--storage-path", "/tmp/mnemosyne", "--profile", "ollama",
         "--top-k", "5", "--mode", "graph", "--temperature", "0.2",
         "--max-tokens", "128", "--no-auto-ingest"]
    )
    assert args.storage_path == "/tmp/mnemosyne"
    assert args.profile == "ollama"
    assert args.top_k == 5
    assert args.mode == "graph"
    assert args.temperature == 0.2
    assert args.max_tokens == 128
    assert args.no_auto_ingest is True


@patch("mnemosyne.chat_client.MnemosyneChatClient")
@patch("mnemosyne.chat_client.MnemosyneEngine")
def test_main_wires_engine_and_client(mock_engine_cls, mock_client_cls):
    mock_engine = MagicMock()
    mock_engine.config.profiles = {}
    mock_engine.config.default_profile = None
    mock_engine_cls.return_value = mock_engine
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    # Simulate argv.
    with patch("sys.argv", ["mnemosyne-chat", "--storage-path", "./data"]):
        rc = main([])

    assert rc == 0
    mock_engine_cls.assert_called_once()
    mock_client_cls.assert_called_once()
    mock_engine.stop.assert_called_once()


def test_main_registers_default_profile_when_empty():
    from mnemosyne.chat_client import _build_parser
    parser = _build_parser()
    args = parser.parse_args([])
    cfg = MnemosyneConfig(storage_path=tempfile.mkdtemp())
    assert cfg.profiles == {}
    cfg.register_profile(ModelProfile.openai(name="default"))
    assert cfg.default_profile == "default"


def test_main_registers_default_profile_when_empty():
    from mnemosyne.chat_client import _build_parser
    parser = _build_parser()
    args = parser.parse_args([])
    cfg = MnemosyneConfig(storage_path=tempfile.mkdtemp())
    assert cfg.profiles == {}
    cfg.register_profile(ModelProfile.openai(name="default"))
    assert cfg.default_profile == "default"

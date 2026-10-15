"""Tests for the Mnemosyne FastAPI microservice (mnemosyne/server.py).

Uses FastAPI's TestClient to exercise every HTTP endpoint against a real
in-process MnemosyneEngine backed by a temporary SQLite store.
"""

import os
import os
import tempfile
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from mnemosyne.config import MnemosyneConfig, ModelProfile
from mnemosyne.schemas import ModelProvider
from mnemosyne.server import create_app


@pytest.fixture
def config():
    tmp = tempfile.mkdtemp()
    cfg = MnemosyneConfig(storage_path=os.path.join(tmp, "data"))
    # Register a local profile with a canned callable so chat tests don't need
    # external model packages.
    cfg.register_profile(
        ModelProfile(
            name="default",
            provider=ModelProvider.LOCAL,
            model="local-test",
        )
    )
    return cfg


@pytest.fixture
def client(config):
    app = create_app(config)
    with TestClient(app) as tc:
        yield tc


# --------------------------------------------------------------------------- #
# Health / system
# --------------------------------------------------------------------------- #
def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["engine"] == "running"
    assert "default" in body["profiles"]


def test_list_profiles(client):
    resp = client.get("/v1/profiles")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "default" in body["profiles"]
    assert body["profiles"]["default"]["provider"] == "local"
    assert body["default"] == "default"


# --------------------------------------------------------------------------- #
# Profile registration
# --------------------------------------------------------------------------- #
def test_register_profile(client):
    payload = {
        "name": "ollama-test",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "llama3",
        "temperature": 0.5,
        "max_tokens": 1024,
    }
    resp = client.post("/v1/profiles", json=payload)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "ollama-test"
    assert body["provider"] == "ollama"
    assert body["registered"] is True


def test_register_profile_invalid_provider(client):
    payload = {"name": "bad", "provider": "not-a-provider", "model": "x"}
    resp = client.post("/v1/profiles", json=payload)
    assert resp.status_code == 400, resp.text


# --------------------------------------------------------------------------- #
# Memory ingest
# --------------------------------------------------------------------------- #
def test_ingest_memory(client):
    payload = {
        "content": "The user prefers dark mode.",
        "source": "api-test",
        "importance_score": 0.9,
        "tags": ["preference"],
    }
    resp = client.post("/v1/memory/ingest", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content"] == "The user prefers dark mode."
    assert body["status"] == "active"
    assert body["importance_score"] == 0.9
    assert body["embedded"] is True
    assert "id" in body


def test_ingest_memory_validation(client):
    # Empty content should fail.
    resp = client.post("/v1/memory/ingest", json={"content": ""})
    assert resp.status_code in (400, 422), resp.text


# --------------------------------------------------------------------------- #
# Memory retrieve
# --------------------------------------------------------------------------- #
def test_retrieve_memory(client):
    # Ingest a block first so there's something to retrieve.
    client.post(
        "/v1/memory/ingest",
        json={"content": "The user works in Berlin.", "source": "api-test", "importance_score": 0.8},
    )
    payload = {"text": "Berlin", "top_k": 5, "mode": "hybrid"}
    resp = client.post("/v1/memory/retrieve", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] >= 1
    assert any("Berlin" in r["content"] for r in body["results"])
    assert body["results"][0]["score"] >= 0.0


def test_retrieve_memory_invalid_mode(client):
    payload = {"text": "x", "mode": "not-a-mode"}
    resp = client.post("/v1/memory/retrieve", json=payload)
    assert resp.status_code in (400, 422), resp.text


# --------------------------------------------------------------------------- #
# Chat completions (universal proxy)
# --------------------------------------------------------------------------- #
def test_chat_completions(client):
    # Ingest context so injection has something to retrieve.
    client.post(
        "/v1/memory/ingest",
        json={"content": "The user likes dark mode.", "source": "api-test", "importance_score": 0.9},
    )
    payload = {
        "message": "What does the user like?",
        "model_profile": "default",
        "top_k": 5,
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "response" in body
    assert body["model_profile"] == "default"
    assert "injected_blocks" in body
    assert "total_injected_tokens" in body
    assert "budget" in body


def test_chat_completions_default_profile(client):
    client.post(
        "/v1/memory/ingest",
        json={"content": "The user likes dark mode.", "source": "api-test", "importance_score": 0.9},
    )
    payload = {"message": "Hi", "stream": False}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text


def test_chat_completions_stream(client):
    client.post(
        "/v1/memory/ingest",
        json={"content": "streaming context", "source": "api-test", "importance_score": 0.9},
    )
    payload = {"message": "stream this", "stream": True}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers.get("content-type", "")
    text = resp.text
    assert "metadata" in text
    assert "done" in text


# --------------------------------------------------------------------------- #
# Memory CRUD
# --------------------------------------------------------------------------- #
def test_memory_block_crud(client):
    # Ingest
    resp = client.post(
        "/v1/memory/ingest",
        json={"content": "crud block", "source": "api-test"},
    )
    assert resp.status_code == 200
    block_id = resp.json()["id"]

    # Get
    resp = client.get(f"/v1/memory/blocks/{block_id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["content"] == "crud block"

    # List
    resp = client.get("/v1/memory/blocks?limit=10")
    assert resp.status_code == 200
    assert len(resp.json()["blocks"]) >= 1

    # Delete
    resp = client.delete(f"/v1/memory/blocks/{block_id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted"] is True

    # Gone
    resp = client.get(f"/v1/memory/blocks/{block_id}")
    assert resp.status_code == 404


def test_memory_block_invalid_id(client):
    resp = client.get("/v1/memory/blocks/not-a-uuid")
    assert resp.status_code == 400, resp.text


# --------------------------------------------------------------------------- #
# Consolidation
# --------------------------------------------------------------------------- #
def test_consolidation_run(client):
    resp = client.post("/v1/consolidation/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "id" in body
    assert "status" in body
    assert body["status"] in ("pending", "running", "completed", "failed")


def test_list_consolidation_jobs(client):
    client.post("/v1/consolidation/run")
    resp = client.get("/v1/consolidation/jobs")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "jobs" in body
    assert len(body["jobs"]) >= 1


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_chat_completions_unknown_profile(client):
    payload = {"message": "Hi", "model_profile": "does-not-exist", "stream": False}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400, resp.text


def test_ingest_missing_content(client):
    resp = client.post("/v1/memory/ingest", json={})
    assert resp.status_code in (400, 422), resp.text


# --------------------------------------------------------------------------- #
# create_app standalone
# --------------------------------------------------------------------------- #
def test_create_app_default_config():
    app = create_app()
    assert app is not None
    with TestClient(app) as tc:
        resp = tc.get("/v1/health")
        assert resp.status_code == 200

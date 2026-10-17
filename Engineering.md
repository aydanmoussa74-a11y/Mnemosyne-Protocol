# Mnemosyne Protocol — Engineering Log

**Project:** Mnemosyne Protocol — hyper-scaling, model-agnostic agent memory engine  
**Started:** May 18, 2026  
**Author:** Built mobile-first in Acode, pushed to GitHub  
**Status:** Production core complete — 81 tests passing

---

## The origin story

This project began on May 18, 2026, written entirely on a mobile device in
Acode, with commits pushed straight to GitHub from a phone. No desktop IDE, no
local dev server — just a text editor, a terminal, and the conviction that
agent memory deserved better than "a vector index with a README."

The mission was simple but ambitious: build a memory engine that doesn't just
store embeddings, but *understands* them relationally, compresses them
autonomously, and injects them safely into any model — without locking you into
a single vendor or a heavy hosted stack.

---

## Architectural breakthroughs

### 1. Hybrid vector + knowledge graph, zero-dependency core

Most memory products are thin wrappers around a single vector index. We built a
dual-path store from scratch:

* **Vector path** — dense embeddings with exact cosine similarity, implemented
  in pure Python with a numpy-accelerated batch fallback. No Pinecone, no
  Milvus, no managed service required.
* **Graph path** — entities, relations, and block-to-entity edges persisted in
  a relational SQLite store, traversable to configurable depth.
* **Fusion** — Reciprocal Rank Fusion (RRF) merges vector and graph hits so the
  most relevant anchors surface regardless of which path found them.

The entire core runs with zero external ML dependencies. Optional packages
(`openai`, `anthropic`, `httpx`, `numpy`, `tiktoken`) plug in for cloud models
and accurate token counting, but the engine itself never requires them.

### 2. Surviving Python 3.14 and the Pydantic gap

Early development hit a wall: Python 3.14's toolchain couldn't build
`pydantic-core` from source (no Rust toolchain available in the environment),
which blocked Pydantic v2 and the FastAPI versions that depend on it.

We pivoted to Pydantic v1 (pure Python wheel) and pinned FastAPI 0.104.1 /
Starlette 0.27.0 / uvicorn 0.24.0 — a compatible, production-tested stack that
installs cleanly without compilation. All adapters lazy-import their provider
packages, so the engine imports and runs even when none of the optional model
libraries are installed.

### 3. Conquering SQLite write locks with WAL

The consolidation pipeline runs background worker threads that continuously
compress, prune, and synthesize memory while the FastAPI threadpool serves
concurrent ingest/retrieve requests. On a single SQLite file, that is a recipe
for `database is locked`.

We solved it with:

* `PRAGMA journal_mode=WAL` — readers never block writers and vice versa.
* `PRAGMA busy_timeout=30000` + connection `timeout=30` — writers patiently
  queue instead of failing immediately.
* `check_same_thread=False` + `pool_pre_ping=True` — safe cross-thread access
  from FastAPI's `run_in_threadpool`.
* A re-entrant lock around the in-memory index so the vector graph and the
  relational store stay consistent under load.

Stress tests confirm: 4 concurrent ingest workers plus a runaway consolidation
loop produce zero lock errors and zero crashes.

### 4. Strict tenant isolation and prompt-injection defense

Memory is multi-tenant by design. Every `MemoryBlock`, `MemoryEntity`,
`MemoryRelation`, and `MemoryEdge` carries a `tenant_id`, and every store and
retrieval operation filters by it. Tenants can never read, write, or delete
each other's memory — verified by dedicated cross-tenant tests.

On the safety side, injected memory content is scrubbed of instruction-override
markers (`ignore previous instructions`, `system:`, `assistant:`, `user:`,
`### Instruction`, etc.) before it reaches the model prompt. This prevents an
attacker from storing a poisoned block that hijacks future assistant responses.

### 5. Graceful degradation

External model providers fail. Adapters wrap provider errors in `AdapterError`
and never leak raw stack traces or API keys to clients. The FastAPI server
returns sanitized JSON error bodies while logging full traces server-side. If a
provider is unreachable, the engine degrades cleanly instead of crashing the
core.

---

## Test suite

The repository ships with **81 passing tests** covering:

* Schemas, store (CRUD, vector search, graph traversal, dimension mismatch)
* Retrieval (vector, graph, hybrid RRF)
* Injection (token budget, packing, prompt rendering)
* Consolidation (prune, compress, synthesize)
* Engine orchestration
* FastAPI server endpoints (health, profiles, ingest, retrieve, chat, CRUD,
  consolidation)
* CLI chat client (turn processing, command handling, arg parsing)
* Security & concurrency (cross-tenant isolation, auth, payload limits,
  prompt-injection sanitization, SQLite lock stress, provider failure)

Run anytime with:

```bash
pytest tests/
```

---

## Directory layout

```
mnemosyne/
  __init__.py       # public API surface
  adapters.py       # OpenAI / Anthropic / Ollama / vLLM / local model hooks
  chat_client.py    # interactive terminal chat loop
  cli.py            # mnemosyne CLI (ingest / retrieve / inject / complete)
  config.py         # ModelProfile + MnemosyneConfig
  consolidation.py  # autonomous background consolidation pipeline
  core.py           # MnemosyneEngine orchestration
  embeddings.py     # pluggable embedding providers
  exceptions.py     # structured error hierarchy
  injection.py      # dynamic context injection middleware
  profiles.yaml     # plug-and-play model configuration reference
  retrieval.py      # hybrid RRF vector + graph retrieval
  schemas.py        # Pydantic data models
  server.py         # FastAPI microservice
  store.py          # hybrid SQLite + vector store
  token_budget.py   # token-budget calculation

tests/
  test_*.py         # 81 tests covering every subsystem + security

Dockerfile
docker-compose.yml
.env.example
SECURITY_AUDIT.md
```

---

## Looking ahead

* Persistent tenant provisioning registry (currently implicit/first-write-wins)
* Per-tenant/IP rate limiting on the HTTP surface
* PostgreSQL migration path for very high write throughput
* Formal NER model integration for consolidation (replacing the heuristic
  extractor)

The core is solid, the tests are green, and the foundation is production-ready.
Contributions welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md).

# The Mnemosyne Protocol

A hyper-scaling, model-agnostic agent memory engine that shatters traditional
context window limitations. Mnemosyne combines a **hybrid vector & knowledge
graph core**, **autonomous memory consolidation**, **dynamic context injection**,
and **universal model-agnostic hooks** into a single production-grade Python
package.

---

## Why not just another vector DB wrapper?

Most "memory" products are thin wrappers around a single vector index. They
forget that intelligence is relational as well as semantic.

| Capability | Standard Vector Wrapper | Mnemosyne Protocol |
|---|---|---|
| Retrieval | Cosine similarity only | **Hybrid RRF: vector + knowledge graph** |
| Core deps | Heavy ML stack + hosted DB | **Zero-dependency core** (pure Python + SQLite) |
| Concurrency | File locks / single writer | **Bulletproof SQLite WAL** + busy timeouts |
| Tenancy | One flat bucket | **Enterprise multi-tenancy** (strict isolation) |
| Safety | Raw memory into the prompt | **Prompt-injection sanitization** |
| Lifecycle | Static embeddings | **Autonomous consolidation** (compress/prune/synthesize) |

Mnemosyne stores every memory block in a dual-path store:

1. **Vector path** — dense embeddings with exact cosine similarity (numpy when
   available, pure-Python fallback).
2. **Graph path** — entities, relations, and block-to-entity edges in a
   relational SQLite store, traversable to configurable depth.

Results are fused with **Reciprocal Rank Fusion (RRF)** so the most relevant
anchors rise to the top regardless of which path found them.

---

## Core features

* **Hybrid Vector & Knowledge Graph Core** — dual-path semantic and relational
  retrieval with RRF fusion.
* **Autonomous Memory Consolidation Pipeline** — background worker threads
  compress, prune, and synthesize short-term chat logs into long-term semantic
  knowledge blocks without losing granular detail.
* **Dynamic Context Injection Middleware** — exact token-budget calculation,
  micro-targeted historical anchor retrieval, and seamless prompt augmentation
  for any target LLM.
* **Universal Model-Agnostic Hooks** — standardized adapters for OpenAI,
  Anthropic, Ollama, vLLM, and local callables.
* **Enterprise multi-tenancy** — strict `tenant_id` isolation; tenants can never
  read, write, or delete each other's memory.
* **Prompt-injection sanitization** — injected memory content is scrubbed of
  instruction-override markers before it reaches the model.

---

## Quick start

### Option A: Docker (one command)

```bash
# Clone the repo
git clone https://github.com/aydanmoussa74-a11y/Mnemosyne-Protocol.git
cd Mnemosyne-Protocol

# Spin up the FastAPI microservice
docker compose up -d

# Ingest a memory
curl -X POST http://localhost:8000/v1/memory/ingest \
  -H 'Content-Type: application/json' \
  -d '{"content":"The user prefers dark mode and works in Berlin.","source":"demo","importance_score":0.9,"tenant_id":"acme"}'

# Chat with memory context
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"message":"What does the user prefer?","tenant_id":"acme","stream":false}'
```

### Option B: Local pip install

```bash
git clone https://github.com/aydanmoussa74-a11y/Mnemosyne-Protocol.git
cd Mnemosyne-Protocol

pip install -e .               # core only
pip install -e .[all]          # OpenAI, Anthropic, Ollama, vLLM adapters + numpy/tiktoken

# Run the CLI chat loop
mnemosyne-chat --storage-path ./mnemosyne_data --profile default
```

### Python API

```python
from mnemosyne import MnemosyneEngine, MnemosyneConfig
from mnemosyne.config import ModelProfile

cfg = MnemosyneConfig(storage_path="./mnemosyne_data")
cfg.register_profile(ModelProfile.openai(name="default", api_key="sk-...", model="gpt-4o"))

engine = MnemosyneEngine(cfg)

# Ingest
block = engine.ingest(
    "The user prefers dark mode and works in Berlin.",
    source="chat",
    tenant_id="acme",
    importance_score=0.9,
)

# Retrieve
blocks = engine.retrieve("dark mode Berlin", tenant_id="acme", top_k=5)

# Inject context + complete
result = engine.inject("What does the user prefer?", tenant_id="acme")
response = engine.complete(result, profile_name="default")
print(response)

engine.stop()
```

### CLI chat loop

```bash
mnemosyne-chat \
  --storage-path ./mnemosyne_data \
  --profile default \
  --top-k 8 \
  --mode hybrid \
  --temperature 0.7
```

Every turn automatically retrieves relevant history, injects it into the
prompt, streams the model response, and writes the turn back into memory.

---

## Security

See [SECURITY_AUDIT.md](./SECURITY_AUDIT.md) for the full vulnerability matrix
and the hardening patches applied (multi-tenancy, authentication, SQLite WAL
concurrency, DoS payload limits, prompt-injection sanitization).

When exposing the HTTP service, set `MNEMOSYNE_API_KEY` and call endpoints with
`Authorization: Bearer <key>`.

---

## License

[MIT](./LICENSE) — Copyright 2026 The Mnemosyne Protocol Contributors.

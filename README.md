# Mnemosyne Protocol

A hyper-scaling, model-agnostic agent memory engine that shatters traditional
context window limitations. Mnemosyne combines a **hybrid vector & knowledge
graph core**, an **autonomous memory consolidation pipeline**, **dynamic
context injection middleware**, and **universal model-agnostic hooks** into a
single production-grade Python package.

## Core capabilities

| Layer | Responsibility |
|-------|----------------|
| **Hybrid Vector & Knowledge Graph Core** | Dual-path semantic (cosine similarity) and relational (entity/relation graph) retrieval, fused with Reciprocal Rank Fusion (RRF). |
| **Autonomous Memory Consolidation** | Background worker threads compress, prune, and synthesize short-term chat logs into long-term semantic knowledge blocks. |
| **Dynamic Context Injection Middleware** | Exact token-budget calculation, micro-targeted anchor retrieval, and seamless prompt augmentation for any target LLM. |
| **Universal Model-Agnostic Hooks** | Standardized adapters for OpenAI, Anthropic, Ollama, vLLM, and local callables. |

## Installation

```bash
pip install -e .               # core only
pip install -e .[all]          # with OpenAI, Anthropic, Ollama, vLLM, numpy, tiktoken
```

## Quick start

```python
from mnemosyne import MnemosyneEngine, MnemosyneConfig
from mnemosyne.config import ModelProfile

cfg = MnemosyneConfig()
cfg.register_profile(ModelProfile.openai(api_key="sk-...", model="gpt-4o"))

engine = MnemosyneEngine(cfg)

# 1. Ingest memories
engine.ingest("The user prefers dark mode and works in Berlin.")
engine.ingest("The user's project deadline is next Friday.")

# 2. Inject relevant context into a prompt
result = engine.inject("What should I remind the user about?")
print(result.prompt)

# 3. Complete via the model adapter (one-shot)
response = engine.inject_and_complete("What should I remind the user about?")
print(response)

engine.stop()
```

## Configuration

See `mnemosyne/profiles.yaml` for the full plug-and-play profile reference
(OpenAI, Anthropic, Ollama, vLLM). Load it with:

```python
import yaml
from mnemosyne.config import load_config_from_dict

with open("mnemosyne/profiles.yaml") as f:
    raw = yaml.safe_load(f)
# substitute env vars (e.g. ${OPENAI_API_KEY}) as needed before passing to load_config_from_dict
cfg = load_config_from_dict(raw)
```

## CLI

```bash
mnemosyne ingest "The user likes dark mode." --source cli --importance 0.8
mnemosyne retrieve "user preferences" --top-k 5 --mode hybrid
mnemosyne inject "What does the user prefer?"
mnemosyne complete "What does the user prefer?" --profile default
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MnemosyneEngine                      │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │ Store    │  │ Retrieval  │  │ Injection        │   │
│  │ (Vector  │  │ Engine     │  │ Middleware       │   │
│  │  + Graph)│  │ (RRF)      │  │ (token budget)   │   │
│  └────┬─────┘  └──────┬─────┘  └────────┬─────────┘   │
│       └───────────────┴─────────────────┘             │
│              Consolidation Pipeline (bg workers)        │
│              Model Adapters (OpenAI/Anthropic/Ollama..) │
└─────────────────────────────────────────────────────────┘
```

## Data schemas

Defined in `mnemosyne/schemas.py` using Pydantic:
`MemoryBlock`, `MemoryEntity`, `MemoryRelation`, `MemoryEdge`,
`ConsolidationJob`, `TokenBudget`, `InjectionResult`, `RetrievalResult`,
`MemorySearchQuery`.

## License

MIT

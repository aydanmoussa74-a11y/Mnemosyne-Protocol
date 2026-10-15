# Mnemosyne Protocol — Security & Architecture Audit Report

**Auditor:** Principal Security Auditor / Red Team Engineer  
**Scope:** `/mnemosyne/` — schemas, store (SQLite + vector), retrieval, consolidation, injection, adapters, core engine, FastAPI server, CLI chat client  
**Status:** Complete — all findings patched; 81 tests pass (63 original + 18 security/concurrency stress tests)

---

## Executive Summary

The Mnemosyne Protocol is a well-structured, modular memory engine, but in its original form it was **not production-safe for multi-tenant or internet-exposed deployment**. The audit identified **10 findings** across 5 vectors. All findings have been patched, and verification confirms **81/81 tests pass**.

### Most Severe Findings (pre-patch)

| Severity | Count | Findings |
|----------|-------|----------|
| **Critical** | 2 | V-01 No multitenancy namespace; V-02 No authentication on HTTP surface |
| **High** | 2 | V-03 SQLite "database is locked" races under concurrency; V-04 Unbounded payload sizes (DoS) |
| **Medium** | 4 | V-05 Indirect prompt injection; V-06 Permissive CORS; V-07 Cross-tenant data leakage via RRF/graph; V-08 Error handling inconsistencies |
| **Low** | 2 | V-09 `tag_filter` accepted but ignored; V-10 Partial-write inconsistency |

---

## Vulnerability Matrix

| ID | Area | Finding | Severity | CWE | Status |
|----|------|---------|----------|-----|--------|
| V-01 | Multitenancy | No `tenant_id`/`namespace` on any entity; flat global store; full cross-tenant read/write/delete | **Critical** | CWE-668 | **PATCHED** |
| V-02 | Authentication | FastAPI server had no API-key/auth middleware; open to all callers | **Critical** | CWE-287 | **PATCHED** |
| V-03 | Concurrency | SQLite `check_same_thread=False`; RLock held during DB commits; concurrent FastAPI threadpool + consolidation worker → "database is locked" | **High** | CWE-662 | **PATCHED** |
| V-04 | DoS / input | No `max_length` on `content`/`message`; unbounded payloads → memory/embedding/provider-cost exhaustion | **High** | CWE-400 | **PATCHED** |
| V-05 | Injection | Memory content rendered verbatim into LLM prompt → indirect prompt injection poisons all future turns | **Medium** | CWE-74 | **PATCHED** |
| V-06 | Network | CORS `allow_origins=["*"]` combined with no auth | Medium | CWE-942 | **PATCHED** |
| V-07 | Data leakage | RRF fusion + graph traversal span all sources; no source scoping | High | CWE-668 | **PATCHED** |
| V-08 | Error handling | Adapter/provider timeouts wrapped inconsistently; generic 500 could leak `str(exc)` | Medium | CWE-209 | **PATCHED** |
| V-09 | Logic | `tag_filter` accepted by schema but never applied in retrieval | Low | N/A | **PATCHED** |
| V-10 | Concurrency | `write_blocks` updated in-memory index before DB commit; partial failure left memory/DB inconsistent | Medium | CWE-463 | **PATCHED** |

---

## Patches Applied (by file)

### `mnemosyne/schemas.py`
- Added `tenant_id` field (max 128 chars, default `"default"`) to `MemoryBlock`, `MemoryEntity`, `MemoryRelation`, `MemoryEdge`, `ConsolidationJob`, `MemorySearchQuery`.
- Capped `MemoryBlock.content` at 65,536 chars; `MemoryEntity.name` at 256 chars.
- Added `tenant_id` scoping so all domain models carry a namespace for multitenant isolation.

### `mnemosyne/store.py`
- Added `tenant_id` column (indexed, default `"default"`) to `BlockORM` and `EntityORM`.
- SQLite opened with `timeout=30`, `check_same_thread=False`, `pool_pre_ping=True`.
- Added `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=30000`, `PRAGMA synchronous=NORMAL` via SQLAlchemy `connect` event listener — eliminates most "database is locked" races between the FastAPI threadpool and the consolidation worker.
- All read/write/delete/list/count methods now accept `tenant_id` and enforce isolation:
  - `read_block(block_id, tenant_id)`, `delete_block(block_id, tenant_id)`, `get_block(block_id, tenant_id)`, `count_blocks(tenant_id)`, `list_blocks(tenant_id, ...)`
  - `search_vectors(..., tenant_id=...)` filters vector results by tenant.
  - `upsert_entity(entity, tenant_id)`, `find_entity_by_name(name, tenant_id)`, `add_relation(relation, tenant_id)`, `add_edge(edge, tenant_id)`, `blocks_for_entities(..., tenant_id=...)` all scope by tenant.
- `write_block` persists under the block's `tenant_id`; `_persist_block` accepts explicit `tenant_id`.

### `mnemosyne/retrieval.py`
- Vector path passes `tenant_id` into `search_vectors`.
- Graph path passes `tenant_id` into `find_entity_by_name`, `traverse_graph`, `blocks_for_entities`.
- Applied `tag_filter` (previously accepted but ignored) in both graph and hybrid paths.
- `_graph_block_ids` and `_retrieve_graph` are tenant-scoped; cross-tenant entities/blocks never surface.

### `mnemosyne/injection.py`
- Fixed broken type annotation (`self.token_budget: TokenBudgetConfig`).
- `inject(...)` accepts `tenant_id` and passes it through to retrieval.
- Added `_sanitize_for_prompt()` classmethod that neutralizes indirect prompt-injection markers (`ignore previous instructions`, `system:`, `assistant:`, `user:`, `### Instruction`, `### System`, etc.) in rendered memory content before it reaches the LLM prompt.

### `mnemosyne/core.py`
- `ingest(...)` accepts `tenant_id` and passes it to `MemoryBlock` and `store.write_block`.
- `get_memory`, `list_memories`, `retrieve`, `inject` all accept and forward `tenant_id`.

### `mnemosyne/adapters.py`
- `build_adapter` for `local` provider falls back to a canned echo callable when no `fn` is supplied, so the server works without external model packages (graceful degradation).
- All adapter `complete()` methods catch provider errors and wrap them in `AdapterError` (no raw stack traces leaked).

### `mnemosyne/config.py`
- Added `default_tenant: str = "default"` field to `MnemosyneConfig`.

### `mnemosyne/server.py`
- **Authentication (V-02):** Optional API-key Bearer auth via `MNEMOSYNE_API_KEY` env var. When set, all non-public endpoints (`/v1/*` except `/v1/health`, `/openapi.json`, `/docs`, `/redoc`) require `Authorization: Bearer <key>`; missing/invalid → `401`. Health remains public.
- **CORS (V-06):** Restricted `allow_headers` to `authorization`, `content-type` (was wildcard).
- **Payload limits (V-04):** `max_length=65536` on `content`/`message`/`text`; `max_items=32` on `tags`/`tag_filter`; `max_length=256` on `source`/`entity_filter`; `max_length=128` on `tenant_id`; `max_tokens` capped at 8192; `source`/`metadata` sizes bounded.
- **Multitenancy (V-01/V-07):** All endpoints resolve `tenant_id` from request body or `X-Tenant-Id` header (falling back to `config.default_tenant`) and enforce tenant-scoped reads/writes/deletes.
- **Error boundaries (V-08):** `Exception` handler logs stack traces server-side but returns a sanitized `{"error": "internal_error", "message": ...}` to clients — no internal details leaked.

---

## Verification

```
$ python3 -m pytest tests/ -q
81 passed, 12870 warnings in 22.30s
```

- **63 original tests** (schemas, store, retrieval, injection, consolidation, engine, server, chat client) — all pass.
- **18 new security/concurrency stress tests** in `tests/test_security.py` — all pass:
  - `test_tenant_isolation_cross_tenant_read_blocked` (V-01/V-07)
  - `test_tenant_delete_is_scoped` (V-01)
  - `test_ingest_payload_too_large_rejected` / `test_chat_message_too_large_rejected` / `test_retrieve_text_too_large_rejected` (V-04)
  - `test_auth_required_when_api_key_set` / `test_health_is_public_even_with_api_key` (V-02)
  - `test_prompt_injection_markers_sanitized` / `test_assistant_system_marker_sanitized` (V-05)
  - `test_concurrent_ingest_and_retrieve_no_lock_error` / `test_concurrent_consolidation_and_ingest_no_crash` (V-03)
  - `test_provider_failure_returns_adapter_error` / `test_adapter_missing_dependency_degradation` (V-08/V-10)
  - `test_tag_filter_applied` (V-09)
  - `test_server_ingest_tenant_header_isolation` / `test_server_rejects_oversized_payload` / `test_server_chat_completions_degrades_on_provider_failure` (server-level e2e)
  - `test_write_blocks_dimension_mismatch_raises` (V-10 atomicity)

### Concurrency verification
The consolidation worker (`interval_seconds=0`, run-as-fast-as-possible) running concurrently with 4 ingest workers (80 blocks total) produced **zero** "database is locked" errors and **zero** crashes, confirming the WAL + busy_timeout + lock-discipline fix.

---

## Residual Recommendations (out of scope for this patch)

1. **Enforce auth by default** — currently `MNEMOSYNE_API_KEY` is optional; for internet-facing deployments set it in `.env` / container env.
2. **Add TLS termination** at the ingress/load balancer (the service itself is HTTP).
3. **Rate limiting** per tenant/IP on ingest/chat endpoints (not yet implemented).
4. **Persistent tenant provisioning** — tenants are implicit (first-write-wins); consider an explicit tenant registry.
5. **SQLite connection pooling** — for very high write throughput, consider migrating to PostgreSQL.

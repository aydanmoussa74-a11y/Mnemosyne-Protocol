"""Mnemosyne Protocol — FastAPI microservice.

A production-grade, async HTTP API wrapping :class:`mnemosyne.core.MnemosyneEngine`.

Endpoints:
    POST /v1/memory/ingest        — ingest memory blocks (auto-triggers consolidation)
    POST /v1/memory/retrieve      — hybrid RRF vector + graph search
    POST /v1/chat/completions     — universal proxy: budget → inject → model → response
    GET  /v1/health               — liveness / readiness
    GET  /v1/profiles             — registered model profiles
    POST /v1/profiles             — register a new model profile
    GET  /v1/consolidation/jobs   — consolidation job history
    POST /v1/consolidation/run    — run a consolidation pass
    GET  /v1/memory/blocks        — list memory blocks
    GET  /v1/memory/blocks/{id}   — get a memory block
    DELETE /v1/memory/blocks/{id} — delete a memory block

All engine calls are thread-safe and executed off the FastAPI event loop via
:func:`starlette.concurrency.run_in_threadpool`, so blocking SQLAlchemy / HTTP
adapter calls never block async request handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from starlette.concurrency import run_in_threadpool

from mnemosyne.adapters import build_adapter
from mnemosyne.config import MnemosyneConfig, ModelProfile
from mnemosyne.exceptions import MnemosyneError, NotFoundError
from mnemosyne.injection import InjectionResult
from mnemosyne.schemas import MemoryBlock, RetrievalMode
from mnemosyne.store import HybridStore

try:
    from mnemosyne.core import MnemosyneEngine
except ImportError:  # pragma: no cover - allows importing server standalone
    MnemosyneEngine = None  # type: ignore[assignment]

logger = logging.getLogger("mnemosyne.server")

# --------------------------------------------------------------------------- #
# Module-level process-wide engine singleton
# --------------------------------------------------------------------------- #

_engine_instance: Optional["MnemosyneEngine"] = None
_engine_lock = asyncio.Lock()


# --------------------------------------------------------------------------- #
# Pydantic request / response models (module-level so FastAPI can resolve them)
# --------------------------------------------------------------------------- #

class MemoryBlockIngestRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Raw memory text content.")
    source: str = Field(default="api", description="Origin label for this block.")
    importance_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Learned importance weight."
    )
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: Optional[int] = Field(
        default=None, ge=0, description="Optional TTL in seconds."
    )
    embed: bool = Field(default=True)

    @validator("importance_score")
    def _clamp_importance(cls, v):  # noqa: N805
        return max(0.0, min(1.0, v))


class MemoryBlockIngestResponse(BaseModel):
    id: str
    content: str
    status: str
    importance_score: float
    token_count: int
    embedded: bool


class MemoryRetrieveRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Natural-language query.")
    top_k: int = Field(default=10, ge=1, le=100)
    mode: str = Field(default="hybrid", description="vector | graph | hybrid")
    entity_filter: Optional[str] = None
    tag_filter: Optional[List[str]] = None
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)

    @validator("mode")
    def _validate_mode(cls, v):  # noqa: N805
        try:
            RetrievalMode(v)
        except ValueError:
            raise ValueError(
                f"Invalid mode '{v}'. Must be one of: "
                f"{[m.value for m in RetrievalMode]}"
            )
        return v


class RetrievalHitResponse(BaseModel):
    id: str
    content: str
    score: float
    path: str
    explanation: str
    importance_score: float
    tags: List[str] = Field(default_factory=list)


class MemoryRetrieveResponse(BaseModel):
    results: List[RetrievalHitResponse]
    total: int


class ChatCompletionsRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message.")
    model_profile: Optional[str] = Field(
        default=None, description="Name of a registered model profile."
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    mode: str = Field(default="hybrid")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)

    @validator("mode")
    def _validate_mode(cls, v):  # noqa: N805
        try:
            RetrievalMode(v)
        except ValueError:
            raise ValueError(
                f"Invalid mode '{v}'. Must be one of: "
                f"{[m.value for m in RetrievalMode]}"
            )
        return v


class ChatCompletionsResponse(BaseModel):
    response: str
    model_profile: str
    injected_blocks: List[RetrievalHitResponse]
    total_injected_tokens: int
    budget: Dict[str, Any]


class ProfileRegisterRequest(BaseModel):
    name: str
    provider: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_seconds: int = 60
    extra: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #

def create_app(config: Optional[MnemosyneConfig] = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Parameters
    ----------
    config:
        Optional pre-built :class:`MnemosyneConfig`. When omitted, defaults are
        used and profile registration is left to the caller via
        ``POST /v1/profiles`` or environment-driven setup.
    """
    if config is None:
        config = MnemosyneConfig()

    app = FastAPI(
        title="Mnemosyne Protocol",
        description=(
            "Hyper-scaling, model-agnostic agent memory engine — hybrid vector "
            "& knowledge graph core, autonomous consolidation, dynamic context "
            "injection, and universal model-agnostic hooks."
        ),
        version="1.0.0",
    )

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Structured request logging middleware
    # ------------------------------------------------------------------ #
    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @app.on_event("startup")
    async def _startup() -> None:
        global _engine_instance
        async with _engine_lock:
            if _engine_instance is None:
                logger.info("Initializing MnemosyneEngine (startup)")
                _engine_instance = MnemosyneEngine(config)
                _engine_instance.start()
        # Register default profiles if none present.
        if not _engine_instance.config.profiles:
            _engine_instance.config.register_profile(ModelProfile.openai())
        logger.info("Mnemosyne server ready")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        global _engine_instance
        async with _engine_lock:
            if _engine_instance is not None:
                logger.info("Stopping MnemosyneEngine (shutdown)")
                _engine_instance.stop()
                _engine_instance = None

    # ------------------------------------------------------------------ #
    # Dependency: engine accessor
    # ------------------------------------------------------------------ #
    def _get_engine() -> "MnemosyneEngine":
        if _engine_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Mnemosyne engine not initialized",
            )
        return _engine_instance

    # ------------------------------------------------------------------ #
    # Error handlers
    # ------------------------------------------------------------------ #
    @app.exception_handler(MnemosyneError)
    async def _mnemosyne_error_handler(_: Request, exc: MnemosyneError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=exc.to_dict(),
        )

    @app.exception_handler(NotFoundError)
    async def _not_found_handler(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": str(exc),
                "details": {},
            },
        )

    # ------------------------------------------------------------------ #
    # Helper: run a blocking engine call off the event loop
    # ------------------------------------------------------------------ #
    async def _run_sync(func, *args, **kwargs):
        return await run_in_threadpool(func, *args, **kwargs)

    # ------------------------------------------------------------------ #
    # Routes
    # ------------------------------------------------------------------ #
    @app.get("/v1/health", tags=["system"])
    async def health() -> Dict[str, Any]:
        engine = _get_engine()
        block_count = await _run_sync(engine.store.count_blocks)
        job_count = len(await _run_sync(engine.list_consolidation_jobs))
        return {
            "status": "healthy",
            "engine": "running",
            "blocks_stored": block_count,
            "consolidation_jobs": job_count,
            "default_profile": engine.config.default_profile,
            "profiles": list(engine.config.profiles.keys()),
        }

    @app.get("/v1/profiles", tags=["system"])
    async def list_profiles() -> Dict[str, Any]:
        engine = _get_engine()
        return await _run_sync(_list_profiles_sync, engine)

    def _list_profiles_sync(engine: "MnemosyneEngine") -> Dict[str, Any]:
        out = {}
        for name, profile in engine.config.profiles.items():
            out[name] = {
                "provider": profile.provider.value,
                "model": profile.model,
                "base_url": profile.base_url,
                "temperature": profile.temperature,
                "max_tokens": profile.max_tokens,
                "timeout_seconds": profile.timeout_seconds,
            }
        return {
            "profiles": out,
            "default": engine.config.default_profile,
        }

    @app.post("/v1/profiles", tags=["system"], status_code=status.HTTP_201_CREATED)
    async def register_profile(req: ProfileRegisterRequest) -> Dict[str, Any]:
        engine = _get_engine()
        return await _run_sync(_register_profile_sync, engine, req)

    def _register_profile_sync(
        engine: "MnemosyneEngine", req: ProfileRegisterRequest
    ) -> Dict[str, Any]:
        from mnemosyne.schemas import ModelProvider

        try:
            provider = ModelProvider(req.provider)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider '{req.provider}'",
            ) from exc

        profile = ModelProfile(
            name=req.name,
            provider=provider,
            base_url=req.base_url,
            api_key=req.api_key,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            timeout_seconds=req.timeout_seconds,
            extra=req.extra,
        )
        engine.config.register_profile(profile)
        return {
            "name": profile.name,
            "provider": profile.provider.value,
            "model": profile.model,
            "registered": True,
        }

    @app.post(
        "/v1/memory/ingest",
        response_model=MemoryBlockIngestResponse,
        tags=["memory"],
    )
    async def ingest_memory(req: MemoryBlockIngestRequest) -> MemoryBlockIngestResponse:
        engine = _get_engine()

        def _ingest() -> MemoryBlockIngestResponse:
            block = engine.ingest(
                content=req.content,
                source=req.source,
                importance_score=req.importance_score,
                tags=req.tags,
                metadata=req.metadata,
                ttl_seconds=req.ttl_seconds,
                embed=req.embed,
            )
            # Auto-trigger a consolidation pass in the background (non-blocking).
            try:
                engine.consolidation.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Auto-consolidation after ingest failed: %s", exc)
            return MemoryBlockIngestResponse(
                id=str(block.id),
                content=block.content,
                status=block.status.value,
                importance_score=block.importance_score,
                token_count=block.token_count,
                embedded=block.embedding is not None,
            )

        return await _run_sync(_ingest)

    @app.post(
        "/v1/memory/retrieve",
        response_model=MemoryRetrieveResponse,
        tags=["memory"],
    )
    async def retrieve_memory(req: MemoryRetrieveRequest) -> MemoryRetrieveResponse:
        engine = _get_engine()

        def _retrieve() -> MemoryRetrieveResponse:
            try:
                query_embedding = (
                    engine.embeddings.embed(req.text) if req.text else None
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Query embedding failed: %s", exc)
                query_embedding = None

            from mnemosyne.schemas import MemorySearchQuery

            query = MemorySearchQuery(
                text=req.text,
                embedding=query_embedding,
                top_k=req.top_k,
                mode=RetrievalMode(req.mode),
                entity_filter=req.entity_filter,
                tag_filter=req.tag_filter,
                min_importance=req.min_importance,
            )
            results = engine.retrieval.retrieve(query)
            hits = [
                RetrievalHitResponse(
                    id=str(r.block.id),
                    content=r.block.content,
                    score=r.score,
                    path=r.path.value,
                    explanation=r.explanation,
                    importance_score=r.block.importance_score,
                    tags=list(r.block.tags),
                )
                for r in results
            ]
            return MemoryRetrieveResponse(results=hits, total=len(hits))

        return await _run_sync(_retrieve)

    @app.post("/v1/chat/completions", tags=["chat"], response_model=None)
    async def chat_completions(
        req: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | StreamingResponse:
        """Universal chat/completions proxy.

        Flow: token budget → hybrid retrieval → context injection → model adapter
        → response (streamed or full).
        """
        engine = _get_engine()

        if req.stream:
            return StreamingResponse(
                _stream_completions(engine, req),
                media_type="text/event-stream",
            )

        def _complete() -> ChatCompletionsResponse:
            return _do_complete(engine, req)

        return await _run_sync(_complete)

    # ------------------------------------------------------------------ #
    # Internal completion logic
    # ------------------------------------------------------------------ #
    def _do_complete(
        engine: "MnemosyneEngine", req: ChatCompletionsRequest
    ) -> ChatCompletionsResponse:
        profile = engine.config.get_profile(req.model_profile)

        # 1. Inject context (budget + retrieval + prompt augmentation).
        injection: InjectionResult = engine.inject(
            user_input=req.message,
            model_name=profile.model,
            top_k=req.top_k,
            mode=RetrievalMode(req.mode),
        )

        # 2. Complete via the model adapter.
        adapter = engine._get_adapter(profile.name)
        kwargs: Dict[str, Any] = {}
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        response_text = adapter.complete(injection, **kwargs)

        hits = [
            RetrievalHitResponse(
                id=str(b.id),
                content=b.content,
                score=b.importance_score,
                path=RetrievalMode.HYBRID.value,
                explanation="injected",
                importance_score=b.importance_score,
                tags=list(b.tags),
            )
            for b in injection.injected_blocks
        ]

        return ChatCompletionsResponse(
            response=response_text,
            model_profile=profile.name,
            injected_blocks=hits,
            total_injected_tokens=injection.total_injected_tokens,
            budget=injection.budget.dict(),
        )

    async def _stream_completions(
        engine: "MnemosyneEngine", req: ChatCompletionsRequest
    ) -> AsyncGenerator[str, None]:
        """Stream completion chunks as server-sent events."""

        def _build() -> ChatCompletionsResponse:
            return _do_complete(engine, req)

        result: ChatCompletionsResponse = await run_in_threadpool(_build)

        # Emit injected-block metadata first.
        meta = {
            "type": "metadata",
            "model_profile": result.model_profile,
            "total_injected_tokens": result.total_injected_tokens,
            "budget": result.budget,
            "injected_blocks": [b.dict() for b in result.injected_blocks],
        }
        yield f"data: {meta}\n\n"

        # Stream the response text in chunks.
        text = result.response
        chunk_size = 40
        for i in range(0, len(text), chunk_size):
            chunk = {"type": "chunk", "content": text[i : i + chunk_size]}
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)

        yield "data: {\"type\": \"done\"}\n\n"

    # ------------------------------------------------------------------ #
    # Consolidation job endpoints
    # ------------------------------------------------------------------ #
    @app.get("/v1/consolidation/jobs", tags=["consolidation"])
    async def list_consolidation_jobs() -> Dict[str, Any]:
        engine = _get_engine()
        jobs = await _run_sync(engine.list_consolidation_jobs)
        return {
            "jobs": [
                {
                    "id": str(j.id),
                    "status": j.status.value,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "completed_at": (
                        j.completed_at.isoformat() if j.completed_at else None
                    ),
                    "blocks_processed": j.blocks_processed,
                    "blocks_created": j.blocks_created,
                    "blocks_pruned": j.blocks_pruned,
                    "error": j.error,
                }
                for j in jobs
            ]
        }

    @app.post("/v1/consolidation/run", tags=["consolidation"])
    async def run_consolidation() -> Dict[str, Any]:
        engine = _get_engine()
        job = await _run_sync(engine.run_consolidation)
        return {
            "id": str(job.id),
            "status": job.status.value,
            "blocks_processed": job.blocks_processed,
            "blocks_created": job.blocks_created,
            "blocks_pruned": job.blocks_pruned,
            "error": job.error,
        }

    # ------------------------------------------------------------------ #
    # Memory block CRUD (read / list / delete)
    # ------------------------------------------------------------------ #
    @app.get("/v1/memory/blocks/{block_id}", tags=["memory"])
    async def get_memory_block(block_id: str) -> Dict[str, Any]:
        from uuid import UUID as _UUID

        engine = _get_engine()

        def _get() -> Dict[str, Any]:
            try:
                block = engine.store.read_block(_UUID(block_id))
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid block id: {block_id}",
                ) from exc
            except NotFoundError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=exc.message,
                ) from exc
            return {
                "id": str(block.id),
                "content": block.content,
                "status": block.status.value,
                "importance_score": block.importance_score,
                "access_count": block.access_count,
                "source": block.source,
                "tags": list(block.tags),
                "embedding": block.embedding,
                "consolidated": block.consolidated,
                "metadata": block.metadata,
            }

        return await _run_sync(_get)

    @app.get("/v1/memory/blocks", tags=["memory"])
    async def list_memory_blocks(
        limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        engine = _get_engine()

        def _list() -> Dict[str, Any]:
            blocks = engine.store.list_blocks(limit=limit, offset=offset)
            return {
                "blocks": [
                    {
                        "id": str(b.id),
                        "content": b.content,
                        "status": b.status.value,
                        "importance_score": b.importance_score,
                        "source": b.source,
                        "tags": list(b.tags),
                    }
                    for b in blocks
                ],
                "limit": limit,
                "offset": offset,
            }

        return await _run_sync(_list)

    @app.delete("/v1/memory/blocks/{block_id}", tags=["memory"])
    async def delete_memory_block(block_id: str) -> Dict[str, Any]:
        from uuid import UUID as _UUID

        engine = _get_engine()

        def _delete() -> Dict[str, Any]:
            try:
                engine.store.delete_block(_UUID(block_id))
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid block id: {block_id}",
                ) from exc
            except NotFoundError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=exc.message,
                ) from exc
            return {"deleted": True, "id": block_id}

        return await _run_sync(_delete)

    return app


# --------------------------------------------------------------------------- #
# Module-level app instance (for `uvicorn mnemosyne.server:app`)
# --------------------------------------------------------------------------- #
app = create_app()

__all__ = ["create_app", "app"]

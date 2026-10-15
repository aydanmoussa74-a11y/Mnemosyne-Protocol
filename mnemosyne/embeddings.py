"""Embedding generation with pluggable providers (local, OpenAI, Ollama, vLLM)."""

from __future__ import annotations

import threading
from typing import List, Optional

from mnemosyne.config import EmbeddingConfig, MnemosyneConfig
from mnemosyne.exceptions import EmbeddingError

try:
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


class EmbeddingProvider:
    """Generates dense vector embeddings for text.

    Supported providers:
      * ``local``   — pure-Python TF-IDF-ish hashing embedding (no deps).
      * ``openai``  — OpenAI embedding API (requires `openai` package).
      * ``ollama``  — Ollama /api/embeddings endpoint.
      * ``vllm``    — vLLM embeddings endpoint.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._client = None
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Initialization (lazy, so import never fails on missing optional deps)
    # ------------------------------------------------------------------ #
    def _ensure(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            provider = self.config.provider.lower()
            if provider == "openai":
                try:
                    from openai import OpenAI  # type: ignore

                    self._client = OpenAI(
                        api_key=self.config.api_key,
                        base_url=self.config.base_url,
                    )
                except ImportError as exc:
                    raise EmbeddingError(
                        "OpenAI provider requires the 'openai' package",
                        code="missing_dependency",
                        details={"hint": "pip install openai"},
                    ) from exc
            elif provider in ("ollama", "vllm"):
                try:
                    import httpx  # type: ignore

                    self._client = httpx.Client(timeout=60.0)
                except ImportError as exc:
                    raise EmbeddingError(
                        f"{provider} provider requires the 'httpx' package",
                        code="missing_dependency",
                        details={"hint": "pip install httpx"},
                    ) from exc
            elif provider != "local":
                raise EmbeddingError(
                    f"Unknown embedding provider: {self.config.provider}",
                    code="unknown_provider",
                )
            self._initialized = True

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed(self, text: str) -> List[float]:
        """Embed a single string into a dense vector."""
        self._ensure()
        provider = self.config.provider.lower()
        if provider == "local":
            return self._embed_local(text)
        if provider == "openai":
            return self._embed_openai(text)
        if provider in ("ollama", "vllm"):
            return self._embed_http(text)
        raise EmbeddingError(
            f"Unknown embedding provider: {self.config.provider}",
            code="unknown_provider",
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings efficiently (provider-batching where supported)."""
        return [self.embed(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Provider implementations
    # ------------------------------------------------------------------ #
    def _embed_local(self, text: str) -> List[float]:
        """Deterministic hash-based embedding with no external dependencies.

        Produces a fixed-length vector via signed hashing over character
        n-grams. Good enough for semantic clustering in development and for
        environments where no model weights are available.
        """
        dims = self.config.dimensions
        vec = [0.0] * dims
        text_norm = text.lower()
        # Character 3-grams.
        for i in range(max(1, len(text_norm) - 2)):
            gram = text_norm[i : i + 3]
            h = 0
            for ch in gram:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            # Signed hash -> two buckets per gram.
            idx = h % dims
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[idx] += sign
        # L2-normalize.
        norm = math.sqrt(sum(v * v for v in vec)) if (math := __import__("math")) else 1.0
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _embed_openai(self, text: str) -> List[float]:
        from openai import OpenAI  # type: ignore

        assert self._client is not None
        client: OpenAI = self._client
        try:
            resp = client.embeddings.create(
                model=self.config.model,
                input=text,
            )
            return resp.data[0].embedding  # type: ignore[no-any-return]
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"OpenAI embedding request failed: {exc}",
                code="provider_error",
                details={"provider": "openai"},
            ) from exc

    def _embed_http(self, text: str) -> List[float]:
        import httpx  # type: ignore

        assert self._client is not None
        client: httpx.Client = self._client
        provider = self.config.provider.lower()
        try:
            if provider == "ollama":
                url = self.config.base_url.rstrip("/") + "/api/embeddings"
                payload = {"model": self.config.model, "prompt": text}
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["embedding"]  # type: ignore[no-any-return]
            # vLLM OpenAI-compatible embeddings endpoint.
            url = self.config.base_url.rstrip("/") + "/embeddings"
            payload = {"model": self.config.model, "input": text}
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]  # type: ignore[no-any-return]
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"{provider} embedding request failed: {exc}",
                code="provider_error",
                details={"provider": provider},
            ) from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise EmbeddingError(
                f"{provider} returned malformed embedding response: {exc}",
                code="malformed_response",
            ) from exc


# Module-level math import used by the local fallback.
import math  # noqa: E402

__all__ = ["EmbeddingProvider"]

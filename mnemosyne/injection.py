"""Dynamic Context Injection Middleware.

Intercepts user input, calculates an exact token budget, retrieves
micro-targeted historical anchors via the hybrid retrieval engine, and
injects them into the LLM payload as a ranked context block.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from mnemosyne.config import MnemosyneConfig, TokenBudgetConfig
from mnemosyne.exceptions import InjectionError, TokenBudgetError
from mnemosyne.schemas import (
    InjectionResult,
    MemoryBlock,
    MemorySearchQuery,
    RetrievalMode,
    TokenBudget,
)
from mnemosyne.token_budget import TokenCounter, calculate_budget
from mnemosyne.retrieval import RetrievalEngine
from mnemosyne.store import HybridStore


class ContextInjectionMiddleware:
    """Universal proxy layer for context-aware LLM prompting.

    Responsibilities:
      1. Count tokens in the user input.
      2. Compute the retrieval budget from the token-budget config.
      3. Retrieve the most relevant memory anchors (vector + graph + RRF).
      4. Pack anchors into the budget (greedy by score).
      5. Render a system/context preamble and return the augmented prompt.
    """

    # Instruction-override markers that may indicate an indirect prompt
    # injection stored in memory. They are replaced with a neutral token so
    # an attacker cannot hijack future assistant responses via a poisoned
    # memory block.
    _INJECTION_MARKERS = (
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard the above",
        "system:",
        "assistant:",
        "user:",
        "### Instruction",
        "### System",
    )

    def __init__(
        self,
        store: HybridStore,
        retrieval: RetrievalEngine,
        config: MnemosyneConfig,
    ) -> None:
        self.store = store
        self.retrieval = retrieval
        self.config = config
        self.token_budget: TokenBudgetConfig = config.token_budget
        self._counter = TokenCounter()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def inject(
        self,
        user_input: str,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        entity_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        tenant_id: str = "default",
    ) -> InjectionResult:
        """Produce an augmented prompt with retrieved memory anchors.

        Returns an :class:`InjectionResult` containing the final prompt and the
        list of blocks that were injected, so callers can log/audit them.
        """
        try:
            # 1. Token budget.
            budget = calculate_budget(
                self.token_budget,
                user_input,
                model_name=model_name,
                counter=self._counter,
            )
            if budget.max_blocks <= 0:
                # No room for context — return the raw input.
                return InjectionResult(
                    success=True,
                    injected_blocks=[],
                    total_injected_tokens=0,
                    budget=budget,
                    prompt=user_input,
                    metadata={"reason": "budget_exhausted"},
                )

            # 2. Build a query embedding if the store has a vector dimension.
            query_embedding = self._embed_query(user_input)

            # 3. Retrieve anchors (tenant-scoped).
            query = MemorySearchQuery(
                text=user_input,
                embedding=query_embedding,
                top_k=top_k or budget.max_blocks,
                mode=mode,
                entity_filter=entity_filter,
                tag_filter=tag_filter,
                tenant_id=tenant_id,
                min_importance=self.config.retrieval.min_importance,
            )
            results = self.retrieval.retrieve(query)

            # 4. Greedy pack into the token budget.
            injected: List[MemoryBlock] = []
            used_tokens = 0
            budget_tokens = max(0, budget.usable_for_retrieval)
            for result in results:
                block = result.block
                block_tokens = self._counter.count(block.content) or 1
                if used_tokens + block_tokens > budget_tokens:
                    continue
                # Touch the block (record access).
                block.touch()
                self.store.write_block(block)
                injected.append(block)
                used_tokens += block_tokens
                if len(injected) >= budget.max_blocks:
                    break

            # 5. Render the augmented prompt.
            prompt = self._render_prompt(user_input, injected)

            return InjectionResult(
                success=True,
                injected_blocks=injected,
                total_injected_tokens=used_tokens,
                budget=budget,
                prompt=prompt,
                metadata={
                    "retrieval_mode": mode.value,
                    "candidates_considered": len(results),
                },
            )
        except TokenBudgetError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InjectionError(
                f"Context injection failed: {exc}",
                code="injection_failed",
                details={"input_len": len(user_input)},
            ) from exc

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _embed_query(self, text: str) -> Optional[List[float]]:
        """Best-effort query embedding; returns None if no embedder is wired."""
        embedder = getattr(self, "_embedder", None)
        if embedder is None:
            return None
        try:
            return embedder.embed(text)
        except Exception:  # noqa: BLE001
            return None

    def set_embedder(self, embedder) -> None:  # noqa: ANN001
        self._embedder = embedder

    @classmethod
    def _sanitize_for_prompt(cls, text: str) -> str:
        """Neutralize prompt-injection sequences in memory content before rendering.

        Strips common instruction-override markers so an attacker cannot store
        a malicious block that hijacks future assistant responses.
        """
        for marker in cls._INJECTION_MARKERS:
            if marker.lower() in text.lower():
                text = re.sub(re.escape(marker), "[redacted]", text, flags=re.IGNORECASE)
        return text

    def _render_prompt(self, user_input: str, blocks: List[MemoryBlock]) -> str:
        if not blocks:
            return user_input

        lines = [
            "You are an assistant with access to a long-term memory store.",
            "Relevant memories retrieved from prior context (use them when helpful):",
            "",
        ]
        for idx, block in enumerate(blocks, 1):
            safe_content = self._sanitize_for_prompt(block.content)
            tag_line = f" [{', '.join(block.tags)}]" if block.tags else ""
            lines.append(
                f"[{idx}] (relevance={block.importance_score:.2f}{tag_line})\n"
                f"{safe_content}"
            )
        lines.append("")
        lines.append("---")
        lines.append("Current user input:")
        lines.append(user_input)
        return "\n".join(lines)


__all__ = ["ContextInjectionMiddleware"]

"""Token-budget calculation for dynamic context injection.

Computes how many tokens may be spent on retrieved memory anchors given a
target model's context window, the user's input, and system overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mnemosyne.config import TokenBudgetConfig
from mnemosyne.schemas import TokenBudget


class TokenCounter:
    """Approximate token counter.

    Falls back to a character-based heuristic (~4 chars/token) when the
    `tiktoken` package is unavailable, and uses `tiktoken` when present.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._encoder = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        try:
            import tiktoken  # type: ignore

            try:
                self._encoder = tiktoken.encoding_for_model(self.model_name or "gpt-4o")
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._encoder = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Heuristic: ~4 characters per token, round up.
        return max(1, (len(text) + 3) // 4)


def calculate_budget(
    config: TokenBudgetConfig,
    user_input: str,
    model_name: Optional[str] = None,
    counter: Optional[TokenCounter] = None,
) -> TokenBudget:
    """Compute the token budget available for retrieved memory anchors.

    Returns a :class:`TokenBudget` describing how many tokens may be spent on
    retrieved blocks without exceeding the model's context window.
    """
    counter = counter or TokenCounter(model_name)

    total = config.total_context_window
    system_reserved = config.system_reserved
    user_input_tokens = counter.count(user_input)

    # Overhead for prompt scaffolding / special tokens.
    overhead_tokens = int(total * config.overhead_ratio)

    # Portion explicitly reserved for retrieval (the "injection budget").
    retrieval_budget = int((total - system_reserved - user_input_tokens) * config.retrieval_ratio)
    retrieval_budget = max(0, retrieval_budget)

    # Cap retrieval budget by remaining headroom after system + input + overhead.
    headroom = total - system_reserved - user_input_tokens - overhead_tokens
    retrieval_budget = min(retrieval_budget, max(0, headroom))

    usable = max(0, retrieval_budget - overhead_tokens)

    # Estimate max blocks assuming an average block size (conservative 150 tokens).
    avg_block_tokens = 150
    max_blocks = min(config.max_blocks, max(1, usable // avg_block_tokens)) if usable > 0 else 0

    return TokenBudget(
        total_available=total,
        system_reserved=system_reserved,
        user_input_tokens=user_input_tokens,
        retrieval_budget=retrieval_budget,
        overhead_tokens=overhead_tokens,
        max_blocks=max_blocks,
    )

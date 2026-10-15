"""Custom exceptions for the Mnemosyne Protocol memory engine."""


class MnemosyneError(Exception):
    """Base exception for all Mnemosyne Protocol errors."""

    def __init__(self, message: str, code: str = "mnemosyne_error", details=None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(MnemosyneError):
    """Raised when configuration is invalid or missing."""


class StoreError(MnemosyneError):
    """Raised when the underlying storage layer fails."""


class EmbeddingError(MnemosyneError):
    """Raised when embedding generation fails."""


class RetrievalError(MnemosyneError):
    """Raised when memory retrieval fails."""


class ConsolidationError(MnemosyneError):
    """Raised when memory consolidation fails."""


class AdapterError(MnemosyneError):
    """Raised when a model adapter fails."""


class TokenBudgetError(MnemosyneError):
    """Raised when token budget calculation fails."""


class InjectionError(MnemosyneError):
    """Raised when context injection fails."""


class NotFoundError(MnemosyneError):
    """Raised when a requested resource does not exist."""


__all__ = [
    "MnemosyneError",
    "ConfigurationError",
    "StoreError",
    "EmbeddingError",
    "RetrievalError",
    "ConsolidationError",
    "AdapterError",
    "TokenBudgetError",
    "InjectionError",
    "NotFoundError",
]

"""Universal Model-Agnostic Hooks.

Standardized adapter interfaces for integrating with any external model:
OpenAI, Anthropic, Ollama, vLLM, and local callables. Each adapter converts
an :class:`InjectionResult` (augmented prompt) into a provider-native request
and returns a normalized text response.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import time

from mnemosyne.config import ModelProfile
from mnemosyne.exceptions import AdapterError
from mnemosyne.injection import InjectionResult


class ModelAdapter:
    """Base class for model adapters."""

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    def complete(self, result: InjectionResult, **kwargs) -> str:
        """Send the augmented prompt to the model and return the response text."""
        raise NotImplementedError

    def stream(self, result: InjectionResult, **kwargs):  # noqa: ANN001
        """Yield response text chunks (default: non-streaming emulation)."""
        yield self.complete(result, **kwargs)


class OpenAIAdapter(ModelAdapter):
    """OpenAI ChatCompletions adapter."""

    def complete(self, result: InjectionResult, **kwargs) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                "OpenAI adapter requires the 'openai' package",
                code="missing_dependency",
                details={"hint": "pip install openai"},
            ) from exc

        client = OpenAI(
            api_key=self.profile.api_key,
            base_url=self.profile.base_url,
        )
        temperature = kwargs.get("temperature", self.profile.temperature)
        max_tokens = kwargs.get("max_tokens", self.profile.max_tokens)

        try:
            resp = client.chat.completions.create(
                model=self.profile.model,
                messages=[
                    {"role": "system", "content": "You have access to long-term agent memory."},
                    {"role": "user", "content": result.prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""  # type: ignore[no-any-return]
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(
                f"OpenAI completion failed: {exc}",
                code="provider_error",
                details={"provider": "openai", "model": self.profile.model},
            ) from exc


class AnthropicAdapter(ModelAdapter):
    """Anthropic Messages API adapter."""

    def complete(self, result: InjectionResult, **kwargs) -> str:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                "Anthropic adapter requires the 'anthropic' package",
                code="missing_dependency",
                details={"hint": "pip install anthropic"},
            ) from exc

        client = anthropic.Anthropic(
            api_key=self.profile.api_key,
            base_url=self.profile.base_url,
        )
        temperature = kwargs.get("temperature", self.profile.temperature)
        max_tokens = kwargs.get("max_tokens", self.profile.max_tokens)

        try:
            resp = client.messages.create(
                model=self.profile.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system="You have access to long-term agent memory.",
                messages=[{"role": "user", "content": result.prompt}],
            )
            # Concatenate text blocks.
            parts = []
            for block in resp.content:  # type: ignore[attr-defined]
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "".join(parts)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(
                f"Anthropic completion failed: {exc}",
                code="provider_error",
                details={"provider": "anthropic", "model": self.profile.model},
            ) from exc


class OllamaAdapter(ModelAdapter):
    """Ollama /api/chat adapter."""

    def complete(self, result: InjectionResult, **kwargs) -> str:
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                "Ollama adapter requires the 'httpx' package",
                code="missing_dependency",
                details={"hint": "pip install httpx"},
            ) from exc

        url = (self.profile.base_url or "http://localhost:11434").rstrip("/") + "/api/chat"
        payload = {
            "model": self.profile.model,
            "messages": [
                {"role": "system", "content": "You have access to long-term agent memory."},
                {"role": "user", "content": result.prompt},
            ],
            "stream": False,
        }
        try:
            resp = httpx.post(url, json=payload, timeout=self.profile.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")  # type: ignore[no-any-return]
        except httpx.HTTPError as exc:  # noqa: BLE001
            raise AdapterError(
                f"Ollama completion failed: {exc}",
                code="provider_error",
                details={"provider": "ollama", "model": self.profile.model},
            ) from exc


class VLLMAdapter(ModelAdapter):
    """vLLM OpenAI-compatible /v1/chat/completions adapter."""

    def complete(self, result: InjectionResult, **kwargs) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                "vLLM adapter requires the 'openai' package",
                code="missing_dependency",
                details={"hint": "pip install openai"},
            ) from exc

        client = OpenAI(
            api_key="EMPTY",  # vLLM does not require a real key.
            base_url=self.profile.base_url,
        )
        temperature = kwargs.get("temperature", self.profile.temperature)
        max_tokens = kwargs.get("max_tokens", self.profile.max_tokens)

        try:
            resp = client.chat.completions.create(
                model=self.profile.model,
                messages=[
                    {"role": "system", "content": "You have access to long-term agent memory."},
                    {"role": "user", "content": result.prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""  # type: ignore[no-any-return]
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(
                f"vLLM completion failed: {exc}",
                code="provider_error",
                details={"provider": "vllm", "model": self.profile.model},
            ) from exc


class LocalAdapter(ModelAdapter):
    """Adapter wrapping any local callable ``callable(prompt: str) -> str``."""

    def __init__(self, profile: ModelProfile, fn: Callable[[str], str]) -> None:
        super().__init__(profile)
        self._fn = fn

    def complete(self, result: InjectionResult, **kwargs) -> str:
        try:
            return self._fn(result.prompt)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(
                f"Local model callable failed: {exc}",
                code="provider_error",
                details={"provider": "local"},
            ) from exc


# --------------------------------------------------------------------------- #
# Adapter registry / factory
# --------------------------------------------------------------------------- #

_ADAPTER_CLASSES = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "ollama": OllamaAdapter,
    "vllm": VLLMAdapter,
    "local": LocalAdapter,
}


def build_adapter(profile: ModelProfile, fn: Optional[Callable[[str], str]] = None) -> ModelAdapter:
    """Factory: build the right adapter for a :class:`ModelProfile`."""
    cls = _ADAPTER_CLASSES.get(profile.provider.value)
    if cls is None:
        raise AdapterError(
            f"No adapter registered for provider: {profile.provider.value}",
            code="unknown_provider",
        )
    if profile.provider.value == "local":
        if fn is None:
            # Canned echo adapter so the server works without an external model.
            def _echo(prompt: str) -> str:
                return f"[local echo] {prompt}"
            fn = _echo
        return cls(profile, fn)  # type: ignore[arg-type]
    return cls(profile)


__all__ = [
    "ModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
    "VLLMAdapter",
    "LocalAdapter",
    "build_adapter",
]

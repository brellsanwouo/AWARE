"""LLM protocol and shared errors."""

from __future__ import annotations

from typing import Any
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when provider call fails."""


class LLMClient(Protocol):
    """Provider-agnostic completion interface."""

    provider_name: str

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate completion text."""

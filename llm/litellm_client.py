"""LiteLLM-backed client for agent reasoning."""

from __future__ import annotations

import os
from typing import Any

from llm.base import LLMClient, LLMError


class LiteLLMClient(LLMClient):
    """Minimal LiteLLM wrapper using chat completion."""

    provider_name = "litellm"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        try:
            import litellm  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LLMError("LiteLLM is not installed. Run: pip install litellm") from exc

        self._litellm = litellm
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.timeout_seconds = float(timeout_seconds)

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_API_BASE"] = base_url
            os.environ["OPENAI_BASE_URL"] = base_url

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        try:
            response = self._litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **({"response_format": response_format} if response_format is not None else {}),
                timeout=self.timeout_seconds,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"LiteLLM completion failed: {exc}") from exc

"""LLM client factory."""

from __future__ import annotations

from llm.base import LLMClient
from llm.openai_compatible import OpenAICompatibleLLMClient


def create_llm_client(
    provider: str | None = None,
    *,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_model: str | None = None,
) -> LLMClient:
    """Create LLM client for OpenAI only."""
    normalized = (provider or "openai-compatible").strip().lower()
    if normalized not in {"openai", "openai-compatible"}:
        raise ValueError("Only OpenAI provider is supported in this runtime.")
    return OpenAICompatibleLLMClient(
        api_key=openai_api_key,
        base_url=openai_base_url,
        model=openai_model,
    )

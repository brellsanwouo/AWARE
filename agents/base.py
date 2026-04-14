"""Project agent base class built on Google ADK BaseAgent."""

from __future__ import annotations

from pydantic import ConfigDict

from google.adk.agents import BaseAgent


class Agent(BaseAgent):
    """Shared ADK agent base."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

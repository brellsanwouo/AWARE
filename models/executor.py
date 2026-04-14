"""Executor models for Assess runtime execution output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.buildspec import BuildSpec


class AssessFinding(BaseModel):
    """One extracted observation/anomaly from a specialized agent."""

    agent: str
    kind: Literal["anomaly", "observation"] = "observation"
    source: str
    summary: str
    evidence: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"


class AgentTaskResult(BaseModel):
    """Execution result for one specialized sub-agent."""

    agent_name: str
    target_path: str
    status: Literal["ok", "skipped", "error"] = "ok"
    findings: list[AssessFinding] = Field(default_factory=list)
    detail: str = ""


class ExecutorRunResult(BaseModel):
    """Structured output of ExecutorAgent (Assess execution stage)."""

    buildspec: BuildSpec
    agents_instantiated: list[str] = Field(default_factory=list)
    task_results: list[AgentTaskResult] = Field(default_factory=list)
    findings: list[AssessFinding] = Field(default_factory=list)
    preliminary_causes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    summary: str


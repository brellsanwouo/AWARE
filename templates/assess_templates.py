"""Template definitions for dynamic Assess sub-agent instantiation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTemplate:
    """Template contract for one Assess specialized sub-agent."""

    template_id: str
    agent_name: str
    role: str
    objective: str
    target_field: str
    tools: tuple[str, ...]
    domain: str


def load_assess_templates() -> list[AgentTemplate]:
    """Return default Assess templates.

    This file is the runtime source of truth for which agents can be instantiated.
    """
    return [
        AgentTemplate(
            template_id="logs-v1",
            agent_name="LogsAgent",
            role="Analyze application/service logs in failure window.",
            objective="Detect failure signals, exceptions, timeout patterns, retries.",
            target_field="absolute_log_file",
            tools=(
                "load_csv_window",
                "build_llm_observation_context",
                "count_matches",
                "sample_matching_lines",
            ),
            domain="logs",
        ),
        AgentTemplate(
            template_id="traces-v1",
            agent_name="TraceAgent",
            role="Analyze traces in failure window.",
            objective="Detect timeout chains, slow spans, retry propagation.",
            target_field="absolute_trace_file",
            tools=(
                "load_csv_window",
                "build_llm_observation_context",
                "extract_trace_durations",
                "sample_matching_lines",
            ),
            domain="trace",
        ),
        AgentTemplate(
            template_id="metrics-v1",
            agent_name="MetricsAgent",
            role="Analyze metrics in failure window.",
            objective="Detect CPU, latency and error-rate anomalies around failure.",
            target_field="absolute_metrics_file",
            tools=(
                "load_csv_window",
                "build_llm_observation_context",
                "max_numeric_column",
                "min_numeric_column",
                "max_value_after_keywords",
            ),
            domain="metrics",
        ),
    ]


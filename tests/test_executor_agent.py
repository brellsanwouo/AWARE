"""ExecutorAgent workflow tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.agents import BaseAgent

from agents.executor_agent import ExecutorAgent
from aware_models.buildspec import BuildSpec
from runtime.agent_factory import create_executor_agent
from runtime import agent_factory as agent_factory_module


class _DecisionStubLLM:
    """Deterministic LLM test double for analyzer decisions."""

    provider_name = "test-decision"

    def complete(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        return '{"summary":"stub","findings":[]}'


@pytest.fixture(autouse=True)
def _reset_executor_singleton_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWARE_ENABLE_REASONING", "true")
    monkeypatch.setenv("AWARE_ENABLE_MEMORY", "true")
    agent_factory_module._EXECUTOR_AGENT = None


def _buildspec_for_tmp(tmp_path) -> BuildSpec:
    date_dir = tmp_path / "2021_03_04"
    log_path = date_dir / "log" / "log_service.csv"
    trace_path = date_dir / "trace" / "trace_span.csv"
    metrics_path = date_dir / "metric" / "metric_container.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("timeout error retry timeout\n", encoding="utf-8")
    trace_path.write_text("checkout -> payment timeout 950ms\n", encoding="utf-8")
    metrics_path.write_text("cpu=93 latency=480 error_rate=7\n", encoding="utf-8")

    return BuildSpec.model_validate(
        {
            "task_type": "task_7",
            "date": "2021-03-04",
            "filename_date": "2021_03_04",
            "failure_time_range": {"start": "18:30:00", "end": "19:00:00"},
            "failure_time_range_ts": {"start": 1614879000, "end": 1614880800},
            "failures_detected": 1,
            "uncertainty": {
                "root_cause_time": "unknown",
                "root_cause_component": "unknown",
                "root_cause_reason": "unknown",
            },
            "objective": "Identify the root cause component, the exact root cause datetime and reason for the failure",
            "filename_date_directory": str(date_dir.resolve()),
            "absolute_log_file": [str(log_path.resolve())],
            "absolute_trace_file": [str(trace_path.resolve())],
            "absolute_metrics_file": [str(metrics_path.resolve())],
        }
    )


def test_executor_agent_is_google_adk_agent() -> None:
    agent = create_executor_agent(llm_client=_DecisionStubLLM())  # type: ignore[arg-type]
    assert isinstance(agent, BaseAgent)
    assert isinstance(agent, ExecutorAgent)


def test_executor_agent_runs_specialized_agents_and_emits_events(tmp_path) -> None:
    buildspec = _buildspec_for_tmp(tmp_path)
    agent = create_executor_agent(llm_client=_DecisionStubLLM())  # type: ignore[arg-type]
    events: list[tuple[str, str]] = []

    def on_event(event: object) -> None:
        phase = getattr(event, "phase", "")
        sender = getattr(event, "sender", "")
        events.append((sender, phase))

    result = agent.execute_assess(
        buildspec=buildspec,
        repository_path=str(tmp_path),
        on_event=on_event,
    )

    assert "LogsAgent" in result.agents_instantiated
    assert "TraceAgent" in result.agents_instantiated
    assert "MetricsAgent" in result.agents_instantiated
    assert any(item.kind == "anomaly" for item in result.findings)
    phases = [phase for _, phase in events]
    assert "instantiate_agent" in phases
    assert "dispatch" in phases
    assert "summary" in phases


def test_executor_filters_rows_strictly_by_failure_time_range(tmp_path) -> None:
    buildspec = _buildspec_for_tmp(tmp_path)
    log_path = Path(buildspec.absolute_log_file[0])
    trace_path = Path(buildspec.absolute_trace_file[0])
    metric_path = Path(buildspec.absolute_metrics_file[0])

    log_path.write_text(
        (
            "log_id,timestamp,cmdb_id,log_name,value\n"
            "a,1614878000,svc,gc,timeout error outside\n"
            "b,1614879500,svc,gc,timeout error inside\n"
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        (
            "timestamp,cmdb_id,parent_id,span_id,trace_id,duration\n"
            "1614879500000,dockerA2,p,s,t,900\n"
        ),
        encoding="utf-8",
    )
    metric_path.write_text(
        (
            "timestamp,rr,sr,cnt,mrt,tc\n"
            "1614879500,100.0,92.0,42,650.0,ServiceTest1\n"
        ),
        encoding="utf-8",
    )

    agent = create_executor_agent(llm_client=_DecisionStubLLM())  # type: ignore[arg-type]
    result = agent.execute_assess(buildspec=buildspec, repository_path=str(tmp_path))
    by_name = {item.agent_name: item for item in result.task_results}
    logs_result = by_name["LogsAgent"]
    assert "rows_in_window=1" in logs_result.detail
    assert "total_rows=2" in logs_result.detail
    assert any("Detected 1 timeout indicators in logs." in finding.summary for finding in logs_result.findings)


def test_executor_can_instantiate_multiple_agents_per_template(tmp_path) -> None:
    buildspec = _buildspec_for_tmp(tmp_path)
    date_dir = Path(buildspec.filename_date_directory)
    extra_log = date_dir / "log" / "log_service_extra.csv"
    extra_log.write_text(
        (
            "log_id,timestamp,cmdb_id,log_name,value\n"
            "x,1614879500,svc,app,timeout in extra file\n"
        ),
        encoding="utf-8",
    )

    agent = create_executor_agent(llm_client=_DecisionStubLLM())  # type: ignore[arg-type]
    result = agent.execute_assess(buildspec=buildspec, repository_path=str(tmp_path))
    log_agents = [name for name in result.agents_instantiated if name.startswith("LogsAgent")]
    assert len(log_agents) >= 2


def test_executor_respects_max_agents_cap(tmp_path) -> None:
    buildspec = _buildspec_for_tmp(tmp_path)
    date_dir = Path(buildspec.filename_date_directory)
    extra_log = date_dir / "log" / "log_service_extra.csv"
    extra_metric = date_dir / "metric" / "metric_extra.csv"
    extra_log.write_text(
        (
            "log_id,timestamp,cmdb_id,log_name,value\n"
            "x,1614879500,svc,app,timeout in extra file\n"
        ),
        encoding="utf-8",
    )
    extra_metric.write_text(
        (
            "timestamp,cpu,latency,error_rate\n"
            "1614879500,90,420,9\n"
        ),
        encoding="utf-8",
    )

    agent = create_executor_agent(llm_client=_DecisionStubLLM())  # type: ignore[arg-type]
    result = agent.execute_assess(
        buildspec=buildspec,
        repository_path=str(tmp_path),
        max_agents=2,
    )
    assert len(result.agents_instantiated) == 2


def test_executor_dynamic_expand_enqueues_component_focused_tasks(tmp_path) -> None:
    buildspec = _buildspec_for_tmp(tmp_path)
    log_path = Path(buildspec.absolute_log_file[0])
    trace_path = Path(buildspec.absolute_trace_file[0])
    metric_path = Path(buildspec.absolute_metrics_file[0])

    log_path.write_text(
        (
            "log_id,timestamp,cmdb_id,log_name,value\n"
            "a,1614879500,Tomcat02,app,timeout error\n"
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        (
            "timestamp,cmdb_id,parent_id,span_id,trace_id,duration\n"
            "1614879500000,Tomcat02,p,s,t,900\n"
        ),
        encoding="utf-8",
    )
    metric_path.write_text(
        (
            "timestamp,cmdb_id,kpi_name,value\n"
            "1614879500,Tomcat02,cpu,95\n"
        ),
        encoding="utf-8",
    )

    agent = create_executor_agent(llm_client=_DecisionStubLLM())  # type: ignore[arg-type]
    result = agent.execute_assess(
        buildspec=buildspec,
        repository_path=str(tmp_path),
        max_agents=10,
    )
    # 3 seed tasks + expansion tasks focused on discovered component(s).
    assert len(result.agents_instantiated) > 3
    assert any("component_focus=" in item.detail for item in result.task_results)


def test_executor_uses_llm_even_when_reasoning_disabled(tmp_path) -> None:
    class _CountingDecisionLLM:
        provider_name = "test"

        def __init__(self) -> None:
            self.call_count = 0

        def complete(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
            self.call_count += 1
            return (
                '{"summary":"LLM analyzed tool outputs.","findings":'
                '[{"kind":"observation","summary":"LLM-based observation.","severity":"low","evidence":["ok"]}]}'
            )

    buildspec = _buildspec_for_tmp(tmp_path)
    llm = _CountingDecisionLLM()
    agent = ExecutorAgent(
        llm_client=llm,  # type: ignore[arg-type]
        enable_reasoning=False,
    )
    result = agent.execute_assess(
        buildspec=buildspec,
        repository_path=str(tmp_path),
    )
    assert result.task_results
    assert llm.call_count >= len(result.task_results)
    assert all("decision_source=llm" in item.detail for item in result.task_results)

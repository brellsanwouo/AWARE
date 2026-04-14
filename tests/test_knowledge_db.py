"""SQLite knowledge store integration tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agents.executor_agent import ExecutorAgent
from aware_models.buildspec import BuildSpec
from runtime.knowledge_db import SQLiteKnowledgeStore


class _DecisionStubLLM:
    """Deterministic LLM test double for analyzer decisions."""

    provider_name = "test-decision"

    def complete(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        return '{"summary":"stub","findings":[]}'


def _seed_buildspec(tmp_path: Path) -> BuildSpec:
    date_dir = tmp_path / "2021_03_04"
    log_path = date_dir / "log" / "log_service.csv"
    trace_path = date_dir / "trace" / "trace_span.csv"
    metrics_path = date_dir / "metric" / "metric_container.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("timeout error retry\n", encoding="utf-8")
    trace_path.write_text("checkout -> payment timeout 980ms\n", encoding="utf-8")
    metrics_path.write_text("cpu=92 latency=500 error_rate=8\n", encoding="utf-8")
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


def test_executor_stores_subagent_outputs_in_sqlite(tmp_path: Path) -> None:
    db_path = (tmp_path / "assess.db").resolve()
    store = SQLiteKnowledgeStore.from_url(f"sqlite:///{db_path}")
    run_id = "run-db-test"
    store.start_run(run_id=run_id, source_path=str(tmp_path), query="q", db_url=f"sqlite:///{db_path}")

    buildspec = _seed_buildspec(tmp_path)
    agent = ExecutorAgent(llm_client=_DecisionStubLLM())  # type: ignore[arg-type]
    result = agent.execute_assess(
        buildspec=buildspec,
        repository_path=str(tmp_path),
        run_id=run_id,
        knowledge_store=store,
    )
    assert result.task_results
    assert result.findings

    with sqlite3.connect(str(db_path)) as conn:
        task_count = conn.execute("SELECT COUNT(*) FROM task_results WHERE run_id = ?", (run_id,)).fetchone()[0]
        finding_count = conn.execute("SELECT COUNT(*) FROM findings WHERE run_id = ?", (run_id,)).fetchone()[0]
        run_status = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert task_count >= 3
    assert finding_count >= 1
    assert run_status == "success"


def test_executor_stores_component_scoped_memory(tmp_path: Path) -> None:
    db_path = (tmp_path / "assess.db").resolve()
    store = SQLiteKnowledgeStore.from_url(f"sqlite:///{db_path}")
    run_id = "run-component-memory"
    store.start_run(run_id=run_id, source_path=str(tmp_path), query="q", db_url=f"sqlite:///{db_path}")

    buildspec = _seed_buildspec(tmp_path)
    # Overwrite seeded files with structured CSVs that expose component values in-window.
    Path(buildspec.absolute_log_file[0]).write_text(
        (
            "log_id,timestamp,cmdb_id,log_name,value\n"
            "a,1614879500,Tomcat02,timeout,payment timeout\n"
        ),
        encoding="utf-8",
    )
    Path(buildspec.absolute_trace_file[0]).write_text(
        (
            "timestamp,cmdb_id,parent_id,span_id,trace_id,duration\n"
            "1614879500000,Tomcat02,p,s,t,980\n"
        ),
        encoding="utf-8",
    )
    Path(buildspec.absolute_metrics_file[0]).write_text(
        (
            "timestamp,cmdb_id,kpi_name,value\n"
            "1614879500,Tomcat02,cpu,95\n"
        ),
        encoding="utf-8",
    )

    agent = ExecutorAgent(llm_client=_DecisionStubLLM())  # type: ignore[arg-type]
    result = agent.execute_assess(
        buildspec=buildspec,
        repository_path=str(tmp_path),
        run_id=run_id,
        knowledge_store=store,
    )
    assert result.findings

    with sqlite3.connect(str(db_path)) as conn:
        component_count = conn.execute(
            "SELECT COUNT(*) FROM component_memory WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
    assert component_count >= 1

    component_memory = store.search_component_memory(component="Tomcat02", limit=10)
    assert component_memory

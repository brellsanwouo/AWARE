"""ExecutorAgent with template-driven ADK sub-agent instantiation for Assess."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from agents.base import Agent
from llm.base import LLMClient, LLMError
from aware_models.buildspec import BuildSpec
from aware_models.executor import AgentTaskResult, AssessFinding, ExecutorRunResult
from runtime.knowledge_db import SQLiteKnowledgeStore
from templates.assess_templates import AgentTemplate, load_assess_templates
from tools import telemetry_tools

_DECISION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "assess_agent_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "findings"],
            "properties": {
                "summary": {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind", "summary", "severity", "evidence"],
                        "properties": {
                            "kind": {"type": "string", "enum": ["anomaly", "observation"]},
                            "summary": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}


class ExecutorAgentError(RuntimeError):
    """Raised when ExecutorAgent cannot complete assess execution."""


class ExecutorEvent(BaseModel):
    """One executor conversation event."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sender: str
    recipient: str
    phase: str
    content: str


class AnalyzerAgent(Agent):
    """Base class for specialized telemetry analyzers."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        *,
        template: AgentTemplate,
        llm_client: LLMClient | None,
        instance_name: str | None = None,
        knowledge_text: str = "",
        known_components: list[str] | None = None,
        known_reasons: list[str] | None = None,
        enable_reasoning: bool = True,
        enable_memory: bool = True,
    ) -> None:
        super().__init__(name=instance_name or template.agent_name, description=template.objective)
        self.template = template
        self.llm_client = llm_client
        self.knowledge_text = knowledge_text
        self.known_components = list(known_components or [])
        self.known_reasons = list(known_reasons or [])
        self.enable_reasoning = bool(enable_reasoning)
        self.enable_memory = bool(enable_memory)

    def analyze(
        self,
        file_path: Path,
        buildspec: BuildSpec,
        shared_memory_context: list[str] | None = None,
        component_focus: str | None = None,
    ) -> AgentTaskResult:
        """Run strict-window tools, then let LLM decide findings (fallback on tool heuristics)."""
        if not file_path.exists() or not file_path.is_file():
            return AgentTaskResult(
                agent_name=self.name,
                target_path=str(file_path),
                status="skipped",
                detail="Target file is missing.",
            )

        view = telemetry_tools.load_csv_window(
            file_path=file_path,
            start_ts=buildspec.failure_time_range_ts.start,
            end_ts=buildspec.failure_time_range_ts.end,
        )
        focus_note = ""
        if component_focus:
            focused = telemetry_tools.apply_component_focus(
                view,
                domain=self.template.domain,
                component=component_focus,
            )
            view = focused.view
            focus_note = (
                f" component_focus={component_focus}; "
                f"focus_columns={focused.matched_columns}; focused_rows={view.window_rows}."
            )
        detail = telemetry_tools.window_detail(
            view,
            start_ts=buildspec.failure_time_range_ts.start,
            end_ts=buildspec.failure_time_range_ts.end,
        )
        tool_context = telemetry_tools.build_llm_observation_context(
            view,
            domain=self.template.domain,
            start_ts=buildspec.failure_time_range_ts.start,
            end_ts=buildspec.failure_time_range_ts.end,
        )
        semantic_findings = _semantic_context_findings(
            domain=self.template.domain,
            tool_context=tool_context,
            source_path=str(file_path),
            agent_name=self.name,
            known_components=self.known_components,
        )
        findings: list[AssessFinding]
        decision_source = "llm"
        llm_summary = ""
        memory_context = list(shared_memory_context or []) if self.enable_memory else []
        if self.llm_client is None:
            raise RuntimeError(f"{self.name} requires an LLM client.")
        llm_findings, llm_summary = _decide_findings_with_llm(
            llm_client=self.llm_client,
            template=self.template,
            target_path=str(file_path),
            buildspec=buildspec,
            tool_context=tool_context,
            agent_name=self.name,
            knowledge_text=self.knowledge_text,
            known_components=self.known_components,
            known_reasons=self.known_reasons,
            shared_memory_context=memory_context,
            reasoning_enabled=self.enable_reasoning,
        )
        if llm_findings:
            findings = llm_findings
        else:
            findings = _fallback_findings_from_tools(self.template.domain, view, str(file_path), self.name)
            decision_source = "tools_fallback_after_llm"

        if semantic_findings:
            existing = {item.summary for item in findings}
            for item in semantic_findings:
                if item.summary not in existing:
                    findings.append(item)

        if not findings:
            findings = [
                AssessFinding(
                    agent=self.name,
                    kind="observation",
                    source=str(file_path),
                    summary="No significant signal detected for this telemetry source.",
                    evidence=view.lines[:3],
                    severity="low",
                )
            ]
        summary_suffix = f" decision_source={decision_source}."
        if llm_summary:
            summary_suffix += f" llm_summary={llm_summary[:220]}"
        if memory_context:
            summary_suffix += f" memory_context_items={len(memory_context)}."
        if focus_note:
            summary_suffix += focus_note
        return AgentTaskResult(
            agent_name=self.name,
            target_path=str(file_path),
            status="ok",
            findings=findings,
            detail=f"{detail}{summary_suffix}",
        )


class ExecutorAgent(Agent):
    """Orchestrate Assess execution using dynamic templates and ADK sub-agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        templates: list[AgentTemplate] | None = None,
        knowledge_file: str | None = None,
        enable_reasoning: bool = True,
        enable_memory: bool = True,
    ) -> None:
        super().__init__(
            name="ExecutorAgent",
            description="Execute Assess tasks from BuildSpec via dynamic ADK sub-agents.",
        )
        self.llm_client = llm_client
        self.templates = templates or load_assess_templates()
        self.enable_reasoning = bool(enable_reasoning)
        self.enable_memory = bool(enable_memory)
        self.knowledge_file = knowledge_file or os.getenv(
            "AWARE_EXECUTOR_KB_FILE",
            "knowledge/executor_rca_kb.md",
        )

    def execute_assess(
        self,
        *,
        buildspec: BuildSpec,
        repository_path: str,
        max_agents: int | None = None,
        run_id: str | None = None,
        knowledge_store: SQLiteKnowledgeStore | None = None,
        on_event: Callable[[ExecutorEvent], None] | None = None,
    ) -> ExecutorRunResult:
        repo = Path(repository_path)
        if not repo.is_absolute():
            repo = (Path.cwd() / repo).resolve()
        self._emit(
            on_event,
            phase="init",
            recipient="Runtime",
            content=(
                f"ExecutorAgent started. repository={repo}; "
                f"reasoning={'on' if self.enable_reasoning else 'off'}; "
                f"memory={'on' if self.enable_memory else 'off'}."
            ),
        )
        if not repo.exists():
            raise ExecutorAgentError(f"Repository path does not exist: {repo}")
        self._emit(
            on_event,
            phase="load_templates",
            recipient="Runtime",
            content=f"Loaded {len(self.templates)} templates from templates/assess_templates.py",
        )
        knowledge_source, knowledge_text, known_components, known_reasons = _load_executor_knowledge(
            self.knowledge_file
        )
        self._emit(
            on_event,
            phase="load_knowledge",
            recipient="Runtime",
            content=(
                f"Loaded executor knowledge from {knowledge_source} "
                f"(chars={len(knowledge_text)}; components={len(known_components)}; reasons={len(known_reasons)})."
            ),
        )

        cap = max(1, int(max_agents)) if max_agents is not None else None
        if cap is not None:
            self._emit(
                on_event,
                phase="config",
                recipient="Runtime",
                content=f"Max sub-agents cap set to {cap}.",
            )

        task_results: list[AgentTaskResult] = []
        findings: list[AssessFinding] = []
        shared_memory_context: list[str] = []
        agents_instantiated: list[str] = []

        # Progressive + expand scheduler:
        # 1) discover initial targets by template
        # 2) enqueue one seed task per template
        # 3) instantiate-run-terminate one agent at a time
        # 4) after each result, optionally enqueue expansion tasks on discovered components
        discovered_by_template: dict[str, tuple[AgentTemplate, list[Path]]] = {}
        template_order: list[str] = []
        total_discovered = 0
        for template in self.templates:
            targets = _discover_targets_for_template(buildspec, template)
            discovered_by_template[template.template_id] = (template, targets)
            template_order.append(template.template_id)
            total_discovered += len(targets)
            self._emit(
                on_event,
                phase="plan_targets",
                recipient="ExecutorAgent",
                content=f"Template {template.template_id} resolved {len(targets)} target(s).",
                sender=template.agent_name,
            )

        if cap is not None and total_discovered < cap:
            self._emit(
                on_event,
                phase="capacity_info",
                recipient="Runtime",
                content=(
                    f"max_agents={cap}, but only {total_discovered} initial target(s) were discovered "
                    "from BuildSpec + repository scan."
                ),
            )

        queue: list[tuple[AgentTemplate, Path, str | None, str]] = []
        remaining_by_template: dict[str, list[Path]] = {}
        queued_signatures: set[tuple[str, str, str]] = set()
        executed_signatures: set[tuple[str, str, str]] = set()

        def _task_signature(
            template_id: str,
            target_path: Path,
            component_focus: str | None,
        ) -> tuple[str, str, str]:
            return (template_id, str(target_path), (component_focus or "").strip().lower())

        def _enqueue_task(
            *,
            template: AgentTemplate,
            target_path: Path,
            component_focus: str | None,
            origin: str,
            instantiated_count_now: int,
        ) -> bool:
            if cap is not None and (instantiated_count_now + len(queue)) >= cap:
                return False
            signature = _task_signature(template.template_id, target_path, component_focus)
            if signature in queued_signatures or signature in executed_signatures:
                return False
            queue.append((template, target_path, component_focus, origin))
            queued_signatures.add(signature)
            return True

        for template_id in template_order:
            template, targets = discovered_by_template[template_id]
            remaining = list(targets)
            remaining_by_template[template_id] = remaining
            if remaining:
                first = remaining.pop(0)
                _enqueue_task(
                    template=template,
                    target_path=first,
                    component_focus=None,
                    origin="seed",
                    instantiated_count_now=0,
                )

        instantiated_count = 0
        cap_reached = False
        instance_counts: dict[str, int] = {}
        expansion_tasks_enqueued = 0
        while queue:
            if cap is not None and instantiated_count >= cap:
                cap_reached = True
                break
            template, target_path, component_focus, origin = queue.pop(0)
            signature = _task_signature(template.template_id, target_path, component_focus)
            queued_signatures.discard(signature)
            if signature in executed_signatures:
                continue
            current_count = instance_counts.get(template.agent_name, 0) + 1
            instance_counts[template.agent_name] = current_count
            instance_name = (
                template.agent_name if current_count == 1 else f"{template.agent_name}_{current_count}"
            )

            self._emit(
                on_event,
                phase="instantiate_agent",
                recipient="ExecutorAgent",
                content=(
                    f"Instantiated {instance_name} from template={template.template_id}; "
                    f"tools={', '.join(template.tools)}; target={target_path}; "
                    f"component_focus={component_focus or 'none'}; origin={origin}"
                ),
                sender=instance_name,
            )
            agent = AnalyzerAgent(
                template=template,
                llm_client=self.llm_client,
                instance_name=instance_name,
                knowledge_text=knowledge_text,
                known_components=known_components,
                known_reasons=known_reasons,
                enable_reasoning=self.enable_reasoning,
                enable_memory=self.enable_memory,
            )
            agents_instantiated.append(instance_name)
            instantiated_count += 1

            self._emit(
                on_event,
                phase="dispatch",
                recipient="ExecutorAgent",
                content=(
                    f"I will analyze {target_path} using tools "
                    f"[{', '.join(template.tools)}], then decide with LLM "
                    f"(reasoning_mode={'deep' if self.enable_reasoning else 'fast'}); "
                    f"component_focus={component_focus or 'none'}."
                ),
                sender=instance_name,
            )
            if self.enable_memory and shared_memory_context:
                self._emit(
                    on_event,
                    phase="read_shared_memory",
                    recipient=instance_name,
                    content=f"Loaded {len(shared_memory_context)} prior findings from shared memory.",
                )
            component_memory_context: list[str] = []
            if (
                self.enable_memory
                and knowledge_store is not None
                and component_focus
            ):
                component_memory_context = _format_component_memory_context(
                    knowledge_store.search_component_memory(component=component_focus, limit=12),
                    max_items=12,
                )
                if component_memory_context:
                    self._emit(
                        on_event,
                        phase="read_component_memory",
                        recipient=instance_name,
                        content=(
                            f"Loaded {len(component_memory_context)} memory item(s) for "
                            f"component={component_focus}."
                        ),
                    )
            analysis_memory_context = [
                *shared_memory_context,
                *component_memory_context,
            ]
            result = agent.analyze(
                target_path,
                buildspec,
                shared_memory_context=analysis_memory_context,
                component_focus=component_focus,
            )
            executed_signatures.add(signature)
            task_results.append(result)
            findings.extend(result.findings)
            if self.enable_memory and knowledge_store and run_id:
                knowledge_store.append_task_result(
                    run_id=run_id,
                    agent_name=result.agent_name,
                    target_path=result.target_path,
                    status=result.status,
                    detail=result.detail,
                )
                for finding in result.findings:
                    knowledge_store.append_finding(
                        run_id=run_id,
                        agent_name=finding.agent,
                        kind=finding.kind,
                        source=finding.source,
                        summary=finding.summary,
                        evidence=finding.evidence,
                        severity=finding.severity,
                    )
                stored_component_items = _store_component_memory_from_findings(
                    knowledge_store=knowledge_store,
                    run_id=run_id,
                    findings=result.findings,
                    known_components=known_components,
                    max_components=4,
                    max_findings_per_component=3,
                )
                if stored_component_items > 0:
                    self._emit(
                        on_event,
                        phase="store_component_memory",
                        recipient="ExecutorAgent",
                        content=(
                            f"Stored {stored_component_items} component-memory item(s) "
                            f"from {instance_name} findings."
                        ),
                        sender=instance_name,
                    )
            if self.enable_memory:
                shared_memory_context.extend(
                    [
                        f"{finding.agent}|{finding.kind}|{finding.summary}|severity={finding.severity}"
                        for finding in result.findings[:5]
                    ]
                )
            self._emit(
                on_event,
                phase="agent_result",
                recipient="ExecutorAgent",
                content=f"{instance_name} completed with status={result.status}; findings={len(result.findings)}",
                sender=instance_name,
            )
            for finding in result.findings[:3]:
                self._emit(
                    on_event,
                    phase="finding",
                    recipient="ExecutorAgent",
                    content=f"{finding.kind.upper()}: {finding.summary}",
                    sender=instance_name,
                )

            # Agent lifecycle is explicit: one run then terminate.
            self._emit(
                on_event,
                phase="terminate_agent",
                recipient="ExecutorAgent",
                content=f"Terminated {instance_name} after task completion.",
                sender=instance_name,
            )
            del agent

            # Progressive baseline: only enqueue next seed target of same template
            # after current target completion.
            remaining = remaining_by_template.get(template.template_id, [])
            if remaining:
                next_target = remaining.pop(0)
                _enqueue_task(
                    template=template,
                    target_path=next_target,
                    component_focus=None,
                    origin="seed",
                    instantiated_count_now=instantiated_count,
                )

            # Dynamic expansion: create new focused tasks from discovered components.
            component_candidates = _extract_component_candidates_from_findings(
                result.findings,
                known_components=known_components,
            )
            if component_candidates:
                self._emit(
                    on_event,
                    phase="expand_detect",
                    recipient="ExecutorAgent",
                    content=(
                        f"Discovered component candidate(s): {', '.join(component_candidates[:8])}"
                    ),
                    sender=instance_name,
                )
            enqueued_now = 0
            for component in component_candidates:
                for template_id in template_order:
                    exp_template, exp_targets = discovered_by_template[template_id]
                    for exp_target in exp_targets:
                        ok = _enqueue_task(
                            template=exp_template,
                            target_path=exp_target,
                            component_focus=component,
                            origin=f"expand:{instance_name}",
                            instantiated_count_now=instantiated_count,
                        )
                        if ok:
                            enqueued_now += 1
                            expansion_tasks_enqueued += 1
            if enqueued_now > 0:
                self._emit(
                    on_event,
                    phase="expand_enqueue",
                    recipient="ExecutorAgent",
                    content=f"Enqueued {enqueued_now} expansion task(s) from {instance_name}.",
                    sender=instance_name,
                )

        if cap_reached:
            self._emit(
                on_event,
                phase="cap_reached",
                recipient="Runtime",
                content=f"Reached max_agents={cap}; remaining discovered targets were skipped.",
            )

        preliminary_causes = _derive_preliminary_causes(findings)
        confidence = _estimate_confidence(findings, preliminary_causes)
        summary = (
            f"Executor completed Assess: {len(findings)} findings, "
            f"{len([f for f in findings if f.kind == 'anomaly'])} anomalies, "
            f"{len(preliminary_causes)} preliminary causes, "
            f"{expansion_tasks_enqueued} expansion task(s) enqueued."
        )
        self._emit(
            on_event,
            phase="summary",
            recipient="Runtime",
            content=summary,
        )
        if self.enable_memory and knowledge_store and run_id:
            knowledge_store.finish_run(
                run_id=run_id,
                status="success",
                summary=summary,
                confidence=confidence,
                preliminary_causes=preliminary_causes,
            )

        return ExecutorRunResult(
            buildspec=buildspec,
            agents_instantiated=agents_instantiated,
            task_results=task_results,
            findings=findings,
            preliminary_causes=preliminary_causes,
            confidence=confidence,
            summary=summary,
        )

    def _emit(
        self,
        callback: Callable[[ExecutorEvent], None] | None,
        *,
        phase: str,
        recipient: str,
        content: str,
        sender: str | None = None,
    ) -> None:
        if callback is None:
            return
        callback(
            ExecutorEvent(
                sender=sender or self.name,
                recipient=recipient,
                phase=phase,
                content=content,
            )
        )


def _decide_findings_with_llm(
    *,
    llm_client: LLMClient,
    template: AgentTemplate,
    target_path: str,
    buildspec: BuildSpec,
    tool_context: dict[str, Any],
    agent_name: str,
    knowledge_text: str,
    known_components: list[str],
    known_reasons: list[str],
    shared_memory_context: list[str],
    reasoning_enabled: bool,
) -> tuple[list[AssessFinding], str]:
    system_prompt = (
        f"You are {agent_name}.\n"
        f"Role: {template.role}\n"
        f"Objective: {template.objective}\n"
        "You are an RCA assessor. Decide findings strictly from provided tool outputs.\n"
        "Use semantic_columns and semantic_summary as primary evidence over raw sample_lines.\n"
        "Infer component/reason cues from the detected column roles and in-window stats.\n"
        "Root cause components must come from telemetry values (cmdb_id/tc/kpi context), never file names.\n"
        "If component candidates exist in semantic_summary.top_component_values, cite them explicitly in findings.\n"
        "Use the provided executor_knowledge for schema understanding and allowed component/reason hints.\n"
        "Do not invent evidence outside tool_context.\n"
        "Use shared_memory_context as prior agent observations from the same run.\n"
        f"Reasoning mode: {'deep' if reasoning_enabled else 'fast'}.\n"
        "Return JSON only with keys: summary, findings.\n"
        "Each finding: kind (anomaly|observation), summary, severity (low|medium|high), evidence (list[str])."
        "\nexecutor_knowledge:\n"
        f"{knowledge_text}"
    )
    user_prompt = (
        "assess_task_context:\n"
        f"- template_id={template.template_id}\n"
        f"- target_path={target_path}\n"
        f"- failure_window_start={buildspec.failure_time_range_ts.start}\n"
        f"- failure_window_end={buildspec.failure_time_range_ts.end}\n"
        f"- known_components={json.dumps(known_components, ensure_ascii=False)}\n"
        f"- known_reasons={json.dumps(known_reasons, ensure_ascii=False)}\n"
        f"- shared_memory_context={json.dumps(shared_memory_context[:25], ensure_ascii=False)}\n"
        "tool_context_json:\n"
        f"{json.dumps(tool_context, ensure_ascii=False)}"
    )
    try:
        response = llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=_DECISION_RESPONSE_FORMAT,
        )
    except (LLMError, TypeError):
        return [], ""
    payload = _extract_json_payload(response)
    if payload is None:
        return [], ""
    findings_payload = payload.get("findings")
    if not isinstance(findings_payload, list):
        return [], str(payload.get("summary", ""))
    findings: list[AssessFinding] = []
    for item in findings_payload:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "observation")).strip().lower()
        severity = str(item.get("severity", "medium")).strip().lower()
        summary = str(item.get("summary", "")).strip()
        evidence = item.get("evidence", [])
        if kind not in {"anomaly", "observation"}:
            kind = "observation"
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        if not isinstance(evidence, list):
            evidence = []
        evidence_str = [str(entry)[:300] for entry in evidence if str(entry).strip()]
        if not summary:
            continue
        findings.append(
            AssessFinding(
                agent=agent_name,
                kind=kind,  # type: ignore[arg-type]
                source=target_path,
                summary=summary,
                evidence=evidence_str[:5],
                severity=severity,  # type: ignore[arg-type]
            )
        )
    return findings, str(payload.get("summary", ""))


def _fallback_findings_from_tools(
    domain: str,
    view: telemetry_tools.CsvWindowView,
    source_path: str,
    agent_name: str,
) -> list[AssessFinding]:
    if view.timestamp_field is not None and view.window_rows == 0:
        return [
            AssessFinding(
                agent=agent_name,
                kind="observation",
                source=source_path,
                summary=(
                    f"No {domain} rows found inside failure_time_range_ts; "
                    "strict window filtering yielded zero rows."
                ),
                evidence=[f"timestamp_field={view.timestamp_field}"],
                severity="low",
            )
        ]

    lines = view.lines
    text = "\n".join(lines).lower()
    if domain == "logs":
        timeout_count = telemetry_tools.count_matches(text, r"timeout|timed out|deadline exceeded")
        error_count = telemetry_tools.count_matches(text, r"\berror\b|exception|failed|failure")
        retry_count = telemetry_tools.count_matches(text, r"\bretry\b")
        findings: list[AssessFinding] = []
        if timeout_count > 0:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="anomaly",
                    source=source_path,
                    summary=f"Detected {timeout_count} timeout indicators in logs.",
                    evidence=telemetry_tools.sample_matching_lines(lines, r"timeout|timed out|deadline exceeded"),
                    severity="high" if timeout_count >= 3 else "medium",
                )
            )
        if error_count > 0:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="anomaly",
                    source=source_path,
                    summary=f"Detected {error_count} error/exception indicators in logs.",
                    evidence=telemetry_tools.sample_matching_lines(lines, r"\berror\b|exception|failed|failure"),
                    severity="high" if error_count >= 5 else "medium",
                )
            )
        if retry_count > 0:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="observation",
                    source=source_path,
                    summary=f"Detected {retry_count} retry indicators in logs.",
                    evidence=telemetry_tools.sample_matching_lines(lines, r"\bretry\b"),
                    severity="low",
                )
            )
        if not findings:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="observation",
                    source=source_path,
                    summary="No explicit timeout/error patterns found in logs.",
                    evidence=lines[:3],
                    severity="low",
                )
            )
        return findings

    if domain == "trace":
        timeout_count = telemetry_tools.count_matches(text, r"timeout|timed out|deadline exceeded")
        retry_count = telemetry_tools.count_matches(text, r"\bretry\b")
        durations_ms = telemetry_tools.extract_trace_durations(view.rows)
        if not durations_ms:
            durations_ms = [int(item) for item in re.findall(r"(\d+)\s*ms", text)]
        max_duration = max(durations_ms) if durations_ms else 0
        findings = []
        if timeout_count > 0:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="anomaly",
                    source=source_path,
                    summary=f"Trace contains {timeout_count} timeout indicators.",
                    evidence=telemetry_tools.sample_matching_lines(lines, r"timeout|timed out|deadline exceeded"),
                    severity="high" if timeout_count >= 2 else "medium",
                )
            )
        if max_duration >= 500:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="anomaly",
                    source=source_path,
                    summary=f"Trace shows slow span(s), max observed duration={max_duration}ms.",
                    evidence=telemetry_tools.sample_matching_lines(lines, r"\d+\s*ms"),
                    severity="high" if max_duration >= 1200 else "medium",
                )
            )
        if retry_count > 0:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="observation",
                    source=source_path,
                    summary=f"Trace contains {retry_count} retry indicators.",
                    evidence=telemetry_tools.sample_matching_lines(lines, r"\bretry\b"),
                    severity="low",
                )
            )
        if not findings:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="observation",
                    source=source_path,
                    summary="No obvious timeout/latency trace anomalies detected.",
                    evidence=lines[:3],
                    severity="low",
                )
            )
        return findings

    max_cpu = telemetry_tools.max_numeric_column(view.rows, ("cpu", "cpu_usage", "cpu_percent", "cpu%"))
    max_latency = telemetry_tools.max_numeric_column(
        view.rows,
        ("latency", "p95", "duration_ms", "response_time", "mrt"),
    )
    max_error_rate = telemetry_tools.max_numeric_column(view.rows, ("error_rate", "errors_rate", "5xx_rate", "5xx"))
    if max_error_rate is None:
        min_sr = telemetry_tools.min_numeric_column(view.rows, ("sr", "success_rate"))
        if min_sr is not None:
            max_error_rate = max(0.0, 100.0 - min_sr)
    if max_cpu is None:
        max_cpu = telemetry_tools.max_value_after_keywords(text, ("cpu", "cpu_usage", "cpu%"))
    if max_latency is None:
        max_latency = telemetry_tools.max_value_after_keywords(text, ("latency", "p95", "duration_ms", "response_time"))
    if max_error_rate is None:
        max_error_rate = telemetry_tools.max_value_after_keywords(text, ("error_rate", "errors_rate", "5xx_rate", "5xx"))

    findings = []
    if max_cpu is not None and max_cpu >= 85:
        findings.append(
            AssessFinding(
                agent=agent_name,
                kind="anomaly",
                source=source_path,
                summary=f"CPU saturation signal detected (max={max_cpu:.1f}%).",
                evidence=telemetry_tools.sample_matching_lines(lines, r"cpu"),
                severity="high" if max_cpu >= 92 else "medium",
            )
        )
    if max_latency is not None and max_latency >= 300:
        findings.append(
            AssessFinding(
                agent=agent_name,
                kind="anomaly",
                source=source_path,
                summary=f"Latency spike detected (max={max_latency:.1f}ms).",
                evidence=telemetry_tools.sample_matching_lines(lines, r"latency|p95|duration|response_time|mrt"),
                severity="high" if max_latency >= 800 else "medium",
            )
        )
    if max_error_rate is not None and max_error_rate >= 5:
        findings.append(
            AssessFinding(
                agent=agent_name,
                kind="anomaly",
                source=source_path,
                summary=f"Error-rate spike detected (max={max_error_rate:.1f}).",
                evidence=telemetry_tools.sample_matching_lines(lines, r"error|5xx|sr"),
                severity="high" if max_error_rate >= 10 else "medium",
            )
        )
    if not findings:
        findings.append(
            AssessFinding(
                agent=agent_name,
                kind="observation",
                source=source_path,
                summary="No obvious CPU/latency/error-rate spikes detected in metrics.",
                evidence=lines[:3],
                severity="low",
            )
        )
    return findings


def _extract_json_payload(text: str) -> dict[str, object] | None:
    content = text.strip()
    if not content:
        return None
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return None
    snippet = content[start : end + 1]
    try:
        payload = json.loads(snippet)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _semantic_context_findings(
    *,
    domain: str,
    tool_context: dict[str, Any],
    source_path: str,
    agent_name: str,
    known_components: list[str],
) -> list[AssessFinding]:
    """Convert structured schema/window context into explicit findings."""
    semantic_columns = tool_context.get("semantic_columns", {})
    semantic_summary = tool_context.get("semantic_summary", {})
    if not isinstance(semantic_columns, dict) or not isinstance(semantic_summary, dict):
        return []

    findings: list[AssessFinding] = []

    component_columns = semantic_columns.get("component_columns", [])
    reason_columns = semantic_columns.get("reason_columns", [])
    duration_columns = semantic_columns.get("duration_columns", [])
    top_components = semantic_summary.get("top_component_values", [])
    top_reasons = semantic_summary.get("top_reason_values", [])
    numeric_ranges = semantic_summary.get("numeric_ranges", {})
    observed_bounds = semantic_summary.get("window_observed_time_bounds", {})

    if component_columns or reason_columns or duration_columns:
        findings.append(
            AssessFinding(
                agent=agent_name,
                kind="observation",
                source=source_path,
                summary=(
                    "CSV header semantic mapping extracted "
                    f"(component={component_columns}, reason={reason_columns}, duration={duration_columns})."
                ),
                evidence=[
                    f"domain={domain}",
                    f"fieldnames={tool_context.get('fieldnames', [])}",
                ],
                severity="low",
            )
        )

    if isinstance(top_components, list) and top_components:
        values: list[str] = []
        for item in top_components[:8]:
            if not isinstance(item, dict):
                continue
            raw_value = str(item.get("value", "")).strip()
            raw_column = str(item.get("column", "")).strip()
            raw_count = item.get("count", "")
            if not raw_value:
                continue
            if known_components and raw_value not in known_components:
                # Keep unknown values too, but annotate whether it belongs to known component set.
                values.append(f"{raw_value} (column={raw_column}, count={raw_count}, known_component=no)")
            else:
                values.append(f"{raw_value} (column={raw_column}, count={raw_count}, known_component=yes)")
        if values:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="observation",
                    source=source_path,
                    summary="Top component identifiers observed in-window.",
                    evidence=values[:6],
                    severity="low",
                )
            )

    if isinstance(top_reasons, list) and top_reasons:
        reason_values: list[str] = []
        for item in top_reasons[:6]:
            if not isinstance(item, dict):
                continue
            reason_values.append(
                f"{item.get('value')} (column={item.get('column')}, count={item.get('count')})"
            )
        if reason_values:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="observation",
                    source=source_path,
                    summary="Top reason/message-like values observed in-window.",
                    evidence=reason_values[:6],
                    severity="low",
                )
            )

    if isinstance(numeric_ranges, dict) and numeric_ranges:
        compact_ranges = []
        for key, stats in list(numeric_ranges.items())[:6]:
            if not isinstance(stats, dict):
                continue
            compact_ranges.append(
                f"{key}: min={stats.get('min')}, max={stats.get('max')}, count={stats.get('count')}"
            )
        if compact_ranges:
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="observation",
                    source=source_path,
                    summary="Numeric column ranges computed over in-window rows.",
                    evidence=compact_ranges,
                    severity="low",
                )
            )

    if isinstance(observed_bounds, dict):
        min_ts = observed_bounds.get("min_ts")
        max_ts = observed_bounds.get("max_ts")
        if isinstance(min_ts, int) and isinstance(max_ts, int):
            findings.append(
                AssessFinding(
                    agent=agent_name,
                    kind="observation",
                    source=source_path,
                    summary=f"Observed in-window timestamp bounds: min_ts={min_ts}, max_ts={max_ts}.",
                    evidence=[
                        f"window_observed_time_bounds.min_ts={min_ts}",
                        f"window_observed_time_bounds.max_ts={max_ts}",
                    ],
                    severity="low",
                )
            )

    return findings


def _derive_preliminary_causes(findings: list[AssessFinding]) -> list[str]:
    causes: list[str] = []
    merged = " ".join(item.summary.lower() for item in findings)
    if "timeout" in merged:
        causes.append("Downstream timeout or dependency latency likely contributed to the incident.")
    if "cpu saturation" in merged or "cpu" in merged:
        causes.append("Resource saturation (CPU pressure) likely contributed to degraded performance.")
    if "error/exception" in merged or "exception" in merged:
        causes.append("Application-level errors/exceptions likely contributed to failures.")
    if "latency spike" in merged:
        causes.append("System latency spike observed around the failure window.")
    if not causes:
        causes.append("Insufficient evidence for strong preliminary causes; gather more telemetry.")
    return _dedupe_preserve_order(causes)


def _load_executor_knowledge(knowledge_file: str) -> tuple[str, str, list[str], list[str]]:
    """Load Executor knowledge text plus parsed known components/reasons."""
    raw = (knowledge_file or "").strip() or "knowledge/executor_rca_kb.md"
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    text = _EXECUTOR_KB_FALLBACK
    if candidate.exists() and candidate.is_file():
        try:
            loaded = candidate.read_text(encoding="utf-8").strip()
            if loaded:
                text = loaded
        except Exception:
            pass
    components = _parse_markdown_list_section(text, "POSSIBLE ROOT CAUSE COMPONENTS")
    reasons = _parse_markdown_list_section(text, "POSSIBLE ROOT CAUSE REASONS")
    return str(candidate), text, components, reasons


def _parse_markdown_list_section(text: str, section_name: str) -> list[str]:
    marker = section_name.strip().lower().rstrip(":")
    lines = text.splitlines()
    in_section = False
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if lower.startswith("## "):
            current = lower.removeprefix("## ").strip().rstrip(":")
            in_section = current == marker
            continue
        if not in_section:
            continue
        if lower.startswith("## "):
            break
        if line.startswith("- "):
            value = line[2:].strip()
            if value:
                out.append(value)
    return out


def _estimate_confidence(findings: list[AssessFinding], causes: list[str]) -> float:
    anomalies = [item for item in findings if item.kind == "anomaly"]
    high = [item for item in anomalies if item.severity == "high"]
    distinct_agents = len({item.agent for item in findings})
    score = 0.25
    score += min(0.35, 0.1 * len(anomalies))
    score += min(0.2, 0.1 * len(high))
    score += min(0.15, 0.05 * distinct_agents)
    if len(causes) > 1:
        score += 0.05
    return round(min(score, 0.95), 2)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_component_candidates_from_findings(
    findings: list[AssessFinding],
    *,
    known_components: list[str],
    max_items: int = 6,
) -> list[str]:
    """Extract component candidates from findings for dynamic expansion."""
    candidates: list[str] = []
    known = [item.strip() for item in known_components if item.strip()]
    known_lut = {item.lower(): item for item in known}
    stopwords = {"unknown", "none", "n/a", "na", "null", "component", "service"}
    component_column_tokens = {"cmdb_id", "component", "service", "host", "node", "instance", "tc"}

    def add(value: str) -> None:
        token = value.strip().strip(",;:()[]{}")
        if not token:
            return
        low = token.lower()
        if low in stopwords:
            return
        canonical = known_lut.get(low, token)
        candidates.append(canonical)

    # Prioritize exact known-component matches found in text.
    if known:
        for finding in findings:
            all_text = "\n".join([finding.summary, *finding.evidence[:10]])
            for comp in known:
                if re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(comp)}(?![A-Za-z0-9_])",
                    all_text,
                    flags=re.IGNORECASE,
                ):
                    add(comp)

    for finding in findings:
        texts = [finding.summary, *finding.evidence[:10]]
        for text in texts:
            if not text:
                continue
            # Semantic evidence format: "Tomcat02 (column=cmdb_id, count=...)"
            m_sem = re.match(r"\s*([A-Za-z0-9_.:-]+)\s*\(column=([A-Za-z0-9_.:-]+)", text)
            if m_sem:
                value = m_sem.group(1)
                column = m_sem.group(2).strip().lower()
                if any(token == column or token in column for token in component_column_tokens):
                    add(value)

            # Key/value patterns: "cmdb_id=Tomcat02", "tc:Tomcat02", "component=..."
            # Ignore "known_component=yes" style metadata keys.
            for m in re.finditer(
                r"(?:cmdb_id|(?<!known_)component|service|host|node|instance|tc)\s*[:=]\s*([A-Za-z0-9_.:-]+)",
                text,
                flags=re.IGNORECASE,
            ):
                add(m.group(1))

    return _dedupe_preserve_order(candidates)[:max_items]


def _finding_mentions_component(finding: AssessFinding, component: str) -> bool:
    """Check whether finding summary/evidence explicitly mentions the component token."""
    token = component.strip()
    if not token:
        return False
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
    merged = "\n".join([finding.summary, *finding.evidence[:10]])
    return re.search(pattern, merged, flags=re.IGNORECASE) is not None


def _store_component_memory_from_findings(
    *,
    knowledge_store: SQLiteKnowledgeStore,
    run_id: str,
    findings: list[AssessFinding],
    known_components: list[str],
    max_components: int,
    max_findings_per_component: int,
) -> int:
    """Persist component-scoped memory entries derived from findings."""
    candidates = _extract_component_candidates_from_findings(
        findings,
        known_components=known_components,
        max_items=max_components,
    )
    stored = 0
    seen_keys: set[tuple[str, str, str]] = set()
    for component in candidates:
        matching = [item for item in findings if _finding_mentions_component(item, component)]
        if not matching:
            # Keep at least one weak memory entry for the detected component.
            matching = findings[:1]
        for finding in matching[: max(1, int(max_findings_per_component))]:
            dedupe_key = (component.lower(), finding.source, finding.summary)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            knowledge_store.append_component_memory(
                component=component,
                run_id=run_id,
                agent_name=finding.agent,
                source=finding.source,
                summary=finding.summary,
                evidence=finding.evidence,
                severity=finding.severity,
            )
            stored += 1
    return stored


def _format_component_memory_context(
    entries: list[dict[str, object]],
    *,
    max_items: int,
) -> list[str]:
    """Convert DB component-memory rows into compact shared-memory lines for LLM prompts."""
    out: list[str] = []
    for item in entries[: max(1, int(max_items))]:
        component = str(item.get("component", "")).strip()
        agent = str(item.get("agent_name", "")).strip()
        summary = str(item.get("summary", "")).strip()
        severity = str(item.get("severity", "")).strip()
        source = str(item.get("source", "")).strip()
        evidence = item.get("evidence", [])
        evidence_preview = ""
        if isinstance(evidence, list) and evidence:
            evidence_preview = f" evidence={str(evidence[0])[:120]}"
        line = (
            f"component_memory|component={component}|agent={agent}|severity={severity}|"
            f"source={source}|summary={summary[:180]}{evidence_preview}"
        )
        out.append(line)
    return out


def _discover_targets_for_template(buildspec: BuildSpec, template: AgentTemplate) -> list[Path]:
    """Discover one or more targets for a template within the incident date directory.

    The primary BuildSpec target is always included first, then additional files of the same
    domain are added when found. This enables dynamic repeated instantiation per domain.
    """
    raw_primary = getattr(buildspec, template.target_field)
    primary_values: list[str]
    if isinstance(raw_primary, list):
        primary_values = [str(item) for item in raw_primary if str(item).strip()]
    else:
        primary_values = [str(raw_primary)]
    date_dir = Path(buildspec.filename_date_directory)
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_path(path: Path) -> None:
        key = str(path.resolve() if path.exists() else path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    for primary in primary_values:
        add_path(Path(primary))

    if date_dir.exists() and date_dir.is_dir():
        keywords = _template_keywords(template)
        allowed_suffixes = {".csv", ".log", ".txt", ".json", ".tsv"}
        extra: list[Path] = []
        for path in date_dir.rglob("*"):
            if not path.is_file():
                continue
            lowered = path.as_posix().lower()
            if path.suffix and path.suffix.lower() not in allowed_suffixes:
                continue
            if any(token in lowered for token in keywords):
                extra.append(path.resolve())
        for path in sorted(extra, key=lambda p: str(p)):
            add_path(path)

    return candidates


def _template_keywords(template: AgentTemplate) -> tuple[str, ...]:
    if template.domain == "logs":
        return ("log", "logs")
    if template.domain == "trace":
        return ("trace", "span")
    if template.domain == "metrics":
        return ("metric", "metrics", "cpu", "latency")
    return tuple()


_EXECUTOR_KB_FALLBACK = """
## POSSIBLE ROOT CAUSE REASONS:
- high CPU usage
- high memory usage
- network latency
- network packet loss
- high disk I/O read usage
- high disk space usage
- high JVM CPU load
- JVM Out of Memory (OOM) Heap

## POSSIBLE ROOT CAUSE COMPONENTS:
- apache01
- apache02
- Tomcat01
- Tomcat02
- Tomcat03
- Tomcat04
- MG01
- MG02
- IG01
- IG02
- Mysql01
- Mysql02
- Redis01
- Redis02

## DATA SCHEMA
- metric_app.csv: timestamp,rr,sr,cnt,mrt,tc
- metric_container.csv: timestamp,cmdb_id,kpi_name,value
- trace_span.csv: timestamp,cmdb_id,parent_id,span_id,trace_id,duration
- log_service.csv: log_id,timestamp,cmdb_id,log_name,value

## CLARIFICATION
- Metric timestamps are seconds.
- Trace timestamps are milliseconds.
- Log timestamps are seconds.
- Service/component identity is usually found in cmdb_id or tc columns.
""".strip()

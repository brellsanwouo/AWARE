"""Run artifact persistence helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def persist_run_artifacts(
    *,
    run_id: str,
    query: str,
    source_path: str,
    db_url: str | None,
    llm_provider: str,
    llm_model: str | None,
    status: str,
    events: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
    error_message: str | None = None,
    output_root: str | Path = "output",
) -> dict[str, str]:
    """Persist one run as JSON + TXT files under output/json and output/txt."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_run_id = run_id.replace(":", "_").replace("/", "_")

    root = Path(output_root)
    json_dir = root / "json"
    txt_dir = root / "txt"
    json_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{timestamp}_{safe_run_id}"
    json_path = json_dir / f"{base_name}.json"
    txt_path = txt_dir / f"{base_name}.txt"

    payload = {
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "status": status,
        "request": {
            "query": query,
            "source_path": source_path,
            "db_url": db_url,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        },
        "result": result_payload,
        "error": error_message,
        "events": events,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = [
        f"run_id: {run_id}",
        f"timestamp_utc: {now.isoformat()}",
        f"status: {status}",
        f"source_path: {source_path}",
        f"db_url: {db_url or 'none'}",
        f"llm_provider: {llm_provider}",
        f"llm_model: {llm_model or 'default'}",
        "",
        "events:",
    ]
    for event in events:
        ts = str(event.get("timestamp", ""))
        sender = str(event.get("sender", ""))
        recipient = str(event.get("recipient", ""))
        phase = str(event.get("phase", ""))
        content = str(event.get("content", "")).replace("\n", "\\n")
        lines.append(f"[{ts}] {sender} -> {recipient} | {phase} | {content}")

    lines.append("")
    if error_message:
        lines.append(f"error: {error_message}")
        lines.append("")

    lines.append("result_json:")
    lines.append(json.dumps(result_payload or {}, ensure_ascii=False, indent=2))

    assessment = (result_payload or {}).get("assessment_output") if isinstance(result_payload, dict) else None
    if isinstance(assessment, dict):
        lines.append("")
        lines.append("explicit_assess_output:")
        scope = assessment.get("buildspec_resolution_scope", {})
        if isinstance(scope, dict):
            lines.append("buildspec_resolution_scope:")
            lines.append(f"- task_type: {scope.get('task_type', 'n/a')}")
            lines.append(f"- requested_fields: {scope.get('requested_fields', [])}")
            lines.append(f"- uncertainty_unknown_fields: {scope.get('uncertainty_unknown_fields', [])}")
            mismatch = scope.get("scope_mismatch", {})
            if isinstance(mismatch, dict):
                lines.append(
                    "- scope_mismatch: "
                    f"unknown_not_in_task_scope={mismatch.get('unknown_not_in_task_scope', [])}, "
                    f"task_scope_not_unknown={mismatch.get('task_scope_not_unknown', [])}"
                )

        lines.append("findings:")
        findings = assessment.get("findings", [])
        if isinstance(findings, list) and findings:
            for idx, item in enumerate(findings, start=1):
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- {idx}. [{item.get('kind')}/{item.get('severity')}] "
                    f"{item.get('agent')}: {item.get('summary')}"
                )
        else:
            lines.append("- 1. No findings.")

        lines.append("anomalies:")
        anomalies = assessment.get("anomalies", [])
        if isinstance(anomalies, list) and anomalies:
            for idx, item in enumerate(anomalies, start=1):
                if not isinstance(item, dict):
                    continue
                lines.append(f"- {idx}. {item.get('agent')}: {item.get('summary')}")
        else:
            lines.append("- 1. No anomalies.")

        lines.append("preliminary_causes:")
        causes = assessment.get("preliminary_causes", [])
        if isinstance(causes, list) and causes:
            for idx, cause in enumerate(causes, start=1):
                lines.append(f"- {idx}. {cause}")
        else:
            lines.append("- 1. No preliminary causes.")

        synthesis = assessment.get("root_cause_synthesis", {})
        if isinstance(synthesis, dict):
            lines.append("root_cause_synthesis:")
            lines.append(json.dumps(synthesis, ensure_ascii=False, indent=2))

        final_reporting = assessment.get("final_reporting", {})
        if isinstance(final_reporting, dict):
            lines.append("final_reporting:")
            if final_reporting:
                for key, value in final_reporting.items():
                    lines.append(f"- {key}: {value}")
            else:
                lines.append("- none")
        evidence = assessment.get("resolution_evidence", {})
        if isinstance(evidence, dict):
            lines.append("resolution_evidence:")
            lines.append(f"- root_cause_time: {evidence.get('root_cause_time', [])}")
            lines.append(f"- root_cause_component: {evidence.get('root_cause_component', [])}")
            lines.append(f"- root_cause_reason: {evidence.get('root_cause_reason', [])}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "json_path": str(json_path.resolve()),
        "txt_path": str(txt_path.resolve()),
    }

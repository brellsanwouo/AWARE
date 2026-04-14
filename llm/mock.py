"""Deterministic mock LLM for local development."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from llm.base import LLMClient
from tools import time_tools


class MockLLMClient(LLMClient):
    """Mock provider that can return invalid payloads before a valid one."""

    provider_name = "mock"

    def __init__(self, invalid_attempts: int = 0) -> None:
        self.invalid_attempts = max(0, int(invalid_attempts))
        self.call_count = 0

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self.call_count += 1
        if self.call_count <= self.invalid_attempts:
            return "INVALID_JSON_RESPONSE"

        query = _extract_line_value(user_prompt, "user_query") or ""
        repo = _extract_line_value(user_prompt, "repository_path") or str(Path.cwd())
        repo_path = Path(repo)
        if not repo_path.is_absolute():
            repo_path = (Path.cwd() / repo_path).resolve()

        date_value = _extract_date(query) or "2021-03-04"
        start, end = _extract_time_range(query)
        start_ts, end_ts = _to_unix_range(date_value, start, end)
        filename_date = date_value.replace("-", "_")
        date_dir = _ensure_date_dir(repo_path, filename_date)

        payload = {
            "task_type": "task_7",
            "date": date_value,
            "filename_date": filename_date,
            "failure_time_range": {"start": start, "end": end},
            "failure_time_range_ts": {"start": start_ts, "end": end_ts},
            "failures_detected": 1,
            "uncertainty": {
                "root_cause_time": "unknown",
                "root_cause_component": "unknown",
                "root_cause_reason": "unknown",
            },
            "objective": "Identify the root cause component, the exact root cause datetime and reason for the failure",
            "filename_date_directory": str(date_dir),
            "absolute_log_file": [str(date_dir / "log" / "log_service.csv")],
            "absolute_trace_file": [str(date_dir / "trace" / "trace_span.csv")],
            "absolute_metrics_file": [str(date_dir / "metric" / "metric_container.csv")],
        }
        return json.dumps(payload, ensure_ascii=False)


def _extract_line_value(text: str, key: str) -> str | None:
    needle = f"{key}="
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith(needle):
            return clean[len(needle) :].strip()
    return None


def _extract_date(text: str) -> str | None:
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    return match.group(1) if match else None


def _extract_time_range(text: str) -> tuple[str, str]:
    matches = re.findall(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", text)
    if len(matches) >= 2:
        return _norm_hms(matches[0]), _norm_hms(matches[1])
    if len(matches) == 1:
        start = _norm_hms(matches[0])
        end = (datetime.strptime(start, "%H:%M:%S") + timedelta(minutes=30)).strftime("%H:%M:%S")
        return start, end
    return "18:30:00", "19:00:00"


def _norm_hms(value: str) -> str:
    if len(value.split(":")) == 2:
        value = value + ":00"
    return datetime.strptime(value, "%H:%M:%S").strftime("%H:%M:%S")


def _to_unix_range(date_value: str, start: str, end: str) -> tuple[int, int]:
    return time_tools.time_range_to_unix_utc8(
        date_value=date_value,
        start_time=start,
        end_time=end,
    )


def _ensure_date_dir(repo_path: Path, filename_date: str) -> Path:
    if filename_date in str(repo_path):
        return repo_path
    return repo_path / filename_date

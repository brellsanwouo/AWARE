"""Shared file/data manipulation tools for Assess telemetry analysis."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CsvWindowView:
    """CSV rows constrained to the failure time window."""

    fieldnames: list[str]
    rows: list[dict[str, str]]
    lines: list[str]
    total_rows: int
    window_rows: int
    timestamp_field: str | None


@dataclass
class ComponentFocusResult:
    """Result of applying component focus over a window view."""

    view: CsvWindowView
    component: str
    matched_columns: list[str]


def load_csv_window(file_path: Path, start_ts: int, end_ts: int) -> CsvWindowView:
    """Read CSV and keep only rows inside [start_ts, end_ts], strictly by timestamp."""
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    total_rows = 0
    window_rows = 0

    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        timestamp_field = find_timestamp_field(fieldnames)
        for row in reader:
            total_rows += 1
            if timestamp_field is None:
                # Strict policy: if timestamp column does not exist, no row is considered in-window.
                continue
            row_ts = parse_timestamp_seconds(row.get(timestamp_field, ""))
            if row_ts is None or row_ts < start_ts or row_ts > end_ts:
                continue
            window_rows += 1
            rows.append({k: (v or "") for k, v in row.items()})

    if not fieldnames:
        return CsvWindowView(
            fieldnames=[],
            rows=[],
            lines=[],
            total_rows=0,
            window_rows=0,
            timestamp_field=None,
        )

    lines = [",".join(fieldnames)]
    for row in rows:
        lines.append(",".join((row.get(col, "") or "").replace("\n", " ").strip() for col in fieldnames))

    return CsvWindowView(
        fieldnames=fieldnames,
        rows=rows,
        lines=lines,
        total_rows=total_rows,
        window_rows=window_rows,
        timestamp_field=find_timestamp_field(fieldnames),
    )


def apply_component_focus(
    view: CsvWindowView,
    *,
    domain: str,
    component: str,
) -> ComponentFocusResult:
    """Filter in-window rows to a component value using semantic component columns."""
    focus = str(component or "").strip()
    if not focus:
        return ComponentFocusResult(
            view=view,
            component=focus,
            matched_columns=[],
        )

    semantic = detect_semantic_columns(view.fieldnames, domain=domain)
    component_columns = semantic.get("component_columns", [])
    if not component_columns:
        return ComponentFocusResult(
            view=CsvWindowView(
                fieldnames=view.fieldnames,
                rows=[],
                lines=[",".join(view.fieldnames)] if view.fieldnames else [],
                total_rows=view.total_rows,
                window_rows=0,
                timestamp_field=view.timestamp_field,
            ),
            component=focus,
            matched_columns=[],
        )

    lowered_focus = focus.lower()
    filtered_rows: list[dict[str, str]] = []
    for row in view.rows:
        matched = False
        for column in component_columns:
            value = (row.get(column, "") or "").strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered == lowered_focus or lowered_focus in lowered or lowered in lowered_focus:
                matched = True
                break
        if matched:
            filtered_rows.append(row)

    lines = [",".join(view.fieldnames)] if view.fieldnames else []
    for row in filtered_rows:
        lines.append(",".join((row.get(col, "") or "").replace("\n", " ").strip() for col in view.fieldnames))

    filtered = CsvWindowView(
        fieldnames=view.fieldnames,
        rows=filtered_rows,
        lines=lines,
        total_rows=view.total_rows,
        window_rows=len(filtered_rows),
        timestamp_field=view.timestamp_field,
    )
    return ComponentFocusResult(
        view=filtered,
        component=focus,
        matched_columns=component_columns,
    )


def find_timestamp_field(fieldnames: list[str]) -> str | None:
    """Locate best timestamp-like column in headers."""
    targets = {
        "timestamp",
        "ts",
        "time",
        "event_time",
        "start_time",
        "end_time",
        "datetime",
    }
    for field in fieldnames:
        if normalize_header(field) in targets:
            return field
    return None


def normalize_header(value: str) -> str:
    """Normalize header for robust matching."""
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def parse_timestamp_seconds(raw: str) -> int | None:
    """Parse timestamp values in seconds, milliseconds, or ISO datetime."""
    value = str(raw or "").strip()
    if not value:
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        try:
            numeric = float(value)
        except ValueError:
            return None
        if abs(numeric) >= 1_000_000_000_000:
            return int(numeric // 1000)
        return int(numeric)
    iso = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def window_detail(view: CsvWindowView, *, start_ts: int, end_ts: int) -> str:
    """Human-readable detail for diagnostics and logs."""
    return (
        "Analyzed strict time-window slice: "
        f"window=[{start_ts},{end_ts}], "
        f"rows_in_window={view.window_rows}, total_rows={view.total_rows}, "
        f"timestamp_field={view.timestamp_field or 'not_found'}."
    )


def build_llm_observation_context(
    view: CsvWindowView,
    *,
    domain: str,
    start_ts: int,
    end_ts: int,
    sample_limit: int = 12,
) -> dict[str, Any]:
    """Build compact tool output context for LLM-based decision making."""
    sample_lines = view.lines[:sample_limit] if view.lines else []
    semantic_columns = detect_semantic_columns(view.fieldnames, domain=domain)
    semantic_summary = summarize_semantic_window(
        rows=view.rows,
        semantic_columns=semantic_columns,
        timestamp_field=view.timestamp_field,
    )
    return {
        "domain": domain,
        "window": {"start": start_ts, "end": end_ts},
        "timestamp_field": view.timestamp_field,
        "total_rows": view.total_rows,
        "rows_in_window": view.window_rows,
        "header_line": view.lines[0] if view.lines else "",
        "fieldnames": view.fieldnames,
        "semantic_columns": semantic_columns,
        "semantic_summary": semantic_summary,
        "sample_lines": sample_lines,
    }


def detect_semantic_columns(fieldnames: list[str], *, domain: str) -> dict[str, list[str]]:
    """Infer semantic column roles from CSV header names."""
    by_norm = {name: normalize_header(name) for name in fieldnames}

    def match(tokens: tuple[str, ...]) -> list[str]:
        selected: list[str] = []
        for original, norm in by_norm.items():
            if any(norm == token or token in norm for token in tokens):
                selected.append(original)
        return selected

    component = match(("cmdb_id", "component", "service", "host", "node", "instance", "tc"))
    reason = match(("value", "message", "msg", "error", "exception", "reason", "log_name", "kpi_name", "status"))
    duration = match(("duration", "duration_ms", "latency", "mrt", "response_time", "p95", "p99"))
    error_rate = match(("error_rate", "errors_rate", "5xx_rate", "5xx", "sr", "rr"))

    # Domain-specific light bias.
    if domain == "trace":
        component = _prepend_unique(component, match(("cmdb_id", "service", "component")))
        duration = _prepend_unique(duration, match(("duration", "latency")))
    elif domain == "logs":
        reason = _prepend_unique(reason, match(("log_name", "value", "message", "error", "exception")))
    elif domain == "metrics":
        component = _prepend_unique(component, match(("tc", "cmdb_id", "service", "host")))
        duration = _prepend_unique(duration, match(("mrt", "latency", "response_time", "duration")))

    numeric_candidates = [
        name for name in fieldnames if _looks_numeric_column(normalize_header(name))
    ]
    return {
        "component_columns": component[:6],
        "reason_columns": reason[:8],
        "duration_columns": duration[:6],
        "error_rate_columns": error_rate[:6],
        "numeric_candidate_columns": numeric_candidates[:10],
    }


def summarize_semantic_window(
    *,
    rows: list[dict[str, str]],
    semantic_columns: dict[str, list[str]],
    timestamp_field: str | None,
) -> dict[str, Any]:
    """Summarize semantic signals from all in-window rows."""
    components = _top_values(rows, semantic_columns.get("component_columns", []), limit=6)
    reasons = _top_values(rows, semantic_columns.get("reason_columns", []), limit=6)
    numeric_ranges = _numeric_ranges(rows, semantic_columns.get("numeric_candidate_columns", []), limit=8)
    min_ts, max_ts = _window_observed_bounds(rows, timestamp_field)
    return {
        "window_observed_time_bounds": {
            "min_ts": min_ts,
            "max_ts": max_ts,
        },
        "top_component_values": components,
        "top_reason_values": reasons,
        "numeric_ranges": numeric_ranges,
    }


def _prepend_unique(base: list[str], preferred: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in preferred + base:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _looks_numeric_column(norm: str) -> bool:
    numeric_tokens = (
        "latency",
        "duration",
        "mrt",
        "p95",
        "p99",
        "cpu",
        "error",
        "rate",
        "rr",
        "sr",
        "cnt",
        "qps",
        "rps",
        "value",
    )
    return any(token == norm or token in norm for token in numeric_tokens)


def _top_values(rows: list[dict[str, str]], columns: list[str], *, limit: int) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for column in columns:
            raw = (row.get(column, "") or "").strip()
            if not raw:
                continue
            counter[(column, raw[:140])] += 1
    results: list[dict[str, Any]] = []
    for (column, value), count in counter.most_common(limit):
        results.append({"column": column, "value": value, "count": count})
    return results


def _numeric_ranges(rows: list[dict[str, str]], columns: list[str], *, limit: int) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for column in columns:
        values: list[float] = []
        for row in rows:
            raw = (row.get(column, "") or "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if not values:
            continue
        out[column] = {
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "count": len(values),
        }
        if len(out) >= limit:
            break
    return out


def _window_observed_bounds(rows: list[dict[str, str]], timestamp_field: str | None) -> tuple[int | None, int | None]:
    if not timestamp_field:
        return None, None
    values: list[int] = []
    for row in rows:
        ts = parse_timestamp_seconds(row.get(timestamp_field, ""))
        if ts is not None:
            values.append(ts)
    if not values:
        return None, None
    return min(values), max(values)


def count_matches(text: str, pattern: str) -> int:
    """Count regex matches case-insensitively."""
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def sample_matching_lines(lines: list[str], pattern: str, limit: int = 3) -> list[str]:
    """Get short evidence snippets matching a pattern."""
    regex = re.compile(pattern, flags=re.IGNORECASE)
    matches: list[str] = []
    for line in lines:
        if regex.search(line):
            matches.append(line.strip()[:300])
            if len(matches) >= limit:
                break
    return matches


def extract_trace_durations(rows: list[dict[str, str]]) -> list[int]:
    """Extract numeric durations from trace-like rows."""
    durations: list[int] = []
    for row in rows:
        for key, value in row.items():
            if normalize_header(key) not in {"duration", "latency", "duration_ms", "mrt"}:
                continue
            try:
                durations.append(int(float(str(value).strip())))
            except ValueError:
                continue
    return durations


def max_numeric_column(rows: list[dict[str, str]], column_names: tuple[str, ...]) -> float | None:
    """Return max numeric value among candidate columns."""
    targets = {item.lower() for item in column_names}
    values: list[float] = []
    for row in rows:
        for key, value in row.items():
            if normalize_header(key) not in targets:
                continue
            try:
                values.append(float(str(value).strip()))
            except ValueError:
                continue
    return max(values) if values else None


def min_numeric_column(rows: list[dict[str, str]], column_names: tuple[str, ...]) -> float | None:
    """Return min numeric value among candidate columns."""
    targets = {item.lower() for item in column_names}
    values: list[float] = []
    for row in rows:
        for key, value in row.items():
            if normalize_header(key) not in targets:
                continue
            try:
                values.append(float(str(value).strip()))
            except ValueError:
                continue
    return min(values) if values else None


def max_value_after_keywords(text: str, keywords: tuple[str, ...]) -> float | None:
    """Fallback extraction for non-structured lines (keyword then number)."""
    values: list[float] = []
    for keyword in keywords:
        pattern = rf"{re.escape(keyword)}[^\d-]{{0,16}}(-?\d+(?:\.\d+)?)"
        for item in re.findall(pattern, text, flags=re.IGNORECASE):
            try:
                values.append(float(item))
            except ValueError:
                continue
    return max(values) if values else None

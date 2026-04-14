"""Telemetry tools tests for header/semantic understanding."""

from __future__ import annotations

from tools import telemetry_tools


def test_detect_semantic_columns_and_summary() -> None:
    fieldnames = ["timestamp", "cmdb_id", "log_name", "value", "mrt", "sr"]
    semantic = telemetry_tools.detect_semantic_columns(fieldnames, domain="logs")
    assert "cmdb_id" in semantic["component_columns"]
    assert "log_name" in semantic["reason_columns"] or "value" in semantic["reason_columns"]

    rows = [
        {
            "timestamp": "1614868200",
            "cmdb_id": "MG01",
            "log_name": "timeout error",
            "value": "timeout to payment",
            "mrt": "190.42",
            "sr": "100.0",
        },
        {
            "timestamp": "1614868210",
            "cmdb_id": "IG02",
            "log_name": "retry failed",
            "value": "retry failed",
            "mrt": "42.88",
            "sr": "99.0",
        },
    ]
    summary = telemetry_tools.summarize_semantic_window(
        rows=rows,
        semantic_columns=semantic,
        timestamp_field="timestamp",
    )
    assert isinstance(summary["top_component_values"], list)
    assert isinstance(summary["top_reason_values"], list)
    assert isinstance(summary["numeric_ranges"], dict)
    bounds = summary["window_observed_time_bounds"]
    assert bounds["min_ts"] == 1614868200
    assert bounds["max_ts"] == 1614868210


def test_apply_component_focus_filters_rows() -> None:
    view = telemetry_tools.CsvWindowView(
        fieldnames=["timestamp", "cmdb_id", "value"],
        rows=[
            {"timestamp": "1614868200", "cmdb_id": "Tomcat01", "value": "ok"},
            {"timestamp": "1614868210", "cmdb_id": "Tomcat02", "value": "timeout"},
            {"timestamp": "1614868220", "cmdb_id": "Tomcat02", "value": "retry"},
        ],
        lines=[
            "timestamp,cmdb_id,value",
            "1614868200,Tomcat01,ok",
            "1614868210,Tomcat02,timeout",
            "1614868220,Tomcat02,retry",
        ],
        total_rows=3,
        window_rows=3,
        timestamp_field="timestamp",
    )
    focused = telemetry_tools.apply_component_focus(
        view,
        domain="logs",
        component="Tomcat02",
    )
    assert focused.component == "Tomcat02"
    assert "cmdb_id" in focused.matched_columns
    assert focused.view.window_rows == 2

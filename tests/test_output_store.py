"""Artifact persistence tests."""

from __future__ import annotations

from pathlib import Path

from runtime.output_store import persist_run_artifacts


def test_persist_run_artifacts_creates_json_and_txt(tmp_path: Path) -> None:
    artifacts = persist_run_artifacts(
        run_id="run-test123",
        query="q",
        source_path="/tmp/src",
        db_url=None,
        llm_provider="openai-compatible",
        llm_model="gpt-test",
        status="success",
        events=[
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "sender": "ParserAgent",
                "recipient": "Runtime",
                "phase": "success",
                "content": "done",
            }
        ],
        result_payload={"ok": True},
        error_message=None,
        output_root=tmp_path,
    )

    json_path = Path(artifacts["json_path"])
    txt_path = Path(artifacts["txt_path"])
    assert json_path.exists()
    assert txt_path.exists()
    assert "/json/" in artifacts["json_path"]
    assert "/txt/" in artifacts["txt_path"]

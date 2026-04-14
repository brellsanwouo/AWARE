"""UI server tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from llm.mock import MockLLMClient
from ui.server import create_app


def test_ui_health() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ui_parse_stream_returns_result(tmp_path, monkeypatch) -> None:
    from ui import server as ui_server
    from agents import parser_agent as parser_module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(ui_server, "create_llm_client", lambda **_: MockLLMClient(invalid_attempts=1))
    monkeypatch.setattr(parser_module, "create_llm_client", lambda **_: MockLLMClient(invalid_attempts=1))
    client = TestClient(create_app())
    params = {
        "query": "On 2021-03-04 between 18:30:00 and 19:00:00 checkout timeout",
        "repo": str(tmp_path),
        "max_attempts": 4,
    }

    with client.stream("GET", "/api/parse-stream", params=params) as response:
        assert response.status_code == 200
        buffer = ""
        for chunk in response.iter_text():
            buffer += chunk

    payload_lines = [line for line in buffer.splitlines() if line.startswith("data: ")]
    assert payload_lines
    payloads = [json.loads(line[6:]) for line in payload_lines]
    assert any(item.get("type") == "event" for item in payloads)
    assert any(item.get("type") == "result" for item in payloads)

"""Environment helpers tests."""

from __future__ import annotations

from runtime.env import env_bool


def test_env_bool_parsing(monkeypatch) -> None:
    monkeypatch.setenv("AWARE_ENABLE_REASONING", "true")
    assert env_bool("AWARE_ENABLE_REASONING", False) is True

    monkeypatch.setenv("AWARE_ENABLE_REASONING", "0")
    assert env_bool("AWARE_ENABLE_REASONING", True) is False

    monkeypatch.setenv("AWARE_ENABLE_REASONING", "invalid")
    assert env_bool("AWARE_ENABLE_REASONING", True) is True

"""Environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_env() -> list[Path]:
    """Load `.env` variables once from likely locations using setdefault semantics."""
    loaded: list[Path] = []
    for path in _candidate_env_paths():
        if not path.exists() or not path.is_file():
            continue
        _load_dotenv_file(path)
        loaded.append(path)
    return loaded


def _candidate_env_paths() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.getenv("AWARE_ENV_FILE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(Path.cwd() / ".env")
    # Project root when launched from another current working directory.
    candidates.append(Path(__file__).resolve().parents[1] / ".env")

    seen: set[str] = set()
    unique: list[Path] = []
    for item in candidates:
        try:
            key = str(item.resolve())
        except Exception:
            key = str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _load_dotenv_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip()
        clean_value = _clean_value(value)
        if clean_key:
            os.environ.setdefault(clean_key, clean_value)


def _clean_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def env_bool(name: str, default: bool = True) -> bool:
    """Read a boolean env var with tolerant parsing."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    return default

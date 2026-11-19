import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

BASE_ENV = {
    "APP_PROVIDER": "stub",
    "APP_ENV": "test",
    "LOG_LEVEL": "warning",
    "LOCAL_BASE_URL": "http://host.docker.internal:11434/v1",
    "LOCAL_API_KEY": "not-needed",
    "LOCAL_MODEL": "qwen2.5:7b-instruct",
    "REMOTE_BASE_URL": "https://api.gapgpt.app/v1",
    "REMOTE_API_KEY": "replace-me",
    "REMOTE_MODEL": "gemini-2.5-flash-lite",
    "EMBEDDING_PROVIDER": "",
    "EMBEDDING_BASE_URL": "",
    "EMBEDDING_MODEL": "",
}


def apply_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CONFIG_DIR", str(ROOT / "config"))
    monkeypatch.setenv("DATA_DIR", str(ROOT / "data"))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))


def copy_config(destination: Path) -> None:
    destination.mkdir()
    for name in ("models.yaml", "roles.yaml", "limits.yaml"):
        shutil.copy(ROOT / "config" / name, destination / name)

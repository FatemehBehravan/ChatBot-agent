import pytest

from src.settings import load_config
from tests.support import apply_env


@pytest.fixture
def config(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    return load_config()

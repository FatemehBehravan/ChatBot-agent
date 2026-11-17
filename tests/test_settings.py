import os
import subprocess
import sys

import pytest

from src.errors import ConfigError
from src.settings import load_config
from tests.support import BASE_ENV, ROOT, apply_env, copy_config


def test_repository_config_loads(config):
    assert config.app_provider == "stub"
    assert config.provider.type == "stub"
    assert config.provider.model == "stub"
    assert "u-anon" in config.users
    assert config.users["u-cust-1001"].role == "customer"
    assert config.roles.roles["customer"].scope == "own_customer"
    assert abs(
        config.limits.context_budget.system_prompt
        + config.limits.context_budget.tool_schemas
        + config.limits.context_budget.history
        + config.limits.context_budget.tool_results
        + config.limits.context_budget.retrieved_docs
        + config.limits.context_budget.output_reserve
        - 1.0
    ) < 1e-6
    assert "delivery assistant" in config.system_prompt.lower()


def test_missing_environment_variable_fails_fast(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    monkeypatch.delenv("LOCAL_BASE_URL")
    with pytest.raises(ConfigError, match=r"LOCAL_BASE_URL is not set"):
        load_config()


def test_provider_switch_is_configuration_only(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    monkeypatch.setenv("APP_PROVIDER", "local")
    local = load_config()
    assert local.app_provider == "local"
    assert local.provider.type == "openai_compatible"
    assert local.provider.model == "qwen2.5:7b-instruct"

    monkeypatch.setenv("APP_PROVIDER", "remote")
    remote = load_config()
    assert remote.provider.type == "openai_compatible"
    assert remote.provider.model == "replace-with-a-low-cost-model"
    assert remote.provider.context_window == 128000


def test_unknown_provider_is_rejected(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    monkeypatch.setenv("APP_PROVIDER", "missing")
    with pytest.raises(ConfigError, match="is not defined"):
        load_config()


def test_context_budget_must_sum_to_one(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    config_dir = tmp_path / "config"
    copy_config(config_dir)
    limits_path = config_dir / "limits.yaml"
    limits_path.write_text(
        limits_path.read_text(encoding="utf-8").replace("system_prompt: 0.10", "system_prompt: 0.50", 1),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    with pytest.raises(ConfigError, match="context_budget fractions must sum to 1"):
        load_config()


def test_unknown_role_scope_fails(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    config_dir = tmp_path / "config"
    copy_config(config_dir)
    roles_path = config_dir / "roles.yaml"
    roles_path.write_text(
        roles_path.read_text(encoding="utf-8").replace("scope: none", "scope: everywhere", 1),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_config()


def test_process_exits_before_serving_when_config_is_invalid(tmp_path):
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.pop("LOCAL_BASE_URL", None)
    env["CONFIG_DIR"] = str(ROOT / "config")
    env["DATA_DIR"] = str(ROOT / "data")
    env["STATE_DIR"] = str(tmp_path / "state")
    result = subprocess.run(
        [sys.executable, "-m", "src.main"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "Configuration error" in result.stderr
    assert "LOCAL_BASE_URL" in result.stderr

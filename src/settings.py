"""Typed configuration from config/*.yaml and the process environment.

Invalid files, unknown keys, unresolved ${VAR} placeholders, and a non-stub
active provider abort startup with ConfigError before the server listens.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.errors import ConfigError
from src.security.identity import Identity, load_users

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}
_APP_ENV_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,32}")


class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_1m_tokens: float | None = None
    output_per_1m_tokens: float | None = None

    @model_validator(mode="after")
    def prices_are_not_negative(self) -> "PricingConfig":
        for name in ("input_per_1m_tokens", "output_per_1m_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")
        return self


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["openai_compatible", "stub"]
    base_url: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    model: str | None = None
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    supports_tool_calling: bool
    timeout_seconds: float | None = None
    data_policy: Literal["full", "masked_only"]
    pricing: PricingConfig | None = None

    @model_validator(mode="after")
    def check_provider(self) -> "ProviderConfig":
        if self.max_output_tokens > self.context_window:
            raise ValueError("max_output_tokens cannot exceed context_window")
        if self.type == "stub":
            if not self.model:
                self.model = "stub"
            return self
        if not self.base_url:
            raise ValueError("base_url is required for an openai_compatible provider")
        if not self.model:
            raise ValueError("model is required for an openai_compatible provider")
        if self.api_key is None or self.api_key == "":
            raise ValueError("api_key is required for an openai_compatible provider")
        if self.timeout_seconds is None or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 for an openai_compatible provider")
        return self


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = ""
    base_url: str = ""
    model: str = ""

    @model_validator(mode="after")
    def complete_when_enabled(self) -> "EmbeddingsConfig":
        if self.provider and (not self.base_url or not self.model):
            raise ValueError("embeddings.base_url and embeddings.model are required when embeddings.provider is set")
        return self


class CircuitBreakerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_threshold: int = Field(ge=1)
    cooldown_seconds: float = Field(gt=0)


class RouteCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttl_seconds: float = Field(ge=0)
    max_entries: int = Field(ge=1)


class RoutingConfig(BaseModel):
    """Which model serves a step. Thresholds live here so a change does not need code."""

    model_config = ConfigDict(extra="forbid")

    default_provider: str
    fallback_order: list[str] = Field(min_length=1)
    cheap_input_tokens: int = Field(default=500, ge=0)
    hosted_budget_usd: float = Field(default=0.02, ge=0)
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=lambda: CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=30)
    )
    cache: RouteCacheConfig = Field(default_factory=lambda: RouteCacheConfig(ttl_seconds=300, max_entries=128))


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig]
    embeddings: EmbeddingsConfig
    routing: RoutingConfig

    @model_validator(mode="after")
    def providers_are_consistent(self) -> "ModelsConfig":
        names = set(self.providers)
        if "stub" not in names:
            raise ValueError("providers.stub is required")
        if self.providers["stub"].type != "stub":
            raise ValueError("providers.stub must have type: stub")
        if self.routing.default_provider not in names:
            raise ValueError(
                f"routing.default_provider {self.routing.default_provider!r} is not a configured provider"
            )
        unknown = [name for name in self.routing.fallback_order if name not in names]
        if unknown:
            raise ValueError(
                "routing.fallback_order references unknown providers: " + ", ".join(unknown)
            )
        if self.routing.default_provider not in self.routing.fallback_order:
            raise ValueError("routing.default_provider must be listed in routing.fallback_order")
        return self


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    tools: list[str] = Field(min_length=1)
    scope: Literal["none", "own_customer", "all"]
    pii: Literal["own_full", "masked", "none"]
    confirmation_required: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def tools_are_well_formed(self) -> "RoleConfig":
        if any(not isinstance(tool, str) or not tool for tool in self.tools):
            raise ValueError("tools must be non-empty strings")
        if len(set(self.tools)) != len(self.tools):
            raise ValueError("tools contains a duplicate")
        missing = sorted(set(self.confirmation_required) - set(self.tools))
        if missing:
            raise ValueError("confirmation_required is not a subset of tools: " + ", ".join(missing))
        return self


class RolesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pii_fields: list[str] = Field(default_factory=lambda: ["phone", "full_name"])
    untrusted_fields: list[str] = Field(default_factory=lambda: ["notes", "text"])
    roles: dict[str, RoleConfig]

    @model_validator(mode="after")
    def at_least_one_role(self) -> "RolesConfig":
        if not self.roles:
            raise ValueError("at least one role is required")
        for name in (*self.pii_fields, *self.untrusted_fields):
            if not isinstance(name, str) or not name:
                raise ValueError("pii_fields and untrusted_fields must be non-empty strings")
        return self


class TokenLimitWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)


class TokenLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_request: TokenLimitWindow
    per_session: dict[str, int]
    per_user: dict[str, Any]

    @model_validator(mode="after")
    def session_and_user_limits(self) -> "TokenLimits":
        session_max = self.per_session.get("max_total_tokens")
        if not isinstance(session_max, int) or session_max <= 0:
            raise ValueError("token_limits.per_session.max_total_tokens must be a positive integer")
        user_max = self.per_user.get("max_total_tokens_per_window")
        window = self.per_user.get("window")
        if not isinstance(user_max, int) or user_max <= 0:
            raise ValueError("token_limits.per_user.max_total_tokens_per_window must be a positive integer")
        if not isinstance(window, str) or not window:
            raise ValueError("token_limits.per_user.window is required")
        return self


class CostLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_session_usd: float = Field(gt=0)
    per_user_usd_per_window: float = Field(gt=0)


class RateLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests_per_minute_per_user: int = Field(gt=0)
    max_concurrent_requests_per_user: int = Field(gt=0)


class ContextBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: float = Field(gt=0, lt=1)
    tool_schemas: float = Field(gt=0, lt=1)
    history: float = Field(gt=0, lt=1)
    tool_results: float = Field(gt=0, lt=1)
    retrieved_docs: float = Field(gt=0, lt=1)
    output_reserve: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def sums_to_one(self) -> "ContextBudget":
        total = (
            self.system_prompt
            + self.tool_schemas
            + self.history
            + self.tool_results
            + self.retrieved_docs
            + self.output_reserve
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"context_budget fractions must sum to 1.0 (got {total:.6f})")
        return self


class SummarizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_ratio: float = Field(gt=0, lt=1)
    keep_last_turns: int = Field(ge=1)


class ToolOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens_per_result: int = Field(gt=0)


class AgentLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1)
    max_identical_tool_calls: int = Field(ge=1)


class LimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_limits: TokenLimits
    cost_limits: CostLimits
    rate_limits: RateLimits
    context_budget: ContextBudget
    summarization: SummarizationConfig
    tool_output: ToolOutputConfig
    agent_loop: AgentLoopConfig


@dataclass(frozen=True)
class AppConfig:
    models: ModelsConfig
    roles: RolesConfig
    limits: LimitsConfig
    users: dict[str, Identity]
    system_prompt: str
    project_root: Path
    data_dir: Path
    state_dir: Path
    app_provider: str
    log_level: str
    app_env: str
    port: int
    database_url: str

    @property
    def provider(self) -> ProviderConfig:
        name = self.app_provider
        if name == "route":
            name = self.models.routing.default_provider
        return self.models.providers[name]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> AppConfig:
    root = project_root()
    config_dir = Path(os.environ.get("CONFIG_DIR", root / "config"))
    data_dir = Path(os.environ.get("DATA_DIR", root / "data"))
    state_dir = Path(os.environ.get("STATE_DIR", root / ".state"))
    models = _parse(config_dir / "models.yaml", ModelsConfig)
    roles = _parse(config_dir / "roles.yaml", RolesConfig)
    limits = _parse(config_dir / "limits.yaml", LimitsConfig)
    _require_directory(data_dir, "DATA_DIR")
    users = load_users(data_dir / "users.json")
    _check_user_roles(users, roles, data_dir / "users.json")
    app_provider = _active_provider(models)
    system_prompt = _read_system_prompt(root / "prompts" / "system.md")
    return AppConfig(
        models=models,
        roles=roles,
        limits=limits,
        users=users,
        system_prompt=system_prompt,
        project_root=root,
        data_dir=data_dir,
        state_dir=state_dir,
        app_provider=app_provider,
        log_level=_log_level(),
        app_env=_app_env(),
        port=_port(),
        database_url=os.environ.get("DATABASE_URL", "").strip(),
    )


def _parse(path: Path, model: type[BaseModel]) -> Any:
    data = _load_yaml(path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path.name}: invalid configuration\n{exc}") from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing configuration file: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: YAML parse error: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path.name}: configuration root must be a mapping")
    return _expand_env(loaded, path.name)


def _expand_env(value: Any, source: str) -> Any:
    if isinstance(value, str):
        return _expand_string(value, source)
    if isinstance(value, dict):
        return {key: _expand_env(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, source) for item in value]
    return value


def _expand_string(value: str, source: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(
                f"{source}: environment variable {name} is not set (referenced as ${{{name}}})"
            )
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def _active_provider(models: ModelsConfig) -> str:
    raw = os.environ.get("APP_PROVIDER", "stub").strip()
    if not raw:
        raise ConfigError("APP_PROVIDER is empty. Set it to stub, local, remote, or route.")
    if raw == "route":
        return raw
    if raw not in models.providers:
        known = ", ".join(sorted([*models.providers, "route"]))
        raise ConfigError(f"APP_PROVIDER={raw!r} is not defined. Known providers: {known}.")
    return raw


def _read_system_prompt(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"missing system prompt: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"system prompt is empty: {path}")
    return text


def _check_user_roles(users: dict[str, Identity], roles: RolesConfig, path: Path) -> None:
    for user in users.values():
        if user.role not in roles.roles:
            raise ConfigError(f"{path}: user {user.user_id} has unknown role {user.role!r}")


def _require_directory(path: Path, name: str) -> None:
    if not path.is_dir():
        raise ConfigError(f"{name} does not exist: {path}")


def _log_level() -> str:
    level = os.environ.get("LOG_LEVEL", "info").strip().lower()
    if level not in _LOG_LEVELS:
        allowed = ", ".join(sorted(_LOG_LEVELS))
        raise ConfigError(f"LOG_LEVEL must be one of {allowed} (got {level!r})")
    return level


def _app_env() -> str:
    value = os.environ.get("APP_ENV", "dev").strip()
    if not _APP_ENV_PATTERN.fullmatch(value):
        raise ConfigError(f"APP_ENV must be 1-32 letters, digits, '_' or '-' (got {value!r})")
    return value


def _port() -> int:
    raw = os.environ.get("PORT", "8000").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= 65535:
        raise ConfigError(f"PORT must be an integer from 1 to 65535 (got {raw!r})")
    return int(raw)

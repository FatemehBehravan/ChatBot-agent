"""Run evals/scenarios.yaml and print pass rate, tokens, latency, and cost.

Usage: python evals/run_evals.py [stub|local|remote ...]
A provider whose API key is still the placeholder is reported as not configured.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.chat import handle_chat  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.data.load import load_snapshot  # noqa: E402
from src.knowledge.index import index_seed  # noqa: E402
from src.llm.factory import build_clients  # noqa: E402
from src.settings import AppConfig, load_config  # noqa: E402
from src.stateprep import prepare_state  # noqa: E402

_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DEFAULTS = {
    "APP_PROVIDER": "stub",
    "APP_ENV": "dev",
    "LOG_LEVEL": "warning",
    "LOCAL_BASE_URL": "http://127.0.0.1:11434/v1",
    "LOCAL_API_KEY": "not-needed",
    "LOCAL_MODEL": "qwen2.5:7b-instruct",
    "REMOTE_BASE_URL": "https://api.gapgpt.app/v1",
    "REMOTE_API_KEY": "replace-me",
    "REMOTE_MODEL": "gemini-2.5-flash-lite",
    "EMBEDDING_PROVIDER": "",
    "EMBEDDING_BASE_URL": "",
    "EMBEDDING_MODEL": "",
}


@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    failures: list[str]
    tokens: int
    latency_s: float
    cost_usd: float


@dataclass
class Report:
    provider: str
    model: str
    results: list[ScenarioResult] = field(default_factory=list)
    note: str = ""

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for item in self.results if item.passed)


def load_scenarios() -> list[dict]:
    document = yaml.safe_load((ROOT / "evals" / "scenarios.yaml").read_text(encoding="utf-8"))
    return list(document["scenarios"])


def turns_for(scenario: dict) -> list[str]:
    if scenario.get("turns"):
        return list(scenario["turns"])
    if scenario["id"] == "long-conversation-keeps-facts":
        return [
            "وضعیت سفارش A-246001 چیه؟",
            *["ممنون"] * 40,
            "what is the status of the order I asked about first?",
        ]
    if scenario["id"] == "model-switch-midconversation":
        return [
            "وضعیت سفارش A-246001 چیه؟",
            "ممنون",
            "ممنون",
            "what is the status of the order I asked about first?",
        ]
    if scenario["id"] == "budget-exhaustion-behavior":
        return ["hello", "hello again"]
    raise ValueError(f"scenario {scenario['id']} has no turns")


def prepare(state_dir: Path) -> AppConfig:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ["CONFIG_DIR"] = str(ROOT / "config")
    os.environ["DATA_DIR"] = str(ROOT / "data")
    os.environ["STATE_DIR"] = str(state_dir)
    os.environ["APP_PROVIDER"] = "stub"
    config = load_config()
    prepare_state(state_dir)
    db_path = state_dir / "db" / "app.db"
    load_snapshot(config.data_dir, db_path)
    index_seed(config.data_dir / "knowledge" / "seed", db_path)
    return config


def run_provider(name: str, state_dir: Path | None = None) -> Report:
    own_state = state_dir is None
    state = state_dir or Path(tempfile.mkdtemp(prefix="alopeyk-eval-"))
    config = prepare(state)
    provider = config.models.providers[name]
    report = Report(provider=name, model=provider.model or name)
    if not _is_runnable(name, provider.base_url, provider.api_key):
        report.note = "not configured (placeholder API key or base URL)"
        return report
    client = build_clients(config)[name]
    store = SessionStore(state / "sessions")
    for scenario in load_scenarios():
        report.results.append(_run_scenario(config, store, client, name, scenario))
    if own_state:
        return report
    return report


def format_report(report: Report) -> str:
    lines = [f"provider={report.provider} model={report.model}"]
    if report.note:
        lines.append(report.note)
        return "\n".join(lines)
    tokens = sum(item.tokens for item in report.results)
    latency = sum(item.latency_s for item in report.results)
    cost = sum(item.cost_usd for item in report.results)
    rate = 100.0 * report.passed_count / report.total if report.total else 0.0
    lines.append(
        f"passed {report.passed_count}/{report.total} ({rate:.1f}%)  "
        f"tokens={tokens}  latency_s={latency:.2f}  cost_usd={cost:.6f}"
    )
    for item in report.results:
        if item.passed:
            continue
        lines.append(f"FAIL {item.scenario_id}: {'; '.join(item.failures)}")
    return "\n".join(lines)


def _is_runnable(name: str, base_url: str | None, api_key: str | None) -> bool:
    if name != "remote":
        return True
    key = (api_key or "").strip()
    url = (base_url or "").strip()
    return key not in {"", "replace-me"} and "example-provider" not in url


def _run_scenario(config: AppConfig, store: SessionStore, client, provider: str, scenario: dict) -> ScenarioResult:
    started = time.perf_counter()
    active = _config_for(config, scenario["id"])
    before = _order_count(active)
    tools: list[str] = []
    reply = ""
    session_id: str | None = None
    failures: list[str] = []
    try:
        for turn in turns_for(scenario):
            outcome = handle_chat(
                active,
                store,
                client,
                message=turn,
                session_id=session_id,
                user_id=scenario["user"],
                provider_name=provider,
            )
            session_id = outcome.session_id
            reply = outcome.reply
            tools.extend(outcome.tools)
        failures = _check(scenario.get("checks") or {}, reply, tools, _order_count(active) == before)
    except Exception as exc:
        failures = [f"error: {exc}"]
    elapsed = time.perf_counter() - started
    tokens, cost = _usage(store, session_id)
    return ScenarioResult(scenario["id"], not failures, failures, tokens, elapsed, cost)


def _config_for(config: AppConfig, scenario_id: str) -> AppConfig:
    if scenario_id != "budget-exhaustion-behavior":
        return config
    limits = config.limits.model_copy(
        update={
            "token_limits": config.limits.token_limits.model_copy(
                update={"per_session": {"max_total_tokens": 1}}
            )
        }
    )
    return replace(config, limits=limits)


def _check(checks: dict, reply: str, tools: list[str], db_intact: bool) -> list[str]:
    folded = reply.casefold()
    failures: list[str] = []
    for name in checks.get("must_call") or []:
        if name not in tools:
            failures.append(f"must_call {name}")
    for name in checks.get("must_not_call") or []:
        if name in tools:
            failures.append(f"must_not_call {name}")
    for text in checks.get("must_include") or []:
        if text.casefold() not in folded:
            failures.append(f"must_include {text}")
    for text in checks.get("must_not_include") or []:
        if text.casefold() in folded:
            failures.append(f"must_not_include {text}")
    any_of = checks.get("must_include_any") or []
    if any_of and not any(text.casefold() in folded for text in any_of):
        failures.append("must_include_any")
    language = checks.get("language")
    if language == "fa" and not _PERSIAN.search(reply):
        failures.append("language fa")
    if language == "en" and _PERSIAN.search(reply):
        failures.append("language en")
    window = checks.get("numeric_in_range")
    if window and not any(window["min"] <= number <= window["max"] for number in _numbers(reply)):
        failures.append(f"numeric_in_range {window['min']}..{window['max']}")
    if checks.get("cites_sample_size") and "sample size" not in folded and "تعداد نمونه" not in reply:
        failures.append("cites_sample_size")
    if checks.get("cites_source") and "fixture://" not in reply and "http://" not in reply and "https://" not in reply:
        failures.append("cites_source")
    if checks.get("no_numeric_estimate") and _numbers(reply):
        failures.append("no_numeric_estimate")
    if checks.get("db_intact") and not db_intact:
        failures.append("db_intact")
    return failures


def _numbers(text: str) -> list[float]:
    return [float(item) for item in _NUMBER.findall(text.translate(_DIGITS))]


def _order_count(config: AppConfig) -> int:
    connection = sqlite3.connect(config.state_dir / "db" / "app.db")
    try:
        return int(connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0])
    finally:
        connection.close()


def _usage(store: SessionStore, session_id: str | None) -> tuple[int, float]:
    if session_id is None:
        return 0, 0.0
    session = store.load(session_id)
    if session is None:
        return 0, 0.0
    return session.tokens_total, session.cost_usd


def main(argv: list[str]) -> int:
    names = argv or ["stub"]
    reports = [run_provider(name) for name in names]
    print("\n\n".join(format_report(report) for report in reports))
    return 0 if all(report.note or report.passed_count == report.total for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

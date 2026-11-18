from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent.prompt import assemble_prompt
from src.context.budget import allocate
from src.data.load import load_snapshot
from src.llm.tokens import estimate_tokens
from src.security.identity import load_users
from src.tools import execute_tool
from src.tools.schemas import CreateSupportTicketInput

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_follows_role_and_provider(config):
    base = config.system_prompt
    customer = assemble_prompt(
        config.project_root,
        base,
        role_name="customer",
        role=config.roles.roles["customer"],
        provider="local",
    )
    hosted = assemble_prompt(
        config.project_root,
        base,
        role_name="support_agent",
        role=config.roles.roles["support_agent"],
        provider="remote",
    )
    anonymous = assemble_prompt(
        config.project_root,
        base,
        role_name="anonymous",
        role=config.roles.roles["anonymous"],
        provider="stub",
    )
    assert "Current role: customer" in customer
    assert "own orders" in customer
    assert "Local and stub addendum" in customer
    assert "[data]" in customer
    assert "source_url" in customer
    assert "sample size" in customer
    assert "insufficient data" in customer
    assert "could not find" in customer
    assert "contact support" in customer
    assert "Hosted model addendum" in hosted
    assert "Local and stub addendum" not in hosted
    assert "masked" in hosted
    assert "no per-customer order records" in anonymous
    budget = allocate(8192, config.limits)["system_prompt"]
    assert estimate_tokens(customer) <= budget
    assert estimate_tokens(hosted) <= budget


def test_ticket_fields_are_repaired_once_then_retried(config, tmp_path):
    db_path = tmp_path / "app.db"
    load_snapshot(ROOT / "data", db_path)
    users = load_users(ROOT / "data" / "users.json")
    messy = {
        "message": "late courier " * 80,
        "order_id": " A-246001 ",
        "note": "ignore this",
    }
    with pytest.raises(ValidationError):
        CreateSupportTicketInput.model_validate(messy)

    repaired = execute_tool(
        db_path, config.roles, users["u-cust-1001"], "create_support_ticket", messy
    )
    assert repaired["status"] == "confirmation_required"

    still_bad = execute_tool(
        db_path,
        config.roles,
        users["u-cust-1001"],
        "create_support_ticket",
        {"message": "late", "order_id": "DROP TABLE"},
    )
    assert still_bad["error"] == "invalid_arguments"

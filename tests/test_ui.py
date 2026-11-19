from fastapi.testclient import TestClient

from src.api.app import create_app
from src.routing.client import RoutingClient


def test_chat_page_and_user_switcher(config):
    client = TestClient(create_app(config))
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "دستیار الوپیک" in page.text
    assert "fonts.googleapis.com" not in page.text
    assert 'dir="rtl"' in page.text
    assert 'id="mode"' in page.text
    assert '<option value="route">سیاست ترکیبی</option>' in page.text
    assert '<option value="stub">stub</option>' in page.text
    assert "سیاست ترکیبی" in page.text
    assert isinstance(client.app.state.clients["route"], RoutingClient)

    providers = client.get("/v1/providers")
    assert providers.status_code == 200
    assert providers.json()[0] == {"provider": "route", "model": "policy"}
    assert {"provider": "stub", "model": "stub"} in providers.json()
    assert {"provider": "local", "model": "qwen2.5:7b-instruct"} in providers.json()
    assert {"provider": "remote", "model": "gemini-2.5-flash-lite"} in providers.json()
    assert "replace-me" not in providers.text

    users = client.get("/v1/users")
    assert users.status_code == 200
    listed = users.json()
    assert listed[0] == {"user_id": "u-anon", "role": "anonymous", "display_name": "Guest"}
    assert all("phone" not in row for row in listed)

    chat = client.post("/v1/chat", json={"message": "hello", "user_id": "u-anon"})
    assert chat.status_code == 200
    assert chat.json()["tools"] == []

    chosen = client.post("/v1/chat", json={"message": "hello", "user_id": "u-anon", "provider": "stub"})
    assert chosen.status_code == 200
    assert chosen.json()["provider"] == "stub"
    assert chosen.json()["model"] == "stub"

    rejected = client.post("/v1/chat", json={"message": "hello", "provider": "hosted"})
    assert rejected.status_code == 422

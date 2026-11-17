from fastapi.testclient import TestClient

from src.api.app import create_app


def test_chat_page_and_user_switcher(config):
    client = TestClient(create_app(config))
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "دستیار الوپیک" in page.text
    assert "fonts.googleapis.com" not in page.text
    assert 'dir="rtl"' in page.text

    users = client.get("/v1/users")
    assert users.status_code == 200
    listed = users.json()
    assert listed[0] == {"user_id": "u-anon", "role": "anonymous", "display_name": "Guest"}
    assert all("phone" not in row for row in listed)

    chat = client.post("/v1/chat", json={"message": "hello", "user_id": "u-anon"})
    assert chat.status_code == 200
    assert chat.json()["tools"] == []

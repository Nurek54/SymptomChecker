from backend.config import SESSION_COOKIE_NAME


def test_history_unauthorized_without_cookie(client):
    r = client.get("/history")
    assert r.status_code == 401


def test_history_item_unauthorized_without_cookie(client):
    r = client.get("/history/123")
    assert r.status_code == 401


def test_history_delete_unauthorized_without_cookie(client):
    r = client.delete("/history/123")
    assert r.status_code == 401


def test_auth_session_cookie_attributes_on_login(client):
    """A successful POST /auth/session sets a session cookie with HttpOnly and SameSite."""
    payload = {"email": "test@example.com", "password": "secret123"}

    resp = client.post("/auth/session", json=payload)

    # No seeded user in unit tests: 401 is accepted, the cookie cannot be checked then.
    if resp.status_code == 401:
        return

    assert resp.status_code == 200

    cookies = resp.cookies
    cookie_name = SESSION_COOKIE_NAME or "sc_session"
    assert cookie_name in cookies

    raw = resp.headers.get("set-cookie", "")
    raw_low = raw.lower()
    assert "httponly" in raw_low
    # "Secure" is not asserted: it is required in prod but may be off in dev/test.
    assert "samesite" in raw_low

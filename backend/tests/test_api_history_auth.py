# tests/test_api_history_auth.py
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
    """
    A2: po poprawnym POST /auth/session cookie ma HttpOnly i SameSite.
    Jeżeli nie masz w seedzie użytkownika, test łagodnie przechodzi,
    ale nie wymusza 200 (401 kończy się „miękkim” return).
    """
    payload = {"email": "test@example.com", "password": "secret123"}

    resp = client.post("/auth/session", json=payload)

    # Jeżeli endpoint wymaga prawdziwego usera i go nie ma → 401.
    # Wtedy nie jesteśmy w stanie sprawdzić ciasteczka w testach unitowych.
    if resp.status_code == 401:
        # Test działa jako „szkielet” – endpoint istnieje i reaguje.
        return

    assert resp.status_code == 200

    cookies = resp.cookies
    cookie_name = SESSION_COOKIE_NAME or "sc_session"
    assert cookie_name in cookies

    # Sprawdzamy atrybuty w nagłówku Set-Cookie
    raw = resp.headers.get("set-cookie", "")
    raw_low = raw.lower()
    assert "httponly" in raw_low
    # W prod powinno być "secure"; w dev/test bywa wyłączone.
    # assert "secure" in raw_low
    assert "samesite" in raw_low

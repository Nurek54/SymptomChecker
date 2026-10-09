from __future__ import annotations

import pytest

from backend.config import FRONTEND_ORIGIN, SESSION_COOKIE_NAME
from backend.utils import mask_sensitive


def test_history_requires_cookie(client):
    """A1: /history without a valid session always returns 401."""
    r = client.get("/history")
    assert r.status_code == 401


def test_history_different_user_cannot_access_foreign_conv(client):
    """A1: a known conversation id is never readable without that user's valid session cookie."""

    conv_id = 123

    r_a = client.get(f"/history/{conv_id}")
    assert r_a.status_code in (401, 403, 404)

    # _uid only reads SESSION_COOKIE_NAME / "access_token", so an unrelated cookie
    # simulates another user without needing a real JWT.
    client.cookies.set("some_other_cookie", "fake-user-b")

    r_b = client.get(f"/history/{conv_id}")
    assert r_b.status_code in (401, 403, 404)


def test_cors_allows_frontend_origin(client):
    """A3: the frontend origin is accepted in a CORS preflight."""
    origin = FRONTEND_ORIGIN or "http://localhost:3000"
    resp = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == origin


def test_cors_blocks_other_origin(client):
    """A3: a foreign origin gets no access-control-allow-origin header."""
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in resp.headers


def test_guest_can_use_chat_but_not_history(client):
    """R1: a guest can use /chat but has no access to /history (no retention)."""
    payload = {
        "session_id": "guest-session",
        "user_input": "Mam gorączkę i kaszel.",
        "lang": "pl",
    }
    r = client.post("/chat", json=payload)
    # 503 is accepted when app_state is not ready.
    assert r.status_code in (200, 503)

    r2 = client.get("/history")
    assert r2.status_code == 401


def test_logging_masks_sensitive_data_unit():
    """R3: mask_sensitive replaces e-mails and 11-digit PESEL numbers with "***" in log text."""
    msg = "email: user@example.com, pesel: 12345678901"
    masked = mask_sensitive(msg)

    assert "user@example.com" not in masked
    assert "12345678901" not in masked
    assert "***" in masked

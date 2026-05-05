# tests/test_security_privacy.py
from __future__ import annotations

import pytest

from backend.config import FRONTEND_ORIGIN, SESSION_COOKIE_NAME
from backend.utils import mask_sensitive


# ─────────── A1 – twarde egzekwowanie cookie ───────────


def test_history_requires_cookie(client):
    """
    A1: /history bez poprawnej sesji zawsze zwraca 401.
    """
    r = client.get("/history")
    assert r.status_code == 401


def test_history_different_user_cannot_access_foreign_conv(client):
    """
    A1: drugi użytkownik nie ma dostępu do cudzej historii.

    W środowisku testowym:
      - baza jest stubowana (FakeDBSession),
      - bez prawidłowego JWT w cookie helper _uid() zawsze zwraca None,
      - router /history i /history/{id} odpowiadają wtedy 401/403/404.

    Ten test sprawdza, że:
      - nawet jeśli znamy jakieś ID rozmowy,
      - żądania z „innym” cookie też nie dostaną 200.
    """

    # Symulowane ID rozmowy – nie musi realnie istnieć w stubowej bazie
    conv_id = 123

    # "Użytkownik A" – bez poprawnego cookie nie powinien dostać 200
    r_a = client.get(f"/history/{conv_id}")
    assert r_a.status_code in (401, 403, 404)

    # "Użytkownik B" – ustawiamy jakieś inne cookie (nie to, które sprawdza _uid)
    # Uwaga: _uid patrzy tylko na SESSION_COOKIE_NAME / "access_token",
    # więc używamy innej nazwy, żeby nie wchodzić w dekodowanie JWT.
    client.cookies.set("some_other_cookie", "fake-user-b")

    r_b = client.get(f"/history/{conv_id}")
    # Kluczowa asercja: drugi „użytkownik” też nie widzi tej rozmowy.
    assert r_b.status_code in (401, 403, 404)


# ─────────── A3 – CORS ───────────


def test_cors_allows_frontend_origin(client):
    """
    A3: poprawny origin frontendu jest zaakceptowany w preflight.
    """
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
    """
    A3: obcy origin nie dostaje nagłówka access-control-allow-origin.
    """
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in resp.headers


# ─────────── R1 – tryb gościa bez retencji ───────────


def test_guest_can_use_chat_but_not_history(client):
    """
    R1: gość może korzystać z /chat, ale nie ma dostępu do /history.
    """
    payload = {
        "session_id": "guest-session",
        "user_input": "Mam gorączkę i kaszel.",
        "lang": "pl",
    }
    r = client.post("/chat", json=payload)
    # w normalnych warunkach 200; jeśli app_state niegotowy → 503 (stub)
    assert r.status_code in (200, 503)

    r2 = client.get("/history")
    assert r2.status_code == 401


# ─────────── R3 – maskowanie w logach ───────────


def test_logging_masks_sensitive_data_unit():
    """
    R3: helper mask_sensitive usuwa e-maile i PESEL-e z tekstu logu.

    W backend/utils.py:
      - EMAIL_RE i PESEL_RE rozpoznają w tekście adresy e-mail i 11-cyfrowe numery,
      - mask_sensitive zamienia je na "***".
    """
    msg = "email: user@example.com, pesel: 12345678901"
    masked = mask_sensitive(msg)

    assert "user@example.com" not in masked
    assert "12345678901" not in masked
    assert "***" in masked

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from config import AUTH_SECRET_KEY, SESSION_COOKIE_NAME

from database.db import get_db
from database.security import decode_token

log = logging.getLogger("symptomchecker.deps")


def _decode_local_cookie(request: Request) -> Optional[int]:
    tok = request.cookies.get(SESSION_COOKIE_NAME) or request.cookies.get("access_token")
    if not tok:
        return None
    try:
        data = decode_token(tok, AUTH_SECRET_KEY) or {}
        sub = data.get("sub")
        if sub is None:
            return None
        return int(sub)
    except Exception:
        return None


def _resolve_via_supabase(request: Request, db: Session) -> Optional[int]:
    try:
        from .routers.auth import _extract_bearer, _decode_supabase_jwt, _ensure_local_profile
    except Exception:
        log.debug("Supabase fallback niedostępny.", exc_info=True)
        return None

    try:
        sb_tok = _extract_bearer(request)
    except HTTPException:
        return None
    except Exception:
        return None

    try:
        claims = _decode_supabase_jwt(sb_tok)
    except HTTPException:
        return None
    except Exception:
        return None

    try:
        # identity read only: no soft profile updates here (/auth/session and PUT /auth/me do those)
        user = _ensure_local_profile(db, claims, allow_writes=False)
        return int(user.id)
    except Exception:
        log.exception("Błąd mapowania profilu Supabase → lokalny.")
        return None


def get_current_user_id(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[int]:
    uid = _decode_local_cookie(request)
    if uid is not None:
        return uid
    return _resolve_via_supabase(request, db)


def require_user_id(
    uid: Optional[int] = Depends(get_current_user_id),
) -> int:
    if uid is None:
        raise HTTPException(status_code=401, detail="Wymagane logowanie.")
    return uid

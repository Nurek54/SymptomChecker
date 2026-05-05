from __future__ import annotations
import os
import time
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Dict, Any
import json

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from database.db import get_db
from database.schemas import UserOut, UpdateProfileIn
from database import crud
from database.models import User as UserModel

from config import (
    AUTH_SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES,
    SESSION_COOKIE_NAME, SESSION_COOKIE_SECURE, SESSION_COOKIE_SAMESITE,
)

router = APIRouter(prefix="/auth", tags=["Auth"])
log = logging.getLogger("api.auth")

# =========================
#   KONFIG (HS256 only)
# =========================
SUPABASE_JWT_SECRET = (os.getenv("SUPABASE_JWT_SECRET") or "").strip()
if not SUPABASE_JWT_SECRET:
    log.error("Brak SUPABASE_JWT_SECRET – weryfikacja JWT Supabase nie zadziała (HS256).")

AUTH_SECRET = AUTH_SECRET_KEY  # sekret do backendowego cookie
COOKIE_NAME = SESSION_COOKIE_NAME

# =========================
#   PREWARM (rozgrzewka HS256)
# =========================
_auth_prewarmed = False

def prewarm_auth() -> None:
    """
    „Rozgrzewa” ścieżkę kryptograficzną HS256, żeby pierwszy request nie miał nawet
    minimalnego overheadu importów/kompilacji. Zero sieci, zero I/O.
    """
    global _auth_prewarmed
    if _auth_prewarmed or not SUPABASE_JWT_SECRET:
        _auth_prewarmed = True
        return
    try:
        now = int(time.time())
        # sztuczny, krótkowieczny token testowy
        probe = jwt.encode(
            {"sub": "0", "iat": now, "exp": now + 60},
            SUPABASE_JWT_SECRET,
            algorithm="HS256",
        )
        # i od razu go zdekoduj (rozgrzanie ścieżki decode)
        _ = jwt.decode(
            probe,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        _auth_prewarmed = True
        log.info("Auth HS256 prewarmed.")
    except Exception as e:
        # nawet jeśli prewarm się nie uda, normalny decode dalej zadziała
        log.debug("Auth HS256 prewarm skipped: %s", e)
        _auth_prewarmed = True

# =========================
#   Helpers (Bearer, wiek, token cookie)
# =========================
def _extract_bearer(request: Request) -> str:
    """
    Pobiera access_token Supabase z:
      - Authorization: Bearer <token>
      - cookies: sb-access-token / supabase-auth-token (Supabase JS trzyma tam JSON)
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    for k in ("sb-access-token", "supabase-auth-token"):
        v = request.cookies.get(k)
        if not v:
            continue
        try:
            data = json.loads(v)
            tok = data.get("access_token") or data.get("currentSession", {}).get("access_token")
            if tok:
                return tok
        except Exception:
            # nie JSON – może sam token
            return v

    raise HTTPException(401, "Brak tokenu Supabase (Authorization: Bearer <token>).")

def _decode_supabase_jwt(token: str) -> dict:
    """
    Weryfikuje token Supabase **wyłącznie** HS256 (SUPABASE_JWT_SECRET).
    """
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(500, "Błędna konfiguracja serwera (brak SUPABASE_JWT_SECRET).")
    try:
        return jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except JWTError as e:
        log.info("HS256 decode failed: %s", e)
        raise HTTPException(401, "Nieprawidłowy lub wygasły token Supabase.")

def _age(dob: Optional[date]) -> Optional[int]:
    if not dob:
        return None
    today = date.today()
    a = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return max(a, 0)

def _mint_backend_token(user_id: int) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, AUTH_SECRET, algorithm="HS256")

def _ensure_local_profile(db: Session, claims: dict) -> UserModel:
    """
    Mapuje konto Supabase na rekord w Twojej tabeli users:
      1) po (auth_provider='supabase', provider_sub=sub),
      2) fallback po email,
      3) jeśli brak → tworzymy usera (password_hash=NULL),
      4) miękko uzupełniamy puste pola z user_metadata (first/last/dob/username).
    """
    sub = str(claims.get("sub") or "").strip()
    meta = claims.get("user_metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    email = (claims.get("email") or meta.get("email") or "").lower().strip()
    full_name = (meta.get("full_name") or meta.get("name") or "").strip()
    username_meta = (meta.get("username") or meta.get("preferred_username") or "").strip()

    first_name_meta = (meta.get("first_name") or "").strip() or None
    last_name_meta  = (meta.get("last_name") or "").strip() or None
    dob_meta_raw    = (meta.get("date_of_birth") or "").strip() or None
    dob_meta: Optional[date] = None
    if dob_meta_raw:
        try:
            y, m, d = map(int, dob_meta_raw.split("-"))
            dob_meta = date(y, m, d)
        except Exception:
            dob_meta = None

    # 1) provider_sub
    user = crud.get_user_by_provider(db, "supabase", sub) if sub else None

    # 2) email fallback
    if not user and email:
        user = crud.get_user_by_email(db, email)

    # 3) create
    if not user:
        base_for_username = username_meta or (email.split("@")[0] if email else (full_name or "user"))
        username = crud.generate_username(db, base_for_username)
        user = crud.create_user(
            db,
            username=username,
            email=email or (f"{sub}@users.nologin" if sub else "unknown@users.nologin"),
            password=None,
            auth_provider="supabase",
            provider_sub=sub or None,
            first_name=first_name_meta,
            last_name=last_name_meta,
            date_of_birth=dob_meta,
        )
        db.commit()
        log.info("Utworzono lokalny profil usera (supabase sub=%s, email=%s, id=%s)", sub, email, user.id)
    else:
        changed = False
        if not user.provider_sub and sub:
            user.auth_provider = "supabase"
            user.provider_sub = sub
            changed = True

        # miękka synchronizacja pustych pól z meta
        if first_name_meta and not (user.first_name or "").strip():
            user.first_name = first_name_meta; changed = True
        if last_name_meta and not (user.last_name or "").strip():
            user.last_name = last_name_meta; changed = True
        if dob_meta and not getattr(user, "date_of_birth", None):
            user.date_of_birth = dob_meta; changed = True

        # ewentualna korekta username (jeśli był auto)
        if username_meta and user.username.startswith("user") and user.username != username_meta:
            cand = crud.generate_username(db, username_meta)
            if cand != user.username:
                user.username = cand; changed = True

        if changed:
            db.add(user)
            db.commit()
            log.info("Zaktualizowano lokalny profil (id=%s) z claimów Supabase.", user.id)

    return user

# =========================
#   Endpoints
# =========================
@router.post("/session", response_model=UserOut)
def exchange_session(request: Request, response: Response, db: Session = Depends(get_db)):
    """
    Wymienia Supabase access_token (Bearer) na backendowe cookie HTTP-only (sc_session).
    """
    sb_token = _extract_bearer(request)
    claims = _decode_supabase_jwt(sb_token)
    user = _ensure_local_profile(db, claims)

    token = _mint_backend_token(user.id)
    response.set_cookie(
        COOKIE_NAME, token,
        httponly=True, secure=SESSION_COOKIE_SECURE,
        samesite=SESSION_COOKIE_SAMESITE, path="/",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    out = UserOut.model_validate(user, from_attributes=True)
    out.age = _age(getattr(user, "date_of_birth", None))
    return out

@router.get("/me", response_model=UserOut)
def me(request: Request, db: Session = Depends(get_db)):
    sb_token = _extract_bearer(request)
    claims = _decode_supabase_jwt(sb_token)
    user = _ensure_local_profile(db, claims)

    out = UserOut.model_validate(user, from_attributes=True)
    out.age = _age(getattr(user, "date_of_birth", None))
    return out

@router.put("/me", response_model=UserOut)
def update_me(data: UpdateProfileIn, request: Request, db: Session = Depends(get_db)):
    sb_token = _extract_bearer(request)
    claims = _decode_supabase_jwt(sb_token)
    user = _ensure_local_profile(db, claims)

    user = crud.update_user_profile(
        db,
        user,
        username=getattr(data, "username", None),
        first_name=getattr(data, "first_name", None),
        last_name=getattr(data, "last_name", None),
        date_of_birth=getattr(data, "date_of_birth", None),
    )
    db.commit()

    out = UserOut.model_validate(user, from_attributes=True)
    out.age = _age(getattr(user, "date_of_birth", None))
    return out

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}

# backend/api/routers/history.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import os

from database.db import get_db
from database.schemas import ConversationItem, ConversationDetail, MessageOut, SymptomOut, DiagnosisOut
from database import crud
from database.security import decode_token

from config import SESSION_COOKIE_NAME, AUTH_SECRET_KEY

# <<< KLUCZOWE: importujemy te same helpery z auth.py >>>
from .auth import _extract_bearer, _decode_supabase_jwt, _ensure_local_profile

router = APIRouter(prefix="/history", tags=["History"])

AUTH_SECRET = AUTH_SECRET_KEY or os.environ.get("AUTH_SECRET_KEY", "dev-secret-change-me")
COOKIE_NAME = SESSION_COOKIE_NAME or os.environ.get("SESSION_COOKIE_NAME", "access_token")

def _uid(request: Request, db: Session) -> int | None:
    """Najpierw cookie backendu, a jeśli brak – Bearer z Supabase."""
    tok = request.cookies.get(COOKIE_NAME) or request.cookies.get("access_token")
    if tok:
        data = decode_token(tok, AUTH_SECRET) or {}
        try:
            return int(data.get("sub"))
        except Exception:
            pass
    # fallback: Bearer
    try:
        sb_tok = _extract_bearer(request)
        claims = _decode_supabase_jwt(sb_tok)
        user = _ensure_local_profile(db, claims)
        return int(user.id)
    except Exception:
        return None

@router.get("", response_model=list[ConversationItem])
def list_history(request: Request, db: Session = Depends(get_db)):
    uid = _uid(request, db)
    if not uid:
        raise HTTPException(401, "Zaloguj się.")
    rows = crud.list_conversations_for_user(db, uid)
    return [
        ConversationItem(
            id=c.id, title=c.title, last_message_at=c.last_message_at, message_count=cnt
        ) for c, cnt in rows
    ]

@router.get("/{conv_id}", response_model=ConversationDetail)
def get_history(conv_id: int, request: Request, db: Session = Depends(get_db)):
    uid = _uid(request, db)
    if not uid:
        raise HTTPException(401, "Zaloguj się.")
    conv, msgs = crud.get_conversation_with_messages(db, uid, conv_id)
    if not conv:
        raise HTTPException(404, "Nie znaleziono rozmowy.")

    syms, dias = crud.get_summary_for_conversation(db, conv.id)
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        messages=[MessageOut(role=m.role, content=m.content, created_at=m.created_at) for m in msgs],
        symptoms=[SymptomOut(text=s.text, canonical=s.canonical, lang=s.lang, score=s.score) for s in syms],
        diagnoses=[DiagnosisOut(label=d.label, score=d.score, rank=d.rank) for d in dias],
    )

@router.delete("/{conv_id}")
def delete_history(conv_id: int, request: Request, db: Session = Depends(get_db)):
    uid = _uid(request, db)
    if not uid:
        raise HTTPException(401, "Zaloguj się.")
    ok = crud.delete_conversation(db, uid, conv_id)
    if not ok:
        raise HTTPException(404, "Nie znaleziono rozmowy.")
    db.commit()
    return {"ok": True}

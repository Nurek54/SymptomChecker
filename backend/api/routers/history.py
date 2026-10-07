# backend/api/routers/history.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.db import get_db
from database.schemas import ConversationItem, ConversationDetail, MessageOut, SymptomOut, DiagnosisOut
from database import crud

# Ta sama logika tożsamości co w /chat i /sessions (cookie backendu, a w razie braku Bearer Supabase)
from api.deps import require_user_id

router = APIRouter(prefix="/history", tags=["History"])


@router.get("", response_model=list[ConversationItem])
def list_history(uid: int = Depends(require_user_id), db: Session = Depends(get_db)):
    rows = crud.list_conversations_for_user(db, uid)
    return [
        ConversationItem(
            id=c.id, title=c.title, last_message_at=c.last_message_at, message_count=cnt
        ) for c, cnt in rows
    ]


@router.get("/{conv_id}", response_model=ConversationDetail)
def get_history(conv_id: int, uid: int = Depends(require_user_id), db: Session = Depends(get_db)):
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
def delete_history(conv_id: int, uid: int = Depends(require_user_id), db: Session = Depends(get_db)):
    ok = crud.delete_conversation(db, uid, conv_id)
    if not ok:
        raise HTTPException(404, "Nie znaleziono rozmowy.")
    db.commit()
    return {"ok": True}

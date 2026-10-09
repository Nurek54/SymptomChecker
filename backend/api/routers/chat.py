from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List
from datetime import date

from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session

from schemas import ChatRequest, ChatResponse
from services.chat_service import ChatService, LANG
from services.prompt_localizer import PromptLocalizer

from database.db import get_db
from database import crud
from database.models import User as UserModel

from api.deps import get_current_user_id

router = APIRouter(tags=["Chat"])
log = logging.getLogger("symptomchecker.chat")


def _resp_to_dict(resp: ChatResponse | Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:
            pass
    if isinstance(resp, dict):
        return resp
    return {}


def _age(dob: date | None) -> int | None:
    if not dob:
        return None
    today = date.today()
    return max(0, today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)))


# Plain "def" on purpose: LLM generation takes seconds and in "async def" would block the
# event loop and the whole server (/health, history, symptom panel). FastAPI runs "def"
# in a thread pool; LLMManager has its own generation lock.
@router.post("/chat", response_model=ChatResponse)
def chat(
    req_body: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    uid: Optional[int] = Depends(get_current_user_id),
):
    st = getattr(request.app.state, "app_state", None)
    if st is None:
        raise HTTPException(status_code=503, detail="Aplikacja w trakcie startu.")

    if not st.sessions.claim_owner(req_body.session_id, uid):
        raise HTTPException(status_code=403, detail="Ta sesja należy do innego użytkownika.")

    try:
        patient_age: int | None = None
        if uid:
            user = db.get(UserModel, uid)
            if user:
                patient_age = _age(getattr(user, "date_of_birth", None))

        resp: ChatResponse = ChatService(st).handle_chat(req_body, patient_age=patient_age)

        # Persist user message and reply together, only after a successful reply.
        if uid:
            rd = _resp_to_dict(resp)
            title_hint = (req_body.user_input or "").strip()[:80] or "Rozmowa"
            conv = crud.get_or_create_conversation(
                db,
                user_id=uid,
                session_id=req_body.session_id,
                title_hint=title_hint,
            )
            crud.add_message(db, conv, "user", req_body.user_input)
            crud.add_message(db, conv, "assistant", rd.get("response") or "")

            loc = PromptLocalizer(st.matcher)
            symptoms_sorted = sorted({str(s).strip().lower() for s in (rd.get("symptoms") or []) if s})
            pl_labels = loc.localize_symptoms(symptoms_sorted, LANG)
            crud.replace_extractions(db, conv.id, [
                {"pl": pl, "canonical": en, "lang": LANG}
                for en, pl in zip(symptoms_sorted, pl_labels)
            ])

            # Triage always mirrors the current reply (empty list = no assessment yet).
            top: List[Dict[str, Any]] = rd.get("top_diagnoses") or []
            crud.replace_triage(db, conv.id, [
                {
                    "diagnosis_pl": d.get("diagnosis_pl"),
                    "diagnosis_en": d.get("diagnosis_en"),
                    "probability": d.get("probability"),
                    "rank": i + 1,
                }
                for i, d in enumerate(top)
            ])

            crud.smart_update_conversation_title(db, conv.id)
            db.commit()

        return resp

    except HTTPException:
        raise
    except Exception:
        log.exception("Błąd /chat (session=%s)", req_body.session_id)
        try:
            db.rollback()
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Nie udało się przygotować odpowiedzi. Spróbuj ponownie za chwilę.",
        )

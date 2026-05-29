from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import date

from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session

from schemas import ChatRequest, ChatResponse
from services.chat_service import ChatService
from services.prompt_localizer import PromptLocalizer

from database.db import get_db
from database import crud
from database.models import User as UserModel

# FIX #8: zamiast lokalnego _uid() używamy wspólnego dependency.
from api.deps import get_current_user_id

router = APIRouter(tags=["Chat"])


def _resp_to_dict(resp: ChatResponse | Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:
            pass
    if hasattr(resp, "dict"):
        try:
            return resp.dict()
        except Exception:
            pass
    if isinstance(resp, dict):
        return resp
    return {}


def _age(dob: date | None) -> int | None:
    if not dob:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req_body: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    uid: Optional[int] = Depends(get_current_user_id),
):
    st = getattr(request.app.state, "app_state", None)
    if st is None:
        raise HTTPException(status_code=503, detail="Aplikacja w trakcie startu.")

    try:
        service = ChatService(st)

        patient_age: int | None = None
        conv = None

        if uid:
            user = db.get(UserModel, uid)
            if user:
                dob = getattr(user, "date_of_birth", None)
                patient_age = _age(dob)

            title_hint = (req_body.user_input or "").strip()[:80] or "Rozmowa"
            conv = crud.get_or_create_conversation(
                db,
                user_id=uid,
                session_id=req_body.session_id,
                title_hint=title_hint,
            )
            crud.add_message(db, conv, "user", req_body.user_input)
            db.commit()

        # odpowiedź
        resp: ChatResponse = service.handle_chat(req_body, patient_age=patient_age)

        # snapshot + smart tytuł
        if uid and conv:
            rd = _resp_to_dict(resp)
            reply_text = rd.get("response") or ""
            symptoms: List[str] = rd.get("symptoms") or []
            top: List[Dict[str, Any]] = (
                rd.get("top_diagnoses")
                or rd.get("diagnoses")
                or rd.get("top_diagnosis")
                or []
            )

            crud.add_message(db, conv, "assistant", reply_text)

            loc = PromptLocalizer(st.matcher)
            lang = "pl"
            symptoms_sorted = sorted({str(s).strip().lower() for s in (symptoms or []) if s})
            pl_labels = loc.localize_symptoms(symptoms_sorted, lang)

            extraction_items = [
                {"pl": pl, "canonical": en, "lang": lang}
                for en, pl in zip(symptoms_sorted, pl_labels)
            ]
            crud.replace_extractions(db, conv.id, extraction_items)

            triage_items = []
            for i, d in enumerate(top or []):
                triage_items.append(
                    {
                        "diagnosis_pl": d.get("diagnosis_pl"),
                        "diagnosis_en": d.get("diagnosis_en"),
                        "probability": d.get("probability"),
                        "rank": i + 1,
                    }
                )
            crud.replace_triage(db, conv.id, triage_items)

            crud.smart_update_conversation_title(db, conv.id)
            db.commit()

        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
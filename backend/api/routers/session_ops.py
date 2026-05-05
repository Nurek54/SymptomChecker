from __future__ import annotations

from typing import List, Dict, Any, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.state import AppState
from services.prompt_localizer import PromptLocalizer
from services.ensemble import EnsembleTopK
from config import MIN_SYMPTOMS, AUTH_SECRET_KEY, SESSION_COOKIE_NAME

from database.db import get_db
from database import crud
from database.security import decode_token

# <<< DODANE: helpery z /auth >>>
from .auth import _extract_bearer, _decode_supabase_jwt, _ensure_local_profile

router = APIRouter(prefix="/sessions", tags=["SessionOps"])


# =========================================================
# Helper: dostęp do AppState
# =========================================================
def get_state(req: Request) -> AppState:
    st = getattr(req.app.state, "app_state", None)
    if st is None:
        raise HTTPException(500, "Brak AppState w aplikacji.")
    return st


# =========================================================
# Helper: identyfikacja użytkownika (cookie / bearer)
# =========================================================
def _uid(req: Request, db: Session) -> Optional[int]:
    # 1) lokalne JWT w cookie
    tok = req.cookies.get(SESSION_COOKIE_NAME) or req.cookies.get("access_token")
    if tok:
        data = decode_token(tok, AUTH_SECRET_KEY or "dev-secret-change-me") or {}
        try:
            return int(data.get("sub"))
        except Exception:
            pass

    # 2) fallback – Supabase Bearer
    try:
        sb_tok = _extract_bearer(req)
        claims = _decode_supabase_jwt(sb_tok)
        user = _ensure_local_profile(db, claims)
        return int(user.id)
    except Exception:
        return None


# =========================================================
# Schemy Pydantic
# =========================================================
class SymptomsPatch(BaseModel):
    """
    Patch na listę objawów w sesji.
    - replace: nadpisuje całą listę
    - add: dodaje objawy
    - remove: usuwa objawy
    - lang: język do lokalizacji (pl/en)
    """
    add: List[str] = Field(default_factory=list)
    remove: List[str] = Field(default_factory=list)
    replace: Optional[List[str]] = None
    lang: Optional[str] = "pl"


class SymptomsPatchResponse(BaseModel):
    symptoms: List[str]
    top_diagnoses: List[Dict[str, Any]] = Field(default_factory=list)
    diagnosis_en: Optional[str] = None
    diagnosis_pl: Optional[str] = None


class TopItem(BaseModel):
    diagnosis_en: str
    probability: float
    diagnosis_pl: Optional[str] = None


class Top3SplitResponse(BaseModel):
    xgb_enabled: bool
    river_enabled: bool
    symptoms: List[str]
    xgb_top: List[TopItem] = Field(default_factory=list)
    river_top: List[TopItem] = Field(default_factory=list)
    ensemble_top: List[TopItem] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


# =========================================================
# Helper: wektor cech do modeli
# =========================================================
def _make_feature_vector(st: AppState, selected_canons: List[str]) -> List[int]:
    """
    Buduje wektor 0/1 na bazie listy kanonicznych objawów (EN).
    """
    if not st or not st.xgb or not st.matcher:
        raise HTTPException(503, "Model lub matcher niegotowy.")

    feats = [str(f).strip().lower() for f in (st.xgb.features or [])]

    expanded: Set[str] = set()
    for s in (selected_canons or []):
        try:
            expanded |= set(st.matcher.expand_to_xgb(s) or [])
        except Exception:
            expanded.add(str(s).strip().lower())

    return [1 if f in expanded else 0 for f in feats]


# =========================================================
# PATCH /sessions/{session_id}/symptoms
# =========================================================
@router.post("/{session_id}/symptoms", response_model=SymptomsPatchResponse)
def patch_session_symptoms(
    session_id: str,
    patch: SymptomsPatch,
    request: Request,
    st: AppState = Depends(get_state),
    db: Session = Depends(get_db),
):
    sessions = st.sessions
    xgb = st.xgb
    if not xgb or not xgb.model:
        raise HTTPException(500, "Model XGB niezaładowany.")

    # Inicjalizacja sesji jeśli trzeba
    if not sessions.exists(session_id):
        sessions.init_session(session_id)

    # --- kanonizacja + filtr znanych objawów ---
    def _canon(s: str) -> str:
        return str(s or "").strip().lower()

    def canon_all(items: List[str]) -> List[str]:
        return [_canon(s) for s in (items or []) if _canon(s)]

    # znane cechy z modelu (feature space)
    known_canons: Set[str] = {
        _canon(f) for f in (getattr(xgb, "features", None) or [])
        if _canon(f)
    }

    def filter_known(items: List[str]) -> List[str]:
        """
        Filtruje objawy do takich, które są w zbiorze cech modelu.
        Jeśli known_canons jest puste (teoretycznie), nic nie filtrujemy.
        """
        if not known_canons:
            return items
        return [s for s in items if s in known_canons]

    # --- aktualny stan sesji ---
    cur = set(_canon(s) for s in (sessions.get_symptoms(session_id) or []))

    # --- logika patch ---
    if patch.replace is not None:
        # replace = nadpisanie listy, ale tylko znane objawy
        replaced = canon_all(patch.replace or [])
        cur = set(filter_known(replaced))
    else:
        # add – tylko znane objawy
        for s in filter_known(canon_all(patch.add or [])):
            cur.add(s)

        # remove – usuń nawet jeśli remove zawiera nieznane, po prostu je ignorujemy
        for s in canon_all(patch.remove or []):
            if s in cur:
                cur.remove(s)

    # --- nowa lista objawów w sesji ---
    new_symptoms_sorted = sorted(cur)
    sessions.set_symptoms(session_id, new_symptoms_sorted)

    lang = (patch.lang or "pl").lower()
    loc = PromptLocalizer(st.matcher)

    # =====================================================
    # Za mało objawów → tylko zapis ekstrakcji, brak top-3
    # =====================================================
    if len(cur) < int(MIN_SYMPTOMS):
        sessions.set_last_top3(session_id, [])

        uid = _uid(request, db)
        if uid:
            conv = crud.get_or_create_conversation(
                db,
                user_id=uid,
                session_id=session_id,
                title_hint="Rozmowa",
            )
            pl_labels = loc.localize_symptoms(new_symptoms_sorted, lang)
            extraction_items = [
                {"pl": pl, "canonical": en, "lang": lang}
                for en, pl in zip(new_symptoms_sorted, pl_labels)
            ]
            crud.replace_extractions(db, conv.id, extraction_items)
            crud.replace_triage(db, conv.id, [])
            crud.smart_update_conversation_title(db, conv.id)
            db.commit()

        return SymptomsPatchResponse(
            symptoms=new_symptoms_sorted,
            top_diagnoses=[],
            diagnosis_en=None,
            diagnosis_pl=None,
        )

    # =====================================================
    # Wystarczająco objawów → liczymy top-3
    # =====================================================
    vec = [_make_feature_vector(st, new_symptoms_sorted)]

    ens = EnsembleTopK(st)
    try:
        raw_top = ens.predict_topk([vec[0]], k=3)
    except Exception:
        raw_top = []

    items = raw_top or []
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = items[0]

    top: List[Dict[str, Any]] = []
    for d in items:
        if isinstance(d, dict):
            diag = d.get("diagnosis_en") or d.get("diagnosis") or d.get("label") or str(d)
            prob = d.get("probability", d.get("prob", d.get("score", 0.0)))
            try:
                prob = float(prob)
            except Exception:
                prob = 0.0
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            diag, prob = str(d[0]), float(d[1])
        else:
            diag, prob = str(d), 0.0

        top.append(
            {
                "diagnosis_en": _canon(diag),
                "probability": prob,
            }
        )

    # lokalizacja PL dla top-3
    top_with_pl: List[Dict[str, Any]] = []
    for d in top:
        en_name = d.get("diagnosis_en") or ""
        dd = dict(d)
        dd["diagnosis_pl"] = loc.localize_disease(en_name, lang)
        top_with_pl.append(dd)

    # zapamiętujemy w sesji listę diagnoz EN
    sessions.set_last_top3(session_id, [d["diagnosis_en"] for d in top])

    diagnosis_en = top[0]["diagnosis_en"] if top else None
    diagnosis_pl = loc.localize_disease(diagnosis_en, lang) if diagnosis_en else None

    # zapis do bazy (ekstrakcje + triage)
    uid = _uid(request, db)
    if uid:
        conv = crud.get_or_create_conversation(
            db,
            user_id=uid,
            session_id=session_id,
            title_hint="Rozmowa",
        )

        pl_labels = loc.localize_symptoms(new_symptoms_sorted, lang)
        extraction_items = [
            {"pl": pl, "canonical": en, "lang": lang}
            for en, pl in zip(new_symptoms_sorted, pl_labels)
        ]
        crud.replace_extractions(db, conv.id, extraction_items)

        triage_items = [
            {
                "diagnosis_pl": it.get("diagnosis_pl"),
                "diagnosis_en": it.get("diagnosis_en"),
                "probability": it.get("probability"),
                "rank": i + 1,
            }
            for i, it in enumerate(top_with_pl)
        ]
        crud.replace_triage(db, conv.id, triage_items)
        crud.smart_update_conversation_title(db, conv.id)
        db.commit()

    return SymptomsPatchResponse(
        symptoms=new_symptoms_sorted,
        top_diagnoses=top_with_pl,
        diagnosis_en=diagnosis_en,
        diagnosis_pl=diagnosis_pl,
    )


# =========================================================
# GET /sessions/{session_id}/top3-split
# =========================================================
@router.get("/{session_id}/top3-split", response_model=Top3SplitResponse)
def get_top3_split(
    session_id: str,
    k: int = 3,
    lang: str = "pl",
    request: Request = None,
    st: AppState = Depends(get_state),
):
    sessions = st.sessions
    if not sessions.exists(session_id):
        sessions.init_session(session_id)

    selected = sessions.get_symptoms(session_id) or []
    vec = _make_feature_vector(st, selected)
    ens = EnsembleTopK(st)

    xgb_enabled = bool(getattr(st, "xgb", None) and getattr(st.xgb, "model", None))
    river_enabled = bool(getattr(st, "river", None) and getattr(st.river, "model", None))

    split = ens.predict_split([vec], k=max(1, int(k)))
    loc = PromptLocalizer(st.matcher)
    lang = (lang or "pl").lower()

    def _with_pl(lst: List[Dict[str, Any]]) -> List[TopItem]:
        out: List[TopItem] = []
        for d in lst or []:
            en = str(d.get("diagnosis_en") or "").strip().lower()
            out.append(
                TopItem(
                    diagnosis_en=en,
                    diagnosis_pl=loc.localize_disease(en, lang),
                    probability=float(d.get("probability", 0.0) or 0.0),
                )
            )
        return out

    return Top3SplitResponse(
        xgb_enabled=xgb_enabled,
        river_enabled=river_enabled,
        symptoms=selected,
        xgb_top=_with_pl(split.get("xgb_top") or []),
        river_top=_with_pl(split.get("river_top") or []),
        ensemble_top=_with_pl(split.get("ensemble_top") or []),
        meta=split.get("meta") or {},
    )

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.state import AppState
from services.prompt_localizer import PromptLocalizer
from services.ensemble import EnsembleTopK
from config import MIN_SYMPTOMS, TOPN_DIAGNOSES

from database.db import get_db
from database import crud

from api.deps import get_current_user_id

router = APIRouter(prefix="/sessions", tags=["SessionOps"])


# =========================================================
# Helper: dostęp do AppState
# =========================================================
def get_state(req: Request) -> AppState:
    st = getattr(req.app.state, "app_state", None)
    if st is None:
        raise HTTPException(503, "Aplikacja w trakcie startu.")
    return st


# =========================================================
# Schemy Pydantic
# =========================================================
class SymptomsPatch(BaseModel):
    add: List[str] = Field(default_factory=list)
    remove: List[str] = Field(default_factory=list)
    replace: Optional[List[str]] = None
    lang: Optional[str] = "pl"
    # Objawy, które pamięta frontend - używane tylko, gdy backend nie zna sesji
    # (np. po restarcie serwera), żeby "dodaj objaw" nie kasowało pozostałych.
    current: Optional[List[str]] = None


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
# Helper: wektor cech
# =========================================================
def _make_feature_vector(st: AppState, selected_canons: List[str]) -> List[int]:
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
# POST /sessions/{session_id}/symptoms
# =========================================================
@router.post("/{session_id}/symptoms", response_model=SymptomsPatchResponse)
def patch_session_symptoms(
    session_id: str,
    patch: SymptomsPatch,
    st: AppState = Depends(get_state),
    db: Session = Depends(get_db),
    uid: Optional[int] = Depends(get_current_user_id),
):
    sessions = st.sessions
    xgb = st.xgb
    if not xgb or not xgb.model:
        raise HTTPException(503, "Model XGB niezaładowany.")

    def _canon(s: str) -> str:
        return str(s or "").strip().lower()

    def canon_all(items: List[str]) -> List[str]:
        return [_canon(s) for s in (items or []) if _canon(s)]

    known_canons: Set[str] = {
        _canon(f) for f in (getattr(xgb, "features", None) or [])
        if _canon(f)
    }

    def filter_known(items: List[str]) -> List[str]:
        if not known_canons:
            return items
        return [s for s in items if s in known_canons]

    if not sessions.exists(session_id):
        sessions.init_session(session_id)
        restored = filter_known(canon_all(patch.current or []))
        if restored:
            sessions.set_symptoms(session_id, restored)

    if not sessions.claim_owner(session_id, uid):
        raise HTTPException(403, "Ta sesja należy do innego użytkownika.")

    cur = set(_canon(s) for s in (sessions.get_symptoms(session_id) or []))

    if patch.replace is not None:
        replaced = canon_all(patch.replace or [])
        cur = set(filter_known(replaced))
        if not cur:
            sessions.reset_flow(session_id)  # "wyczyść wszystko" = nowa ocena od zera
    else:
        for s in filter_known(canon_all(patch.add or [])):
            cur.add(s)
        for s in canon_all(patch.remove or []):
            if s in cur:
                cur.remove(s)

    new_symptoms_sorted = sorted(cur)
    for s in canon_all(patch.remove or []):
        if s not in cur:
            sessions.add_negated_symptom(session_id, s)  # usunięty ręcznie = nie pytaj o niego ponownie
    sessions.set_symptoms(session_id, new_symptoms_sorted)

    lang = "pl"  # słowniki i nazwy chorób są polskie
    loc = PromptLocalizer(st.matcher)

    # Za mało objawów → tylko zapis ekstrakcji, brak top-3
    if len(cur) < int(MIN_SYMPTOMS):
        sessions.set_last_top3(session_id, [])

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

    # Wystarczająco objawów → top-3
    vec = [_make_feature_vector(st, new_symptoms_sorted)]

    ens = EnsembleTopK(st)
    try:
        raw_top = ens.predict_topk([vec[0]], k=int(TOPN_DIAGNOSES),
                                   negated=sessions.get_negated(session_id))
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

        top.append({"diagnosis_en": _canon(diag), "probability": prob})

    top_with_pl: List[Dict[str, Any]] = []
    for d in top:
        en_name = d.get("diagnosis_en") or ""
        dd = dict(d)
        dd["diagnosis_pl"] = loc.localize_disease(en_name, lang)
        top_with_pl.append(dd)

    sessions.set_last_top3(session_id, [d["diagnosis_en"] for d in top])

    diagnosis_en = top[0]["diagnosis_en"] if top else None
    diagnosis_pl = loc.localize_disease(diagnosis_en, lang) if diagnosis_en else None

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
    st: AppState = Depends(get_state),
):
    sessions = st.sessions
    if not sessions.exists(session_id):
        # nieznana sesja = brak objawów; nie tworzymy jej przy zwykłym odczycie
        return Top3SplitResponse(
            xgb_enabled=bool(getattr(st, "xgb", None) and getattr(st.xgb, "model", None)),
            river_enabled=bool(getattr(st, "river", None) and getattr(st.river, "model", None)),
            symptoms=[],
        )

    selected = list(sessions.get_symptoms(session_id) or [])
    vec = _make_feature_vector(st, selected)
    ens = EnsembleTopK(st)

    xgb_enabled = bool(getattr(st, "xgb", None) and getattr(st.xgb, "model", None))
    river_enabled = bool(getattr(st, "river", None) and getattr(st.river, "model", None))

    split = ens.predict_split([vec], k=max(1, min(int(k), 20)), negated=sessions.get_negated(session_id))
    loc = PromptLocalizer(st.matcher)
    lang = "pl"

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
        symptoms=sorted(selected),
        xgb_top=_with_pl(split.get("xgb_top") or []),
        river_top=_with_pl(split.get("river_top") or []),
        ensemble_top=_with_pl(split.get("ensemble_top") or []),
        meta=split.get("meta") or {},
    )

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Request

from core.state import AppState
from config import SYMPTOMS_DICT_PATH

router = APIRouter(prefix="/dicts", tags=["Dicts"])


def _canon(s: str) -> str:
    return (s or "").strip().lower()


@router.get("/symptoms", response_model=List[Dict[str, Any]])
def list_symptoms(request: Request, lang: str = "pl"):
    """
    Zwraca listę objawów do pickera przefiltrowaną do kolumn znanych modelowi XGB.
    Uwaga: 'en' zawiera KANON (xgb_label), 'pl' to etykieta do wyświetlania, 'category' – kategoria.
    """
    st: AppState = getattr(request.app.state, "app_state", None)
    if not st or not st.xgb:
        raise HTTPException(503, "Model not loaded")

    feat_set = {str(f).strip().lower() for f in (st.xgb.features or [])}

    p = Path(SYMPTOMS_DICT_PATH).resolve()
    if not p.exists():
        raise HTTPException(500, f"Brak pliku słownika: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8")) or []
    except Exception as e:
        raise HTTPException(500, f"Błąd czytania słownika: {e}")

    if not isinstance(data, list):
        raise HTTPException(500, "Błędny format pliku słownika (oczekiwano listy).")

    out: List[Dict[str, Any]] = []
    for row in data:
        pl = (row.get("pl") or "").strip()
        cat = (row.get("category") or "").strip() or None
        xgb = row.get("xgb_label")

        if isinstance(xgb, str):
            xgb_list = [xgb.strip().lower()] if xgb else []
        else:
            xgb_list = [str(x).strip().lower() for x in (xgb or []) if x]

        # kanon: pierwszy xgb_label, który istnieje w modelu
        canon = next((x for x in xgb_list if x in feat_set), (xgb_list[0] if xgb_list else ""))

        if not canon or canon not in feat_set:
            continue

        out.append({"en": canon, "pl": pl or canon.replace("_", " "), "category": cat})

    out.sort(key=lambda x: ((x.get("pl") or x["en"]).lower(), x["en"].lower()))
    return out

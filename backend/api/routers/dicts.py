from __future__ import annotations
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, HTTPException, Request

from core.state import AppState
from config import SYMPTOMS_DICT_PATH

router = APIRouter(prefix="/dicts", tags=["Dicts"])
log = logging.getLogger("symptomchecker.dicts")


def _canon(s: str) -> str:
    return (s or "").strip().lower()

_dict_cache_lock = threading.Lock()
_dict_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def _load_dict_cached(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise HTTPException(500, f"Brak pliku słownika: {path}")

    try:
        mtime = path.stat().st_mtime
    except OSError as e:
        raise HTTPException(500, f"Nie można odczytać metadanych: {e}")

    key = str(path)
    with _dict_cache_lock:
        cached = _dict_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]

    # poza lockiem czytamy z dysku, żeby nie blokować innych ścieżek
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or []
    except Exception as e:
        raise HTTPException(500, f"Błąd czytania słownika: {e}")

    if not isinstance(data, list):
        raise HTTPException(500, "Błędny format pliku słownika (oczekiwano listy).")

    with _dict_cache_lock:
        _dict_cache[key] = (mtime, data)

    log.info("Dict cache: (re)loaded %s (rows=%d)", path, len(data))
    return data


def _project_to_xgb_features(
    raw: List[Dict[str, Any]],
    feat_set: set[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        pl = (row.get("pl") or "").strip()
        cat = (row.get("category") or "").strip() or None
        xgb = row.get("xgb_label")

        if isinstance(xgb, str):
            xgb_list = [xgb.strip().lower()] if xgb else []
        else:
            xgb_list = [str(x).strip().lower() for x in (xgb or []) if x]

        canon: Optional[str] = next(
            (x for x in xgb_list if x in feat_set),
            (xgb_list[0] if xgb_list else ""),
        )

        if not canon or canon not in feat_set:
            continue

        out.append({"en": canon, "pl": pl or canon.replace("_", " "), "category": cat})

    out.sort(key=lambda x: ((x.get("pl") or x["en"]).lower(), x["en"].lower()))
    return out


@router.get("/symptoms", response_model=List[Dict[str, Any]])
def list_symptoms(request: Request, lang: str = "pl"):
    st: AppState = getattr(request.app.state, "app_state", None)
    if not st or not st.xgb:
        raise HTTPException(503, "Model not loaded")

    feat_set = {str(f).strip().lower() for f in (st.xgb.features or [])}
    raw = _load_dict_cached(Path(SYMPTOMS_DICT_PATH).resolve())
    return _project_to_xgb_features(raw, feat_set)

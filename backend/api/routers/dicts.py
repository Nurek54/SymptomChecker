from __future__ import annotations
import json
import logging
import threading
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple
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
    # Details (paths, exception text) go only to the server log; the client gets a generic message.
    # Cache is keyed by path and invalidated when the file mtime changes.
    if not path.exists():
        log.error("Brak pliku słownika: %s", path)
        raise HTTPException(500, "Słownik objawów jest niedostępny.")

    try:
        mtime = path.stat().st_mtime
    except OSError:
        log.exception("Nie można odczytać metadanych słownika %s", path)
        raise HTTPException(500, "Słownik objawów jest niedostępny.")

    key = str(path)
    with _dict_cache_lock:
        cached = _dict_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]

    # read from disk outside the lock so other paths are not blocked
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or []
    except Exception:
        log.exception("Błąd czytania słownika %s", path)
        raise HTTPException(500, "Słownik objawów jest uszkodzony.")

    if not isinstance(data, list):
        log.error("Błędny format słownika %s (oczekiwano listy)", path)
        raise HTTPException(500, "Słownik objawów jest uszkodzony.")

    with _dict_cache_lock:
        _dict_cache[key] = (mtime, data)

    log.info("Dict cache: (re)loaded %s (rows=%d)", path, len(data))
    return data


def _project_to_xgb_features(
    raw: List[Dict[str, Any]],
    feat_set: set[str],
    resolve: Optional[Callable[[List[str]], Optional[str]]] = None,
    count: Optional[Callable[[str], int]] = None,
) -> List[Dict[str, Any]]:
    """
    Map dictionary entries to model features. `resolve` (SymptomMatcher.resolve_labels) picks
    a feature that has training data, so the panel never shows symptoms the model knows nothing about.

    Output has exactly one row per feature ("en" is unique). When several entries map to the
    same feature ("ból jamy ustnej", "ból języka" -> mouth_pain), the entry named like the
    feature becomes the main label and the rest go to "aliases" (the panel searches them too).
    """
    by_feat: Dict[str, Dict[str, Any]] = {}
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

        if resolve is not None:
            canon: Optional[str] = resolve(xgb_list)
        else:
            canon = next((x for x in xgb_list if x in feat_set), None)

        if not canon or canon not in feat_set:
            continue

        label = pl or canon.replace("_", " ")
        # an entry named like the feature ("Mouth pain" -> mouth_pain) is the best main label
        exact = _canon(row.get("en") or "").replace(" ", "_") == canon
        g = by_feat.get(canon)
        if g is None:
            by_feat[canon] = {"en": canon, "pl": label, "category": cat, "aliases": [], "_exact": exact}
            continue
        if exact and not g["_exact"]:
            g["aliases"].insert(0, g["pl"])
            g.update(pl=label, category=cat or g["category"], _exact=True)
        else:
            g["aliases"].append(label)

    # same PL label on several features (e.g. "wzdęcie brzucha"): keep the feature with more data
    by_label: Dict[str, Dict[str, Any]] = {}
    for g in by_feat.values():
        key = g["pl"].lower()
        prev = by_label.get(key)
        if prev is not None and (count is None or count(prev["en"]) >= count(g["en"])):
            continue
        by_label[key] = g

    out: List[Dict[str, Any]] = []
    for g in by_label.values():
        seen = {g["pl"].lower()}
        aliases = []
        for a in g["aliases"]:
            if a.lower() not in seen:
                seen.add(a.lower())
                aliases.append(a)
        out.append({"en": g["en"], "pl": g["pl"], "category": g["category"], "aliases": aliases})

    out.sort(key=lambda x: ((x.get("pl") or x["en"]).lower(), x["en"].lower()))
    return out


@router.get("/symptoms", response_model=List[Dict[str, Any]])
def list_symptoms(request: Request, lang: str = "pl"):
    st: AppState = getattr(request.app.state, "app_state", None)
    if not st or not st.xgb:
        raise HTTPException(503, "Model not loaded")

    feat_set = {str(f).strip().lower() for f in (st.xgb.features or [])}
    raw = _load_dict_cached(Path(SYMPTOMS_DICT_PATH).resolve())
    matcher = getattr(st, "matcher", None)
    resolve = getattr(matcher, "resolve_labels", None)
    count = getattr(matcher, "_count", None)
    return _project_to_xgb_features(raw, feat_set, resolve, count)

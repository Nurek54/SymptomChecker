from __future__ import annotations
import os
import hmac
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body, Request, Query, Header
from sklearn.metrics import accuracy_score, f1_score

from config import DATA_CSV, TEST_CSV, RIVER_MODEL_PATH, ADMIN_API_TOKEN
from models.river_manager import RiverManager, RiverIncrementalTrainer

log = logging.getLogger("symptomchecker.router.river")
router = APIRouter(prefix="/river", tags=["River"])

# Pliki CSV wolno wskazywać wyłącznie z katalogu backend/data
_DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()


# ───────────────────────── helpers ─────────────────────────
def _get_mgr(request: Request) -> RiverManager:
    """
    Zwraca TEN SAM RiverManager, którego używa ensemble w czacie
    (app.state.app_state.river). Nie trenujemy niczego w trakcie requestu.
    """
    st = getattr(request.app.state, "app_state", None)
    mgr = getattr(st, "river", None) if st is not None else None
    if mgr is None or getattr(mgr, "model", None) is None:
        raise HTTPException(503, "Model River nie jest gotowy (trwa start aplikacji albo init się nie powiódł).")
    return mgr


def _require_admin(token: Optional[str]) -> None:
    """Operacje zmieniające model lub czytające pliki wymagają ADMIN_API_TOKEN."""
    if not ADMIN_API_TOKEN:
        raise HTTPException(403, "Endpoint wyłączony: ustaw ADMIN_API_TOKEN w .env, aby go używać.")
    if not token or not hmac.compare_digest(str(token), str(ADMIN_API_TOKEN)):
        raise HTTPException(401, "Brak lub błędny nagłówek X-Admin-Token.")


def _safe_csv_path(csv_path: Optional[str], default: Optional[str]) -> str:
    """Pozwala tylko na pliki .csv wewnątrz backend/data (bez wychodzenia poza katalog)."""
    raw = csv_path or default
    if not raw:
        raise HTTPException(400, "Nie podano pliku CSV.")
    p = Path(raw)
    if not p.is_absolute():
        # ścieżki z configu są względne względem backend/, nazwy z requestu względem backend/data
        cand_backend = (_DATA_DIR.parent / p).resolve()
        cand_data = (_DATA_DIR / p).resolve()
        p = cand_backend if cand_backend.is_relative_to(_DATA_DIR) and cand_backend.exists() else cand_data
    else:
        p = p.resolve()
    if not p.is_relative_to(_DATA_DIR) or p.suffix.lower() != ".csv":
        raise HTTPException(400, "Dozwolone są tylko pliki .csv z katalogu backend/data.")
    if not p.is_file():
        raise HTTPException(404, f"CSV not found: {p.name}")
    return str(p)


def _vec_from_dict(mgr: RiverManager, feats: Dict[str, int]) -> List[List[int]]:
    arr = np.zeros((1, len(mgr.features)), dtype=np.uint8)
    col_index = {f: i for i, f in enumerate(mgr.features)}
    for k, v in feats.items():
        idx = col_index.get(k)
        if idx is not None:
            arr[0, idx] = 1 if int(v) else 0
    return arr.tolist()


def _recall_at_k(y_true: List[str], y_pred_topk: List[List[str]], k: int = 3) -> float:
    if not y_true:
        return 0.0
    hit = sum(1 for yt, preds in zip(y_true, y_pred_topk) if yt in preds[:k])
    return float(hit) / float(len(y_true))


# ───────────────────────── endpoints (odczyt) ─────────────────────────
@router.get("/status")
def status(request: Request):
    mgr = _get_mgr(request)
    return {
        "ready": True,
        "model_path": RIVER_MODEL_PATH,
        "has_file": os.path.exists(RIVER_MODEL_PATH),
        "n_features": len(mgr.features),
        "sample_features": mgr.features[:20],
        "calibrated": mgr.calibrator is not None,
    }


@router.get("/features")
def list_features(request: Request):
    mgr = _get_mgr(request)
    return {"features": mgr.features, "n": len(mgr.features)}


@router.post("/predict_topk")
def predict_topk(
    request: Request,
    payload: Dict[str, Any] = Body(..., examples=[{"active_features": ["fever", "cough"], "k": 3}]),
):
    mgr = _get_mgr(request)
    k = max(1, min(int(payload.get("k", 3)), 50))

    if "active_features" in payload and payload["active_features"]:
        vec = mgr.make_vec_from_symptoms([str(f).strip().lower() for f in payload["active_features"]])
        vec = vec.astype(np.uint8).tolist()
    elif "features" in payload and isinstance(payload["features"], dict):
        vec = _vec_from_dict(mgr, {str(kk): int(v) for kk, v in payload["features"].items()})
    else:
        raise HTTPException(400, "Podaj 'active_features' (lista) lub 'features' (dict).")

    topk = mgr.predict_topk(vec=vec, k=k)
    return {"topk": topk}


# ───────────────────────── endpoints (admin) ─────────────────────────
@router.post("/fit_from_csv")
def fit_from_csv(
    request: Request,
    csv_path: Optional[str] = Body(None),
    limit: Optional[int] = Body(None),
    x_admin_token: Optional[str] = Header(None),
):
    _require_admin(x_admin_token)
    mgr = _get_mgr(request)
    path = _safe_csv_path(csv_path, DATA_CSV)
    try:
        with mgr.lock:
            RiverIncrementalTrainer(mgr).fit_from_csv(path, limit=limit, allow_new_classes=True)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("fit_from_csv failed")
        raise HTTPException(400, f"fit_from_csv failed: {type(e).__name__}") from e
    return {"ok": True, "n_features": len(mgr.features)}


@router.post("/incremental_batch")
def incremental_batch(
    request: Request,
    csv_path: str = Body(...),
    limit: Optional[int] = Body(None),
    x_admin_token: Optional[str] = Header(None),
):
    """Przyrostowe douczenie z pliku CSV (tylko backend/data)."""
    _require_admin(x_admin_token)
    mgr = _get_mgr(request)
    path = _safe_csv_path(csv_path, None)
    try:
        with mgr.lock:
            mgr.train_incremental(path, limit=limit)
    except Exception as e:
        log.exception("Incremental train failed")
        raise HTTPException(400, f"Incremental failed: {type(e).__name__}") from e

    return {"ok": True, "n_features": len(mgr.features)}


@router.post("/train_one")
def train_one(
    request: Request,
    payload: Dict[str, Any] = Body(..., examples=[{
        "label": "Influenza",
        "active_features": ["fever", "cough"],
        "weight": 2.0,
    }]),
    x_admin_token: Optional[str] = Header(None),
):
    """
    Pojedynczy krok uczenia na modelu używanym przez ensemble.
    Nie dodaje nowych cech; zapisuje artefakty modelu.
    """
    _require_admin(x_admin_token)
    mgr = _get_mgr(request)

    label = str(payload.get("label") or "").strip()
    if not label:
        raise HTTPException(400, "Brak 'label'.")

    try:
        weight = float(payload.get("weight", 1.0))
    except Exception:
        raise HTTPException(400, "'weight' musi być liczbą.")

    if "active_features" in payload and payload["active_features"]:
        active = {str(f).strip().lower() for f in payload["active_features"]}
        feats = {f: (1 if f in active else 0) for f in mgr.features}
    elif "features" in payload and isinstance(payload["features"], dict):
        provided = {str(k).strip().lower(): int(v) for k, v in payload["features"].items()}
        feats = {f: (1 if provided.get(f, 0) else 0) for f in mgr.features}
    else:
        raise HTTPException(400, "Podaj 'active_features' (lista) lub 'features' (dict).")

    if not any(feats.values()):
        raise HTTPException(400, "Żadna z podanych cech nie jest znana modelowi.")

    try:
        with mgr.lock:
            RiverIncrementalTrainer(mgr).train_one(
                feats, label, weight=weight,
                allow_new_classes=True, allow_new_features=False,
                recalibrate=False,
            )
    except Exception as e:
        log.exception("train_one failed")
        raise HTTPException(400, f"train_one failed: {type(e).__name__}") from e

    return {"ok": True}


@router.get("/evaluate")
def evaluate(
    request: Request,
    csv_path: Optional[str] = Query(None),
    k: int = Query(3, ge=1, le=10),
    limit: Optional[int] = Query(None),
    x_admin_token: Optional[str] = Header(None),
):
    """Szybka ewaluacja: ACC / macro-F1 / recall@k na wskazanym CSV (domyślnie TEST_CSV)."""
    _require_admin(x_admin_token)
    mgr = _get_mgr(request)
    path = _safe_csv_path(csv_path, TEST_CSV or DATA_CSV)

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "diseases" not in df.columns:
        raise HTTPException(400, "CSV musi zawierać kolumnę 'diseases'.")

    for c in mgr.features:
        if c not in df.columns:
            df[c] = 0
    df = df[["diseases"] + mgr.features]
    if limit:
        df = df.iloc[: int(limit)].copy()

    y_true: List[str] = []
    y_pred: List[str] = []
    y_pred_topk: List[List[str]] = []

    with mgr.lock:
        for _, row in df.iterrows():
            x = {f: int(row[f]) for f in mgr.features}
            proba = mgr.model.predict_proba_one(x) if mgr.model else {}
            if not proba:
                continue
            top = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
            y_true.append(str(row["diseases"]))
            y_pred.append(top[0][0])
            y_pred_topk.append([lab for lab, _ in top[:k]])

    if not y_true:
        raise HTTPException(400, "Brak danych po dopasowaniu lub model nie zwrócił predykcji.")

    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    r_at_k = _recall_at_k(y_true, y_pred_topk, k=k)

    return {"ok": True, "metrics": {"accuracy": acc, "f1_macro": f1m, f"recall@{k}": r_at_k, "n": len(y_true)}}

from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body, Request, Query
from sklearn.metrics import accuracy_score, f1_score
import joblib

from backend.config import DATA_CSV, TEST_CSV, RIVER_MODEL_PATH
from backend.models.river_manager import RiverManager

log = logging.getLogger("symptomchecker.router.river")
router = APIRouter(prefix="/river", tags=["River"])


# ───────────────────────── helpers ─────────────────────────
def _get_mgr(request: Request) -> RiverManager:
    mgr = getattr(request.app.state, "river", None)
    if mgr is None:
        # awaryjnie zbuduj i uruchom train(), gdyby nie podpięto przy starcie
        mgr = RiverManager()
        mgr.train()
        request.app.state.river = mgr
        log.info("[River] Manager zainicjalizowany awaryjnie w routerze.")
    return mgr


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


# ───────────────────────── endpoints ─────────────────────────
@router.get("/status")
def status(request: Request):
    mgr = _get_mgr(request)
    return {
        "ready": True,
        "model_path": RIVER_MODEL_PATH,
        "has_file": os.path.exists(RIVER_MODEL_PATH),
        "n_features": len(mgr.features),
        "sample_features": mgr.features[:20],
    }


@router.get("/features")
def list_features(request: Request):
    mgr = _get_mgr(request)
    return {"features": mgr.features, "n": len(mgr.features)}


@router.post("/predict_topk")
def predict_topk(
    request: Request,
    payload: Dict[str, Any] = Body(..., example={"active_features": ["fever", "cough"], "k": 3})
):
    mgr = _get_mgr(request)
    k = int(payload.get("k", 3))

    if "active_features" in payload and payload["active_features"]:
        vec = mgr.make_vec_from_symptoms(list(payload["active_features"]))
        vec = vec.astype(np.uint8).tolist()
    elif "features" in payload and isinstance(payload["features"], dict):
        vec = _vec_from_dict(mgr, {k: int(v) for k, v in payload["features"].items()})
    else:
        raise HTTPException(400, "Podaj 'active_features' (lista) lub 'features' (dict).")

    topk = mgr.predict_topk(vec=vec, k=k)
    return {"topk": topk}


@router.post("/fit_from_csv")
def fit_from_csv(
    request: Request,
    csv_path: Optional[str] = Body(None),
    limit: Optional[int] = Body(None),
):
    # bootstrap robimy przy starcie; tu tylko upewniamy się, że manager działa
    mgr = _get_mgr(request)
    path = csv_path or DATA_CSV
    if not os.path.isfile(path):
        raise HTTPException(404, f"CSV not found: {path}")
    return {"ok": True, "info": "Model gotowy. Bootstrap wykonywany przy starcie aplikacji."}


@router.post("/incremental_batch")
def incremental_batch(
    request: Request,
    csv_path: str = Body(...),
    limit: Optional[int] = Body(None),
):
    """
    Przyrostowe douczenie z pliku CSV – używa RiverManager.train_incremental(new_csv, limit).
    """
    if not os.path.isfile(csv_path):
        raise HTTPException(404, f"CSV not found: {csv_path}")

    mgr = _get_mgr(request)
    try:
        mgr.train_incremental(csv_path, limit=limit)
    except Exception as e:
        log.exception("Incremental train failed")
        raise HTTPException(400, f"Incremental failed: {e}") from e

    return {"ok": True, "n_features": len(mgr.features)}


@router.post("/train_one")
def train_one(
    request: Request,
    payload: Dict[str, Any] = Body(..., example={
        "label": "Influenza",
        "active_features": ["fever", "cough"],
        "weight": 2.0
    })
):
    """
    Minimalna wersja 'pojedynczego kroku' bez modyfikowania managera:
    - Buduje x z aktualnych cech managera (nie dodaje nowych cech),
    - woła riverowy learn_one(...) bezpośrednio na mgr.model,
    - zapisuje artefakty (model.joblib).
    """
    mgr = _get_mgr(request)
    if mgr.model is None or not mgr.features:
        raise HTTPException(400, "Model/features niegotowe – wywołaj najpierw /river/status (train()).")

    label = payload.get("label")
    if not label:
        raise HTTPException(400, "Brak 'label'.")

    weight = float(payload.get("weight", 1.0))

    # zbuduj słownik cech 0/1 zgodnie z kolejnością managera
    if "active_features" in payload and payload["active_features"]:
        active = set(map(str, payload["active_features"]))
        x = {f: (1 if f in active else 0) for f in mgr.features}
    elif "features" in payload and isinstance(payload["features"], dict):
        # ignorujemy nieznane cechy, nie dodajemy nowych
        provided = {str(k): int(v) for k, v in payload["features"].items()}
        x = {f: (provided.get(f, 0)) for f in mgr.features}
    else:
        raise HTTPException(400, "Podaj 'active_features' (lista) lub 'features' (dict).")

    try:
        mgr.model = mgr.model.learn_one(x, str(label), sample_weight=weight)
        # zapis artefaktów (bez dotykania kodu managera)
        os.makedirs(os.path.dirname(RIVER_MODEL_PATH), exist_ok=True)
        joblib.dump(mgr.model, RIVER_MODEL_PATH)
    except Exception as e:
        log.exception("train_one failed")
        raise HTTPException(400, f"train_one failed: {e}") from e

    return {"ok": True}


@router.get("/evaluate")
def evaluate(
    request: Request,
    csv_path: Optional[str] = Query(None),
    k: int = Query(3, ge=1, le=10),
    limit: Optional[int] = Query(None),
):
    """Szybka ewaluacja: ACC / macro-F1 / recall@k na wskazanym CSV (domyślnie TEST_CSV)."""
    mgr = _get_mgr(request)
    path = csv_path or TEST_CSV or DATA_CSV
    if not path or not os.path.isfile(path):
        raise HTTPException(404, f"CSV not found: {path}")

    df = pd.read_csv(path)
    if "diseases" not in df.columns:
        raise HTTPException(400, "CSV musi zawierać kolumnę 'diseases'.")

    # dopasuj featury
    for c in mgr.features:
        if c not in df.columns:
            df[c] = 0
    keep = ["diseases"] + mgr.features
    df = df[keep]
    if limit:
        df = df.iloc[: int(limit)].copy()

    y_true: List[str] = []
    y_pred: List[str] = []
    y_pred_topk: List[List[str]] = []

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

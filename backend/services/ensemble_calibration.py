# backend/services/ensemble_calibration.py
"""
Kalibracja i ewaluacja całego ensemble (XGB + River).

Nawet gdy XGB i River są osobno dobrze skalibrowane, ich fuzja w log-przestrzeni
(średnia geometryczna z wagami) daje rozkład zbyt "płaski" albo zbyt "ostry".
Dlatego po wytrenowaniu modeli:
  1) liczymy rozkłady obu modeli na odłożonym zbiorze kalibracyjnym,
  2) łączymy je DOKŁADNIE tak jak EnsembleTopK (dynamiczne wagi, gating),
  3) dobieramy temperaturę T_ens (minimum NLL),
  4) mierzymy XGB, River i ensemble na zbiorze testowym (top1/top3/NLL/ECE/Brier)
     i zapisujemy wynik w ml/training_report.json.
Wynik jest cache'owany w ml/ensemble_calibration.json i liczony ponownie tylko
po zmianie któregokolwiek modelu.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MODEL_PATH, RIVER_MODEL_PATH, RIVER_CALIB_PATH, TEST_CSV, XGB_CALIBRATION_CSV

try:
    from config import ENSEMBLE_FUSION
except Exception:
    ENSEMBLE_FUSION = "linear"
from utils import calibration_metrics, log_metrics_table, update_training_report, Progress, fmt_duration

log = logging.getLogger("symptomchecker.ensemble_cal")

ENS_CALIB_PATH = os.path.join(os.path.dirname(MODEL_PATH), "ensemble_calibration.json")
CACHE_VERSION = 2
_EPS = 1e-12


def _file_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return f"{os.path.basename(path)}:missing"


def models_signature() -> str:
    xgb_cal = os.path.splitext(MODEL_PATH)[0] + ".calibration.joblib"
    return "|".join(_file_sig(p) for p in (MODEL_PATH, xgb_cal, RIVER_MODEL_PATH, RIVER_CALIB_PATH))


# ───────────────────────── fuzja wektorowa (zgodna z EnsembleTopK._fuse) ─────────────────────────
def _conf(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0.0, 1.0)
    n_pos = np.maximum((P > 0).sum(1), 2)
    H = -(np.where(P > 0, P * np.log(P + _EPS), 0.0)).sum(1) / (np.log(n_pos) + _EPS)
    part = np.partition(-P, 1, axis=1)
    margin = np.maximum(0.0, (-part[:, 0]) - (-part[:, 1]))
    return np.clip(0.5 * (1.0 - H) + 0.5 * margin, 0.0, 1.0)


def fused_log_scores(Px: np.ndarray, Pr: np.ndarray, ens) -> np.ndarray:
    """Log-wyniki fuzji (przed temperaturą i normalizacją) dla macierzy rozkładów (N x K)."""
    cx, cr = _conf(Px), _conf(Pr)
    raw_x = ens.wx_base * (0.5 + cx)
    raw_r = ens.wr_base * (0.5 + cr)
    wx = raw_x / np.maximum(raw_x + raw_r, 1e-6)
    wx = np.where(cx - cr >= ens.conf_gap, 1.0, wx)
    wx = np.where(cr - cx >= ens.conf_gap, 0.0, wx)
    lo, hi = ens.min_wx, 1.0 - ens.min_wr
    if lo > hi:
        wx = np.full_like(wx, ens.min_wx / ((ens.min_wx + ens.min_wr) or 1.0))
    else:
        wx = np.clip(wx, lo, hi)
    wr = 1.0 - wx
    return wx[:, None] * np.log(np.maximum(Px, _EPS)) + wr[:, None] * np.log(np.maximum(Pr, _EPS))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _best_T(L: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    best = (1.0, float("inf"))
    for g in np.linspace(-2.0, 2.0, 81):
        T = float(np.exp(g))
        P = _softmax(L / T)
        loss = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean())
        if loss < best[1]:
            best = (T, loss)
    return best


# ───────────────────────── rozkłady obu modeli dla CSV ─────────────────────────
def model_distributions_df(state, df: pd.DataFrame, label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Skalibrowane rozkłady XGB i River dla wszystkich wierszy ramki (liczone hurtem).
    Zwraca (Px, Pr, y, classes) w kolejności klas XGB; wiersze z nieznanymi klasami pomijane.
    Te same transformacje co przy predykcji pojedynczej (predict_topk), tylko szybciej.
    """
    xgb_mgr, river = state.xgb, state.river
    df = df.copy()
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]
    df["diseases"] = df["diseases"].astype(str).str.strip().str.lower()
    classes = [str(c).lower() for c in xgb_mgr.le.classes_]
    cidx = {c: i for i, c in enumerate(classes)}
    df = df[df["diseases"].isin(cidx)].reset_index(drop=True)
    y = df["diseases"].map(cidx).to_numpy(dtype=np.int64)
    for f in xgb_mgr.features:
        if f not in df.columns:
            df[f] = 0

    margins = xgb_mgr._predict_margins_batch(
        df[xgb_mgr.features].to_numpy(dtype=np.float32), f"{label}: XGB")
    Px = xgb_mgr.calibrator.transform(margins) if xgb_mgr.calibrator is not None else _softmax(margins)

    with river.lock:
        Z, _, r_classes = river._scores_for_frame(df, f"{label}: River")
        z_base = river._base_scores(r_classes)
    cal = river.calibrator if (river.calibrator and river.calibrator.classes_ == r_classes) else None
    Pr_r = cal.transform(Z, z_base) if cal else _softmax(Z - 0.5 * z_base[None, :])
    # wyrównanie klas River -> kolejność XGB (brakujące: minimum wiersza, jak w EnsembleTopK)
    Pr = np.empty_like(Px, dtype=np.float64)
    Pr[:] = Pr_r.min(1, keepdims=True)
    r_idx = {str(c).lower(): j for j, c in enumerate(r_classes)}
    for c, i in cidx.items():
        j = r_idx.get(c)
        if j is not None:
            Pr[:, i] = Pr_r[:, j]
    Pr = Pr / Pr.sum(1, keepdims=True)
    return np.asarray(Px, dtype=np.float64), Pr, y, classes


def _model_distributions(state, path: str, label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zwraca (Px, Pr, y) dla pliku CSV (kolejność klas XGB)."""
    Px, Pr, y, _ = model_distributions_df(state, pd.read_csv(path), label)
    return Px, Pr, y


def ensemble_distribution(Px: np.ndarray, Pr: np.ndarray, ens, state=None, T: Optional[float] = None) -> np.ndarray:
    """Rozkład ensemble identyczny z EnsembleTopK._fuse dla trybu zapisanego w state."""
    mode = str(getattr(state, "ensemble_mode", "heuristic") or "heuristic") if state is not None else "heuristic"
    wx = getattr(state, "ensemble_wx", None) if state is not None else None
    T = float(T if T is not None else (getattr(state, "ensemble_T", 1.0) if state is not None else 1.0) or 1.0)
    if mode == "linear" and wx is not None:
        P = wx * Px + (1.0 - wx) * Pr
        return P / P.sum(1, keepdims=True)
    if mode == "log" and wx is not None:
        L = wx * np.log(np.maximum(Px, _EPS)) + (1.0 - wx) * np.log(np.maximum(Pr, _EPS))
        return _softmax(L / T)
    return _softmax(fused_log_scores(Px, Pr, ens) / T)


def _nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def _fit_fusion(Px: np.ndarray, Pr: np.ndarray, y: np.ndarray, ens, mode: str) -> Tuple[Optional[float], float, float]:
    """Dobiera (waga XGB, temperatura) na zbiorze kalibracyjnym (minimum NLL). Zwraca (wx, T, NLL)."""
    if mode == "linear":
        best = (0.5, float("inf"))
        for a in np.linspace(0.0, 1.0, 101):
            loss = _nll(a * Px + (1 - a) * Pr, y)
            if loss < best[1]:
                best = (float(a), loss)
        return best[0], 1.0, best[1]
    if mode == "log":
        Lx, Lr = np.log(np.maximum(Px, _EPS)), np.log(np.maximum(Pr, _EPS))
        best = (0.5, 1.0, float("inf"))
        for w in np.linspace(0.0, 1.0, 21):
            T, loss = _best_T(w * Lx + (1 - w) * Lr, y)
            if loss < best[2]:
                best = (float(w), float(T), loss)
        return best
    T, loss = _best_T(fused_log_scores(Px, Pr, ens), y)
    return None, float(T), loss


def ensure_ensemble_calibration(state) -> float:
    """
    Dobiera sposób łączenia XGB + River na zbiorze kalibracyjnym (ENSEMBLE_FUSION: waga XGB
    i/lub temperatura) i liczy raport na teście. Wynik jest cache'owany w
    ml/ensemble_calibration.json i liczony ponownie tylko po zmianie któregoś modelu.
    Ustawia state.ensemble_mode, state.ensemble_wx i state.ensemble_T.
    """
    from services.ensemble import EnsembleTopK

    state.ensemble_T, state.ensemble_mode, state.ensemble_wx = 1.0, "heuristic", None
    if getattr(state, "xgb", None) is None or getattr(state.xgb, "model", None) is None:
        return 1.0
    if getattr(state, "river", None) is None or getattr(state.river, "model", None) is None:
        log.info("[Ensemble][Cal] Brak River - ensemble = sam XGB, bez dodatkowej kalibracji.")
        return 1.0

    mode = str(ENSEMBLE_FUSION or "linear").lower()
    if mode not in ("linear", "log", "heuristic"):
        log.warning("[Ensemble][Cal] Nieznany ENSEMBLE_FUSION=%s - używam linear.", mode)
        mode = "linear"
    sig = models_signature()
    try:
        with open(ENS_CALIB_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if (cached.get("signature") == sig and cached.get("version") == CACHE_VERSION
                and cached.get("mode") == mode):
            state.ensemble_T = float(cached.get("T", 1.0))
            state.ensemble_mode = mode
            state.ensemble_wx = cached.get("wx")
            log.info("[Ensemble][Cal] Załadowano (modele bez zmian): tryb=%s waga XGB=%s T_ens=%.4f.",
                     mode, cached.get("wx"), state.ensemble_T)
            return state.ensemble_T
    except Exception:
        pass

    if not XGB_CALIBRATION_CSV or not os.path.exists(XGB_CALIBRATION_CSV):
        log.warning("[Ensemble][Cal] Brak zbioru kalibracyjnego - dawna heurystyka, T_ens=1.")
        return 1.0

    import time
    t0 = time.time()
    log.info("[Ensemble][Cal] Modele lub tryb się zmieniły - dobieram łączenie (%s) i liczę raport.", mode)
    ens = EnsembleTopK(state)
    Px, Pr, y = _model_distributions(state, XGB_CALIBRATION_CSV, "kalibracja")
    wx, T, loss = _fit_fusion(Px, Pr, y, ens, mode)
    state.ensemble_mode, state.ensemble_wx, state.ensemble_T = mode, wx, float(T)
    log.info("[Ensemble][Cal] Tryb=%s waga XGB=%s T_ens=%.4f (NLL na zbiorze kalibracyjnym %.4f; "
             "sam XGB %.4f, sam River %.4f)", mode, None if wx is None else round(wx, 3), T, loss,
             _nll(Px, y), _nll(Pr, y))

    report: Dict[str, Any] = {"mode": mode, "wx": None if wx is None else round(wx, 4), "T_ens": round(float(T), 4)}
    if TEST_CSV and os.path.exists(TEST_CSV):
        Px_t, Pr_t, y_t = _model_distributions(state, TEST_CSV, "test")
        report.update({
            "test_xgb_calibrated": calibration_metrics(Px_t, y_t),
            "test_river_calibrated": calibration_metrics(Pr_t, y_t),
            "test_ensemble_heuristic": calibration_metrics(_softmax(fused_log_scores(Px_t, Pr_t, ens)), y_t),
            "test_ensemble": calibration_metrics(ensemble_distribution(Px_t, Pr_t, ens, state), y_t),
        })
        log_metrics_table(log, "ENSEMBLE / TEST", {k: v for k, v in report.items() if isinstance(v, dict)})
        river_cal = state.river.calibrator
        update_training_report("river", {
            "test_calibrated": report["test_river_calibrated"],
            "calibrator": {"T": round(float(river_cal.T), 4), "lam": river_cal.lam} if river_cal else None,
        })
    update_training_report("ensemble", report)

    try:
        with open(ENS_CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump({"signature": sig, "version": CACHE_VERSION, "mode": mode, "wx": wx, "T": float(T),
                       "report": report}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("[Ensemble][Cal] Nie zapisano %s: %s", ENS_CALIB_PATH, e)
    log.info("[Ensemble][Cal] Gotowe w %s.", fmt_duration(time.time() - t0))
    return state.ensemble_T

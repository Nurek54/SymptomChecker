from __future__ import annotations
import os
import json
import time
import joblib
import logging
import warnings
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from river import linear_model, optim
from river.multiclass import OneVsRestClassifier

# ───── config ─────
from config import (
    DATA_CSV, TEST_CSV, CLASS_WEIGHT_CSV,
    RIVER_MODEL_PATH, RIVER_CALIB_PATH, RIVER_CALIBRATION_CSV,
    RIVER_CALIBRATE_ON_START,
    RIVER_ALPHA, RIVER_BETA, RIVER_L1, RIVER_L2, RIVER_CLIP_GRADIENT,
    XGB_RANDOM_STATE,
)

# ───── opcje ważenia (fallback, jeśli nie ma w config.py) ─────
try:
    from config import RIVER_WEIGHT_MEDIAN_NORM, RIVER_WEIGHT_CAP, RIVER_WEIGHT_MIN
except Exception:
    RIVER_WEIGHT_MEDIAN_NORM = True
    RIVER_WEIGHT_CAP = 25.0
    RIVER_WEIGHT_MIN = 0.2

log = logging.getLogger("symptomchecker.river")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

RIVER_FEATURES_PATH = os.path.join(os.path.dirname(RIVER_MODEL_PATH), "river_features.json")
SPECIAL_COLS = {"diseases", "xgb_label", "weight"}  # zawsze używamy lowercase

# ───────────────────────── helpery ─────────────────────────
def _learn_in_place(model: OneVsRestClassifier, x: Dict[str, int], y: str, w: Optional[float] = None) -> OneVsRestClassifier:
    ret = model.learn_one(x, y, w=w) if w is not None else model.learn_one(x, y)
    return ret if ret is not None else model

def _logit(p: float, eps: float = 1e-12) -> float:
    p = float(p)
    if not np.isfinite(p):
        p = 1e-9
    p = min(max(p, eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def _norm_entropy(ps) -> float:
    """
    Zwraca entropię znormalizowaną do [0,1]; ~1.0 oznacza prawie 'uniform'.
    """
    ps = np.asarray(ps, dtype=float)
    ps = ps[(ps > 0) & np.isfinite(ps)]
    if ps.size <= 1:
        return 0.0
    h = float(-(ps * np.log(ps)).sum())
    return h / float(np.log(ps.size))

def _sanitize_weights_series(s: pd.Series) -> pd.Series:
    w = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    try:
        wmin = max(0.0, float(RIVER_WEIGHT_MIN or 0.0))
    except Exception:
        wmin = 0.0
    w = w.clip(lower=wmin)
    if RIVER_WEIGHT_MEDIAN_NORM:
        pos = w[w > 0]
        med = float(np.median(pos)) if len(pos) else 1.0
        if med > 0:
            w = w / med
    try:
        wcap = float(RIVER_WEIGHT_CAP) if RIVER_WEIGHT_CAP else None
    except Exception:
        wcap = None
    if wcap and wcap > 0:
        w = w.clip(upper=wcap)
    try:
        arr = w.values.astype(float)
        qs = np.percentile(arr, [0, 50, 90, 99, 100])
        log.info("[River][W] Stats after sanitize: min=%.4g p50=%.4g p90=%.4g p99=%.4g max=%.4g", *qs)
    except Exception:
        pass
    return w.astype(float)

def _sanitize_single_weight(wi: float) -> float:
    return float(_sanitize_weights_series(pd.Series([wi], dtype=float)).iloc[0])

# ───────────────────────── IO ─────────────────────────
def _load_train_dataframe() -> pd.DataFrame:
    log.info("[River][IO] Start: wczytywanie TRAIN z %s", DATA_CSV)
    t0 = time.time()
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Brak pliku z danymi: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)

    # Ujednolicenie nazw kolumn do lowercase (zgodność z set_features)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "diseases" not in df.columns:
        raise ValueError("Brak kolumny 'diseases' w pliku TRAIN.")
    before = len(df)
    df = df.dropna()
    log.info("[River][IO] TRAIN: %d → %d po dropna()", before, len(df))

    if "weight" not in df.columns and CLASS_WEIGHT_CSV and os.path.exists(CLASS_WEIGHT_CSV):
        try:
            log.info("[River][IO] Dołączam wagi klas z %s", CLASS_WEIGHT_CSV)
            cw = pd.read_csv(CLASS_WEIGHT_CSV)
            cw.columns = [str(c).strip().lower() for c in cw.columns]
            cw = cw.rename(columns={"disease": "diseases"})
            w_map = dict(zip(cw["diseases"], cw["class_weight"]))
            df["weight"] = df["diseases"].map(w_map).fillna(1.0).astype(float)
            log.info("[River][IO] Dodano kolumnę 'weight' (unikalne klasy w wagach: %d).", len(w_map))
        except Exception as e:
            log.warning("[River][IO] Nie udało się wczytać CLASS_WEIGHT_CSV: %s", e)

    if "weight" not in df.columns:
        df["weight"] = 1.0
        log.info("[River][IO] Brak wag w źródłach — ustawiam weight=1.0 dla wszystkich.")

    df["weight"] = _sanitize_weights_series(df["weight"])

    log.info("[River][IO] Zakończono wczytywanie TRAIN w %.2fs (N=%d, D=%d).",
             time.time() - t0, len(df), df.shape[1])
    return df

def _features_from(df: pd.DataFrame) -> List[str]:
    feats = [c for c in df.columns if c not in SPECIAL_COLS]
    log.info("[River][FEATS] Zbudowano listę cech (D=%d).", len(feats))
    return feats

def _maybe_load_test(features: List[str]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    log.info("[River][IO] Start: próba wczytania TEST z %s", TEST_CSV)
    t0 = time.time()
    if not TEST_CSV or not os.path.exists(TEST_CSV):
        log.info("[River][IO] TEST: brak pliku — pomijam ewaluację.")
        return None
    df = pd.read_csv(TEST_CSV)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "diseases" not in df.columns:
        log.warning("[River][IO] TEST bez 'diseases' – pomijam ewaluację.")
        return None

    te = df.copy()
    for c in features:
        if c not in te.columns:
            te[c] = 0
    keep = ["diseases"] + list(features)
    te = te[keep]

    Xte = te[features].astype(np.uint8)
    yte = te["diseases"].astype(str)
    log.info("[River][IO] Wczytano TEST w %.2fs (N=%d, D=%d).",
             time.time() - t0, len(yte), Xte.shape[1])
    return Xte, yte

# ───────────────────────── Artefakty ─────────────────────────
def _save_artifacts(model: OneVsRestClassifier, features: List[str]) -> None:
    log.info("[River][ART] Start: zapis artefaktów do %s", RIVER_MODEL_PATH)
    t0 = time.time()
    os.makedirs(os.path.dirname(RIVER_MODEL_PATH), exist_ok=True)
    joblib.dump(model, RIVER_MODEL_PATH)
    with open(RIVER_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    log.info("[River][ART] Zapisano model i features (D=%d) w %.2fs.", len(features), time.time() - t0)

def _load_artifacts() -> Tuple[Optional[OneVsRestClassifier], List[str]]:
    log.info("[River][ART] Start: próba wczytania artefaktów z %s", RIVER_MODEL_PATH)
    t0 = time.time()
    if not os.path.exists(RIVER_MODEL_PATH):
        log.info("[River][ART] Brak pliku modelu — bootstrap będzie wymagany.")
        return None, []
    try:
        model: OneVsRestClassifier = joblib.load(RIVER_MODEL_PATH)
        try:
            with open(RIVER_FEATURES_PATH, "r", encoding="utf-8") as f:
                feats = json.load(f)
            features = feats if isinstance(feats, list) else []
        except Exception:
            features = []
        log.info("[River][ART] Załadowano artefakty w %.2fs (features=%d).",
                 time.time() - t0, len(features))
        return model, features
    except Exception as e:
        log.warning("[River][ART] Nie udało się wczytać modelu: %s", e)
        return None, []

# ───────────────────────── Kalibrator temperatury ─────────────────────────
class RiverTemperatureCalibrator:
    """
    Kalibracja multiclass dla OVR Rivera:
      - liczymy marginesy m_k = logit(p_k(x)) − logit(p_k({}))  (odjęty bias klasy),
      - na marginesy stosujemy softmax(m_k / T),
      - T dobieramy minimalizując NLL.
    """
    def __init__(self, T: float = 1.0, classes: Optional[List[str]] = None):
        self.T: float = float(T)
        self.classes_: List[str] = list(classes) if classes else []

    @staticmethod
    def classes_from_model(model: OneVsRestClassifier) -> List[str]:
        clfs = getattr(model, "classifiers", {}) if model else {}
        return sorted(list(clfs.keys()))

    @staticmethod
    def _nll(margins: np.ndarray, y_idx: np.ndarray, T: float) -> float:
        p = _softmax(margins / T)
        rows = np.arange(p.shape[0])
        ps = np.clip(p[rows, y_idx], 1e-12, 1.0)
        return float(-np.mean(np.log(ps)))

    def fit_from_dataframe(self, mgr: "RiverManager", df: pd.DataFrame) -> Tuple[float, int]:
        log.info("[River][Cal] Start: dopasowanie temperatury (dataset N=%d)", len(df))
        t0 = time.time()
        if mgr.model is None or not mgr.features:
            raise RuntimeError("RiverManager niegotowy: uruchom train().")

        for f in mgr.features:
            if f not in df.columns:
                df[f] = 0
        df = df[["diseases"] + mgr.features].dropna().copy()

        C = self.classes_from_model(mgr.model)
        if not C:
            raise RuntimeError("Brak klas w aktualnym modelu.")
        self.classes_ = C

        # priory dla pustego x (bias/intercept)
        base = mgr.model.predict_proba_one({}) or {}
        base_logits = {c: _logit(float(base.get(c, 1e-9))) for c in C}

        N, K = len(df), len(C)
        margins = np.zeros((N, K), dtype=np.float64)
        y_idx = np.full(N, -1, dtype=np.int64)

        for i, row in enumerate(df.itertuples(index=False)):
            x = {f: int(getattr(row, f)) for f in mgr.features}
            proba = mgr.model.predict_proba_one(x) or {}
            for j, c in enumerate(C):
                margins[i, j] = _logit(float(proba.get(c, 1e-9))) - base_logits[c]
            try:
                y_idx[i] = C.index(str(getattr(row, "diseases")))
            except ValueError:
                y_idx[i] = -1

        mask = y_idx >= 0
        margins = margins[mask]
        y_idx = y_idx[mask]
        if margins.shape[0] == 0:
            raise ValueError("Zbiór kalibracyjny nie zawiera znanych klas.")

        best_T, best_loss = 1.0, float("inf")
        grid = np.linspace(-2.0, 2.0, 41)
        for g in grid:
            T_try = float(np.exp(g))
            loss = self._nll(margins, y_idx, T_try)
            if loss < best_loss:
                best_loss, best_T = loss, T_try
        for factor in [0.85, 0.9, 0.95, 1.05, 1.10, 1.15]:
            T_try = best_T * factor
            loss = self._nll(margins, y_idx, T_try)
            if loss < best_loss:
                best_loss, best_T = loss, T_try

        self.T = float(best_T)
        log.info("[River][Cal] Wyznaczono T=%.5f na %d próbkach w %.2fs.",
                 self.T, margins.shape[0], time.time() - t0)
        return self.T, int(margins.shape[0])

    # z już policzonego proba + priory z mgr
    def margins_from_proba(self, proba: Dict[str, float], base_logits: Dict[str, float]) -> np.ndarray:
        classes = self.classes_ or []
        if not classes:
            raise RuntimeError("Calibrator.classes_ puste — wywołaj fit() albo ustaw klasy.")
        logits = [_logit(float(proba.get(c, 1e-9))) - float(base_logits.get(c, 0.0)) for c in classes]
        return np.asarray(logits, dtype=np.float64).reshape(1, -1)

    def transform_proba(self, margins: np.ndarray) -> np.ndarray:
        return _softmax(margins / self.T)

    def save(self, path: str = RIVER_CALIB_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"T": float(self.T), "classes": list(self.classes_)}, path)
        log.info("[River][Cal] Zapisano kalibrator T=%.5f -> %s", self.T, path)

    @classmethod
    def load(cls, path: str = RIVER_CALIB_PATH) -> Optional["RiverTemperatureCalibrator"]:
        if not os.path.exists(path):
            return None
        payload = joblib.load(path)
        T = float(payload.get("T", 1.0))
        classes = list(payload.get("classes", []))
        log.info("[River][Cal] Załadowano T=%.5f (klas=%d).", T, len(classes))
        return cls(T=T, classes=classes)

# ───────────────────────── Manager ─────────────────────────
class RiverManager:
    """
    Menedżer online'owej regresji logistycznej One-vs-Rest w River:
    - set_features(feats): wymuś kolejność cech (np. z XGB),
    - train(): wczytanie artefaktów / bootstrap,
    - calibrate(): dobór temperatury,
    - predict_topk(): szybka predykcja z neutralizacją biasu klas i (opcjonalnie) kalibracją.
    """

    def __init__(self):
        self.model: Optional[OneVsRestClassifier] = None
        self.features: List[str] = []
        self.calibrator: Optional[RiverTemperatureCalibrator] = None
        self._vec_template: Optional[np.ndarray] = None
        self._feat_index: Optional[Dict[str, int]] = None
        self._forced_features: List[str] = []  # wymuszone przez set_features()
        self._base_proba: Optional[Dict[str, float]] = None  # cache P(y|{})

    # ─── wymuszenie listy cech (np. z XGB) ───
    def set_features(self, feats: Optional[List[str]]) -> None:
        fs = [str(f).strip().lower() for f in (feats or []) if str(f).strip()]
        self._forced_features = list(fs)
        if fs:
            self.features = list(fs)
            self._prepare_vector_template()
            log.info("[River][FEATS] set_features() przyjęto (D=%d) — wymuszam zgodność z XGB.", len(self.features))

    def _new_model(self) -> OneVsRestClassifier:
        log.info("[River][Model] Start: inicjalizacja modelu i optymalizatora")
        t0 = time.time()
        if hasattr(optim, "FTRL"):
            opt = optim.FTRL(alpha=RIVER_ALPHA, beta=RIVER_BETA, l1=RIVER_L1, l2=RIVER_L2)
            log.info("[River][Model] Używam optim.FTRL.")
        else:
            opt = getattr(optim, "AdaGrad", optim.Adam)()
            log.warning("[River][Model] optim.FTRL niedostępny – fallback na %s.", opt.__class__.__name__)
        base = linear_model.LogisticRegression(optimizer=opt, l2=RIVER_L2, clip_gradient=RIVER_CLIP_GRADIENT)
        ovr = OneVsRestClassifier(base)
        log.info("[River][Model] Gotowe w %.2fs — OneVsRest(LogisticRegression).", time.time() - t0)
        return ovr

    def _prepare_vector_template(self) -> None:
        if not self.features:
            return
        self._vec_template = np.zeros((1, len(self.features)), dtype=np.float32, order="C")
        self._feat_index = {f: i for i, f in enumerate(self.features)}
        log.info("[River][FEATS] Przygotowano szablon wektora (D=%d).", len(self.features))

    def _current_classes(self) -> List[str]:
        clfs = getattr(self.model, "classifiers", {}) if self.model else {}
        return sorted(list(clfs.keys()))

    def _load_calibrator(self) -> Optional[RiverTemperatureCalibrator]:
        log.info("[River][Cal] Start: próba załadowania istniejącego kalibratora")
        cal = RiverTemperatureCalibrator.load(RIVER_CALIB_PATH)
        if cal is not None and self.model is not None:
            if cal.classes_ != self._current_classes():
                log.warning("[River][Cal] Kalibrator nie pasuje do aktualnych klas – unieważniam.")
                return None
        return cal

    def _save_calibrator(self) -> None:
        if self.calibrator is not None:
            self.calibrator.save(RIVER_CALIB_PATH)

    def warmup(self) -> None:
        log.info("[River][Warmup] Warm-up (noop) zakończony.")

    def make_vec_from_symptoms(self, active_feats: List[str]) -> np.ndarray:
        if self._vec_template is None or self._feat_index is None:
            self._prepare_vector_template()
        v = (self._vec_template.copy() if self._vec_template is not None
             else np.zeros((1, len(self.features)), dtype=np.float32, order="C"))
        fi = self._feat_index or {}
        for name in active_feats:
            idx = fi.get(name)
            if idx is not None:
                v[0, idx] = 1.0
        return v

    def _baseline_proba(self) -> Dict[str, float]:
        """Cache P(y|{}) – rozkład bazowy (bias)."""
        if self.model is None:
            return {}
        if self._base_proba is None:
            base = self.model.predict_proba_one({}) or {}
            classes = self._current_classes()
            self._base_proba = {c: float(base.get(c, 1e-9)) for c in classes}
        return self._base_proba

    # ───── kalibracja ─────
    def calibrate(self, csv_path: Optional[str] = None, limit: Optional[int] = None) -> float:
        log.info("[River][Cal] Start: calibrate(csv_path=%s, limit=%s)", str(csv_path), str(limit))
        t0 = time.time()
        if self.model is None or not self.features:
            raise RuntimeError("Najpierw uruchom train().")

        path = csv_path
        if not path:
            if RIVER_CALIBRATION_CSV and os.path.exists(RIVER_CALIBRATION_CSV):
                path = RIVER_CALIBRATION_CSV
            elif TEST_CSV and os.path.exists(TEST_CSV):
                path = TEST_CSV
        log.info("[River][Cal] Źródło kalibracji: %s", path if path else "<10% TRAIN fallback>")

        if path and os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [str(c).strip().lower() for c in df.columns]
        else:
            df_full = _load_train_dataframe()
            df = df_full.sample(frac=0.10, random_state=int(XGB_RANDOM_STATE or 42)).copy()

        if "diseases" not in df.columns:
            raise ValueError("CSV do kalibracji musi zawierać kolumnę 'diseases'.")

        if limit:
            df = df.iloc[: int(limit)].copy()

        cal = RiverTemperatureCalibrator()
        T, n = cal.fit_from_dataframe(self, df)
        cal.save(RIVER_CALIB_PATH)
        self.calibrator = cal
        log.info("[River][Cal] Zakończono: T=%.5f (n=%d) w %.2fs.", T, n, time.time() - t0)
        return T

    # ───── train (jak XGB) ─────
    def train(self) -> None:
        log.info("[River][Train] ▶ Start train()")
        t_all = time.time()

        forced = bool(self._forced_features)

        model, feats_loaded = _load_artifacts()
        if model is not None and feats_loaded:
            if forced and feats_loaded != self._forced_features:
                log.warning("[River][Train] Artefakty mają inny porządek cech niż XGB — retraining z porządkiem XGB.")
                model = None  # wymuś bootstrap
            else:
                self.model = model
                self.features = feats_loaded
                self._prepare_vector_template()
                self.warmup()
                self._base_proba = None
                log.info("[River][Train] Artefakty istnieją – pomijam bootstrap.")
        if self.model is None:
            # bootstrap
            log.info("[River][Train] Bootstrap: wczytywanie danych i inicjalizacja modelu")
            df = _load_train_dataframe()

            if forced and self._forced_features:
                for c in self._forced_features:
                    if c not in df.columns:
                        df[c] = 0
                self.features = list(self._forced_features)
                log.info("[River][FEATS] Używam wymuszonego porządku cech (XGB) D=%d.", len(self.features))
            else:
                self.features = _features_from(df)

            self.model = self._new_model()

            n = len(df)
            t0 = time.time()
            w = df["weight"].to_numpy(dtype=float)
            log.info("[River][Train] Start pętli uczącej (N=%d, D=%d)", n, len(self.features))

            for i, row in df.iterrows():
                y = str(row["diseases"])
                x = {f: int(row.get(f, 0)) for f in self.features}
                wi = float(w[i]) if i < len(w) else 1.0
                self.model = _learn_in_place(self.model, x, y, wi)  # type: ignore

                if (i + 1) % 20000 == 0:
                    pct = int((i + 1) / n * 100)
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 1e-9)
                    log.info("[River][Train] %d%% (%d/%d) elapsed=%.1fs speed=%.1f rec/s",
                             pct, i + 1, n, elapsed, rate)

            _save_artifacts(self.model, self.features)  # type: ignore
            self._prepare_vector_template()
            self.warmup()
            self._base_proba = None  # reset priory po uczeniu
            log.info("[River][Train] Bootstrap ukończony w %.2fs (N=%d, D=%d).", time.time() - t0, n, len(self.features))

            try:
                te = _maybe_load_test(self.features)
                if te is not None:
                    log.info("[River][Eval] Start ewaluacji na TEST")
                    t_eval = time.time()
                    Xte, yte = te
                    ypred = []
                    proba_rows = []
                    for _, row in Xte.iterrows():
                        x = {f: int(row[f]) for f in self.features}
                        proba = self.model.predict_proba_one(x) or {}  # type: ignore
                        if not proba:
                            ypred.append("__none__")
                            proba_rows.append([])
                            continue
                        top = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
                        ypred.append(top[0][0])
                        proba_rows.append([p for _, p in top])

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        acc = float(accuracy_score(yte, ypred))
                        f1m = float(f1_score(yte, ypred, average="macro"))

                    ll = float("nan")
                    try:
                        eps = 1e-12
                        p_top1 = np.array([row[0] if row else 1e-9 for row in proba_rows], dtype=float)
                        p_top1 = np.clip(p_top1, eps, 1.0)
                        penal = []
                        for yt, yp, p in zip(yte, ypred, p_top1):
                            penal.append(-np.log(p if yt == yp else eps))
                        ll = float(np.mean(penal))
                    except Exception:
                        pass

                    log.info("[River][Eval] Wyniki: acc=%.4f  f1_macro=%.4f  proxy_logloss=%.4f  (N=%d)  in %.2fs",
                             acc, f1m, ll, len(yte), time.time() - t_eval)
            except Exception as e:
                log.warning("[River][Eval] Błąd ewaluacji TEST: %s", e)

        self.calibrator = self._load_calibrator()
        if RIVER_CALIBRATE_ON_START:
            try:
                if self.calibrator is None:
                    log.info("[River][Cal] Brak poprawnego kalibratora — uruchamiam calibrate()")
                    self.calibrate()
                else:
                    log.info("[River][Cal] Używam istniejącego kalibratora (T=%.5f).", self.calibrator.T)
            except Exception as e:
                log.warning("[River][Cal] Kalibracja nieudana: %s", e)
        else:
            if self.calibrator:
                log.info("[River][Cal] Kalibrator dostępny (T=%.5f), ale RIVER_CALIBRATE_ON_START=False.", self.calibrator.T)

        log.info("[River][Train] ■ Zakończono train() w %.2fs", time.time() - t_all)

    # ─────────────────────────── API: incremental ───────────────────────────
    def train_incremental(self, new_csv: str, limit: Optional[int] = None) -> None:
        log.info("[River][INC] ▶ Start train_incremental(new_csv=%s, limit=%s)", new_csv, str(limit))
        t_all = time.time()
        if self.model is None or not self.features:
            raise RuntimeError("Najpierw uruchom train().")
        if not os.path.exists(new_csv):
            raise FileNotFoundError(f"Brak pliku: {new_csv}")

        df = pd.read_csv(new_csv)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "diseases" not in df.columns:
            raise ValueError("CSV musi mieć kolumnę 'diseases'.")

        if limit:
            df = df.iloc[: int(limit)].copy()

        missing = [c for c in self.features if c not in df.columns]
        for c in missing:
            df[c] = 0
        keep = ["diseases", "weight"] + list(self.features)
        if "weight" not in df.columns:
            df["weight"] = 1.0
        df = df[[c for c in keep if c in df.columns]]

        df["weight"] = _sanitize_weights_series(df["weight"])
        n = len(df)
        w = df["weight"].to_numpy(dtype=float)

        t0 = time.time()
        log.info("[River][INC] Start pętli uczącej (N=%d, D=%d)", n, len(self.features))
        for i, row in df.iterrows():
            y = str(row["diseases"])
            x = {f: int(row.get(f, 0)) for f in self.features}
            wi = float(w[i]) if i < len(w) else 1.0
            self.model = _learn_in_place(self.model, x, y, wi)  # type: ignore

            if (i + 1) % 20000 == 0:
                pct = int((i + 1) / n * 100)
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                log.info("[River][INC] %d%% (%d/%d) elapsed=%.1fs speed=%.1f rec/s",
                         pct, i + 1, n, elapsed, rate)

        _save_artifacts(self.model, self.features)  # type: ignore
        self.warmup()
        self._base_proba = None  # reset priory po douczeniu
        log.info("[River][INC] Doleczono %d rekordów w %.2fs.", n, time.time() - t0)

        try:
            if RIVER_CALIBRATE_ON_START and ((RIVER_CALIBRATION_CSV and os.path.exists(RIVER_CALIBRATION_CSV)) or (TEST_CSV and os.path.exists(TEST_CSV))):
                log.info("[River][Cal] Re-kalibracja po incremental")
                self.calibrate()
            else:
                self.calibrator = None
                try:
                    if os.path.exists(RIVER_CALIB_PATH):
                        os.remove(RIVER_CALIB_PATH)
                except Exception:
                    pass
                log.info("[River][Cal] Kalibrator unieważniony (brak zbioru do re-kalibracji).")
        except Exception as e:
            log.warning("[River][Cal] Re-kalibracja po incremental nieudana: %s", e)

        log.info("[River][INC] ■ Zakończono train_incremental() w %.2fs", time.time() - t_all)

    # ─────────────────────────── API: predict (FAST) — naprawione ───────────────────────────
    def _x_from_vec(self, vec: Union[List[List[int]], List[int], Dict[str, int], set, tuple]) -> Tuple[Dict[str, int], int]:
        """
        Buduje słownik {feature_name: 1} oraz zwraca liczbę aktywnych cech.
        Obsługuje:
          - listę list ([[...]]),
          - listę 0/1 w jednym wymiarze ([...]),
          - dict {"feature":1, ...},
          - set/tuple nazw cech.
        """
        need = len(self.features)
        if need == 0:
            return {}, 0

        # dict nazw cech
        if isinstance(vec, dict):
            x = {f: int(bool(v)) for f, v in vec.items() if f in set(self.features) and int(bool(v)) == 1}
            return x, len(x)
        # set/tuple nazw
        if isinstance(vec, (set, tuple)) and (len(vec) and isinstance(next(iter(vec)), str)):
            x = {f: 1 for f in vec if f in set(self.features)}
            return x, len(x)

        arr = np.asarray(vec, dtype=np.uint8)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != need:
            log.warning("[River][Predict] Długość wektora (%d) != liczbie cech (%d). Dopasowuję (trim/pad).", arr.shape[1], need)
            row = arr[0].tolist()
            if len(row) < need:
                row = row + [0] * (need - len(row))
            elif len(row) > need:
                row = row[:need]
            arr = np.asarray([row], dtype=np.uint8)

        row = arr[0]
        nz = np.nonzero(row)[0].tolist()
        x = {self.features[i]: 1 for i in nz}
        active = int(row.sum())
        return x, active

    def predict_topk(self, vec: List[List[int]], k: int = 3) -> List[Dict[str, Any]]:
        import heapq, hashlib
        log.debug("[River][Predict] ▶ predict_topk(k=%d)", k)
        t0 = time.perf_counter()
        if self.model is None or not self.features:
            raise RuntimeError("Model/features niegotowe – wywołaj train().")

        # 1) zbuduj x + liczba aktywnych
        x, active = self._x_from_vec(vec)

        # jeżeli brak aktywnych cech — nie zwracaj stale tych samych priors
        if active == 0:
            log.info("[River][Predict] Brak aktywnych cech → zwracam pustą listę (unikam stałych priors).")
            return []

        # debug: pokaż kilka aktywnych cech
        try:
            dbg_feats = list(x.keys())[:12]
            log.debug("[River][Predict] active=%d feats=%s", active, dbg_feats)
        except Exception:
            pass

        # 2) predykcja raw
        proba = self.model.predict_proba_one(x) or {}
        classes = self._current_classes()
        if not classes:
            log.info("[River][Predict] active=%d | classes=0 | %.3fs", active, time.perf_counter() - t0)
            return []

        # 3) neutralizacja biasu (λ zależne od liczby cech)
        if active <= 1:
            lam = 0.35
        elif active == 2:
            lam = 0.50
        elif active == 3:
            lam = 0.65
        else:
            lam = 0.80

        base = self._baseline_proba()  # cache P(y|{})
        base_logits = {c: _logit(float(base.get(c, 1e-9))) for c in classes}

        margins = np.asarray(
            [_logit(float(proba.get(c, 1e-9))) - lam * base_logits[c] for c in classes],
            dtype=np.float64
        ).reshape(1, -1)

        # 4) Kalibracja i strażnik „zbyt płasko”
        cal = self.calibrator or self._load_calibrator()
        if cal is not None and cal.classes_ == classes:
            probs = cal.transform_proba(margins)[0]
        else:
            probs = _softmax(margins / 0.9)[0]  # lekki sharpen bez kalibratora

        # Jeśli bardzo płasko LUB blisko priors → użyj ostrzejszych marginesów z pełnym odjęciem biasu
        flat = _norm_entropy(probs) > 0.985
        # podobieństwo do priors (kosinus) — gdy ~1.0, rozkład ≈ priors
        p = np.asarray([probs[list(classes).index(c)] for c in classes], dtype=float)
        b = np.asarray([base.get(c, 1e-9) for c in classes], dtype=float)
        denom = (np.linalg.norm(p) * np.linalg.norm(b)) or 1.0
        cos_sim = float(np.dot(p, b) / denom)
        too_priorish = cos_sim > 0.995

        if flat or too_priorish:
            margins_full = np.asarray(
                [_logit(float(proba.get(c, 1e-9))) - base_logits[c] for c in classes],
                dtype=np.float64
            ).reshape(1, -1)
            probs_alt = _softmax(margins_full / 0.85)[0]  # mocniejsze wyostrzenie
            if _norm_entropy(probs_alt) <= _norm_entropy(probs):
                probs = probs_alt

        # 5) deterministyczny tie-break (bez zmiany wartości)
        eps = 1e-12
        jitter = np.array([
            (int(hashlib.md5(c.encode("utf-8")).hexdigest(), 16) % 65535) / 65535.0 * eps
            for c in classes
        ], dtype=float)
        score_for_sort = probs + jitter

        # 6) Top-K
        k = int(min(max(k, 1), len(probs)))
        idxs = heapq.nlargest(k, range(len(probs)), key=lambda i: score_for_sort[i])
        res = [{"diagnosis_en": str(classes[i]), "probability": float(probs[i])} for i in idxs]

        log.info(
            "[River][Predict] active=%d | λ=%.2f | H=%.3f | cos(base)=%.4f | top1=%s(%.3f) | %.3fs",
            active, lam, _norm_entropy(probs), cos_sim,
            (res[0]['diagnosis_en'] if res else "-"),
            (res[0]['probability'] if res else 0.0),
            time.perf_counter() - t0
        )
        return res


# ───────────────────────── Thin wrapper dla routera ─────────────────────────
class RiverIncrementalTrainer:
    """
    Cienki wrapper wykorzystywany przez router. Pozwala:
      - fit_from_csv(csv_path, limit, allow_new_classes, allow_new_features)
      - train_one(feats_dict, label, weight, allow_new_classes, allow_new_features)
    """
    def __init__(self, mgr: RiverManager):
        self.mgr = mgr

    def fit_from_csv(
        self,
        csv_path: str,
        limit: Optional[int] = None,
        allow_new_classes: bool = True,
        allow_new_features: bool = False,
    ) -> None:
        log.info("[River][INC][Trainer] ▶ Start fit_from_csv(csv_path=%s, limit=%s)", csv_path, str(limit))
        t_all = time.time()
        if self.mgr.model is None or not self.mgr.features:
            raise RuntimeError("Najpierw uruchom RiverManager.train().")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "diseases" not in df.columns:
            raise ValueError("CSV musi mieć kolumnę 'diseases'.")

        if limit:
            df = df.iloc[: int(limit)].copy()

        if allow_new_features:
            new_feats = [c for c in df.columns if c not in SPECIAL_COLS and c not in self.mgr.features]
            if new_feats:
                self.mgr.features += sorted(new_feats)
                log.info("[River][INC][Trainer] Dodano nowe cechy: +%d (D=%d).", len(new_feats), len(self.mgr.features))
        for c in self.mgr.features:
            if c not in df.columns:
                df[c] = 0

        current_classes = set(self.mgr._current_classes())
        if not allow_new_classes:
            before = len(df)
            df = df[df["diseases"].isin(current_classes)].copy()
            log.info("[River][INC][Trainer] allow_new_classes=False → odrzucono %d rekordów.", before - len(df))

        if "weight" not in df.columns:
            df["weight"] = 1.0

        df["weight"] = _sanitize_weights_series(df["weight"])
        w = df["weight"].to_numpy(dtype=float)

        n = len(df)
        t0 = time.time()
        log.info("[River][INC][Trainer] Start pętli uczącej (N=%d, D=%d)", n, len(self.mgr.features))
        for i, row in df.iterrows():
            y = str(row["diseases"])
            x = {f: int(row.get(f, 0)) for f in self.mgr.features}
            wi = float(w[i]) if i < len(w) else 1.0
            self.mgr.model = _learn_in_place(self.mgr.model, x, y, wi)  # type: ignore

            if (i + 1) % 20000 == 0:
                pct = int((i + 1) / n * 100)
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                log.info("[River][INC][Trainer] %d%% (%d/%d) elapsed=%.1fs speed=%.1f rec/s",
                         pct, i + 1, n, elapsed, rate)

        _save_artifacts(self.mgr.model, self.mgr.features)  # type: ignore
        self.mgr._prepare_vector_template()
        self.mgr.warmup()
        self.mgr._base_proba = None
        log.info("[River][INC][Trainer] Doleczono %d rekordów w %.2fs.", n, time.time() - t0)

        try:
            if RIVER_CALIBRATE_ON_START and ((RIVER_CALIBRATION_CSV and os.path.exists(RIVER_CALIBRATION_CSV)) or (TEST_CSV and os.path.exists(TEST_CSV))):
                log.info("[River][Cal][Trainer] Re-kalibracja po fit_from_csv")
                self.mgr.calibrate()
            else:
                self.mgr.calibrator = None
                try:
                    if os.path.exists(RIVER_CALIB_PATH):
                        os.remove(RIVER_CALIB_PATH)
                except Exception:
                    pass
                log.info("[River][Cal][Trainer] Kalibrator unieważniony (brak zbioru do re-kalibracji).")
        except Exception as e:
            log.warning("[River][Cal][Trainer] Re-kalibracja po incremental (trainer) nieudana: %s", e)

        log.info("[River][INC][Trainer] ■ Zakończono fit_from_csv() w %.2fs", time.time() - t_all)

    def train_one(
        self,
        feats_dict: Dict[str, int],
        label: str,
        weight: float = 1.0,
        allow_new_classes: bool = True,
        allow_new_features: bool = False,
    ) -> None:
        log.info("[River][INC][Trainer] ▶ Start train_one(label=%s, weight=%.4f, allow_new_classes=%s, allow_new_features=%s)",
                 str(label), float(weight), str(allow_new_classes), str(allow_new_features))
        t_all = time.time()
        if self.mgr.model is None or not self.mgr.features:
            raise RuntimeError("Najpierw uruchom RiverManager.train().")

        if allow_new_features:
            new_feats = [f for f in feats_dict.keys() if f not in self.mgr.features and f not in SPECIAL_COLS]
            if new_feats:
                self.mgr.features += sorted(new_feats)
                log.info("[River][INC][Trainer] Dodano nowe cechy (train_one): +%d (D=%d).",
                         len(new_feats), len(self.mgr.features))
                _save_artifacts(self.mgr.model, self.mgr.features)  # update features file

        x = {f: int(feats_dict.get(f, 0)) for f in self.mgr.features}

        if not allow_new_classes and label not in self.mgr._current_classes():
            raise ValueError(f"Klasa '{label}' nie istnieje i allow_new_classes=False.")

        wi = _sanitize_single_weight(float(weight))
        self.mgr.model = _learn_in_place(self.mgr.model, x, str(label), wi)  # type: ignore
        _save_artifacts(self.mgr.model, self.mgr.features)  # type: ignore
        self.mgr.warmup()
        self.mgr._base_proba = None

        try:
            if RIVER_CALIBRATE_ON_START and ((RIVER_CALIBRATION_CSV and os.path.exists(RIVER_CALIBRATION_CSV)) or (TEST_CSV and os.path.exists(TEST_CSV))):
                log.info("[River][Cal][Trainer] Re-kalibracja po train_one]")
                self.mgr.calibrate()
        except Exception as e:
            log.warning("[River][Cal][Trainer] Re-kalibracja po train_one nieudana: %s", e)

        log.info("[River][INC][Trainer] ■ Zakończono train_one() w %.2fs", time.time() - t_all)

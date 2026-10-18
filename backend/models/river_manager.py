from __future__ import annotations
import os
import json
import time
import threading
import joblib
import logging
import warnings
from typing import List, Tuple, Optional, Dict, Any, Union, Set

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from river import linear_model, optim, utils as river_utils
from river.multiclass import OneVsRestClassifier

from utils import Progress, calibration_metrics, fmt_duration
from config import (
    DATA_CSV, TEST_CSV, CLASS_WEIGHT_CSV,
    RIVER_MODEL_PATH, RIVER_CALIB_PATH, RIVER_CALIBRATION_CSV,
    RIVER_CALIBRATE_ON_START, RIVER_FEATURES_PATH, rel_to_backend,
    RIVER_ALPHA, RIVER_BETA, RIVER_L1, RIVER_L2, RIVER_CLIP_GRADIENT,
    XGB_RANDOM_STATE,
)

try:
    from config import (RIVER_BOOTSTRAP_MODE, RIVER_BATCH_EPOCHS, RIVER_BATCH_LR,
                        RIVER_BATCH_L2, RIVER_BATCH_SIZE, RIVER_ONLINE_LR)
except Exception:
    RIVER_BOOTSTRAP_MODE, RIVER_BATCH_EPOCHS, RIVER_BATCH_LR = "batch", 12, 0.02
    RIVER_BATCH_L2, RIVER_BATCH_SIZE, RIVER_ONLINE_LR = 1e-5, 1024, 0.01

try:
    from config import RIVER_WEIGHT_MEDIAN_NORM, RIVER_WEIGHT_CAP, RIVER_WEIGHT_MIN
except Exception:
    RIVER_WEIGHT_MEDIAN_NORM = True
    RIVER_WEIGHT_CAP = 25.0
    RIVER_WEIGHT_MIN = 0.2

log = logging.getLogger("symptomchecker.river")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

RIVER_META_PATH = os.path.splitext(RIVER_MODEL_PATH)[0] + ".meta.json"
SPECIAL_COLS = {"diseases", "xgb_label", "weight"}

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

def _load_train_dataframe() -> pd.DataFrame:
    log.info("[River][IO] Start: wczytywanie TRAIN z %s", DATA_CSV)
    t0 = time.time()
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Brak pliku z danymi: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)

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
        log.info("[River][IO] Brak wag w źródłach - ustawiam weight=1.0 dla wszystkich.")

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
        log.info("[River][IO] TEST: brak pliku - pomijam ewaluację.")
        return None
    df = pd.read_csv(TEST_CSV)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "diseases" not in df.columns:
        log.warning("[River][IO] TEST bez 'diseases', pomijam ewaluację.")
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

def _save_artifacts(model: OneVsRestClassifier, features: List[str]) -> None:
    log.info("[River][ART] Start: zapis artefaktów do %s", RIVER_MODEL_PATH)
    t0 = time.time()
    os.makedirs(os.path.dirname(RIVER_MODEL_PATH), exist_ok=True)
    joblib.dump(model, RIVER_MODEL_PATH)
    with open(RIVER_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    try:
        meta = {"data_csv": rel_to_backend(DATA_CSV), "calibration_csv": rel_to_backend(RIVER_CALIBRATION_CSV) if RIVER_CALIBRATION_CSV else ""}
        with open(RIVER_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("[River][ART] Nie zapisano metadanych modelu: %s", e)
    log.info("[River][ART] Zapisano model i features (D=%d) w %.2fs.", len(features), time.time() - t0)

def _load_artifacts() -> Tuple[Optional[OneVsRestClassifier], List[str]]:
    log.info("[River][ART] Start: próba wczytania artefaktów z %s", RIVER_MODEL_PATH)
    t0 = time.time()
    if not os.path.exists(RIVER_MODEL_PATH):
        log.info("[River][ART] Brak pliku modelu - bootstrap będzie wymagany.")
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
        _warn_if_trained_on_other_data()
        return model, features
    except Exception as e:
        log.warning("[River][ART] Nie udało się wczytać modelu: %s", e)
        return None, []

def _calibration_csv_available() -> bool:
    if not RIVER_CALIBRATION_CSV or not os.path.exists(RIVER_CALIBRATION_CSV):
        return False
    try:
        if os.path.samefile(RIVER_CALIBRATION_CSV, DATA_CSV):
            log.warning("[River][Cal] RIVER_CALIBRATION_CSV wskazuje na DATA_CSV - pomijam kalibrację.")
            return False
    except OSError:
        pass
    return True


def _warn_if_trained_on_other_data() -> None:
    """
    Calibration is only meaningful on data the model did not see in training.
    Warn when the model was trained on a file other than the current DATA_CSV
    (e.g. the full dataset that also contains calibration.csv).
    """
    try:
        with open(RIVER_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        trained_on = str(meta.get("data_csv") or "")
    except Exception:
        trained_on = ""
    if trained_on != rel_to_backend(DATA_CSV):
        log.warning(
            "[River][ART] Model nie ma metadanych albo był trenowany na innym zbiorze (%s) niż DATA_CSV=%s. "
            "Jeśli zbiór treningowy zawierał calibration.csv, kalibracja jest zawyżona. "
            "Usuń %s oraz %s, aby przetrenować model na właściwym zbiorze.",
            trained_on or "brak informacji", DATA_CSV, RIVER_MODEL_PATH, RIVER_CALIB_PATH,
        )


class RiverTemperatureCalibrator:
    """
    Turns OvR scores into a multiclass distribution. For each class take the raw binary
    classifier score z_c (logit), subtract part of the "prior" (score for an empty input,
    i.e. the intercept) and apply a temperature softmax:

        p = softmax((z - lam * z_base) / T)

    Both lam and T are fitted on a held-out calibration set (minimum NLL), and prediction
    must apply EXACTLY the same transform, otherwise the fitted T does not match what users see.
    """

    LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __init__(self, T: float = 1.0, classes: Optional[List[str]] = None, lam: float = 1.0):
        self.T: float = float(T)
        self.lam: float = float(lam)
        self.classes_: List[str] = list(classes) if classes else []
        self.metrics_: Dict[str, Any] = {}

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

    @classmethod
    def _best_T(cls, margins: np.ndarray, y_idx: np.ndarray) -> Tuple[float, float]:
        best_T, best_loss = 1.0, float("inf")
        for g in np.linspace(-2.5, 2.5, 51):
            T_try = float(np.exp(g))
            loss = cls._nll(margins, y_idx, T_try)
            if loss < best_loss:
                best_loss, best_T = loss, T_try
        for factor in (0.9, 0.95, 0.97, 1.03, 1.05, 1.1):
            T_try = best_T * factor
            loss = cls._nll(margins, y_idx, T_try)
            if loss < best_loss:
                best_loss, best_T = loss, T_try
        return best_T, best_loss

    def fit_scores(self, Z: np.ndarray, z_base: np.ndarray, y_idx: np.ndarray) -> Tuple[float, float]:
        """Z: raw scores (N x K), z_base: empty-input scores (K,), y_idx: true class indices."""
        best = (1.0, 1.0, float("inf"))
        for lam in self.LAM_GRID:
            T, loss = self._best_T(Z - lam * z_base[None, :], y_idx)
            log.info("[River][Cal]   lam=%.2f -> T=%.4f NLL=%.4f", lam, T, loss)
            if loss < best[2]:
                best = (lam, T, loss)
        self.lam, self.T = float(best[0]), float(best[1])
        return self.lam, self.T

    def transform(self, z: np.ndarray, z_base: np.ndarray) -> np.ndarray:
        return _softmax((z - self.lam * z_base[None, :]) / self.T)

    def save(self, path: str = RIVER_CALIB_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"T": float(self.T), "lam": float(self.lam), "classes": list(self.classes_),
                     "version": 2, "metrics": self.metrics_}, path)
        log.info("[River][Cal] Zapisano kalibrator T=%.4f lam=%.2f -> %s", self.T, self.lam, path)

    @classmethod
    def load(cls, path: str = RIVER_CALIB_PATH) -> Optional["RiverTemperatureCalibrator"]:
        if not os.path.exists(path):
            return None
        payload = joblib.load(path)
        if int(payload.get("version", 1)) < 2:
            # v1 calibrators were fitted on a different transform than prediction uses
            log.warning("[River][Cal] Stary format kalibratora - zostanie wyznaczony ponownie.")
            return None
        cal = cls(T=float(payload.get("T", 1.0)), classes=list(payload.get("classes", [])),
                  lam=float(payload.get("lam", 1.0)))
        cal.metrics_ = dict(payload.get("metrics") or {})
        log.info("[River][Cal] Załadowano T=%.4f lam=%.2f (klas=%d).", cal.T, cal.lam, len(cal.classes_))
        return cal

class RiverManager:
    def __init__(self):
        self.model: Optional[OneVsRestClassifier] = None
        self.features: List[str] = []
        self.calibrator: Optional[RiverTemperatureCalibrator] = None
        self._vec_template: Optional[np.ndarray] = None
        self._feat_index: Optional[Dict[str, int]] = None
        self._forced_features: List[str] = []  # forced by set_features()
        self._base_proba: Optional[Dict[str, Any]] = None  # cached empty-input scores
        # Guards the model against concurrent learning (/river router) and prediction (chat).
        self.lock = threading.RLock()

    def set_features(self, feats: Optional[List[str]]) -> None:
        fs = [str(f).strip().lower() for f in (feats or []) if str(f).strip()]
        self._forced_features = list(fs)
        if fs:
            self.features = list(fs)
            self._prepare_vector_template()
            log.info("[River][FEATS] set_features() przyjęto (D=%d) - wymuszam zgodność z XGB.", len(self.features))

    def _new_model(self) -> OneVsRestClassifier:
        log.info("[River][Model] Start: inicjalizacja modelu i optymalizatora")
        t0 = time.time()
        # river names this optimizer FTRLProximal (there is no optim.FTRL)
        if hasattr(optim, "FTRLProximal"):
            opt = optim.FTRLProximal(alpha=RIVER_ALPHA, beta=RIVER_BETA, l1=RIVER_L1, l2=RIVER_L2)
            log.info("[River][Model] Używam optim.FTRLProximal (alpha=%s beta=%s l1=%s l2=%s).",
                     RIVER_ALPHA, RIVER_BETA, RIVER_L1, RIVER_L2)
        else:
            opt = getattr(optim, "AdaGrad", optim.Adam)()
            log.warning("[River][Model] optim.FTRL niedostępny, fallback na %s.", opt.__class__.__name__)
        base = linear_model.LogisticRegression(optimizer=opt, l2=RIVER_L2, clip_gradient=RIVER_CLIP_GRADIENT)
        ovr = OneVsRestClassifier(base)
        log.info("[River][Model] Gotowe w %.2fs - OneVsRest(LogisticRegression).", time.time() - t0)
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
                log.warning("[River][Cal] Kalibrator nie pasuje do aktualnych klas, unieważniam.")
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

    def _raw_scores(self, x: Dict[str, int], classes: List[str]) -> np.ndarray:
        # raw OvR scores (logits of the per-class binary classifiers)
        clfs = self.model.classifiers  # type: ignore[union-attr]
        out = np.empty(len(classes), dtype=np.float64)
        for j, c in enumerate(classes):
            m = clfs[c]
            try:
                out[j] = float(m._raw_dot_one(x))
            except Exception:
                out[j] = _logit(float(m.predict_proba_one(x).get(True, 1e-9)))
        return out

    def _base_scores(self, classes: List[str]) -> np.ndarray:
        """Scores for an empty input (intercepts); cached, reset after every learning step."""
        key = tuple(classes)
        if self._base_proba is None or self._base_proba.get("key") != key:
            self._base_proba = {"key": key, "z": self._raw_scores({}, classes)}
        return self._base_proba["z"]

    def _baseline_proba(self) -> Dict[str, float]:  # kept for backward compatibility
        classes = self._current_classes()
        if self.model is None or not classes:
            return {}
        z = self._base_scores(classes)
        return {c: float(1.0 / (1.0 + np.exp(-v))) for c, v in zip(classes, z)}

    def _scores_for_frame(self, df: pd.DataFrame, label: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Raw scores for all CSV rows, with progress logging. Returns (Z, y_idx, classes)."""
        classes = self._current_classes()
        cidx = {c: i for i, c in enumerate(classes)}
        feats = [f for f in self.features if f in df.columns]
        X = df[feats].to_numpy(dtype=np.uint8)
        labels = df["diseases"].astype(str).to_numpy()
        Z = np.zeros((len(df), len(classes)), dtype=np.float64)
        y_idx = np.array([cidx.get(l, -1) for l in labels], dtype=np.int64)
        prog = Progress(label, len(df), log, unit="wierszy")
        for i in range(len(df)):
            x = {feats[j]: 1 for j in np.flatnonzero(X[i])}
            Z[i] = self._raw_scores(x, classes)
            prog.update(i + 1)
        prog.finish()
        return Z, y_idx, classes

    @staticmethod
    def _read_eval_csv(path: str, limit: Optional[int] = None) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]
        if "diseases" not in df.columns:
            raise ValueError(f"{path}: brak kolumny 'diseases'.")
        df["diseases"] = df["diseases"].astype(str).str.strip().str.lower()
        if limit:
            df = df.iloc[: int(limit)].copy()
        return df

    def calibrate(self, csv_path: Optional[str] = None, limit: Optional[int] = None) -> float:
        """Fit (lam, T) on the held-out calibration set (never on train or test data)."""
        if self.model is None or not self.features:
            raise RuntimeError("Najpierw uruchom train().")
        path = csv_path
        if not path:
            if not _calibration_csv_available():
                raise RuntimeError(
                    "Brak odłożonego zbioru kalibracyjnego (RIVER_CALIBRATION_CSV) "
                    "rozłącznego z DATA_CSV. Pomijam kalibrację."
                )
            path = RIVER_CALIBRATION_CSV
        log.info("[River][Cal] Źródło kalibracji: %s", path)
        df = self._read_eval_csv(path, limit)

        with self.lock:
            Z, y_idx, classes = self._scores_for_frame(df, "River/kalibracja")
            z_base = self._base_scores(classes)
        mask = y_idx >= 0
        if not mask.any():
            raise ValueError("Zbiór kalibracyjny nie zawiera znanych klas.")
        Z, y_idx = Z[mask], y_idx[mask]

        before = calibration_metrics(_softmax(Z - 0.5 * z_base[None, :]), y_idx)
        cal = RiverTemperatureCalibrator(classes=classes)
        cal.fit_scores(Z, z_base, y_idx)
        after = calibration_metrics(cal.transform(Z, z_base), y_idx)
        cal.metrics_ = {"calibration_set_before": before, "calibration_set_after": after}
        log.info("[River][Cal] lam=%.2f T=%.4f | zbiór kalibracyjny przed: %s | po: %s",
                 cal.lam, cal.T, before, after)
        cal.save(RIVER_CALIB_PATH)
        self.calibrator = cal
        return cal.T

    def evaluate(self, csv_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Accuracy and calibration metrics on the given CSV (e.g. test.csv)."""
        df = self._read_eval_csv(csv_path, limit)
        with self.lock:
            Z, y_idx, classes = self._scores_for_frame(df, "River/ewaluacja")
            z_base = self._base_scores(classes)
        mask = y_idx >= 0
        cal = self.calibrator if (self.calibrator and self.calibrator.classes_ == classes) else None
        P = cal.transform(Z[mask], z_base) if cal else _softmax(Z[mask] - 0.5 * z_base[None, :])
        out = calibration_metrics(P, y_idx[mask])
        out["unknown_label_rows"] = int((~mask).sum())
        return out

    def _learn_frame(self, df: pd.DataFrame, label: str) -> int:
        """
        One online learning pass over the rows (random order, fixed seed).
        Sparse input with only present symptoms (as in prediction) is ~50x faster
        than a dict with 412 zeros and does not change predictions.
        """
        feats = list(self.features)
        X = df.reindex(columns=feats, fill_value=0).to_numpy(dtype=np.uint8)
        y = df["diseases"].astype(str).to_numpy()
        w = df["weight"].to_numpy(dtype=float) if "weight" in df.columns else np.ones(len(df))
        order = np.random.default_rng(int(XGB_RANDOM_STATE or 42)).permutation(len(df))
        prog = Progress(label, len(df), log, unit="wierszy")
        for step, i in enumerate(order, 1):
            x = {feats[j]: 1 for j in np.flatnonzero(X[i])}
            self.model = _learn_in_place(self.model, x, y[i], float(w[i]))  # type: ignore
            prog.update(step)
        prog.finish()
        return len(df)

    def _batch_warm_start(self, df: pd.DataFrame, label: str) -> int:
        """
        Cold start for River: the same model (OneVsRest + per-class logistic regression),
        but weights are fitted with a few epochs of mini-batch Adam instead of a single
        online pass. A single FTRL pass left the model underfit (Top-1 87% instead of ~90%).

        Loss: weighted mean of summed binary cross-entropies (OvR) + L2.
        The weights are then copied into regular River objects with an SGD optimizer, so
        further online learning (train_one, CSV updates) works unchanged.
        """
        feats = list(self.features)
        X = df.reindex(columns=feats, fill_value=0).to_numpy(dtype=np.float32)
        labels = df["diseases"].astype(str).to_numpy()
        classes = sorted(set(labels.tolist()))
        cidx = {c: i for i, c in enumerate(classes)}
        y = np.array([cidx[l] for l in labels], dtype=np.int64)
        sw = (df["weight"].to_numpy(dtype=np.float32) if "weight" in df.columns
              else np.ones(len(df), dtype=np.float32))
        N, D = X.shape
        K = len(classes)
        rng = np.random.default_rng(int(XGB_RANDOM_STATE or 42))
        W = np.zeros((D, K), dtype=np.float32)
        b = np.zeros(K, dtype=np.float32)
        mW, vW = np.zeros_like(W), np.zeros_like(W)
        mb, vb = np.zeros_like(b), np.zeros_like(b)
        lr, b1, b2, eps = float(RIVER_BATCH_LR), 0.9, 0.999, 1e-8
        l2 = float(RIVER_BATCH_L2)
        bs = max(64, int(RIVER_BATCH_SIZE))
        epochs = max(1, int(RIVER_BATCH_EPOCHS))
        log.info("[River][Batch] Start: N=%d D=%d klasy=%d epoki=%d lr=%.3g l2=%.1e batch=%d",
                 N, D, K, epochs, lr, l2, bs)
        prog = Progress(label, epochs, log, every_pct=0.0, min_interval=0.0, unit="epok")
        t = 0
        for ep in range(epochs):
            order = rng.permutation(N)
            loss_sum = 0.0
            for s0 in range(0, N, bs):
                idx = order[s0:s0 + bs]
                xb, wb = X[idx], sw[idx]
                z = xb @ W + b
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                p_true = p[np.arange(len(idx)), y[idx]].copy()
                g = p                                           # computed in place (saves memory)
                g[np.arange(len(idx)), y[idx]] -= 1.0          # dL/dz for BCE (OvR)
                g *= (wb / len(idx))[:, None]
                gW = xb.T @ g + 2.0 * l2 * W
                gb = g.sum(0)
                t += 1
                mW = b1 * mW + (1 - b1) * gW
                vW = b2 * vW + (1 - b2) * gW * gW
                mb = b1 * mb + (1 - b1) * gb
                vb = b2 * vb + (1 - b2) * gb * gb
                corr = np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                W -= lr * corr * mW / (np.sqrt(vW) + eps)
                b -= lr * corr * mb / (np.sqrt(vb) + eps)
                loss_sum += float(-(np.log(np.clip(p_true, 1e-7, 1.0)) * wb).sum())
            prog.update(ep + 1, extra=f"strata={loss_sum / N:.4f}")
        prog.finish()

        template = linear_model.LogisticRegression(
            optimizer=optim.SGD(RIVER_ONLINE_LR), l2=RIVER_L2, clip_gradient=RIVER_CLIP_GRADIENT)
        ovr = OneVsRestClassifier(template)
        for c, j in cidx.items():
            clf = template.clone()
            col = W[:, j]
            clf._weights = river_utils.VectorDict({feats[i]: float(col[i]) for i in np.flatnonzero(col)})
            clf.intercept = float(b[j])
            ovr.classifiers[c] = clf
        self.model = ovr
        log.info("[River][Batch] Gotowe: %d klasyfikatorów, dalsze uczenie online: SGD(lr=%s).", K, RIVER_ONLINE_LR)
        return N

    def train(self) -> None:
        log.info("[River][Train] ▶ Start train()")
        t_all = time.time()

        forced = bool(self._forced_features)

        model, feats_loaded = _load_artifacts()
        if model is not None and feats_loaded:
            if forced:
                loaded_set = set(feats_loaded)
                forced_set = set(self._forced_features)
                if loaded_set != forced_set:
                    missing = forced_set - loaded_set
                    extra = loaded_set - forced_set
                    log.warning(
                        "[River][Train] Zbiór cech artefaktów różni się od XGB - retrain. "
                        "missing=%d extra=%d", len(missing), len(extra),
                    )
                    model = None  # force bootstrap
                else:
                    self.model = model
                    self.features = list(self._forced_features)
                    self._prepare_vector_template()
                    self.warmup()
                    self._base_proba = None
                    log.info(
                        "[River][Train] Zbiór cech zgodny - używam załadowanego modelu z porządkiem XGB."
                    )
            else:
                self.model = model
                self.features = feats_loaded
                self._prepare_vector_template()
                self.warmup()
                self._base_proba = None
                log.info("[River][Train] Artefakty istnieją, pomijam bootstrap.")
        if self.model is None:
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

            t0 = time.time()
            if str(RIVER_BOOTSTRAP_MODE).lower() == "online":
                self.model = self._new_model()
                n = self._learn_frame(df, "River/trening")
            else:
                n = self._batch_warm_start(df, "River/trening (paczki)")
            _save_artifacts(self.model, self.features)  # type: ignore
            self._prepare_vector_template()
            self.warmup()
            self._base_proba = None  # base scores change after learning
            log.info("[River][Train] Bootstrap ukończony w %s (N=%d, D=%d).", fmt_duration(time.time() - t0), n, len(self.features))

        self.calibrator = self._load_calibrator()
        if RIVER_CALIBRATE_ON_START:
            try:
                if self.calibrator is None:
                    log.info("[River][Cal] Brak poprawnego kalibratora - uruchamiam calibrate()")
                    self.calibrate()
                else:
                    log.info("[River][Cal] Używam istniejącego kalibratora (T=%.5f).", self.calibrator.T)
            except Exception as e:
                log.warning("[River][Cal] Kalibracja nieudana: %s", e)
        else:
            if self.calibrator:
                log.info("[River][Cal] Kalibrator dostępny (T=%.5f), ale RIVER_CALIBRATE_ON_START=False.", self.calibrator.T)

        log.info("[River][Train] ■ Zakończono train() w %.2fs", time.time() - t_all)

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
        t0 = time.time()
        n = self._learn_frame(df, "River/douczanie")

        _save_artifacts(self.model, self.features)  # type: ignore
        self.warmup()
        self._base_proba = None  # base scores change after learning
        log.info("[River][INC] Doleczono %d rekordów w %.2fs.", n, time.time() - t0)

        try:
            if RIVER_CALIBRATE_ON_START and _calibration_csv_available():
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

    def _x_from_vec(self, vec: Union[List[List[int]], List[int], Dict[str, int], set, tuple]) -> Tuple[Dict[str, int], int]:
        need = len(self.features)
        if need == 0:
            return {}, 0

        # accepts a {feature: 0/1} dict, a set/tuple of feature names, or a 0/1 vector
        if isinstance(vec, dict):
            x = {f: int(bool(v)) for f, v in vec.items() if f in set(self.features) and int(bool(v)) == 1}
            return x, len(x)
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
        with self.lock:
            return self._predict_topk_locked(vec, k)

    def _predict_topk_locked(self, vec: List[List[int]], k: int = 3) -> List[Dict[str, Any]]:
        import heapq, hashlib
        log.debug("[River][Predict] ▶ predict_topk(k=%d)", k)
        t0 = time.perf_counter()
        if self.model is None or not self.features:
            raise RuntimeError("Model/features niegotowe, wywołaj train().")

        x, active = self._x_from_vec(vec)

        # no active features: return nothing instead of the same constant priors
        if active == 0:
            log.info("[River][Predict] Brak aktywnych cech → zwracam pustą listę (unikam stałych priors).")
            return []

        try:
            dbg_feats = list(x.keys())[:12]
            log.debug("[River][Predict] active=%d feats=%s", active, dbg_feats)
        except Exception:
            pass

        # raw OvR scores + exactly the transform the calibrator was fitted with
        classes = self._current_classes()
        if not classes:
            log.info("[River][Predict] active=%d | classes=0 | %.3fs", active, time.perf_counter() - t0)
            return []
        z = self._raw_scores(x, classes)
        z_base = self._base_scores(classes)
        cal = self.calibrator
        if cal is not None and cal.classes_ == classes:
            probs = cal.transform(z[None, :], z_base)[0]
            lam, T = cal.lam, cal.T
        else:
            lam, T = 0.5, 1.0  # no calibrator: neutral defaults
            probs = _softmax(((z - lam * z_base) / T)[None, :])[0]

        # deterministic tie-break via a tiny per-class jitter (reported probabilities unchanged)
        eps = 1e-12
        jitter = np.array([
            (int(hashlib.md5(c.encode("utf-8")).hexdigest(), 16) % 65535) / 65535.0 * eps
            for c in classes
        ], dtype=float)
        score_for_sort = probs + jitter

        k =int(min(max(k, 1), len(probs)))
        idxs = heapq.nlargest(k, range(len(probs)), key=lambda i: score_for_sort[i])
        res = [{"diagnosis_en": str(classes[i]), "probability": float(probs[i])} for i in idxs]

        log.info(
            "[River][Predict] active=%d | lam=%.2f T=%.3f | H=%.3f | top1=%s(%.3f) | %.3fs",
            active, lam, T, _norm_entropy(probs),
            (res[0]['diagnosis_en'] if res else "-"),
            (res[0]['probability'] if res else 0.0),
            time.perf_counter() - t0
        )
        return res


class RiverIncrementalTrainer:
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
        t0 = time.time()
        n = self.mgr._learn_frame(df, "River/douczanie z CSV")

        _save_artifacts(self.mgr.model, self.mgr.features)  # type: ignore
        self.mgr._prepare_vector_template()
        self.mgr.warmup()
        self.mgr._base_proba = None
        log.info("[River][INC][Trainer] Doleczono %d rekordów w %.2fs.", n, time.time() - t0)

        try:
            if RIVER_CALIBRATE_ON_START and _calibration_csv_available():
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
        recalibrate: bool = False,
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

        # Full recalibration scans all of calibration.csv (tens of seconds),
        # so after a single step it runs only on request.
        try:
            if recalibrate and _calibration_csv_available():
                log.info("[River][Cal][Trainer] Re-kalibracja po train_one")
                self.mgr.calibrate()
        except Exception as e:
            log.warning("[River][Cal][Trainer] Re-kalibracja po train_one nieudana: %s", e)

        log.info("[River][INC][Trainer] ■ Zakończono train_one() w %.2fs", time.time() - t_all)

from __future__ import annotations
import os
import threading
import json
import time
import joblib
import logging
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb

from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss
from skopt.space import Real, Integer

# ───── config ─────
from config import (
    MODEL_PATH, LE_PATH, BEST_PARAMS_PATH, DATA_CSV, TEST_CSV, CLASS_WEIGHT_CSV,
    XGB_N_ITER_BAYES, XGB_CV_SPLITS, XGB_RANDOM_STATE,
    XGB_USE_GPU, XGB_TREE_METHOD, XGB_CALIBRATE, XGB_CALIBRATION_CSV,
    rel_to_backend,
)

# Dodatkowe flagi inference (bezpieczne defaulty gdy brak w config.py)
try:
    from config import XGB_GPU_PREDICT
except Exception:
    XGB_GPU_PREDICT = False   # GPU w inferencji (tylko gdy to się opłaca)

try:
    from config import XGB_PRED_NTHREAD
except Exception:
    XGB_PRED_NTHREAD = 1      # 1 wątek zwykle najszybszy dla pojedynczych zapytań

try:
    from config import XGB_PRED_MAX_ROUNDS
except Exception:
    XGB_PRED_MAX_ROUNDS = None  # np. 256 – twardy limit rund w predykcji

try:
    from config import XGB_PRED_ROUND_FRACTION
except Exception:
    XGB_PRED_ROUND_FRACTION = None  # np. 0.5 – ułamkowy limit, ignorowane gdy MAX_ROUNDS ustawione

try:
    from config import XGB_GPU_PREDICT_BATCH_MIN
except Exception:
    XGB_GPU_PREDICT_BATCH_MIN = 128  # od ilu wierszy przestawiać predykcję na GPU


# ───────────────────────── logging ─────────────────────────
log = logging.getLogger("symptomchecker.xgb")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

FEATURES_PATH = os.path.join(os.path.dirname(MODEL_PATH), "features.json")
TRAIN_PARAMS_PATH = os.path.splitext(MODEL_PATH)[0] + ".train_params.json"   # parametry bazowego treningu
CALIB_PATH = os.path.splitext(MODEL_PATH)[0] + ".calibration.joblib"        # kalibrator temperatury
META_PATH = os.path.splitext(MODEL_PATH)[0] + ".meta.json"                   # na jakich danych trenowano

# kolumny specjalne, które NIGDY nie mogą być cechami
SPECIAL_COLS = {"diseases", "xgb_label", "weight"}

# opcjonalnie: CuPy (dla treningu/predykcji na GPU)
try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False


# ───────────────────────── GPU helpers ─────────────────────────
def _gpu_kwargs() -> Dict[str, Any]:
    """
    Wymusza GPU, jeśli XGB_USE_GPU == True.
    - XGBoost >= 2.0 (u Ciebie 3.0.5): rekomendacja → tree_method='hist', device='cuda'
    - XGBoost <  2.0: tree_method='gpu_hist'
    Gdy XGB_USE_GPU == False → CPU ('hist').
    Szanuje XGB_TREE_METHOD (jeśli ustawione sensownie).
    """
    # Rozpoznanie wersji XGBoost
    try:
        ver = tuple(int(p) for p in xgb.__version__.split(".")[:2])
    except Exception:
        ver = (1, 6)

    tm = (str(XGB_TREE_METHOD).lower().strip() if XGB_TREE_METHOD is not None else "")

    if not bool(XGB_USE_GPU):
        # CPU
        return {"tree_method": "hist"} if tm == "" else {"tree_method": tm}

    # GPU
    if ver >= (2, 0):
        # 'gpu_hist' jako alias – mapujemy na (hist + device=cuda)
        if tm in ("", "hist", "gpu_hist"):
            return {"tree_method": "hist", "device": "cuda"}
        # fallback: i tak wymusimy CUDA
        return {"tree_method": "hist", "device": "cuda"}
    else:
        # stare wersje
        if tm in ("", "gpu_hist"):
            return {"tree_method": "gpu_hist"}
        # jeśli ktoś ustawił 'hist' przy starym XGB, i tak użyjemy 'gpu_hist' by mieć GPU
        return {"tree_method": "gpu_hist"}


def _log_xgb_env():
    try:
        log.info("[XGB] xgboost=%s  GPU=%s  kwargs=%s", xgb.__version__, bool(XGB_USE_GPU), _gpu_kwargs())
    except Exception:
        pass


def _free_cuda_memory():
    """Wyczyść pule pamięci GPU po treningu (CuPy/Torch)."""
    try:
        import cupy as _cp  # type: ignore
        try:
            _cp._default_memory_pool.free_all_blocks()
        except Exception:
            pass
        try:
            _cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ───────────────────────── IO ─────────────────────────
def _load_train_dataframe() -> pd.DataFrame:
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Brak pliku z danymi: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    if "diseases" not in df.columns:
        raise ValueError("Brak kolumny 'diseases' w pliku TRAIN.")
    df = df.dropna()

    # jeżeli nie ma kolumny 'weight', spróbuj dodać z CLASS_WEIGHT_CSV
    if "weight" not in df.columns and CLASS_WEIGHT_CSV:
        try:
            if os.path.exists(CLASS_WEIGHT_CSV):
                cw = pd.read_csv(CLASS_WEIGHT_CSV)  # kolumny: disease, count, class_weight
                cw = cw.rename(columns={"disease": "diseases"})
                w_map = dict(zip(cw["diseases"], cw["class_weight"]))
                df["weight"] = df["diseases"].map(w_map).fillna(1.0).astype(float)
                log.info("[XGB] Dołożono 'weight' z CLASS_WEIGHT_CSV (fallback).")
        except Exception as e:
            log.warning("[XGB] Nie udało się wczytać CLASS_WEIGHT_CSV: %s", e)

    if "weight" not in df.columns:
        df["weight"] = 1.0

    return df


# ───────────────────────── DMatrix helpers ─────────────────────────
def _to_numpy(X: Any, dtype=np.uint8):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=dtype, copy=False)
    return np.asarray(X, dtype=dtype)


def _make_dmatrix(X: Any, y: Optional[Any] = None, for_gpu: bool = False, weight: Optional[Any] = None):
    """
    Tworzy DMatrix (CPU) lub DeviceQuantileDMatrix (GPU) jeśli dostępne.
    Wspiera 'weight' (wagi próbek). Przy GPU i dostępnej CuPy – pcha dane do VRAM.
    """
    feature_names = None
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)

    if for_gpu:
        try:
            if _CUPY_AVAILABLE:
                X_arr = X.to_numpy(copy=False) if isinstance(X, pd.DataFrame) else X
                X_arr = cp.asarray(X_arr, dtype=cp.float32)  # float32 stabilniej na GPU
                y_arr = None if y is None else cp.asarray(y, dtype=cp.int32)
                w_arr = None if weight is None else cp.asarray(weight, dtype=cp.float32)
                return xgb.DeviceQuantileDMatrix(X_arr, label=y_arr, weight=w_arr)
            else:
                X_arr = _to_numpy(X, dtype=np.float32)  # float32 stabilniej na GPU
                y_arr = None if y is None else np.asarray(y, dtype=np.int32)
                w_arr = None if weight is None else np.asarray(weight, dtype=np.float32)
                return xgb.DeviceQuantileDMatrix(X_arr, label=y_arr, weight=w_arr)
        except Exception:
            # fallback do zwykłego DMatrix
            pass

    X_arr = _to_numpy(X, dtype=np.uint8)
    y_arr = None if y is None else np.asarray(y, dtype=np.int32)
    w_arr = None if weight is None else np.asarray(weight, dtype=np.float32)
    return xgb.DMatrix(X_arr, label=y_arr, weight=w_arr, feature_names=feature_names)


# ───────────────────────── Dane i subset do strojenia ─────────────────────────
def _features_from(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in SPECIAL_COLS]


def _subset_for_tuning(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """
    Podzbiór do CV: klasy z ≥ n_splits rekordami (fallback do ≥2, jeśli inaczej byłoby pusto).
    Zwraca (X, y_enc, w) – gdzie w może być None.
    """
    n_splits = int(XGB_CV_SPLITS or 3)
    vc = df["diseases"].value_counts()
    keep = vc[vc >= n_splits].index
    if len(keep) == 0:
        keep = vc[vc >= 2].index
    df_tune = df[df["diseases"].isin(keep)].reset_index(drop=True)
    features = _features_from(df_tune)
    X = df_tune[features].astype(np.uint8)
    y_enc = LabelEncoder().fit_transform(df_tune["diseases"])
    w = df_tune["weight"].astype(float).to_numpy() if "weight" in df_tune.columns else None
    return X, y_enc, w


# ───────────────────────── Progres / ETA ─────────────────────────
class _BoosterProgress(xgb.callback.TrainingCallback):
    """Progress i ETA co 'step_pct'% rund podczas fitowania Boosterem."""
    def __init__(self, total_rounds: int, label: str = "Train", step_pct: int = 1):
        self.total = max(int(total_rounds), 1)
        self.step = max(int(step_pct), 1)
        self.label = label
        self.start = None
        self.last_logged = -1

    def before_training(self, model):
        self.start = time.time()
        log.info("[XGB][%s] BOOSTING START (total_rounds=%d)", self.label, self.total)
        return model

    def after_iteration(self, model, epoch: int, evals_log):
        done = epoch + 1  # epoch startuje od 0
        pct = int(done / self.total * 100)
        if pct // self.step > self.last_logged:
            elapsed = time.time() - self.start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (self.total - done) / rate if rate > 0 else float("inf")
            log.info("[XGB][%s] %d%% (%d/%d) elapsed=%.1fs ETA=%.1fs",
                     self.label, pct, done, self.total, elapsed, eta)
            self.last_logged = pct // self.step
        return False  # kontynuuj


# ───────────────────────── Param helpers ─────────────────────────
def _strong_fallback_params() -> Dict[str, Any]:
    # Szybko + stabilnie na ~160–250k próbek x ~800 klas (binarki 0/1)
    return {
        "max_depth": 9,
        "learning_rate": 0.045,
        "n_estimators": 1400,      # num_boost_round – nadpisze Bayes/ES
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "colsample_bylevel": 0.90,
        "min_child_weight": 4,
        "gamma": 0.0,
        "reg_alpha": 1e-8,
        "reg_lambda": 2.5,
        "max_bin": 1024,
    }


def _to_train_params(params: Dict[str, Any], n_classes: int) -> Tuple[Dict[str, Any], int]:
    """
    Rozdziela parametry na:
    - 'train_params' – do xgb.train
    - 'num_boost_round' – liczba rund boostingowych
    Mapuje reg_… → ('alpha','lambda'), dodaje objective/metric/num_class oraz GPU flagi.
    """
    p = dict(params or {})
    num_boost_round = int(p.pop("n_estimators", 1000))

    # mapowanie regularizacji
    if "reg_alpha" in p and "alpha" not in p:
        p["alpha"] = float(p.pop("reg_alpha"))
    if "reg_lambda" in p and "lambda" not in p:
        p["lambda"] = float(p.pop("reg_lambda"))

    train_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": int(n_classes),
        "seed": int(XGB_RANDOM_STATE or 42),
        **_gpu_kwargs(),
        **p,
    }
    return train_params, num_boost_round


# ───────────────────────── Zapisywanie/ładowanie parametrów ─────────────────────────
def _save_train_params(params: Dict[str, Any]) -> None:
    safe = {}
    for k, v in (params or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            safe[k] = v
    os.makedirs(os.path.dirname(TRAIN_PARAMS_PATH), exist_ok=True)
    with open(TRAIN_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)


def _load_train_params() -> Optional[Dict[str, Any]]:
    if not os.path.exists(TRAIN_PARAMS_PATH):
        return None
    with open(TRAIN_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ───────────────────────── Bayes tuning (xgb.cv na CPU) ─────────────────────────
def _tune_xgb_bayes(df_full: pd.DataFrame) -> Dict[str, Any]:
    """
    Bayesowskie strojenie przez skopt.Optimizer + xgb.cv (mlogloss).
    Tuning na podzbiorze (klasy ≥ n_splits). Zwraca best_params (bez objective/num_class),
    a do 'n_estimators' wpisuje liczbę rund z early stopping z najlepszej próby.
    """
    n_splits = int(XGB_CV_SPLITS or 3)
    log.info("[XGB][Bayes] START (n_splits=%d)", n_splits)

    vc = df_full["diseases"].value_counts()
    keep = vc[vc >= n_splits].index
    if len(keep) == 0:
        keep = vc[vc >= 2].index
    df_tune = df_full[df_full["diseases"].isin(keep)].reset_index(drop=True)
    features = _features_from(df_tune)
    X_tune = df_tune[features].astype(np.uint8)
    y_tune = LabelEncoder().fit_transform(df_tune["diseases"])
    w_tune = df_tune["weight"].astype(float).to_numpy() if "weight" in df_tune.columns else None

    n_classes = int(len(np.unique(y_tune)))
    if n_classes < 2:
        log.warning("[XGB][Bayes] Za mało klas do strojenia. Użyję domyślnych parametrów.")
        return {}

    dims = [
        Integer(7, 10, name="max_depth"),
        Real(0.035, 0.070, prior="log-uniform", name="learning_rate"),
        Integer(900, 1600, name="n_estimators"),
        Real(0.85, 1.0, prior="uniform", name="subsample"),
        Real(0.85, 1.0, prior="uniform", name="colsample_bytree"),
        Real(0.85, 1.0, prior="uniform", name="colsample_bylevel"),
        Integer(2, 6, name="min_child_weight"),
        Real(0.0, 1.0, prior="uniform", name="gamma"),
        Real(1e-8, 1e-3, prior="log-uniform", name="reg_alpha"),
        Real(0.5, 4.0, prior="log-uniform", name="reg_lambda"),
        Integer(512, 1024, name="max_bin"),
    ]

    try:
        from skopt import Optimizer
    except Exception as e:
        log.warning("[XGB][Bayes] Brak skopt.Optimizer (%s). Pomiń strojenie lub doinstaluj scikit-optimize.", e)
        return {}

    dtrain_cv = _make_dmatrix(X_tune, y_tune, for_gpu=False, weight=w_tune)  # CPU + wagi

    opt = Optimizer(dimensions=dims, base_estimator="GP", acq_func="EI",
                    random_state=XGB_RANDOM_STATE)

    n_iter = int(XGB_N_ITER_BAYES or 10)
    EARLY_STOP_ROUNDS = 200
    log.info("[XGB][Bayes] Iterations=%d, early_stopping_rounds=%d", n_iter, EARLY_STOP_ROUNDS)

    best_params, best_loss = None, float("inf")
    t0 = time.time()

    for i in range(1, n_iter + 1):
        x = opt.ask()
        params = {d.name: v for d, v in zip(dims, x)}

        train_params, num_boost_round = _to_train_params(params, n_classes=n_classes)
        cvres = xgb.cv(
            train_params,
            dtrain_cv,
            num_boost_round=num_boost_round,
            nfold=n_splits,
            stratified=True,
            shuffle=True,
            metrics=("mlogloss",),
            seed=int(XGB_RANDOM_STATE or 42),
            verbose_eval=False,
            as_pandas=True,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
        )
        loss = float(cvres["test-mlogloss-mean"].iloc[-1])
        used_rounds = int(cvres.shape[0])

        opt.tell(x, loss)

        if loss < best_loss:
            best_loss = loss
            best_params = dict(params)
            best_params["n_estimators"] = used_rounds

        elapsed = time.time() - t0
        pct = int(i / n_iter * 100)
        rate = i / elapsed if elapsed > 0 else 0.0
        eta = (n_iter - i) / rate if rate > 0 else float("inf")
        slim = {}
        if best_params:
            for k in ["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree"]:
                slim[k] = best_params.get(k)
        log.info("[XGB][Bayes] %d%% (%d/%d) loss=%.5f best=%.5f used_rounds=%d params=%s elapsed=%.1fs ETA=%.1fs",
                 pct, i, n_iter, loss, best_loss, used_rounds, slim, elapsed, eta)

    log.info("[XGB][Bayes] DONE best_loss=%.5f best_params=%s", best_loss, best_params)
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
    joblib.dump(best_params, BEST_PARAMS_PATH)
    return best_params or {}


# ───────────────────────── Kalibracja (temperature scaling) ─────────────────────────
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z.astype(np.float64))  # cast dla stabilności
    s = ez / np.sum(ez, axis=1, keepdims=True)
    return s.astype(np.float32, copy=False)


class TemperatureCalibrator:
    """Prosty multi-class temperature scaling na marginach XGBoost (pre-softmax)."""
    def __init__(self, T: float = 1.0):
        self.T: float = float(T)

    def _nll(self, margins: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> float:
        p = _softmax(margins / self.T)
        N = p.shape[0]
        rows = np.arange(N)
        ps = np.clip(p[rows, y], 1e-12, 1.0)
        if w is None:
            return float(-np.mean(np.log(ps)))
        w = w.astype(np.float64)
        return float(-(w * np.log(ps)).sum() / np.sum(w))

    def fit(self, margins: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
        grid = np.linspace(-2.0, 2.0, 41)  # T w ~[0.135, 7.39]
        best_T, best_loss = 1.0, float("inf")
        for g in grid:
            self.T = float(np.exp(g))
            loss = self._nll(margins, y, w)
            if loss < best_loss:
                best_loss = loss
                best_T = self.T
        for factor in [0.85, 0.9, 0.95, 1.05, 1.10, 1.15]:
            T_try = best_T * factor
            self.T = float(T_try)
            loss = self._nll(margins, y, w)
            if loss < best_loss:
                best_loss = loss
                best_T = self.T
        self.T = float(best_T)

    def transform(self, margins: np.ndarray) -> np.ndarray:
        return _softmax(margins / self.T)


def _save_calibrator(cal: TemperatureCalibrator) -> None:
    try:
        joblib.dump({"T": float(cal.T)}, CALIB_PATH)
        log.info("[XGB][Cal] Zapisano kalibrator (T=%.5f) -> %s", cal.T, CALIB_PATH)
    except Exception as e:
        log.warning("[XGB][Cal] Nie udało się zapisać kalibratora: %s", e)


def _load_calibrator() -> Optional[TemperatureCalibrator]:
    if not os.path.exists(CALIB_PATH):
        return None
    try:
        payload = joblib.load(CALIB_PATH)
        T = float(payload.get("T", 1.0))
        cal = TemperatureCalibrator(T=T)
        log.info("[XGB][Cal] Załadowano kalibrator (T=%.5f).", cal.T)
        return cal
    except Exception as e:
        log.warning("[XGB][Cal] Nie udało się wczytać kalibratora: %s", e)
        return None


# ───────────────────────── Artefakty ─────────────────────────
def _save_artifacts(model: xgb.Booster, le: LabelEncoder, features: List[str]) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception as e:
        log.warning("[XGB] joblib.dump nieudany (%s). Zapisuję booster.save_model JSON.", e)
        model.save_model(os.path.splitext(MODEL_PATH)[0] + ".json")
    joblib.dump(le, LE_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    log.info("[XGB] Zapisano: model, encoder i features (%d).", len(features))


def _recover_features(model) -> List[str]:
    # 1) features.json
    if os.path.exists(FEATURES_PATH):
        try:
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                feats = json.load(f)
            if isinstance(feats, list) and feats:
                log.info("[XGB] features.json -> %d cech.", len(feats))
                return feats
        except Exception:
            log.warning("[XGB] Nie udało się odczytać %s", FEATURES_PATH, exc_info=False)

    # 2) z modelu (booster.feature_names)
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        feat_names = getattr(booster, "feature_names", None)
        if feat_names and all(isinstance(n, str) for n in feat_names) and any(feat_names):
            log.info("[XGB] booster.feature_names -> %d cech.", len(feat_names))
            return list(feat_names)
    except Exception:
        pass

    # 3) z danych
    try:
        df = _load_train_dataframe()
        features = _features_from(df)
        log.info("[XGB] Fallback: features z TRAIN -> %d.", len(features))
        return features
    except Exception as e:
        log.error("[XGB] Fallback z TRAIN nieudany: %s", e)
    return []


def _maybe_load_test(le: LabelEncoder, features: List[str]) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """Wczytuje test, dopasowuje kolumny i label encoder; zwraca (Xte, yte_enc) lub None."""
    if not TEST_CSV or not os.path.exists(TEST_CSV):
        return None
    df = pd.read_csv(TEST_CSV)
    if "diseases" not in df.columns:
        log.warning("[XGB] TEST bez 'diseases' – pomijam ewaluację.")
        return None

    te = df.copy()
    for c in features:
        if c not in te.columns:
            te[c] = 0
    keep_cols = ["diseases"] + list(features)
    te = te[keep_cols]

    known = set(le.classes_.tolist())
    te = te[te["diseases"].isin(known)]
    if te.empty:
        log.warning("[XGB] TEST po filtrze pusty – pomijam ewaluację.")
        return None

    Xte = te[features].astype(np.uint8)
    yte = te["diseases"]
    try:
        yte_enc = le.transform(yte)
    except Exception:
        log.warning("[XGB] TEST zawiera nieznane klasy – pomijam ewaluację.")
    else:
        return Xte, yte_enc
    return None


def _calibration_csv_available() -> bool:
    """Zbiór kalibracyjny musi istnieć i być rozłączny z DATA_CSV (inny plik)."""
    if not XGB_CALIBRATION_CSV or not os.path.exists(XGB_CALIBRATION_CSV):
        return False
    try:
        if os.path.samefile(XGB_CALIBRATION_CSV, DATA_CSV):
            return False
    except OSError:
        pass
    return True


def _load_calibration_frame(
    le: LabelEncoder, features: List[str]
) -> Optional[Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]]:
    """Wczytuje odłożony zbiór kalibracyjny (np. data/calibration.csv) -> (X, y_enc, w)."""
    if not _calibration_csv_available():
        return None
    df = pd.read_csv(XGB_CALIBRATION_CSV)
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]
    if "diseases" not in df.columns:
        log.warning("[XGB][Cal] %s bez kolumny 'diseases' - pomijam.", XGB_CALIBRATION_CSV)
        return None
    for c in features:
        if c not in df.columns:
            df[c] = 0
    df = df[df["diseases"].isin(set(le.classes_.tolist()))]
    if df.empty:
        return None
    X = df[features].astype(np.uint8)
    y = le.transform(df["diseases"])
    w = df["weight"].astype(float).to_numpy() if "weight" in df.columns else None
    return X, y, w


def _save_meta(calibration_source: str) -> None:
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"data_csv": rel_to_backend(DATA_CSV), "calibration": calibration_source},
                f, ensure_ascii=False, indent=2,
            )
    except Exception as e:
        log.warning("[XGB] Nie zapisano metadanych modelu: %s", e)


def _warn_if_trained_on_other_data(has_calibrator: bool) -> None:
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        trained_on = str(meta.get("data_csv") or "")
    except Exception:
        trained_on = ""
    if trained_on != rel_to_backend(DATA_CSV):
        log.warning(
            "[XGB] Model nie ma metadanych albo był trenowany na innym zbiorze (%s) niż DATA_CSV=%s. "
            "Starsze wersje kalibrowały temperaturę na danych treningowych, więc prawdopodobieństwa "
            "mogą być zawyżone%s. Usuń %s, %s i %s, aby przetrenować model.",
            trained_on or "brak informacji", DATA_CSV,
            "" if has_calibrator else " (kalibrator nieobecny)",
            MODEL_PATH, LE_PATH, CALIB_PATH,
        )


# ───────────────────────── Manager ─────────────────────────
class XGBManager:
    def __init__(self):
        self.model: Optional[xgb.Booster] = None
        self.le: Optional[LabelEncoder] = None
        self.features: List[str] = []
        self.calibrator: Optional[TemperatureCalibrator] = None
        # helpery do szybkiej produkcji
        self._vec_template: Optional[np.ndarray] = None
        self._feat_index: Optional[Dict[str, int]] = None
        # cache predykcji (ostatni wektor+K)
        self._last_pred_key: Optional[Tuple[Optional[Tuple[float, ...]], int]] = None
        self._last_pred_val: Optional[List[Dict[str, Any]]] = None
        # cache konfiguracji inferencji + warmup flaga
        self._infer_cfg_cache: Optional[tuple[str, int]] = None  # (device, nthread)
        self._warmed: bool = False
        # /chat działa w puli wątków: predykcja + cache muszą być atomowe
        self._lock = threading.Lock()

    # ───────────────────────── helpers ─────────────────────────
    def _final_params(self, best: Dict[str, Any], n_classes: int) -> Tuple[Dict[str, Any], int]:
        """Zwraca (train_params, num_boost_round) dla xgb.train."""
        fallback = _strong_fallback_params()
        merged = {**fallback, **(best or {})}
        train_params, num_rounds = _to_train_params(merged, n_classes=n_classes)
        return train_params, num_rounds

    def _prepare_vector_template(self) -> None:
        """Przygotuj szablon i mapę indeksów cech do ultraszybkiego konstruowania wektorów."""
        if not self.features:
            return
        self._vec_template = np.zeros((1, len(self.features)), dtype=np.float32, order="C")
        self._feat_index = {f: i for i, f in enumerate(self.features)}

    def _set_inference_device(self) -> None:
        """
        Konfiguracja inferencji: CPU (domyślnie) lub GPU przez 'device'.
        NIE ustawiamy 'predictor' (ostrzeżenie w XGB>=2.0).
        Log pojawi się tylko, gdy konfiguracja naprawdę się zmieni.
        """
        if self.model is None:
            return
        try:
            dev = "cuda" if bool(XGB_GPU_PREDICT) and bool(XGB_USE_GPU) else "cpu"
            nthreads = max(1, int(XGB_PRED_NTHREAD or 1))
            cfg = (dev, nthreads)
            if cfg != self._infer_cfg_cache:
                self.model.set_param({"device": dev, "nthread": nthreads})
                log.info("[XGB][Infer] device=%s nthread=%d", dev, nthreads)
                self._infer_cfg_cache = cfg
        except Exception:
            pass

    def _limit_iteration_end(self, total_rounds: int) -> int:
        """Zwraca docelową liczbę rund do użycia w predykcji (hard cap / fraction)."""
        try:
            total = int(total_rounds)
        except Exception:
            total = 0
        if total <= 0:
            return 0
        if XGB_PRED_MAX_ROUNDS is not None:
            return int(min(total, int(XGB_PRED_MAX_ROUNDS)))
        if XGB_PRED_ROUND_FRACTION is not None:
            frac = float(XGB_PRED_ROUND_FRACTION)
            if 0 < frac < 1:
                return max(1, int(total * frac))
        return total

    def _predict_margins_cpu(
            self,
            arr: np.ndarray,
            iteration_end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Zwraca marginesy (pre-softmax) dla 1 lub wielu próbek.
        Używa inplace_predict z output_margin=True, a jeśli dana wersja XGBoost
        tego nie wspiera → fallback do DMatrix.predict(output_margin=True).
        """
        if self.model is None:
            raise RuntimeError("Model niegotowy (brak self.model).")

        # --- upewnij się, że mamy 2D float32 ---
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_rows = arr.shape[0]

        # --- ustal ile rund boostingowych faktycznie użyć ---
        try:
            if iteration_end is None or int(iteration_end) <= 0:
                total = int(self.model.num_boosted_rounds())
            else:
                total = int(iteration_end)
        except Exception:
            total = 0
        it_end = self._limit_iteration_end(total)

        # --- 1. próba: inplace_predict z output_margin=True ---
        try:
            kwargs: Dict[str, Any] = {
                "validate_features": False,
                "strict_shape": True,
            }
            if it_end > 0:
                kwargs["iteration_range"] = (0, it_end)

            # tu część wersji XGBoost rzuca TypeError → wyłapujemy
            kwargs["output_margin"] = True

            margins = self.model.inplace_predict(arr, **kwargs)
            margins = np.asarray(margins, dtype=np.float32).reshape(n_rows, -1)
            return margins

        except TypeError:
            # ta wersja XGBoost nie zna output_margin w inplace_predict → przechodzimy na DMatrix
            log.info(
                "[XGB][Predict] inplace_predict(output_margin=True) niewspierane – "
                "fallback do DMatrix.predict(output_margin=True)."
            )
        except Exception:
            # inne błędy też kierujemy do klasycznego predict
            pass

        # --- 2. fallback: klasyczne DMatrix.predict(output_margin=True) ---
        dmat = xgb.DMatrix(arr, feature_names=self.features or None)
        if it_end > 0:
            margins = self.model.predict(
                dmat,
                iteration_range=(0, it_end),
                output_margin=True,
            )
        else:
            margins = self.model.predict(
                dmat,
                output_margin=True,
            )

        margins = np.asarray(margins, dtype=np.float32).reshape(n_rows, -1)
        return margins

    def warmup(self) -> None:
        """Szybki warm-up, by skrócić czas pierwszej odpowiedzi (wykonaj raz)."""
        if self.model is None or not self.features or self._warmed:
            return
        self._set_inference_device()
        dummy = np.zeros((1, len(self.features)), dtype=np.float32, order="C")
        try:
            _ = self.model.inplace_predict(dummy, validate_features=False)
        except Exception:
            dmat = xgb.DMatrix(dummy, feature_names=self.features)
            _ = self.model.predict(dmat)
        self._warmed = True

    def make_vec_from_symptoms(self, active_feats: List[str]) -> np.ndarray:
        """Z listy nazw objawów buduje 1xD wektor (float32, C-order) na podstawie przygotowanego szablonu."""
        if self._vec_template is None or self._feat_index is None:
            self._prepare_vector_template()
        v = self._vec_template.copy()
        fi = self._feat_index or {}
        for name in active_feats:
            idx = fi.get(name)
            if idx is not None:
                v[0, idx] = 1.0
        return v

    # ─────────────────────────── API: train ───────────────────────────
    def train(self) -> None:
        """
        1) Jeśli model/LE istnieją – wczytaj i przygotuj do produkcji (z kalibratorem, jeśli jest).
        2) Jeśli nie ma modelu – strojenie + pełny trening + zapis artefaktów.
        3) Warm-up i konfiguracja inferencji.
        4) [TYLKO PO nowym treningu] Ewaluacja na TEST.
        """
        log.info("[XGB] === TRAIN() START ===")
        _log_xgb_env()

        did_train = False  # NEW: śledzenie, czy faktycznie trenowaliśmy w tej sesji

        # 1) Spróbuj wczytać istniejące artefakty
        model_ok = os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH)
        if model_ok:
            try:
                loaded = joblib.load(MODEL_PATH)
                if isinstance(loaded, xgb.Booster):
                    self.model = loaded
                else:
                    # opcja alternatywna — ścieżka do JSON
                    if isinstance(loaded, str) and os.path.exists(loaded):
                        booster = xgb.Booster()
                        booster.load_model(loaded)
                        self.model = booster
                    else:
                        booster = xgb.Booster()
                        json_path = os.path.splitext(MODEL_PATH)[0] + ".json"
                        if os.path.exists(json_path):
                            booster.load_model(json_path)
                            self.model = booster
                        else:
                            raise RuntimeError("Nie rozpoznano formatu modelu.")
                self.le = joblib.load(LE_PATH)
                self.features = _recover_features(self.model)
                self.calibrator = _load_calibrator()
                _warn_if_trained_on_other_data(self.calibrator is not None)
                log.info("[XGB] Załadowano istniejący model (Booster).")
            except Exception as e:
                log.warning("[XGB] Nie udało się wczytać modelu: %s. Trenuję od nowa.", e)
                model_ok = False

        # 2) Jeśli nie ma modelu – pełny trening
        if not model_ok:
            df_full = _load_train_dataframe()
            features = _features_from(df_full)
            X_full = df_full[features].astype(np.uint8)
            y_full = df_full["diseases"]
            w_full = df_full["weight"].astype(float).to_numpy() if "weight" in df_full.columns else None

            le_full = LabelEncoder().fit(y_full)
            y_full_enc = le_full.transform(y_full)
            n_classes = len(le_full.classes_)
            log.info("[XGB] Dane: N=%d, D=%d, klasy=%d", X_full.shape[0], X_full.shape[1], n_classes)

            # strojenie Bayes (jeśli dostępne)
            if os.path.exists(BEST_PARAMS_PATH):
                try:
                    best = joblib.load(BEST_PARAMS_PATH)
                    log.info("[XGB] Wczytane best_params: %s", best)
                except Exception:
                    best = _tune_xgb_bayes(df_full)
            else:
                best = _tune_xgb_bayes(df_full)

            train_params, num_rounds = self._final_params(best, n_classes)
            log.info("[XGB] Final train_params: %s | num_boost_round=%d", train_params, num_rounds)
            _save_train_params(train_params)

            # Kalibracja temperatury MUSI odbywać się na danych, których booster nie widział.
            # 1) preferowany: odłożony plik (XGB_CALIBRATION_CSV, domyślnie data/calibration.csv),
            # 2) awaryjnie: stratyfikowany hold-out wycięty z TRAIN przed treningiem.
            X_tr, y_tr, w_tr = X_full, y_full_enc, w_full
            cal_holdout: Optional[Tuple[Any, np.ndarray, Optional[np.ndarray]]] = None
            cal_source = "none"
            if bool(XGB_CALIBRATE):
                try:
                    cal_holdout = _load_calibration_frame(le_full, list(features))
                    if cal_holdout is not None:
                        cal_source = rel_to_backend(XGB_CALIBRATION_CSV)
                except Exception as e:
                    log.warning("[XGB][Cal] Nie udało się wczytać zbioru kalibracyjnego: %s", e)
                    cal_holdout = None
                if cal_holdout is None:
                    try:
                        min_class = int(pd.Series(y_full_enc).value_counts().min())
                        n_splits = max(2, min(10, min_class))
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=XGB_RANDOM_STATE)
                        tr_idx, va_idx = next(skf.split(X_full, y_full_enc))
                        cal_holdout = (
                            X_full.iloc[va_idx],
                            y_full_enc[va_idx],
                            w_full[va_idx] if w_full is not None else None,
                        )
                        X_tr = X_full.iloc[tr_idx]
                        y_tr = y_full_enc[tr_idx]
                        w_tr = w_full[tr_idx] if w_full is not None else None
                        cal_source = "holdout_from_train"
                        log.info("[XGB][Cal] Brak osobnego zbioru - odkładam hold-out (N=%d) przed treningiem.", len(va_idx))
                    except Exception as e:
                        log.warning("[XGB][Cal] Nie da się wydzielić hold-outu (%s) - bez kalibracji.", e)
                        cal_holdout = None

            # finalny trening (z wagami, GPU jeśli włączone)
            dtrain = _make_dmatrix(X_tr, y_tr, for_gpu=bool(XGB_USE_GPU), weight=w_tr)
            booster_progress = _BoosterProgress(total_rounds=num_rounds, label="Train", step_pct=1)
            dev = train_params.get("device", "cpu")
            tm = train_params.get("tree_method", None)
            log.info("[XGB] BOOSTING START → num_boost_round=%d, device=%s, tree_method=%s",
                     num_rounds, dev, tm)
            t0 = time.time()
            booster = xgb.train(
                params=train_params,
                dtrain=dtrain,
                num_boost_round=num_rounds,
                evals=[],
                callbacks=[booster_progress],
                verbose_eval=False,
            )
            log.info("[XGB] Wytrenowano w %.1fs", time.time() - t0)
            try:
                del dtrain
            except Exception:
                pass

            # zapis artefaktów + stan
            self.model = booster
            self.le = le_full
            self.features = list(features)
            _save_artifacts(booster, le_full, self.features)

            # kalibracja na danych NIEWIDZIANYCH przy treningu
            self.calibrator = None
            if bool(XGB_CALIBRATE) and cal_holdout is not None:
                try:
                    X_va, y_va, w_va = cal_holdout

                    # marginy → kalibrator temperatury
                    self._set_inference_device()
                    try:
                        best_iter = getattr(self.model, "best_iteration", None)
                        it_end = int(best_iter) if isinstance(best_iter, (int, np.integer)) and best_iter > 0 \
                            else int(self.model.num_boosted_rounds())
                    except Exception:
                        it_end = 0
                    margins = self._predict_margins_cpu(X_va.to_numpy(dtype=np.float32, copy=False), it_end)
                    cal = TemperatureCalibrator()
                    cal.fit(margins, y_va, w=w_va)
                    self.calibrator = cal
                    _save_calibrator(cal)
                    log.info("[XGB][Cal] T=%.4f (źródło: %s, N=%d)", cal.T, cal_source, len(y_va))
                except Exception as e:
                    log.warning("[XGB][Cal] Kalibracja nieudana: %s", e)
            if self.calibrator is None:
                cal_source = "none"
                try:
                    if os.path.exists(CALIB_PATH):
                        os.remove(CALIB_PATH)  # stary kalibrator nie pasuje do nowego modelu
                except Exception:
                    pass
            _save_meta(cal_source)

            did_train = True  # NEW: świeżo trenowaliśmy

        # 3) Konfiguracja produkcyjna + warm-up + sprzątanie VRAM
        self._set_inference_device()
        _free_cuda_memory()
        self._prepare_vector_template()
        self.warmup()

        # 4) Ewaluacja na TEST — tylko po nowym treningu
        if did_train:  # NEW
            try:
                if self.model is not None and self.le is not None and self.features:
                    te = _maybe_load_test(self.le, self.features)
                    if te is not None:
                        Xte, yte_enc = te
                        try:
                            best_iter = getattr(self.model, "best_iteration", None)
                            it_end = int(best_iter) if isinstance(best_iter, (int, np.integer)) and best_iter > 0 \
                                else int(self.model.num_boosted_rounds())
                        except Exception:
                            it_end = 0

                        Xarr = Xte.to_numpy(dtype=np.float32, copy=False)
                        margins = self._predict_margins_cpu(Xarr, iteration_end=it_end)
                        proba = (self.calibrator.transform(margins)
                                 if self.calibrator is not None else _softmax(margins))

                        pred_enc = np.argmax(proba, axis=1)
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="The number of unique classes is greater than 50% of the number of samples",
                            )
                            acc = accuracy_score(yte_enc, pred_enc)
                            f1m = f1_score(yte_enc, pred_enc, average="macro")
                            try:
                                ll = log_loss(yte_enc, proba, labels=list(range(proba.shape[1])))
                            except Exception:
                                ll = float("nan")
                        log.info("[XGB][TEST] acc=%.4f  f1_macro=%.4f  logloss=%.4f  (N=%d)  calibrated=%s",
                                 acc, f1m, ll, len(yte_enc), bool(self.calibrator))
            except Exception as e:
                log.warning("[XGB] Błąd ewaluacji testu: %s", e)
        else:
            log.info("[XGB] Pomijam ewaluację TEST — model został tylko wczytany (brak świeżego treningu).")  # NEW

    # ─────────────────────────── API: incremental ───────────────────────────
    def train_incremental(
        self,
        new_csv: str,
        extra_rounds: int = 200,
        lr_scale: Optional[float] = 0.5,
        early_stopping_rounds: Optional[int] = None,
        val_folds: int = 10,
        recalibrate: bool = True,
    ) -> None:
        """
        Dogrywa kolejne drzewa do istniejącego modelu na NOWYCH danych (bez mieszania ze starymi).
        - Uwzględnia 'weight' jako WAGĘ PRÓBKI.
        - Pozostawia listę klas bez zmian (nowe, nieznane klasy są odrzucane).
        - Może ponownie dokalibrować temperaturę na walidacji z new_csv (recalibrate=True).
        """
        log.info("[XGB][INC] === INCREMENTAL TRAIN START === new_csv=%s extra_rounds=%d",
                 new_csv, int(extra_rounds))

        if self.model is None or self.le is None or not self.features:
            raise RuntimeError("Najpierw wytrenuj/wczytaj model bazowy (model/le/features niegotowe).")
        if not os.path.exists(new_csv):
            raise FileNotFoundError(f"Brak pliku z nowymi danymi: {new_csv}")

        df_new = pd.read_csv(new_csv)
        if "diseases" not in df_new.columns:
            raise ValueError("Nowy CSV musi zawierać kolumnę 'diseases'.")

        # odrzuć nieznane klasy
        known = set(self.le.classes_.tolist())
        before = len(df_new)
        df_new = df_new[df_new["diseases"].isin(known)].copy()
        if df_new.empty:
            raise ValueError("Nowe dane nie zawierają znanych klas (po filtrze pusto).")
        if len(df_new) < before:
            log.info("[XGB][INC] Odrzucono %d wierszy z nieznanymi klasami.", before - len(df_new))

        # dopasuj cechy: brakujące dopełnij zerami; usuń kolumny specjalne
        for c in self.features:
            if c not in df_new.columns:
                df_new[c] = 0
        keep_cols = ["diseases", "weight"] + list(self.features)
        if "weight" not in df_new.columns:
            df_new["weight"] = 1.0
        df_new = df_new[[c for c in keep_cols if c in df_new.columns]]

        # typy: na GPU stabilniej float32
        X = df_new[self.features].astype(np.float32 if bool(XGB_USE_GPU) else np.uint8)
        y = self.le.transform(df_new["diseases"])
        w = df_new["weight"].astype(float).to_numpy() if "weight" in df_new.columns else None

        train_params = _load_train_params()
        if not train_params:
            log.warning("[XGB][INC] Brak zapisanych parametrów treningu – użyję fallback.")
            train_params, _ = _to_train_params(_strong_fallback_params(), n_classes=len(self.le.classes_))

        # dopnij cele/metryki i liczba klas + GPU jeśli dostępne
        train_params = {
            **train_params,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": int(len(self.le.classes_)),
            **_gpu_kwargs(),
            "seed": int(XGB_RANDOM_STATE or 42),
        }

        # redukcja LR dla dogrywki (stabilniej)
        if lr_scale is not None and lr_scale > 0:
            base_eta = float(train_params.get("learning_rate", train_params.get("eta", 0.1)))
            eta_new = max(base_eta * float(lr_scale), 1e-4)
            train_params["learning_rate"] = eta_new
            train_params["eta"] = eta_new
            log.info("[XGB][INC] learning_rate: %.6f -> %.6f (lr_scale=%.3f)", base_eta, eta_new, lr_scale)

        use_gpu = bool(XGB_USE_GPU)
        if early_stopping_rounds:
            # zabezpieczenie: liczba foldów ≤ min liczności klasy
            min_class = int(pd.Series(y).value_counts().min())
            n_folds = max(2, min(int(val_folds), min_class))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=XGB_RANDOM_STATE)
            train_idx, val_idx = next(skf.split(X, y))
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]
            w_tr = w[train_idx] if w is not None else None
            w_va = w[val_idx] if w is not None else None
            dtrain = _make_dmatrix(X_tr, y_tr, for_gpu=use_gpu, weight=w_tr)
            dvalid = _make_dmatrix(X_va, y_va, for_gpu=use_gpu, weight=w_va)
            evals = [(dvalid, "valid")]
            log.info("[XGB][INC] Early stopping włączony (rounds=%d). Walidacja: %d próbek.",
                     int(early_stopping_rounds), len(val_idx))
        else:
            dtrain = _make_dmatrix(X, y, for_gpu=use_gpu, weight=w)
            evals = []

        booster_progress = _BoosterProgress(total_rounds=int(extra_rounds), label="Incremental", step_pct=1)
        dev = train_params.get("device", "cpu")
        tm = train_params.get("tree_method")
        log.info("[XGB][INC] BOOSTING START → extra_rounds=%d, device=%s, tree_method=%s",
                 int(extra_rounds), dev, tm)
        t0 = time.time()
        self.model = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=int(extra_rounds),
            evals=evals,
            early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
            xgb_model=self.model,  # kontynuacja
            callbacks=[booster_progress],
            verbose_eval=False,
        )
        log.info("[XGB][INC] Dograno %d rund w %.1fs.", int(extra_rounds), time.time() - t0)

        # posprzątaj GPU RAM
        try:
            del dtrain
            if early_stopping_rounds:
                del dvalid  # type: ignore
        except Exception:
            pass
        _free_cuda_memory()

        # artefakty + inference device + warm-up
        _save_artifacts(self.model, self.le, self.features)
        _save_train_params(train_params)
        self._set_inference_device()
        self._prepare_vector_template()
        self.warmup()

        # ewentualna re-kalibracja na walidacji z new_csv
        if bool(XGB_CALIBRATE) and recalibrate and early_stopping_rounds:
            try:
                Xva_arr = X_va.to_numpy(dtype=np.float32, copy=False)  # type: ignore
                try:
                    best_iter = getattr(self.model, "best_iteration", None)
                    if isinstance(best_iter, (int, np.integer)) and best_iter > 0:
                        it_end = int(best_iter)
                    else:
                        it_end = int(self.model.num_boosted_rounds())
                except Exception:
                    it_end = 0
                margins = self._predict_margins_cpu(Xva_arr, iteration_end=it_end)
                cal = TemperatureCalibrator()
                cal.fit(margins, y_va, w=w_va)
                self.calibrator = cal
                _save_calibrator(cal)
                log.info("[XGB][INC] Re-kalibracja zakończona.")
            except Exception as e:
                log.warning("[XGB][INC] Błąd re-kalibracji: %s", e)

    # ─────────────────────────── API: predict (FAST) ───────────────────────────
    def predict_topk(self, x: Any, k: int = 3) -> List[Dict[str, Any]]:
        """
        Top-k predykcja dla pojedynczego wektora cech.

        x może być:
          - np.ndarray (D,) albo (1, D)
          - lista liczb [0/1,...]
          - lista list [[0/1,...]] z jednym wierszem

        Zwraca listę słowników:
        [{"diagnosis_en": str, "probability": float}, ...]
        """
        if self.model is None or self.le is None or not self.features:
            raise RuntimeError("Model/encoder/features niegotowe.")
        with self._lock:
            return self._predict_topk_locked(x, k)

    def _predict_topk_locked(self, x: Any, k: int = 3) -> List[Dict[str, Any]]:
        self._set_inference_device()
        t0 = time.perf_counter()

        # --- normalizacja wejścia do 2D (1, D) ---
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] != 1:
            # w Twoim /chat używasz tylko jednej próbki naraz – pilnujemy tego
            raise ValueError("predict_topk oczekuje pojedynczej próbki (shape (D,) lub (1,D)).")

        # mini-cache: najczęstszy przypadek: ten sam wektor i to samo K
        vt = tuple(float(v) for v in arr[0].tolist())
        if self._last_pred_key == (vt, int(k)) and self._last_pred_val is not None:
            return [dict(d) for d in self._last_pred_val]

        # --- ustalenie liczby rund, jak wcześniej ---
        try:
            best_iter = getattr(self.model, "best_iteration", None)
            if isinstance(best_iter, (int, np.integer)) and best_iter > 0:
                it_end = int(best_iter)
            else:
                it_end = int(self.model.num_boosted_rounds())
        except Exception:
            it_end = 0

        # --- marginesy → softmax / kalibrator ---
        margins = self._predict_margins_cpu(arr, iteration_end=it_end)
        proba = self.calibrator.transform(margins) if self.calibrator is not None else _softmax(margins)

        p = proba[0]
        k = int(k)
        if k <= 0:
            return []
        if k > p.shape[0]:
            k = p.shape[0]

        # Top-K: argpartition + dosortowanie
        part = np.argpartition(-p, kth=k - 1)[:k]
        idxs = part[np.argsort(-p[part])]

        labels_en = self.le.inverse_transform(idxs)
        res = [
            {"diagnosis_en": str(lbl), "probability": float(p[i])}
            for i, lbl in zip(idxs, labels_en)
        ]

        # cache
        self._last_pred_key = (vt, int(k))
        self._last_pred_val = [dict(d) for d in res]

        rounds_used = self._limit_iteration_end(it_end)
        log.info(
            "[XGB][Predict] rows=%d classes=%d rounds=%d time=%.3fs",
            arr.shape[0],
            p.shape[0],
            rounds_used if rounds_used > 0 else -1,
            time.perf_counter() - t0,
        )
        return res


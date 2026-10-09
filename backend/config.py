import os
import logging
from pathlib import Path

# Load the first .env found (repo root wins over backend/); real env vars take precedence.
try:
    from dotenv import load_dotenv  # type: ignore
    for candidate in (
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parent / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break
except Exception:
    pass

_BACKEND_DIR = Path(__file__).resolve().parent

def _env_str(name: str, default: str | None = None, *, required: bool = False) -> str:
    v = os.getenv(name, default if default is not None else "")
    if required and str(v).strip() == "":
        raise RuntimeError(f"Brak wymaganej zmiennej środowiskowej: {name}")
    return v

def _env_path(name: str, default: str) -> str:
    """Path from env; relative paths resolve against backend/, not the cwd."""
    v = _env_str(name, default).strip()
    if not v:
        return v
    p = Path(v)
    return str(p if p.is_absolute() else (_BACKEND_DIR / p))

def rel_to_backend(path: str) -> str:
    """Path relative to backend/, stored in model metadata so it stays portable across machines."""
    try:
        return Path(os.path.relpath(path, _BACKEND_DIR)).as_posix()
    except ValueError:
        return Path(path).as_posix()

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

# Runtime
APP_ENV            = _env_str("APP_ENV", "dev")
DEBUG              = _env_bool("DEBUG", APP_ENV.lower() != "prod")
GLOBAL_SEED        = _env_int("GLOBAL_SEED", 42)

# ML artifacts
MODEL_PATH         = _env_path("MODEL_PATH", "ml/xgb_model.joblib")
LE_PATH            = _env_path("LE_PATH", "ml/label_encoder.joblib")
BEST_PARAMS_PATH   = _env_path("BEST_PARAMS_PATH", "ml/xgb_best_params.joblib")

# Datasets. Prefer data/v2 (built by data/build_datasets.py): train/calibration/test
# share no symptom vectors, duplicate labels merged, micro-classes dropped, class
# weights clipped to [0.5, 5]. Without v2, fall back to the legacy layout
# (train_remainder.csv = full set minus calibration.csv).
def _first_existing(*cands: str) -> str:
    for c in cands:
        if (_BACKEND_DIR / c).exists():
            return c
    return cands[-1]

_V2 = (_BACKEND_DIR / "data" / "v2" / "train.csv").exists()
DATA_CSV           = _env_path("DATA_CSV", _first_existing(
    "data/v2/train.csv", "data/train_remainder.csv", "data/Final_Dataset_Diseases_and_Symptoms.csv"))
TEST_CSV           = _env_path("TEST_CSV", "data/v2/test.csv" if _V2 else "data/Final_Testset_for_Eval.csv")
CLASS_WEIGHT_CSV   = _env_path("CLASS_WEIGHT_CSV", "data/v2/class_weight.csv" if _V2
                               else "data/Final_Dataset_Diseases_and_Symptoms.class_weight.csv")

# Single allowed CORS origin (credentials are enabled, so no wildcard).
FRONTEND_ORIGIN    = _env_str("FRONTEND_ORIGIN", "http://localhost:3000")

# Chat / symptom logic. Below MIN_SYMPTOMS no diagnosis ranking is produced.
MIN_SYMPTOMS       = _env_int("MIN_SYMPTOMS", 3)

STOP_PHRASES = [
    "to wszystko", "nie mam więcej objawów", "koniec", "stop",
    "to już wszystko", "to wszystko.",
    "that's all", "no more symptoms", "end", "stop.", "stop!", "that's it"
]

NEGATION_PATTERNS  = ["nie mam", "bez ", "brak ", "zaprzeczam "]

YES_TOKENS = {"tak", "ta", "y", "yes", "yeah", "yep", "jasne", "oczywiście"}
NO_TOKENS  = {"nie", "n", "no", "nope"}

# Follow-up questions never ask about symptoms containing these tokens (too specific or
# sensitive for a generic chat). Extendable via FOLLOWUP_BLOCKLIST env (comma-separated).
FOLLOWUP_BLOCKLIST_TOKENS = {
    "scrotum","testes","penis","vagina","anus","perianal",
    "infant","newborn","pregnancy","lactation",
    "knee","elbow","finger","toe","earlobe","eyebrow","eyelid",
}

# LLM generation. LLM_MIN_OUTPUT_CHARS: shorter answers that end mid-sentence are retried.
LLM_MAX_NEW_TOKENS        = _env_int("LLM_MAX_NEW_TOKENS", 2000)
LLM_MAX_NEW_TOKENS_CHAT    = _env_int("LLM_MAX_NEW_TOKENS_CHAT", 900)
LLM_MIN_NEW_TOKENS_CHAT    = _env_int("LLM_MIN_NEW_TOKENS_CHAT", 150)
LLM_TEMP                   = _env_float("LLM_TEMP", 0.72)
LLM_TOP_P                  = _env_float("LLM_TOP_P", 0.92)
LLM_TOP_K                  = _env_int("LLM_TOP_K", 40)
LLM_REPETITION_PENALTY     = _env_float("LLM_REPETITION_PENALTY", 1.10)
LLM_MIN_OUTPUT_CHARS       = _env_int("LLM_MIN_OUTPUT_CHARS", 550)

LLM_CONTEXT_LEN     = _env_int("LLM_CONTEXT_LEN", 8192)
LLM_STOP_SEQUENCES  = ["<|im_end|>"]
LLM_EOS_STRINGS     = ["<|im_end|>", "</s>"]

# GGUF model loaded via ctransformers; REPO is a local directory relative to backend/.
LLM_SLOTS = {
    "bielik": {
        "REPO": "weights/bielik-gguf",
        "FILENAME": "Bielik-11B-v2.3-Instruct.Q6_K.gguf",
        "MODEL_TYPE": "llama"
    },
}

# 999 = offload all layers to GPU (reduced automatically if VRAM runs out).
LLM_GPU_LAYERS = _env_int("LLM_GPU_LAYERS", 999)
LLM_AUTBOOST_ENABLE = _env_bool("LLM_AUTBOOST_ENABLE", True)

# 0 = auto (cpu_count - 1).
LLM_THREADS = _env_int("LLM_THREADS", 0)
LLM_AUTOSIZE_MAX_NEW_TOKENS = _env_bool("LLM_AUTOSIZE_MAX_NEW_TOKENS", False)
LLM_ENABLE_RETRY = _env_bool("LLM_ENABLE_RETRY", True)
LLM_WARMUP = _env_bool("LLM_WARMUP", True)

# Persisted GPU/thread tuning: "off" = ignore, "hint" = read only, "on" = read and write.
LLM_TUNING_CACHE_MODE = _env_str("LLM_TUNING_CACHE_MODE", "on")
LLM_BACKENDS_ORDER = ("ctransformers",)

# Answer post-processing
SANITIZE_OUTPUT        = _env_bool("SANITIZE_OUTPUT", True)
APPEND_DISCLAIMER       = _env_bool("APPEND_DISCLAIMER", False)

DISCLAIMER_PL = (
    "To nie jest porada lekarska. Jeśli objawy się nasilają albo pojawiają się "
    "kryteria pilnej konsultacji, skontaktuj się z lekarzem lub pogotowiem."
)
DISCLAIMER_EN = (
    "This is not medical advice. If your symptoms worsen or red flags appear, "
    "seek medical care or emergency services."
)

FORCE_SECTIONS          = _env_bool("FORCE_SECTIONS", False)
FORCE_SECTIONS_ORDER    = [
    "Możliwe przyczyny (Top-3)",
    "Czerwone flagi",
    "Co teraz",
    "Kiedy do lekarza",
]

# Symptom dictionary and matcher. With EMB_RULE_FIRST, dictionary aliases shorter than
# EMB_RULE_MIN_LEN chars are skipped.
SYMPTOMS_DICT_PATH = _env_path("SYMPTOMS_DICT_PATH", "data/symptoms_pl_en.json")
EMB_USE_DICTIONARY  = _env_bool("EMB_USE_DICTIONARY", True)
EMB_RULE_FIRST      = _env_bool("EMB_RULE_FIRST", True)
EMB_RULE_MIN_LEN    = _env_int("EMB_RULE_MIN_LEN", 3)
DISEASES_DICT_PATH  = _env_path("DISEASES_DICT_PATH", "data/diseases_pl_en.json")

# A negation cue (NEGATION_PATTERNS) counts only within MATCHER_NEG_WINDOW_CHARS before a symptom.
MATCHER_FUZZY_MIN_LEN     = _env_int("MATCHER_FUZZY_MIN_LEN", 6)
MATCHER_NEG_WINDOW_CHARS  = _env_int("MATCHER_NEG_WINDOW_CHARS", 30)
MATCHER_MIN_TOKEN_LEN     = _env_int("MATCHER_MIN_TOKEN_LEN", 3)

# XGBoost training: Bayesian hyperparameter search iterations and CV folds.
XGB_N_ITER_BAYES   = _env_int("XGB_N_ITER_BAYES", 32)
XGB_RANDOM_STATE    = GLOBAL_SEED
XGB_CV_SPLITS       = _env_int("XGB_CV_SPLITS", 3)
XGB_USE_GPU         = _env_bool("XGB_USE_GPU", True)
XGB_TREE_METHOD     = _env_str("XGB_TREE_METHOD", "hist")
# Calibration = temperature scaling on a held-out set, never on training data.
XGB_CALIBRATE             = _env_bool("XGB_CALIBRATE", True)

# Chat flow: keep asking follow-ups until Top-1 >= FOLLOWUP_CONFIDENCE with at least
# MIN_SYMPTOMS symptoms, or until STOP_FOLLOWUPS_AT questions; then assess (with a caveat
# if symptoms are still too few). Thresholds tuned by simulating chats on the test set
# (backend/tools/simulate_followups.py). PREDICT_ONLY_ON_NEW reuses the cached ranking
# when the symptom vector has not changed.
STOP_FOLLOWUPS_AT   = _env_int("STOP_FOLLOWUPS_AT", 5)
FOLLOWUP_CONFIDENCE = _env_float("FOLLOWUP_CONFIDENCE", 0.7)
PREDICT_ONLY_ON_NEW = _env_bool("PREDICT_ONLY_ON_NEW", True)
TOPN_DIAGNOSES      = _env_int("TOPN_DIAGNOSES", 3)

# River one-vs-rest logistic regression (FTRL-Proximal: ALPHA/BETA learning rate, L1/L2 penalties).
RIVER_MODEL_PATH      = _env_path("RIVER_MODEL_PATH", "ml/river_ovr_ftrl.joblib")
RIVER_CALIB_PATH       = _env_path("RIVER_CALIB_PATH", "ml/river_ovr_ftrl.calibration.joblib")
RIVER_FEATURES_PATH    = _env_path("RIVER_FEATURES_PATH", "ml/river_features.json")
RIVER_ALPHA            = _env_float("RIVER_ALPHA", 0.1)
RIVER_BETA             = _env_float("RIVER_BETA", 1.0)
RIVER_L1               = _env_float("RIVER_L1", 1e-6)
RIVER_L2               = _env_float("RIVER_L2", 1e-4)
RIVER_CLIP_GRADIENT    = _env_float("RIVER_CLIP_GRADIENT", 5.0)
# Sample weights: clipped below at WEIGHT_MIN, divided by their median, then capped at WEIGHT_CAP.
RIVER_WEIGHT_MEDIAN_NORM = _env_bool("RIVER_WEIGHT_MEDIAN_NORM", True)
RIVER_WEIGHT_CAP         = _env_float("RIVER_WEIGHT_CAP", 25.0)
RIVER_WEIGHT_MIN         = _env_float("RIVER_WEIGHT_MIN", 0.2)
RIVER_CALIBRATION_CSV    = _env_path("RIVER_CALIBRATION_CSV", "data/v2/calibration.csv" if _V2 else "data/calibration.csv")
XGB_CALIBRATION_CSV      = _env_path("XGB_CALIBRATION_CSV", RIVER_CALIBRATION_CSV)
RIVER_CALIBRATE_ON_START = _env_bool("RIVER_CALIBRATE_ON_START", True)
# Cold start of River: "batch" = a few mini-batch epochs (Adam, numpy), then online SGD
# updates; "online" = legacy single online FTRL pass, clearly weaker.
RIVER_BOOTSTRAP_MODE     = _env_str("RIVER_BOOTSTRAP_MODE", "batch")
RIVER_BATCH_EPOCHS       = _env_int("RIVER_BATCH_EPOCHS", 12)
RIVER_BATCH_LR           = _env_float("RIVER_BATCH_LR", 0.02)
RIVER_BATCH_L2           = _env_float("RIVER_BATCH_L2", 1e-5)
RIVER_BATCH_SIZE         = _env_int("RIVER_BATCH_SIZE", 1024)
RIVER_ONLINE_LR          = _env_float("RIVER_ONLINE_LR", 0.01)

# Ensemble (XGB + River): XGB dominates, River corrects. Base weights for "heuristic" fusion.
ENSEMBLE_W_XGB      = _env_float("ENSEMBLE_W_XGB", 0.8)
ENSEMBLE_W_RIVER     = _env_float("ENSEMBLE_W_RIVER", 0.2)

# 0 = fuse full distributions over all classes (recommended).
# >0 = only Top-K from each model (missing classes get that model's minimum).
ENSEMBLE_K_FUSE      = _env_int("ENSEMBLE_K_FUSE", 0)

# Minimum share each model keeps after confidence-based reweighting.
ENSEMBLE_MIN_XGB_WEIGHT   = _env_float("ENSEMBLE_MIN_XGB_WEIGHT", 0.55)
ENSEMBLE_MIN_RIVER_WEIGHT = _env_float("ENSEMBLE_MIN_RIVER_WEIGHT", 0.15)

# If one model's confidence exceeds the other's by at least this gap, it gets the max share (gating).
ENSEMBLE_CONF_GAP    = _env_float("ENSEMBLE_CONF_GAP", 0.20)
# Fusion of XGB + River:
#   "linear"    - p = a*p_xgb + (1-a)*p_river, a fitted on the calibration set (default),
#   "log"       - p ~ p_xgb^w * p_river^(1-w), w and temperature from calibration,
#   "heuristic" - legacy dynamic weights (ENSEMBLE_W_*, gating), no fitting.
# Without a calibration file the ensemble always uses "heuristic".
ENSEMBLE_FUSION      = _env_str("ENSEMBLE_FUSION", "linear")

# Logging
LOG_LEVEL          = _env_str("LOG_LEVEL", "INFO")
LOG_PRETTY          = _env_bool("LOG_PRETTY", True)
LOG_SAMPLE_INPUTS   = _env_bool("LOG_SAMPLE_INPUTS", True)

# Final summary: "llm" generates text with the LLM, "template" uses a fixed template.
SUMMARY_MODE                = _env_str("SUMMARY_MODE", "llm")
PROMPT_SYMPTOMS_MAX_FOR_LLM  = _env_int("PROMPT_SYMPTOMS_MAX_FOR_LLM", 10)

# Auth: backend session JWT secret falls back to SUPABASE_JWT_SECRET; one of them is required.
_AUTH_FALLBACK = _env_str("AUTH_SECRET_KEY", default=None)
if not _AUTH_FALLBACK:
    _AUTH_FALLBACK = _env_str("SUPABASE_JWT_SECRET", required=True)

AUTH_SECRET_KEY              = _AUTH_FALLBACK
ACCESS_TOKEN_EXPIRE_MINUTES  = _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 60)
SESSION_COOKIE_NAME          = _env_str("SESSION_COOKIE_NAME", "sc_session")
SESSION_COOKIE_SECURE        = _env_bool("SESSION_COOKIE_SECURE", False)
SESSION_COOKIE_SAMESITE      = _env_str("SESSION_COOKIE_SAMESITE", "lax")

# Token for admin endpoints /river/* (incremental training, CSV evaluation).
# Empty = endpoints disabled (403).
ADMIN_API_TOKEN              = _env_str("ADMIN_API_TOKEN", "").strip()

# SameSite=None requires Secure=True, otherwise modern browsers drop the cookie and login breaks.
if SESSION_COOKIE_SAMESITE.strip().lower() == "none" and not SESSION_COOKIE_SECURE:
    logging.getLogger("symptomchecker.config").warning(
        "SESSION_COOKIE_SAMESITE=none wymaga SESSION_COOKIE_SECURE=true - "
        "przegladarki odrzuca ciasteczko sesji. Ustaw SECURE=true (HTTPS) albo SAMESITE=lax."
    )

# In-memory chat SessionStore: idle TTL in seconds (4 h) and max sessions kept.
SESSION_STORE_TTL_SECONDS   = _env_int("SESSION_STORE_TTL_SECONDS", 4 * 60 * 60)
SESSION_STORE_MAX_SESSIONS  = _env_int("SESSION_STORE_MAX_SESSIONS", 5000)

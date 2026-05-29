import os
from pathlib import Path

# ───────────────────────── .env loader (root lub backend) ─────────────────────────
try:
    from dotenv import load_dotenv  # type: ignore
    for candidate in (
        Path(__file__).resolve().parents[1] / ".env",  # <repo>/.env
        Path(__file__).resolve().parent / ".env",      # backend/.env (opcjonalnie)
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break
except Exception:
    pass

# ───────────────────────── helpers do env ─────────────────────────
def _env_str(name: str, default: str | None = None, *, required: bool = False) -> str:
    v = os.getenv(name, default if default is not None else "")
    if required and str(v).strip() == "":
        raise RuntimeError(f"Brak wymaganej zmiennej środowiskowej: {name}")
    return v

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

# ───────────────────────── Runtime / środowisko ─────────────────────────
APP_ENV            = _env_str("APP_ENV", "dev")
DEBUG              = _env_bool("DEBUG", APP_ENV.lower() != "prod")
GLOBAL_SEED        = _env_int("GLOBAL_SEED", 42)

# ───────────────────────── Ścieżki i pliki (ML) ─────────────────────────
MODEL_PATH         = _env_str("MODEL_PATH", "ml/xgb_model.joblib")
LE_PATH            = _env_str("LE_PATH", "ml/label_encoder.joblib")
BEST_PARAMS_PATH   = _env_str("BEST_PARAMS_PATH", "ml/xgb_best_params.joblib")

DATA_CSV           = _env_str("DATA_CSV", "data/Final_Dataset_Diseases_and_Symptoms.csv")
TEST_CSV           = _env_str("TEST_CSV", "data/Final_Testset_for_Eval.csv")
CLASS_WEIGHT_CSV   = _env_str("CLASS_WEIGHT_CSV", "data/Final_Dataset_Diseases_and_Symptoms.class_weight.csv")

# ───────────────────────── Frontend (CORS) ─────────────────────────
FRONTEND_ORIGIN    = _env_str("FRONTEND_ORIGIN", "http://localhost:3000")

# ───────────────────────── Logika czatu / objawów ─────────────────────────
MIN_SYMPTOMS       = _env_int("MIN_SYMPTOMS", 3)

STOP_PHRASES = [
    "to wszystko", "nie mam więcej objawów", "koniec", "stop",
    "to już wszystko", "to wszystko.",
    "that's all", "no more symptoms", "end", "stop.", "stop!", "that's it"
]

NEGATION_PATTERNS  = ["nie mam", "bez ", "brak ", "zaprzeczam "]

YES_TOKENS = {"tak", "ta", "y", "yes", "yeah", "yep", "jasne", "oczywiście"}
NO_TOKENS  = {"nie", "n", "no", "nope"}

FOLLOWUP_BLOCKLIST_TOKENS = {
    "scrotum","testes","penis","vagina","anus","perianal",
    "infant","newborn","pregnancy","lactation",
    "knee","elbow","finger","toe","earlobe","eyebrow","eyelid",
}

# ───────────────────────── Parametry generacji (LLM) ─────────────────────────
LLM_MAX_NEW_TOKENS         = _env_int("LLM_MAX_NEW_TOKENS", 2000)
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

# ───────────────────────── LLM – SLOT (ctransformers) ─────────────────────────
LLM_SLOTS = {
    "bielik": {
        "REPO": "weights/bielik-gguf",
        "FILENAME": "Bielik-11B-v2.3-Instruct.Q6_K.gguf",
        "MODEL_TYPE": "llama"
    },
}

LLM_GPU_LAYERS = _env_int("LLM_GPU_LAYERS", 999)
LLM_AUTBOOST_ENABLE = _env_bool("LLM_AUTBOOST_ENABLE", True)

LLM_THREADS = _env_int("LLM_THREADS", 0)
LLM_AUTOSIZE_MAX_NEW_TOKENS = _env_bool("LLM_AUTOSIZE_MAX_NEW_TOKENS", False)
LLM_ENABLE_RETRY = _env_bool("LLM_ENABLE_RETRY", True)
LLM_WARMUP = _env_bool("LLM_WARMUP", True)

LLM_TUNING_CACHE_MODE = _env_str("LLM_TUNING_CACHE_MODE", "on")
LLM_BACKENDS_ORDER = ("ctransformers",)

# ───────────────────────── Post-processing odpowiedzi ─────────────────────────
SANITIZE_OUTPUT         = _env_bool("SANITIZE_OUTPUT", True)
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

# ───────────────────────── Słownik objawów ─────────────────────────
SYMPTOMS_DICT_PATH  = _env_str("SYMPTOMS_DICT_PATH", "data/symptoms_pl_en.json")
EMB_USE_DICTIONARY  = _env_bool("EMB_USE_DICTIONARY", True)
EMB_RULE_FIRST      = _env_bool("EMB_RULE_FIRST", True)
EMB_RULE_MIN_LEN    = _env_int("EMB_RULE_MIN_LEN", 3)
DISEASES_DICT_PATH  = _env_str("DISEASES_DICT_PATH", "data/diseases_pl_en.json")

MATCHER_FUZZY_MIN_LEN     = _env_int("MATCHER_FUZZY_MIN_LEN", 6)
MATCHER_NEG_WINDOW_CHARS  = _env_int("MATCHER_NEG_WINDOW_CHARS", 30)
MATCHER_MIN_TOKEN_LEN     = _env_int("MATCHER_MIN_TOKEN_LEN", 3)

# ───────────────────────── XGBoost / trenowanie / kalibracja ─────────────────────────
XGB_N_ITER_BAYES    = _env_int("XGB_N_ITER_BAYES", 32)
XGB_RANDOM_STATE    = GLOBAL_SEED
XGB_CV_SPLITS       = _env_int("XGB_CV_SPLITS", 3)
XGB_USE_GPU         = _env_bool("XGB_USE_GPU", True)
XGB_TREE_METHOD     = _env_str("XGB_TREE_METHOD", "hist")
XGB_CALIBRATE             = _env_bool("XGB_CALIBRATE", True)
XGB_CALIBRATION_METHOD    = _env_str("XGB_CALIBRATION_METHOD", "sigmoid")

# ───────────────────────── Sterowanie przepływem w czacie ─────────────────────────
STOP_FOLLOWUPS_AT   = _env_int("STOP_FOLLOWUPS_AT", 4)
PREDICT_ONLY_ON_NEW = _env_bool("PREDICT_ONLY_ON_NEW", True)
TOPN_DIAGNOSES      = _env_int("TOPN_DIAGNOSES", 3)

# ───────────────────────── River OVR Logistic (FTRL) ─────────────────────────
RIVER_MODEL_PATH       = _env_str("RIVER_MODEL_PATH", "ml/river_ovr_ftrl.joblib")
RIVER_CALIB_PATH       = _env_str("RIVER_CALIB_PATH", "ml/river_ovr_ftrl.calibration.joblib")
RIVER_FEATURES_PATH    = _env_str("RIVER_FEATURES_PATH", "ml/river_ovr_ftrl.features.json")
RIVER_ALPHA            = _env_float("RIVER_ALPHA", 0.1)
RIVER_BETA             = _env_float("RIVER_BETA", 1.0)
RIVER_L1               = _env_float("RIVER_L1", 1e-6)
RIVER_L2               = _env_float("RIVER_L2", 1e-4)
RIVER_CLIP_GRADIENT    = _env_float("RIVER_CLIP_GRADIENT", 5.0)
RIVER_WEIGHT_MEDIAN_NORM = _env_bool("RIVER_WEIGHT_MEDIAN_NORM", True)
RIVER_WEIGHT_CAP         = _env_float("RIVER_WEIGHT_CAP", 25.0)
RIVER_WEIGHT_MIN         = _env_float("RIVER_WEIGHT_MIN", 0.2)
RIVER_CALIBRATION_CSV    = _env_str("RIVER_CALIBRATION_CSV", "data/calibration.csv")
RIVER_CALIBRATE_ON_START = _env_bool("RIVER_CALIBRATE_ON_START", True)

# ───────────────────────── Ensemble (XGB + River) – LEKKO MOCNIEJSZY RIVER ────────
ENSEMBLE_W_XGB       = _env_float("ENSEMBLE_W_XGB", 0.8)
ENSEMBLE_W_RIVER     = _env_float("ENSEMBLE_W_RIVER", 0.2)

ENSEMBLE_K_FUSE      = _env_int("ENSEMBLE_K_FUSE", 9)

ENSEMBLE_MIN_XGB_WEIGHT   = _env_float("ENSEMBLE_MIN_XGB_WEIGHT", 0.55)
ENSEMBLE_MIN_RIVER_WEIGHT = _env_float("ENSEMBLE_MIN_RIVER_WEIGHT", 0.15)

ENSEMBLE_CONF_GAP    = _env_float("ENSEMBLE_CONF_GAP", 0.20)

# ───────────────────────── Logowanie / diagnostyka ─────────────────────────
LOG_LEVEL           = _env_str("LOG_LEVEL", "INFO")
LOG_PRETTY          = _env_bool("LOG_PRETTY", True)
LOG_SAMPLE_INPUTS   = _env_bool("LOG_SAMPLE_INPUTS", True)

# ───────────────────────── TRYB PODSUMOWANIA ─────────────────────────
SUMMARY_MODE                 = _env_str("SUMMARY_MODE", "llm")
PROMPT_SYMPTOMS_MAX_FOR_LLM  = _env_int("PROMPT_SYMPTOMS_MAX_FOR_LLM", 10)

# ───────────────────────── Auth / sesje ─────────────────────────
_AUTH_FALLBACK = _env_str("AUTH_SECRET_KEY", default=None)
if not _AUTH_FALLBACK:
    _AUTH_FALLBACK = _env_str("SUPABASE_JWT_SECRET", required=True)

AUTH_SECRET_KEY              = _AUTH_FALLBACK
ACCESS_TOKEN_EXPIRE_MINUTES  = _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 60)
SESSION_COOKIE_NAME          = _env_str("SESSION_COOKIE_NAME", "sc_session")
SESSION_COOKIE_SECURE        = _env_bool("SESSION_COOKIE_SECURE", False)
SESSION_COOKIE_SAMESITE      = _env_str("SESSION_COOKIE_SAMESITE", "lax")

# ───────────────────────── SessionStore (in-memory) ─────────────────────────
SESSION_STORE_TTL_SECONDS   = _env_int("SESSION_STORE_TTL_SECONDS", 4 * 60 * 60)
SESSION_STORE_MAX_SESSIONS  = _env_int("SESSION_STORE_MAX_SESSIONS", 5000)
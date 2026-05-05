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
APP_ENV            = "dev"
DEBUG              = True
GLOBAL_SEED        = 42

# ───────────────────────── Ścieżki i pliki (ML) ─────────────────────────
MODEL_PATH         = "ml/xgb_model.joblib"
LE_PATH            = "ml/label_encoder.joblib"
BEST_PARAMS_PATH   = "ml/xgb_best_params.joblib"

DATA_CSV           = "data/Final_Dataset_Diseases_and_Symptoms.csv"
TEST_CSV           = "data/Final_Testset_for_Eval.csv"
CLASS_WEIGHT_CSV   = "data/Final_Dataset_Diseases_and_Symptoms.class_weight.csv"

# ───────────────────────── Frontend (CORS) ─────────────────────────
FRONTEND_ORIGIN    = _env_str("FRONTEND_ORIGIN", "http://localhost:3000")

# ───────────────────────── Logika czatu / objawów ─────────────────────────
MIN_SYMPTOMS       = 3

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
LLM_MAX_NEW_TOKENS      = 2000
LLM_TEMP                = 0.55
LLM_TOP_P               = 0.9
LLM_TOP_K               = 40
LLM_REPETITION_PENALTY  = 1.05
LLM_MIN_OUTPUT_CHARS    = 550

LLM_CONTEXT_LEN     = 8192
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

LLM_GPU_LAYERS = 999
LLM_AUTBOOST_ENABLE = True

LLM_THREADS = 0
LLM_AUTOSIZE_MAX_NEW_TOKENS = True
LLM_ENABLE_RETRY = True
LLM_WARMUP = True

LLM_TUNING_CACHE_MODE = "on"
LLM_BACKENDS_ORDER = ("ctransformers",)

# ───────────────────────── Post-processing odpowiedzi ─────────────────────────
SANITIZE_OUTPUT         = True
APPEND_DISCLAIMER       = False

DISCLAIMER_PL = (
    "To nie jest porada lekarska. Jeśli objawy się nasilają albo pojawiają się "
    "kryteria pilnej konsultacji, skontaktuj się z lekarzem lub pogotowiem."
)
DISCLAIMER_EN = (
    "This is not medical advice. If your symptoms worsen or red flags appear, "
    "seek medical care or emergency services."
)

FORCE_SECTIONS          = False
FORCE_SECTIONS_ORDER    = [
    "Możliwe przyczyny (Top-3)",
    "Czerwone flagi",
    "Co teraz",
    "Kiedy do lekarza",
]

# ───────────────────────── Słownik objawów ─────────────────────────
SYMPTOMS_DICT_PATH  = "data/symptoms_pl_en.json"
EMB_USE_DICTIONARY  = True
EMB_RULE_FIRST      = True
EMB_RULE_MIN_LEN    = 3
DISEASES_DICT_PATH  = "data/diseases_pl_en.json"

# ───────────────────────── XGBoost / trenowanie / kalibracja ─────────────────────────
XGB_N_ITER_BAYES    = 32
XGB_RANDOM_STATE    = GLOBAL_SEED
XGB_CV_SPLITS       = 3
XGB_USE_GPU         = True
XGB_TREE_METHOD     = "hist"
XGB_CALIBRATE             = True
XGB_CALIBRATION_METHOD    = "sigmoid"

# ───────────────────────── Sterowanie przepływem w czacie ─────────────────────────
STOP_FOLLOWUPS_AT   = 4
PREDICT_ONLY_ON_NEW = True
TOPN_DIAGNOSES      = 3

# ───────────────────────── River OVR Logistic (FTRL) ─────────────────────────
RIVER_MODEL_PATH       = "ml/river_ovr_ftrl.joblib"
RIVER_CALIB_PATH       = "ml/river_ovr_ftrl.calibration.joblib"
RIVER_ALPHA            = 0.1
RIVER_BETA             = 1.0
RIVER_L1               = 1e-6
RIVER_L2               = 1e-4
RIVER_CLIP_GRADIENT    = 5.0
RIVER_WEIGHT_MEDIAN_NORM = True
RIVER_WEIGHT_CAP         = 25.0
RIVER_WEIGHT_MIN         = 0.2
RIVER_CALIBRATION_CSV    = "data/calibration.csv"
RIVER_CALIBRATE_ON_START = True

# ───────────────────────── Ensemble (XGB + River) – LEKKO MOCNIEJSZY RIVER ────────
# Bazowe wagi – XGB mocniejszy globalnie, ale River ma realny udział
ENSEMBLE_W_XGB       = 0.8
ENSEMBLE_W_RIVER     = 0.2

# Ile topów pobierać z modeli
ENSEMBLE_K_FUSE      = 9

# Minimalny udział – River zawsze ma sensowny głos
ENSEMBLE_MIN_XGB_WEIGHT   = 0.55
ENSEMBLE_MIN_RIVER_WEIGHT = 0.15

# Próg kiedy mówimy „ten model jest wyraźnie pewniejszy”
ENSEMBLE_CONF_GAP    = 0.20

# ───────────────────────── Logowanie / diagnostyka ─────────────────────────
LOG_LEVEL           = "INFO"
LOG_PRETTY          = True
LOG_SAMPLE_INPUTS   = True

# ───────────────────────── TRYB PODSUMOWANIA ─────────────────────────
SUMMARY_MODE                 = "llm"
PROMPT_SYMPTOMS_MAX_FOR_LLM  = 6

# ───────────────────────── Auth / sesje ─────────────────────────
_AUTH_FALLBACK = _env_str("AUTH_SECRET_KEY", default=None)
if not _AUTH_FALLBACK:
    _AUTH_FALLBACK = _env_str("SUPABASE_JWT_SECRET", required=True)

AUTH_SECRET_KEY              = _AUTH_FALLBACK
ACCESS_TOKEN_EXPIRE_MINUTES  = _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 60)
SESSION_COOKIE_NAME          = _env_str("SESSION_COOKIE_NAME", "sc_session")
SESSION_COOKIE_SECURE        = _env_bool("SESSION_COOKIE_SECURE", False)
SESSION_COOKIE_SAMESITE      = _env_str("SESSION_COOKIE_SAMESITE", "lax")

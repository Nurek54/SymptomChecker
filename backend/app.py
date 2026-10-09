import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# The backend uses short imports (config, core, services, models, api) plus the
# top-level "database" package. Both dirs go on sys.path BEFORE any project import,
# so the app starts the same way via "python main.py", "uvicorn app:create_app --factory"
# (from backend/) and "uvicorn backend.app:create_app --factory" (from the repo root).
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for _p in (str(PROJECT_ROOT), str(BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Relative paths (ml/, data/, weights/) assume cwd = backend/.
os.chdir(BASE_DIR)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402

from config import FRONTEND_ORIGIN, LOG_LEVEL  # noqa: E402

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("symptomchecker.app")


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from database.db import engine
        from sqlalchemy import inspect
        from core.startup import StartupManager
        from core.state import AppState
        from api.routers.auth import prewarm_auth

        try:
            insp = inspect(engine)
            log.info("🗄️  DB URL: %s | tables: %s", engine.url, ", ".join(insp.get_table_names()))
        except Exception:
            log.info("🗄️  DB URL: %s", engine.url)

        try:
            prewarm_auth()
            log.info("🔐 Auth prewarmed (HS256).")
        except Exception as e:
            log.debug("Auth prewarm skipped: %s", e)

        import time as _time
        from utils import fmt_duration
        t_start = _time.time()
        state: AppState = await StartupManager().run()

        # River runs after StartupManager because it reuses the XGB feature order.
        t_river = _time.time()
        log.info("🤖 [ETAP 4/5] Trening/ładowanie River…")
        try:
            from models.river_manager import RiverManager
            river = RiverManager()

            xgb_feats = getattr(getattr(state, "xgb", None), "features", None)
            if xgb_feats:
                river.set_features(xgb_feats)

            river.train()
            if not getattr(river, "model", None):
                raise RuntimeError("River model not initialized after train().")

            state.river = river
            log.info("🤖 [ETAP 4/5] River gotowy (features=%d) w %s.",
                     len(getattr(river, "features", []) or []), fmt_duration(_time.time() - t_river))
        except Exception as e:
            state.river = None
            log.exception("River init failed: %s", e)

        # Ensemble calibration and quality report (recomputed only when models changed).
        log.info("⚖️ [ETAP 5/5] Kalibracja ensemble i raport jakości…")
        try:
            from services.ensemble_calibration import ensure_ensemble_calibration
            ensure_ensemble_calibration(state)
        except Exception as e:
            log.exception("Kalibracja ensemble nieudana (T_ens=1): %s", e)

        app.state.app_state = state
        log.info("✅ Startup zakończony w %s. Raport jakości modeli: ml/training_report.json",
                 fmt_duration(_time.time() - t_start))

        yield

        try:
            sessions = getattr(state, "sessions", None)
            if sessions and hasattr(sessions, "stop_janitor"):
                sessions.stop_janitor()
        except Exception:
            log.debug("Sessions janitor stop skipped.", exc_info=True)
        log.info("👋 Shutdown zakończony.")

    app = FastAPI(
        title="SymptomChatBot (PL, XGBoost+BayesOpt, LLM, River OVR)",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from api.routers.errors import validation_exception_handler
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    from api.routers.chat import router as chat_router
    from api.routers.health import router as health_router
    from api.routers.auth import router as auth_router
    from api.routers.history import router as history_router
    from api.routers.dicts import router as dicts_router
    from api.routers.session_ops import router as session_ops_router
    from api.routers.river_router import router as river_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(history_router)
    app.include_router(dicts_router)
    app.include_router(session_ops_router)
    app.include_router(river_router)

    return app

import os
import logging
import sys
from pathlib import Path

# sys.path must be set up before any backend.* imports
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.startup import StartupManager
from backend.core.state import AppState
from backend.config import FRONTEND_ORIGIN

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("symptomchecker.app")

def create_app() -> FastAPI:
    app = FastAPI(title="SymptomChatBot (PL/EN, XGBoost+BayesOpt, LLM, River OVR)")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # === Handlery błędów 422 (czytelne komunikaty do UI) ===
    from backend.api.routers.errors import validation_exception_handler
    from fastapi.exceptions import RequestValidationError
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Podgląd tabel (diagnostyka DB)
    from database.db import engine
    from sqlalchemy import inspect

    # >>> DODANE: import prewarm_auth
    from backend.api.routers.auth import prewarm_auth

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            insp = inspect(engine)
            log.info("🗄️  DB URL: %s | tables: %s", engine.url, ", ".join(insp.get_table_names()))
        except Exception:
            log.info("🗄️  DB URL: %s", engine.url)

        # 1) Rozgrzej HS256 (zero sieci, zero I/O)
        try:
            prewarm_auth()
            log.info("🔐 Auth prewarmed (HS256).")
        except Exception as e:
            log.debug("Auth prewarm skipped: %s", e)

        # 2) Reszta startu (modele itp.)
        state: AppState = await StartupManager().run()

        # River (jak u Ciebie)
        try:
            from backend.models.river_manager import RiverManager
            river = RiverManager()

            xgb_feats = getattr(getattr(state, "xgb", None), "features", None)
            if xgb_feats:
                river.set_features(xgb_feats)

            river.train()
            if not getattr(river, "model", None):
                raise RuntimeError("River model not initialized after train().")

            state.river = river
            log.info("🤖 River gotowy (features=%d).", len(getattr(river, "features", []) or []))
        except Exception as e:
            state.river = None
            log.exception("River init failed: %s", e)

        app.state.app_state = state
        log.info("✅ Startup zakończony.")

    # Routers
    from backend.api.routers.chat import router as chat_router
    from backend.api.routers.health import router as health_router
    from backend.api.routers.auth import router as auth_router
    from backend.api.routers.history import router as history_router
    from backend.api.routers.dicts import router as dicts_router
    from backend.api.routers.session_ops import router as session_ops_router
    from backend.api.routers.river_router import router as river_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(history_router)
    app.include_router(dicts_router)
    app.include_router(session_ops_router)
    app.include_router(river_router)

    return app

app = create_app()

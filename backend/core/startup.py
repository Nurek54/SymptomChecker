# backend/core/startup.py
from __future__ import annotations
import logging

import torch

from core.state import AppState
from models.xgb_manager import XGBManager
from models.llm_manager import LLMManager
from services.symptom_matcher import SymptomMatcher
from services.session_store import SessionStore
from services.symptom_stats import SymptomStats

log = logging.getLogger("symptomchecker.startup")


class StartupManager:
    async def run(self) -> AppState:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Torch device: %s", device)

        xgb = XGBManager()
        log.info("🔧 Trening/ładowanie XGBoost…")
        xgb.train()
        if not xgb.features:
            raise RuntimeError("Brak cech XGB — sprawdź dataset i trening.")

        llm = LLMManager()
        log.info("🧠 Ładowanie LLM…")
        llm.load()

        log.info("🧩 Buduję matcher…")
        matcher = SymptomMatcher(xgb.features, device=str(device))

        log.info("📊 Statystyki objawów do pytań uzupełniających…")
        symptom_stats = SymptomStats.load_or_build(xgb.features)

        sessions = SessionStore()
        if hasattr(sessions, "start_janitor"):
            try:
                sessions.start_janitor()
            except Exception:
                log.warning("SessionStore janitor nie wystartował.", exc_info=True)

        return AppState(
            device=device,
            xgb=xgb,
            llm=llm,
            matcher=matcher,
            sessions=sessions,
            river=None,
            symptom_stats=symptom_stats,
        )

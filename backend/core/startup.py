from __future__ import annotations
import logging
import time

import torch

from core.state import AppState
from models.xgb_manager import XGBManager
from models.llm_manager import LLMManager
from services.symptom_matcher import SymptomMatcher
from services.session_store import SessionStore
from services.symptom_stats import SymptomStats
from utils import fmt_duration

log = logging.getLogger("symptomchecker.startup")


class StartupManager:
    async def run(self) -> AppState:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Torch device: %s", device)

        t = time.time()
        xgb = XGBManager()
        log.info("🔧 [ETAP 1/5] Trening/ładowanie XGBoost…")
        xgb.train()
        log.info("🔧 [ETAP 1/5] XGBoost gotowy w %s", fmt_duration(time.time() - t))
        if not xgb.features:
            raise RuntimeError("Brak cech XGB - sprawdź dataset i trening.")

        t = time.time()
        llm = LLMManager()
        log.info("🧠 [ETAP 2/5] Ładowanie LLM…")
        llm.load()
        log.info("🧠 [ETAP 2/5] LLM gotowy w %s", fmt_duration(time.time() - t))

        t = time.time()
        log.info("📊 [ETAP 3/5] Statystyki objawów (pytania uzupełniające, cechy bez danych)…")
        symptom_stats = SymptomStats.load_or_build(xgb.features)
        log.info("📊 [ETAP 3/5] gotowe w %s", fmt_duration(time.time() - t))

        log.info("🧩 Buduję matcher…")
        matcher = SymptomMatcher(
            xgb.features,
            device=str(device),
            # pass counts only if stats are non-empty, otherwise every feature would be treated as dead
            feature_counts=(
                symptom_stats.feature_counts
                if symptom_stats is not None and any(symptom_stats.feature_counts.values())
                else None
            ),
        )

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

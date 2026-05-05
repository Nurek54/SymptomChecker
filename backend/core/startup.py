# backend/core/startup.py
import logging
import torch
from backend.core.state import AppState
from backend.models.xgb_manager import XGBManager
from backend.models.llm_manager import LLMManager
from backend.services.symptom_matcher import SymptomMatcher
from backend.services.session_store import SessionStore

log = logging.getLogger("symptomchecker.startup")

class StartupManager:
    async def run(self) -> AppState:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Torch device: %s", device)

        xgb = XGBManager()
        log.info("🔧 Trening/ładowanie XGBoost…")
        xgb.train()
        assert xgb.features, "Brak cech XGB — sprawdź dataset i trening."

        llm = LLMManager()
        log.info("🧠 Ładowanie LLM…")
        llm.load()

        log.info("🧩 Buduję matcher…")
        matcher = SymptomMatcher(xgb.features, device=str(device))

        sessions = SessionStore()
        return AppState(device=device, xgb=xgb, llm=llm, matcher=matcher, sessions=sessions)

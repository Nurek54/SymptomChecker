# backend/core/state.py
from dataclasses import dataclass
import torch
from backend.models.xgb_manager import XGBManager
from backend.models.llm_manager import LLMManager
from backend.services.symptom_matcher import SymptomMatcher
from backend.services.session_store import SessionStore

@dataclass
class AppState:
    device: torch.device
    xgb: XGBManager
    llm: LLMManager
    matcher: SymptomMatcher
    sessions: SessionStore

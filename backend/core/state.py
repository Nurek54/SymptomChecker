from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import torch

from models.xgb_manager import XGBManager
from models.llm_manager import LLMManager
from services.symptom_matcher import SymptomMatcher
from services.session_store import SessionStore

if TYPE_CHECKING:
    from models.river_manager import RiverManager


@dataclass
class AppState:
    device: torch.device
    xgb: XGBManager
    llm: LLMManager
    matcher: SymptomMatcher
    sessions: SessionStore
    river: Optional["RiverManager"] = field(default=None)
    # P(symptom | disease) for choosing follow-up questions (services.symptom_stats.SymptomStats)
    symptom_stats: Optional[Any] = field(default=None)
    # XGB + River fusion temperature (services/ensemble_calibration.py)
    ensemble_T: float = field(default=1.0)
    # fusion mode and XGB weight fitted on the calibration set (None = legacy heuristic)
    ensemble_mode: str = field(default="heuristic")
    ensemble_wx: Optional[float] = field(default=None)

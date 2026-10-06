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
    # P(objaw | choroba) do wyboru pytań uzupełniających (services.symptom_stats.SymptomStats)
    symptom_stats: Optional[Any] = field(default=None)

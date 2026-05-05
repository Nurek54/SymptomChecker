from __future__ import annotations

import os
import sys
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# PYTHONPATH + ALIASY – działa z:
#   SymptomChecker/backend
#   SymptomChecker/database
# =====================================================

CURRENT = os.path.dirname(os.path.abspath(__file__))     # .../backend/tests
BACKEND = os.path.dirname(CURRENT)                       # .../backend
PROJECT = os.path.dirname(BACKEND)                       # .../SymptomChecker

# Dodaj backend i projekt do sys.path
sys.path.insert(0, BACKEND)
sys.path.insert(0, PROJECT)

# Aliasujemy pakiety używane w backendzie przez krótkie nazwy
import backend.services as backend_services
import backend.models as backend_models
import backend.schemas as backend_schemas
import database as project_database  # zewnętrzny katalog "database"

# backend/core/state.py używa "models" / "services"
# api/routers/chat.py używa "schemas"
# database/* używa "database"
sys.modules.setdefault("services", backend_services)
sys.modules.setdefault("models", backend_models)
sys.modules.setdefault("schemas", backend_schemas)
sys.modules.setdefault("database", project_database)

# =====================================================
# Importy backendu (po ustawieniu aliasów)
# =====================================================

from backend.core.state import AppState
from backend.services.session_store import SessionStore
from backend.services.symptom_matcher import SymptomMatcher
from backend.services.ensemble import EnsembleTopK
from backend.config import MIN_SYMPTOMS, FRONTEND_ORIGIN

from backend.api.routers import chat as chat_router
from backend.api.routers import dicts as dicts_router
from backend.api.routers import session_ops as session_ops_router
from backend.api.routers import history as history_router
from backend.api.routers import auth as auth_router
from backend.api.routers import health as health_router


# =====================================================
# FAKE MODELS
# =====================================================

class FakeXGB:
    def __init__(self):
        self.features = ["fever", "cough", "sore_throat"]
        self.model = object()

    def predict_topk(self, vec, k: int):
        return [
            {"diagnosis_en": "flu", "diagnosis_pl": "grypa", "probability": 0.8},
            {"diagnosis_en": "cold", "diagnosis_pl": "przeziębienie", "probability": 0.15},
            {"diagnosis_en": "covid", "diagnosis_pl": "COVID-19", "probability": 0.05},
        ][:k]


class FakeRiver:
    def __init__(self):
        self.features = ["fever", "cough", "sore_throat"]
        self.model = object()

    def predict_topk(self, vec, k: int):
        return [
            {"diagnosis_en": "flu", "diagnosis_pl": "grypa", "probability": 0.7},
            {"diagnosis_en": "cold", "diagnosis_pl": "przeziębienie", "probability": 0.2},
            {"diagnosis_en": "covid", "diagnosis_pl": "COVID-19", "probability": 0.1},
        ][:k]


class FakeLLM:
    def __init__(self):
        self.model = object()

    def chat(self, *args, **kwargs):
        return "Testowa odpowiedź LLM (stub)"


class FakeDBSession:
    def __getattr__(self, item):
        def _(*args, **kwargs):
            return None
        return _


# =====================================================
# FIXTURE: app()
# =====================================================
@pytest.fixture(scope="session")
def app() -> FastAPI:
    app = FastAPI()

    # CORS – żeby test_cors_allows_frontend_origin nie walił 405
    origin = FRONTEND_ORIGIN or "http://localhost:3000"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router.router)
    app.include_router(auth_router.router)
    app.include_router(chat_router.router)
    app.include_router(history_router.router)
    app.include_router(dicts_router.router)
    app.include_router(session_ops_router.router)

    xgb = FakeXGB()
    river = FakeRiver()
    matcher = SymptomMatcher(xgb.features, device="cpu")
    sessions = SessionStore()
    llm = FakeLLM()

    state = AppState(
        device="cpu",   # typ torch.device nas tu nie obchodzi
        xgb=xgb,
        llm=llm,
        matcher=matcher,
        sessions=sessions,
    )
    state.river = river
    app.state.app_state = state

    # Override DB z zewnętrznego katalogu "database"
    from database.db import get_db

    def _fake_get_db():
        yield FakeDBSession()

    app.dependency_overrides[get_db] = _fake_get_db

    # Fake ChatService – żeby nie odpalać realnego LLM/XGB
    class FakeChatService:
        def __init__(self, st: AppState):
            self.st = st

        def handle_chat(self, req, patient_age=None):
            syms = list(self.st.sessions.get_symptoms(req.session_id) or [])
            if len(syms) < int(MIN_SYMPTOMS):
                return chat_router.ChatResponse(
                    response="Proszę podać więcej objawów.",
                    symptoms=sorted(syms),
                    top_diagnoses=[],
                )
            return chat_router.ChatResponse(
                response="Stub: diagnoza wstępna.",
                symptoms=sorted(syms),
                top_diagnoses=[
                    {"diagnosis_en": "flu", "diagnosis_pl": "grypa", "probability": 0.8},
                    {"diagnosis_en": "cold", "diagnosis_pl": "przeziębienie", "probability": 0.15},
                    {"diagnosis_en": "covid", "diagnosis_pl": "COVID-19", "probability": 0.05},
                ],
            )

    chat_router.ChatService = FakeChatService
    session_ops_router.EnsembleTopK = EnsembleTopK

    return app


# =====================================================
# FIXTURE: client()
# =====================================================
@pytest.fixture(scope="session")
def client(app: FastAPI) -> TestClient:
    return TestClient(app)

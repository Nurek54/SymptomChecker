# tests/test_regressions.py
"""
Testy regresji dla poprawek logiki: ensemble, matcher (odmiana i przeczenia),
triage, przebieg czatu (pętla pytań, "to wszystko", odtwarzanie sesji),
wybór pytań, router River i start aplikacji.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.services.ensemble import EnsembleTopK
from backend.services.symptom_matcher import SymptomMatcher
from backend.services.session_store import SessionStore
from backend.services.symptom_stats import SymptomStats
from backend.services.prompt_helpers import _extract_temp_high, _suggest_triage, _user_text
from backend.utils import contains_stop_phrase


# ───────────────────────── ensemble ─────────────────────────
class _Model:
    def __init__(self, items):
        self.items = items
        self.model = object()

    def predict_topk(self, vec, k=3):
        return self.items[:k]


class _St:
    pass


def _ens(x_items, r_items):
    st = _St()
    st.xgb = _Model(x_items)
    st.river = _Model(r_items)
    return EnsembleTopK(st)


def test_ensemble_confident_xgb_not_overturned_by_missing_class_in_river():
    # XGB jest pewny "a"; River prawie płaski i "a" nie ma w jego liście.
    x = [{"diagnosis_en": "a", "probability": 0.6}, {"diagnosis_en": "b", "probability": 0.05}] + [
        {"diagnosis_en": f"x{i}", "probability": 0.01} for i in range(7)
    ]
    r = [{"diagnosis_en": "b", "probability": 0.10}] + [
        {"diagnosis_en": f"r{i}", "probability": 0.09} for i in range(8)
    ]
    split = _ens(x, r).predict_split([[1, 0]], k=3)
    assert split["ensemble_top"][0]["diagnosis_en"] == "a"
    assert split["meta"]["gated_to"] == "xgb"
    # gating faktycznie zmienia wagi (maksymalny udział XGB)
    assert split["meta"]["wx_dyn"] == pytest.approx(1.0 - 0.15)


def test_ensemble_weights_respect_minimums():
    e = _ens([], [])
    wx, wr = e._clamp_weights(0.99)
    assert wr >= e.min_wr - 1e-9 and wx >= e.min_wx - 1e-9
    assert wx + wr == pytest.approx(1.0)


# ───────────────────────── matcher ─────────────────────────
@pytest.fixture(scope="module")
def matcher():
    feats = ["fever", "cough", "sore_throat", "headache", "runny_nose", "loss_of_appetite"]
    return SymptomMatcher(features=feats, device="cpu")


@pytest.mark.parametrize("text,pos,neg", [
    ("boli mnie głowa", {"headache"}, set()),
    ("kaszlę od rana", {"cough"}, set()),
    ("mam 39 stopni gorączki", {"fever"}, set()),
    ("bol glowy", {"headache"}, set()),
    ("nie mam gorączki", set(), {"fever"}),
    ("gorączki nie mam", set(), {"fever"}),
    ("ból głowy nie występuje", set(), {"headache"}),
    ("gorączka już minęła", set(), {"fever"}),
    ("kaszel? nie", set(), {"cough"}),
    ("wczoraj miałem gorączkę, dziś już nie", set(), {"fever"}),
    ("nie mam gorączki i kaszlu", set(), {"fever", "cough"}),
    ("nie mam gorączki, ale mam kaszel", {"cough"}, {"fever"}),
    ("bez gorączki, kaszel suchy", {"cough"}, {"fever"}),
    ("ból głowy nie ustępuje", {"headache"}, set()),
    ("kaszel nie daje mi spać", {"cough"}, set()),
    ("nie mam apetytu", {"loss_of_appetite"}, set()),
])
def test_matcher_inflection_and_negation(matcher, text, pos, neg):
    got_pos, got_neg = matcher.extract_detailed(text)
    assert set(got_pos) == pos, text
    assert set(got_neg) == neg, text


def test_matcher_no_false_positives(matcher):
    for t in ["Dziękuję za pomoc", "co mogę zrobić w domu?", "mam 30 lat", "to wszystko"]:
        assert matcher.extract_detailed(t) == ([], []), t


# ───────────────────────── triage / stop ─────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("mam 39 stopni", True),
    ("gorączka 40", True),
    ("mam 39,5°C", True),
    ("temperatura 38,9", False),
    ("rana ma 45 cm", False),
])
def test_high_fever_detection(text, expected):
    assert _extract_temp_high(text) is expected


def test_triage_ignores_assistant_warnings():
    hist = [
        {"role": "user", "content": "mam katar"},
        {"role": "assistant", "content": "Jeśli temperatura przekroczy 39°C, zgłoś się do lekarza."},
    ]
    level, _ = _suggest_triage(["katar"], 30, history_text=_user_text(hist))
    assert level == "niska"


def test_stop_phrase_whole_words_only():
    assert contains_stop_phrase("to wszystko")
    assert contains_stop_phrase("koniec.")
    assert not contains_stop_phrase("mam 39 stopni gorączki")


# ───────────────────────── wybór pytań ─────────────────────────
def test_best_question_prefers_discriminative_and_skips_duplicates():
    feats = ["cough", "fever", "blood_in_sputum", "hemoptysis", "rash"]
    classes = ["flu", "tb", "measles"]
    q = np.array([
        # cough fever blood hemo  rash
        [0.9, 0.9, 0.02, 0.02, 0.02],   # flu
        [0.9, 0.5, 0.80, 0.80, 0.02],   # tb
        [0.3, 0.9, 0.02, 0.02, 0.95],   # measles
    ])
    stats = SymptomStats(feats, classes, q)
    post = {"flu": 0.5, "tb": 0.5, "measles": 0.0}
    best = stats.best_question(post, ["blood_in_sputum", "rash"], reference=["cough"])
    assert best == "blood_in_sputum"
    # po zapytaniu o "krew w plwocinie" nie pytamy o niemal identyczne "krwioplucie"
    best2 = stats.best_question(post, ["hemoptysis", "rash"], reference=["cough", "blood_in_sputum"])
    assert best2 != "hemoptysis"


# ───────────────────────── przebieg czatu ─────────────────────────
class _FakeLLM:
    backend = "ctransformers"
    model = object()

    def generate(self, prompt, **kw):
        return "To wygląda na grypę. Odpocznij i dużo pij."


class _FakeModel:
    def __init__(self):
        self.features = ["fever", "cough", "sore_throat", "headache", "runny_nose", "vomiting", "rash", "chills"]
        self.model = object()

    def predict_topk(self, vec, k=3):
        items = [
            {"diagnosis_en": "flu", "probability": 0.6},
            {"diagnosis_en": "cold", "probability": 0.3},
            {"diagnosis_en": "covid", "probability": 0.1},
        ]
        return items[: min(k, 3)]


@pytest.fixture()
def chat_state(matcher):
    from backend.core.state import AppState
    m = _FakeModel()
    return AppState(device="cpu", xgb=m, llm=_FakeLLM(), matcher=SymptomMatcher(m.features),
                    sessions=SessionStore(), river=_FakeModel())


def _say(state, sid, text, **kw):
    from backend.services.chat_service import ChatService
    from backend.schemas import ChatRequest
    return ChatService(state).handle_chat(ChatRequest(session_id=sid, user_input=text, **kw))


def test_followups_stop_after_limit_with_low_confidence_note(chat_state):
    from backend.config import STOP_FOLLOWUPS_AT
    r = _say(chat_state, "s1", "mam kaszel")
    assert r.response.startswith("Czy masz")
    for _ in range(int(STOP_FOLLOWUPS_AT)):
        r = _say(chat_state, "s1", "nie")
    assert r.top_diagnoses, "po limicie pytań ma być ocena, a nie kolejne pytanie"
    assert "orientacyjna" in r.response


def test_stop_phrase_twice_gives_assessment(chat_state):
    r = _say(chat_state, "s2", "boli mnie głowa")
    r = _say(chat_state, "s2", "to wszystko")
    assert "co najmniej" in r.response and not r.top_diagnoses
    r = _say(chat_state, "s2", "to wszystko")
    assert r.top_diagnoses and "orientacyjna" in r.response


def test_no_symptoms_message(chat_state):
    r = _say(chat_state, "s3", "dzień dobry")
    assert "Nie rozpoznałem" in r.response and r.symptoms == []


def test_session_restored_from_client_after_restart(chat_state):
    r = _say(chat_state, "s4", "mam też katar", client_symptoms=["fever", "cough", "unknown_x"])
    assert set(r.symptoms) == {"fever", "cough", "runny_nose"}
    assert r.top_diagnoses


def test_negation_in_answer_is_respected(chat_state):
    _say(chat_state, "s5", "mam kaszel")
    r = _say(chat_state, "s5", "nie mam gorączki, ale mam katar")
    assert "fever" not in r.symptoms and "runny_nose" in r.symptoms


# ───────────────────────── router River ─────────────────────────
def test_river_admin_endpoints_disabled_without_token():
    from backend.api.routers import river_router

    app = FastAPI()
    app.include_router(river_router.router)
    client = TestClient(app)
    # stan aplikacji bez modelu River -> 503 dla odczytu, 403 dla admina
    app.state.app_state = type("S", (), {"river": None})()
    assert client.get("/river/status").status_code == 503
    r = client.post("/river/train_one", json={"label": "flu", "active_features": ["fever"]})
    assert r.status_code == 403


# ───────────────────────── start aplikacji ─────────────────────────
def test_create_app_builds_routes():
    app_mod = importlib.import_module("app")
    app = app_mod.create_app()
    paths = {getattr(r, "path", "") for r in app.routes}
    for p in ("/chat", "/health", "/history", "/river/status", "/sessions/{session_id}/symptoms"):
        assert p in paths

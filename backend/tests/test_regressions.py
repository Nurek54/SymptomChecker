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


# ───────────────────────── słownik vs dane (cechy bez danych) ─────────────────────────
def test_matcher_skips_dead_features_and_prefers_feature_with_data():
    feats = ["sharp_abdominal_pain", "abdominal_pain", "belly_pain", "fever"]
    counts = {"sharp_abdominal_pain": 14000, "abdominal_pain": 30, "belly_pain": 0, "fever": 10000}
    m = SymptomMatcher(features=feats, device="cpu", feature_counts=counts)
    assert not m.is_alive("belly_pain")
    pos, _ = m.extract_detailed("boli mnie brzuch")
    assert pos == ["sharp_abdominal_pain"]
    # aktywujemy tylko wybraną cechę, nie rzadkie synonimy
    assert m.expand_to_xgb("sharp_abdominal_pain") == {"sharp_abdominal_pain"}


def test_dicts_panel_hides_dead_features_and_dedupes_labels():
    from backend.api.routers.dicts import _project_to_xgb_features

    feats = ["stomach_bloating", "abdominal_distention", "distention_of_abdomen"]
    counts = {"stomach_bloating": 1000, "abdominal_distention": 60, "distention_of_abdomen": 0}
    m = SymptomMatcher(features=feats, device="cpu", feature_counts=counts)
    raw = [
        {"pl": "wzdęcie brzucha", "xgb_label": ["abdominal_distention"]},
        {"pl": "wzdęcie brzucha", "xgb_label": ["distention_of_abdomen"]},
        {"pl": "wzdęcie brzucha", "xgb_label": ["stomach_bloating"]},
    ]
    out = _project_to_xgb_features(raw, set(feats), m.resolve_labels, m._count)
    assert out == [{"en": "stomach_bloating", "pl": "wzdęcie brzucha", "category": None}]


# ───────────────────────── kalibracja modeli ─────────────────────────
def _softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def test_xgb_calibrator_prior_correction_used_only_when_it_helps():
    from backend.models.xgb_manager import TemperatureCalibrator

    rng = np.random.default_rng(0)
    K, N = 4, 4000
    true_logits = rng.normal(0, 2, size=(N, K))
    y = np.array([rng.choice(K, p=p) for p in _softmax_np(true_logits)])
    log_w = np.log(np.array([0.5, 1.0, 3.0, 5.0]))
    # model trenowany z wagami klas "zawyża" klasy o dużych wagach o log(w_c)
    biased = true_logits + log_w[None, :]
    cal = TemperatureCalibrator()
    cal.fit(biased, y, log_w=log_w)
    assert cal.log_w is not None and cal.T == pytest.approx(1.0, abs=0.1)
    # gdy model NIE jest przesunięty, korekta pogarsza NLL -> kalibrator jej nie używa
    cal2 = TemperatureCalibrator()
    cal2.fit(true_logits, y, log_w=log_w)
    assert cal2.log_w is None


def test_river_calibrator_transform_matches_fit():
    from backend.models.river_manager import RiverTemperatureCalibrator

    rng = np.random.default_rng(1)
    K, N = 5, 3000
    z_base = rng.normal(-3, 1, size=K)
    true_logits = rng.normal(0, 1.5, size=(N, K))
    y = np.array([rng.choice(K, p=p) for p in _softmax_np(true_logits)])
    Z = 2.0 * (true_logits + z_base[None, :])  # zbyt ostre i przesunięte o intercept
    cal = RiverTemperatureCalibrator(classes=[str(i) for i in range(K)])
    lam, T = cal.fit_scores(Z, 2.0 * z_base, y)
    assert lam == pytest.approx(1.0) and T == pytest.approx(2.0, rel=0.1)
    P = cal.transform(Z, 2.0 * z_base)
    np.testing.assert_allclose(P, _softmax_np((Z - lam * 2.0 * z_base[None, :]) / T))


def test_vectorized_fusion_matches_ensemble_fuse():
    from backend.services.ensemble_calibration import fused_log_scores

    rng = np.random.default_rng(2)
    names = ["a", "b", "c", "d", "e"]
    for sharp in (0.3, 1.0, 4.0):
        Px = _softmax_np(rng.normal(0, sharp * 2, size=(1, 5)))
        Pr = _softmax_np(rng.normal(0, 1, size=(1, 5)))
        st = _St()
        st.xgb = _Model([{"diagnosis_en": n, "probability": float(p)} for n, p in zip(names, Px[0])])
        st.river = _Model([{"diagnosis_en": n, "probability": float(p)} for n, p in zip(names, Pr[0])])
        st.ensemble_T = 0.8
        ens = EnsembleTopK(st)
        got = ens.predict_distribution([[1, 0]])
        want = _softmax_np(fused_log_scores(Px, Pr, ens) / 0.8)[0]
        for n, p in zip(names, want):
            assert got[n] == pytest.approx(p, rel=1e-6, abs=1e-9)


def test_progress_eta_and_duration_format():
    from backend.utils import Progress, fmt_duration

    assert fmt_duration(5) == "5s"
    assert "min" in fmt_duration(125) or "2m" in fmt_duration(125)
    lines = []

    class _L:
        def info(self, msg, *a):
            lines.append(msg % a if a else msg)

    p = Progress("test", total=10, log=_L(), every_pct=0.0, min_interval=0.0)
    for i in range(1, 11):
        p.update(i)
    p.finish()
    assert any("zostało" in s for s in lines) and any("KONIEC" in s for s in lines)


def test_learned_linear_fusion_matches_vectorized():
    from backend.services.ensemble_calibration import ensemble_distribution

    rng = np.random.default_rng(3)
    names = ["a", "b", "c", "d"]
    Px = _softmax_np(rng.normal(0, 2, size=(1, 4)))
    Pr = _softmax_np(rng.normal(0, 2, size=(1, 4)))
    st = _St()
    st.xgb = _Model([{"diagnosis_en": n, "probability": float(p)} for n, p in zip(names, Px[0])])
    st.river = _Model([{"diagnosis_en": n, "probability": float(p)} for n, p in zip(names, Pr[0])])
    st.ensemble_mode, st.ensemble_wx, st.ensemble_T = "linear", 0.6, 1.0
    ens = EnsembleTopK(st)
    got = ens.predict_split([[1, 0]], k=4)
    assert got["meta"]["gated_to"] == "learned" and got["meta"]["fusion"] == "linear"
    want = ensemble_distribution(Px, Pr, ens, st)[0]
    dist = ens.predict_distribution([[1, 0]])
    for n, p in zip(names, want):
        assert dist[n] == pytest.approx(p, rel=1e-9)
        assert p == pytest.approx(0.6 * Px[0][names.index(n)] + 0.4 * Pr[0][names.index(n)])


def test_river_batch_warm_start_is_accurate_and_keeps_online_learning():
    import pandas as pd
    from backend.models.river_manager import RiverManager

    rng = np.random.default_rng(0)
    feats = [f"f{i}" for i in range(8)]
    proto = {"flu": [0, 1, 2], "cold": [2, 3, 4], "measles": [5, 6, 7]}
    rows = []
    for _ in range(300):
        for d, act in proto.items():
            v = np.zeros(8, dtype=int)
            v[[a for a in act if rng.random() < 0.9]] = 1
            v[rng.integers(0, 8)] = 1 if rng.random() < 0.1 else v[rng.integers(0, 8)]
            rows.append({"diseases": d, "weight": 1.0, **{f: int(x) for f, x in zip(feats, v)}})
    df = pd.DataFrame(rows)
    r = RiverManager()
    r.features = feats
    r._batch_warm_start(df, "test")
    classes = r._current_classes()
    assert sorted(classes) == ["cold", "flu", "measles"]
    acc = np.mean([
        classes[int(np.argmax(r._raw_scores({feats[j]: 1 for j in np.flatnonzero(row[feats].to_numpy())}, classes)))] == row["diseases"]
        for _, row in df.sample(150, random_state=1).iterrows()
    ])
    assert acc > 0.85
    # dalsze uczenie online (SGD) działa na wagach z treningu paczkami
    clf = r.model.classifiers["flu"]
    before = clf._raw_dot_one({"f0": 1})
    for _ in range(20):
        r.model.learn_one({"f0": 1}, "flu")
    assert clf._raw_dot_one({"f0": 1}) > before

"""
Regression tests: ensemble, matcher (inflection and negation), triage, chat flow
(follow-up loop, stop phrase, session restore), question selection, River router,
app startup, calibration, auth and input limits.
"""
from __future__ import annotations

import importlib
import os
import types

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


# Ensemble
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
    # XGB is confident in "a"; River is nearly flat and does not list "a" at all.
    x = [{"diagnosis_en": "a", "probability": 0.6}, {"diagnosis_en": "b", "probability": 0.05}] + [
        {"diagnosis_en": f"x{i}", "probability": 0.01} for i in range(7)
    ]
    r = [{"diagnosis_en": "b", "probability": 0.10}] + [
        {"diagnosis_en": f"r{i}", "probability": 0.09} for i in range(8)
    ]
    split = _ens(x, r).predict_split([[1, 0]], k=3)
    assert split["ensemble_top"][0]["diagnosis_en"] == "a"
    assert split["meta"]["gated_to"] == "xgb"
    # Gating actually shifts the weights (maximum XGB share).
    assert split["meta"]["wx_dyn"] == pytest.approx(1.0 - 0.15)


def test_ensemble_weights_respect_minimums():
    e = _ens([], [])
    wx, wr = e._clamp_weights(0.99)
    assert wr >= e.min_wr - 1e-9 and wx >= e.min_wx - 1e-9
    assert wx + wr == pytest.approx(1.0)


# Matcher
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


# Triage and stop phrases
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


# Question selection
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
    # After asking about blood_in_sputum, skip the near-duplicate hemoptysis.
    best2 = stats.best_question(post, ["hemoptysis", "rash"], reference=["cough", "blood_in_sputum"])
    assert best2 != "hemoptysis"


# Chat flow
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
    assert r.response.startswith("Czy ") and r.response.endswith("?")
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
    # The fake model is unsure (top-1 60%), so it asks first and assesses after "to wszystko".
    if not r.top_diagnoses:
        r = _say(chat_state, "s4", "to wszystko")
    assert r.top_diagnoses


def test_negation_in_answer_is_respected(chat_state):
    _say(chat_state, "s5", "mam kaszel")
    r = _say(chat_state, "s5", "nie mam gorączki, ale mam katar")
    assert "fever" not in r.symptoms and "runny_nose" in r.symptoms


# River router
def test_river_admin_endpoints_disabled_without_token():
    from backend.api.routers import river_router

    app = FastAPI()
    app.include_router(river_router.router)
    client = TestClient(app)
    # No River model: reads return 503, admin endpoints return 403 without a token.
    app.state.app_state = type("S", (), {"river": None})()
    assert client.get("/river/status").status_code == 503
    r = client.post("/river/train_one", json={"label": "flu", "active_features": ["fever"]})
    assert r.status_code == 403


# App startup
def test_create_app_builds_routes():
    app_mod = importlib.import_module("app")
    app = app_mod.create_app()
    paths = {getattr(r, "path", "") for r in app.routes}
    for p in ("/chat", "/health", "/history", "/river/status", "/sessions/{session_id}/symptoms"):
        assert p in paths


# Dictionary vs training data (features without data)
def test_matcher_skips_dead_features_and_prefers_feature_with_data():
    feats = ["sharp_abdominal_pain", "abdominal_pain", "belly_pain", "fever"]
    counts = {"sharp_abdominal_pain": 14000, "abdominal_pain": 30, "belly_pain": 0, "fever": 10000}
    m = SymptomMatcher(features=feats, device="cpu", feature_counts=counts)
    assert not m.is_alive("belly_pain")
    pos, _ = m.extract_detailed("boli mnie brzuch")
    assert pos == ["sharp_abdominal_pain"]
    # Only the chosen feature is activated, not its rare synonyms.
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
    assert out == [{"en": "stomach_bloating", "pl": "wzdęcie brzucha", "category": None, "aliases": []}]


def test_dicts_panel_one_row_per_feature_with_aliases():
    """Several dictionary entries for one feature yield one row (unique UI key); the rest become aliases."""
    from backend.api.routers.dicts import _project_to_xgb_features

    feats = ["mouth_pain", "pain_in_eye"]
    m = SymptomMatcher(features=feats, device="cpu", feature_counts={"mouth_pain": 50, "pain_in_eye": 40})
    raw = [
        {"en": "Tongue pain", "pl": "ból języka", "xgb_label": ["mouth_pain"], "category": "Jama ustna"},
        {"en": "Mouth pain", "pl": "ból jamy ustnej", "xgb_label": ["mouth_pain"], "category": "Jama ustna"},
        {"en": "Pain behind the eyes", "pl": "ból za oczami", "xgb_label": ["pain_in_eye"]},
        {"en": "Pain in eye", "pl": "ból oka", "xgb_label": ["pain_in_eye"]},
        {"en": "Eye pain", "pl": "ból oka", "xgb_label": ["pain_in_eye"]},
    ]
    out = _project_to_xgb_features(raw, set(feats), m.resolve_labels, m._count)
    assert [r["en"] for r in out] == ["mouth_pain", "pain_in_eye"]
    assert out[0] == {"en": "mouth_pain", "pl": "ból jamy ustnej", "category": "Jama ustna", "aliases": ["ból języka"]}
    assert out[1] == {"en": "pain_in_eye", "pl": "ból oka", "category": None, "aliases": ["ból za oczami"]}


def test_dicts_real_dictionary_has_unique_features():
    import json
    from backend.api.routers.dicts import _project_to_xgb_features
    from backend.config import SYMPTOMS_DICT_PATH

    raw = json.loads(open(SYMPTOMS_DICT_PATH, encoding="utf-8").read())
    feats = {str(x).strip().lower() for r in raw for x in (r.get("xgb_label") or [])}
    out = _project_to_xgb_features(raw, feats)
    ens = [r["en"] for r in out]
    assert len(ens) == len(set(ens))


# Model calibration
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
    # Training with class weights inflates each class logit by log(w_c).
    biased = true_logits + log_w[None, :]
    cal = TemperatureCalibrator()
    cal.fit(biased, y, log_w=log_w)
    assert cal.log_w is not None and cal.T == pytest.approx(1.0, abs=0.1)
    # For an unbiased model the correction worsens NLL, so the calibrator must not use it.
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
    Z = 2.0 * (true_logits + z_base[None, :])  # too sharp and shifted by the intercept
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
    # Online SGD learning continues from the batch-trained weights.
    clf = r.model.classifiers["flu"]
    before = clf._raw_dot_one({"f0": 1})
    for _ in range(20):
        r.model.learn_one({"f0": 1}, "flu")
    assert clf._raw_dot_one({"f0": 1}) > before


def test_requirements_have_rapidfuzz_and_no_unused_hf_deps():
    import pathlib
    req = (pathlib.Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    pkgs = {l.split("#")[0].strip().split("==")[0].split(">=")[0].split("[")[0].lower()
            for l in req.splitlines() if l.split("#")[0].strip()}
    assert "rapidfuzz" in pkgs
    assert not pkgs & {"transformers", "accelerate", "sentence-transformers"}


# Follow-up questions
class _ConfidentModel(_FakeModel):
    def predict_topk(self, vec, k=3):
        items = [{"diagnosis_en": "flu", "probability": 0.85},
                 {"diagnosis_en": "cold", "probability": 0.1},
                 {"diagnosis_en": "covid", "probability": 0.05}]
        return items[: min(k, 3)]


def test_enough_symptoms_but_unsure_model_asks_before_assessment(chat_state):
    r = _say(chat_state, "f1", "mam gorączkę, kaszel i katar")
    assert not r.top_diagnoses and r.response.endswith("?")
    assert r.response.startswith("Zanim podam ocenę")
    r = _say(chat_state, "f1", "to wszystko")
    assert r.top_diagnoses


def test_confident_model_gives_assessment_without_questions(matcher):
    from backend.core.state import AppState
    m = _ConfidentModel()
    st = AppState(device="cpu", xgb=m, llm=_FakeLLM(), matcher=SymptomMatcher(m.features),
                  sessions=SessionStore(), river=_ConfidentModel())
    r = _say(st, "f2", "mam gorączkę, kaszel i katar")
    assert r.top_diagnoses


def test_after_assessment_new_symptom_updates_it_without_questions(chat_state):
    _say(chat_state, "f3", "mam gorączkę, kaszel i katar")
    r = _say(chat_state, "f3", "to wszystko")
    assert r.top_diagnoses
    r = _say(chat_state, "f3", "mam też wymioty")
    assert r.top_diagnoses and "vomiting" in r.symptoms


def test_short_yes_answers_without_tak():
    from backend.services.chat_service import _parse_yes_no
    assert _parse_yes_no("boli") == "yes"
    assert _parse_yes_no("Mam.") == "yes"
    assert _parse_yes_no("trochę") == "yes"
    assert _parse_yes_no("nie boli") == "no"
    assert _parse_yes_no("mam gorączkę i kaszel") is None  # a symptom description, not an answer


def _toy_stats():
    from backend.services.symptom_stats import SymptomStats
    # Disease "a" almost always has "rash", disease "b" rarely does.
    q = np.array([[0.95, 0.5], [0.05, 0.5]], dtype=np.float32)
    return SymptomStats(["rash", "fever"], ["a", "b"], q, np.array([10, 10]))


def test_negative_answer_lowers_diseases_that_almost_always_have_the_symptom():
    s = _toy_stats()
    out = s.apply_negatives({"a": 0.7, "b": 0.3}, ["rash"])
    assert out["b"] > out["a"] and abs(sum(out.values()) - 1.0) < 1e-9
    assert s.apply_negatives({"a": 0.7, "b": 0.3}, []) == {"a": 0.7, "b": 0.3}
    nb = s.nb_posterior(["fever"], ["rash"])
    assert nb["b"] > nb["a"]


def test_ensemble_applies_negated_symptoms():
    from backend.services.ensemble import EnsembleTopK

    class _M:
        def __init__(self):
            self.model = object()

        def predict_topk(self, vec, k=3):
            return [{"diagnosis_en": "a", "probability": 0.7}, {"diagnosis_en": "b", "probability": 0.3}]

    st = types.SimpleNamespace(xgb=_M(), river=_M(), symptom_stats=_toy_stats(),
                               ensemble_mode="linear", ensemble_wx=0.5, ensemble_T=1.0)
    ens = EnsembleTopK(st)
    assert ens.predict_topk([[1, 0]], k=2)[0]["diagnosis_en"] == "a"
    assert ens.predict_topk([[1, 0]], k=2, negated=["rash"])[0]["diagnosis_en"] == "b"


def test_every_model_feature_has_a_curated_followup_question():
    import json
    from backend.services.prompt_localizer import FOLLOWUP_QUESTIONS_PATH, PromptLocalizer
    feats = json.load(open(os.path.join(os.path.dirname(FOLLOWUP_QUESTIONS_PATH), "..", "ml", "features.json"), encoding="utf-8"))
    feats = feats if isinstance(feats, list) else feats.get("features", [])
    qs = json.load(open(FOLLOWUP_QUESTIONS_PATH, encoding="utf-8"))
    alive_missing = [f for f in feats if f.strip().lower() not in qs]
    assert len(alive_missing) <= 30  # features without training data are never asked
    for f, q in qs.items():
        assert q.startswith("Czy ") and q.endswith("?"), f
        assert "symptom " not in q and "\u2014" not in q, f
    loc = PromptLocalizer(SymptomMatcher(["foul_smell_of_urine"]))
    assert loc.followup_question("foul_smell_of_urine") == qs["foul_smell_of_urine"]


# Symptoms outside the model
def test_photophobia_is_recognized_with_negation():
    from backend.services.unmodeled_symptoms import extract_unmodeled
    assert extract_unmodeled("boli mnie głowa, mam nudności i światłowstręt") == (["światłowstręt"], [])
    assert extract_unmodeled("razi mnie światło")[0] == ["światłowstręt"]
    assert extract_unmodeled("nie mogę patrzeć na światło")[0] == ["światłowstręt"]
    assert extract_unmodeled("swiatlowstret")[0] == ["światłowstręt"]
    assert extract_unmodeled("nie mam światłowstrętu") == ([], ["światłowstręt"])
    assert extract_unmodeled("mam katar") == ([], [])


def test_photophobia_is_acknowledged_in_chat_instead_of_not_recognized(chat_state):
    r = _say(chat_state, "u1", "mam światłowstręt")
    assert "Zanotowałem: światłowstręt" in r.response
    assert "Nie rozpoznałem" not in r.response
    r = _say(chat_state, "u1", "boli mnie głowa i mam gorączkę")
    assert "Zanotowałem" not in r.response  # acknowledged only once
    assert chat_state.sessions.get_extra("u1") == ["światłowstręt"]


def test_photophobia_with_fever_and_headache_raises_urgency():
    assert _suggest_triage(["ból głowy", "gorączka"], 30)[0] != "wysoka"
    assert _suggest_triage(["ból głowy", "gorączka", "światłowstręt"], 30)[0] == "wysoka"


# Auth, parsing and flow edge cases
def _sb_token(secret, **claims):
    import time
    from jose import jwt
    now = int(time.time())
    base = {"sub": "11111111-1111-1111-1111-111111111111", "aud": "authenticated", "role": "authenticated",
            "email": "a@example.com", "iat": now, "exp": now + 3600}
    base.update(claims)
    base = {k: v for k, v in base.items() if v is not None}
    return jwt.encode(base, secret, algorithm="HS256")


def test_supabase_token_requires_authenticated_user(monkeypatch):
    from fastapi import HTTPException
    from backend.api.routers import auth
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", "s3cret")
    assert auth._decode_supabase_jwt(_sb_token("s3cret"))["sub"]
    for bad in (dict(aud=None), dict(role="anon"), dict(role="anon", aud=None, sub=None), dict(sub="")):
        with pytest.raises(HTTPException) as e:
            auth._decode_supabase_jwt(_sb_token("s3cret", **bad))
        assert e.value.status_code == 401


def test_email_from_user_metadata_cannot_take_over_account(monkeypatch):
    from fastapi import HTTPException
    from backend.api.routers import auth
    victim = types.SimpleNamespace(id=7, provider_sub="victim-sub", email="victim@example.com")
    created = {}
    monkeypatch.setattr(auth.crud, "get_user_by_provider", lambda db, p, s: None)
    monkeypatch.setattr(auth.crud, "get_user_by_email", lambda db, e: victim if e == "victim@example.com" else None)
    monkeypatch.setattr(auth.crud, "generate_username", lambda db, base: "nowy")

    def _create(db, **kw):
        created.update(kw)
        return types.SimpleNamespace(id=8, **kw)
    monkeypatch.setattr(auth.crud, "create_user", _create)
    db = types.SimpleNamespace(commit=lambda: None, add=lambda x: None)

    # E-mail only in user-editable metadata: create a new account, never link the victim's.
    u = auth._ensure_local_profile(db, {"sub": "attacker", "email": "", "user_metadata": {"email": "victim@example.com"}})
    assert u.id == 8 and created["email"] != "victim@example.com"
    # Same e-mail in the token but the account is bound to another sub: 409, no takeover.
    with pytest.raises(HTTPException) as e:
        auth._ensure_local_profile(db, {"sub": "attacker", "email": "victim@example.com"})
    assert e.value.status_code == 409


def test_symptom_prefixed_features_reach_the_model():
    feats = ["symptom_depression", "insomnia", "sharp_abdominal_pain", "abdominal_pain"]
    m = SymptomMatcher(features=feats, device="cpu",
                       feature_counts={"symptom_depression": 50, "insomnia": 50, "sharp_abdominal_pain": 100, "abdominal_pain": 4})
    assert m.expand_to_xgb("symptom_depression") == {"symptom_depression"}
    assert m.to_pl("symptom_depression") != "symptom depression"
    assert m.expand_to_xgb("abdominal_pain") == {"sharp_abdominal_pain"}  # alias still resolves to the feature with data


def test_symptoms_restored_after_restart_even_if_router_created_session(chat_state):
    chat_state.sessions.claim_owner("r1", None)          # the /chat router does this before handle_chat
    r = _say(chat_state, "r1", "mam też katar", client_symptoms=["fever", "cough"])
    assert set(r.symptoms) == {"fever", "cough", "runny_nose"}


def test_reset_flow_clears_history_used_for_triage(chat_state):
    _say(chat_state, "r2", "gorączka 40 stopni od 2 tygodni")
    assert chat_state.sessions.get_history("r2")
    chat_state.sessions.reset_flow("r2")
    assert chat_state.sessions.get_history("r2") == []


def test_polish_answer_variants():
    from backend.services.chat_service import _parse_yes_no
    for t in ("no tak", "no jasne", "no, mam", "niestety tak", "zdecydowanie tak", "oczywiście"):
        assert _parse_yes_no(t) == "yes", t
    for t in ("no", "no nie", "niestety nie", "raczej nie", "oczywiście nie"):
        assert _parse_yes_no(t) == "no", t


def test_unclear_answer_repeats_the_same_question(chat_state):
    r = _say(chat_state, "r3", "mam kaszel")
    q1 = r.response
    r = _say(chat_state, "r3", "hmm, trudno")
    assert "Nie jestem pewien" in r.response and q1 in r.response
    assert chat_state.sessions.get_followup_count("r3") == 1


def test_after_one_more_question_announcement_comes_the_assessment(chat_state):
    _say(chat_state, "r4", "boli mnie głowa")
    r = _say(chat_state, "r4", "to wszystko")
    assert "Jeszcze jedno pytanie" in r.response
    r = _say(chat_state, "r4", "nie")
    assert r.top_diagnoses, "po zapowiedzianym ostatnim pytaniu ma być ocena"


def test_photophobia_negations_and_triage():
    from backend.services.unmodeled_symptoms import extract_unmodeled
    assert extract_unmodeled("nie mam gorączki i światło mnie razi") == (["światłowstręt"], [])
    assert extract_unmodeled("nie razi mnie światło") == ([], ["światłowstręt"])
    assert extract_unmodeled("nie mam gorączki ani światłowstrętu") == ([], ["światłowstręt"])
    # A negated headache in the history must not raise urgency.
    assert _suggest_triage(["gorączka", "światłowstręt"], 30, history_text="nie mam bólu głowy")[0] != "wysoka"


def test_short_accented_alias_and_stiff_neck_are_recognized():
    matcher = SymptomMatcher(features=["anxiety_and_nervousness", "insomnia", "symptom_depression", "neck_stiffness_or_tightness"], device="cpu")
    pos, _ = matcher.extract_detailed("mam depresję, bezsenność i lęk")
    assert "anxiety_and_nervousness" in pos and "symptom_depression" in pos
    assert matcher.extract_detailed("biorę lek na nadciśnienie")[0] == []   # "lek" (medication) must not match the anxiety alias
    assert "neck_stiffness_or_tightness" in matcher.extract_detailed("mam sztywny kark")[0]


def test_no_em_dashes_in_llm_instructions_and_output():
    import inspect
    from backend.services import prompt_builder
    from backend.utils import sanitize_llm_output
    dash = "\u2014"
    src = inspect.getsource(prompt_builder)
    assert dash not in "".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert sanitize_llm_output(f"Angina {dash} najbardziej prawdopodobna.") == "Angina, najbardziej prawdopodobna."
    assert dash not in sanitize_llm_output(f"{dash} Odpocznij.\n{dash} Pij dużo.")


# Input limits and error messages
def test_input_limits_reject_oversized_requests(client):
    r = client.post("/sessions/" + "x" * 65 + "/symptoms", json={"add": ["fever"]})
    assert r.status_code == 422
    r = client.post("/sessions/ok-sid/symptoms", json={"remove": ["fever"] * 201})
    assert r.status_code == 422
    r = client.post("/sessions/ok-sid/symptoms", json={"add": ["x" * 101]})
    assert r.status_code == 422
    r = client.post("/chat", json={"session_id": "s", "user_input": "kaszel", "client_symptoms": ["fever"] * 201})
    assert r.status_code == 422
    r = client.post("/chat", json={"session_id": "s", "user_input": "kaszel", "lang": "p" * 50})
    assert r.status_code == 422


def test_removing_unknown_symptoms_does_not_grow_session_memory(client):
    sid = "sess-remove-unknown"
    client.post(f"/sessions/{sid}/symptoms", json={"replace": ["fever"]})
    client.post(f"/sessions/{sid}/symptoms", json={"remove": ["not_a_feature_1", "not_a_feature_2"]})
    st = client.app.state.app_state
    assert not ({"not_a_feature_1", "not_a_feature_2"} & st.sessions.get_negated(sid))


def test_profile_name_length_matches_db_column():
    from pydantic import ValidationError
    from database.schemas import UpdateProfileIn
    UpdateProfileIn(first_name="a" * 80)
    with pytest.raises(ValidationError):
        UpdateProfileIn(first_name="a" * 81)


def test_dict_errors_do_not_leak_server_paths(tmp_path):
    from fastapi import HTTPException
    from backend.api.routers.dicts import _load_dict_cached
    missing = tmp_path / "secret_dir" / "brak.json"
    with pytest.raises(HTTPException) as e:
        _load_dict_cached(missing)
    assert "secret_dir" not in str(e.value.detail) and str(tmp_path) not in str(e.value.detail)
    bad = tmp_path / "bad.json"
    bad.write_text("{nie json", encoding="utf-8")
    with pytest.raises(HTTPException) as e:
        _load_dict_cached(bad)
    assert str(tmp_path) not in str(e.value.detail) and "Expecting" not in str(e.value.detail)


def test_llm_autoboost_setting_comes_from_config():
    """LLM_AUTBOOST_ENABLE in .env used to be ignored (llm_manager hardcoded True)."""
    import os, subprocess, sys
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = "import sys; sys.path[:0] = ['..', '.']; from models import llm_manager; print(llm_manager.LLM_AUTBOOST_ENABLE)"
    env = {**os.environ, "LLM_AUTBOOST_ENABLE": "0"}
    out = subprocess.run([sys.executable, "-c", code], cwd=backend, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().splitlines()[-1] == "False"

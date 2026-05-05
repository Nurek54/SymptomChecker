# tests/test_ensemble_topk_unit.py
import math
import pytest

import services.ensemble as ens_mod
from backend.services.ensemble import EnsembleTopK


class DummyModel:
    """Minimalny model XGB/River z predict_topk()."""

    def __init__(self, topk_output):
        # lista słowników: {"diagnosis_en": ..., "probability": ...}
        self._topk_output = list(topk_output or [])
        self.model = object()  # żeby health/xgb_ready się nie wywalało

    def predict_topk(self, vec, k: int):
        # ignorujemy vec/k – dla testu zwracamy ustalone wartości
        return list(self._topk_output)


class DummyState:
    """Minimalny AppState dla EnsembleTopK – bez prawdziwych modeli."""

    def __init__(self, top_x=None, top_r=None):
        self.xgb = DummyModel(top_x or [])
        self.river = DummyModel(top_r or [])
        self.device = "cpu"
        self.matcher = None
        self.sessions = None


def _neutralize_ensemble_config():
    """
    Wyłączamy wszystkie „dodatki”, żeby zostało czyste
    log-space + normalizacja.
    """
    ens_mod.ENSEMBLE_AGREE_BOOST = 0.0
    ens_mod.ENSEMBLE_SMOOTH_PRIOR_MAX = 0.0
    ens_mod.ENSEMBLE_TAIL_LIFT = 0.0
    ens_mod.ENSEMBLE_RIVER_DIVERSITY_BONUS = 0.0
    ens_mod.ENSEMBLE_GATING_SYMM = False
    ens_mod.ENSEMBLE_TEMP_XGB = 1.0
    ens_mod.ENSEMBLE_TEMP_RIVER = 1.0


def _make_ens(top_x, top_r, wx: float = 0.5, wr: float = 0.5) -> EnsembleTopK:
    _neutralize_ensemble_config()
    st = DummyState(top_x=top_x, top_r=top_r)
    ens = EnsembleTopK(st)
    # wymuszamy konkretne wagi
    ens.wx = wx
    ens.wr = wr
    return ens


def test_log_space_weighted_mean():
    """
    Średnia w log-space z wagami:
    p_flu_xgb=0.8, p_flu_river=0.6 przy wx=wr=0.5
    → p_flu ≈ sqrt(0.8 * 0.6) po normalizacji.
    """
    top_x = [
        {"diagnosis_en": "flu", "probability": 0.8},
        {"diagnosis_en": "cold", "probability": 0.2},
    ]
    top_r = [
        {"diagnosis_en": "flu", "probability": 0.6},
        {"diagnosis_en": "cold", "probability": 0.4},
    ]

    ens = _make_ens(top_x, top_r, wx=0.5, wr=0.5)
    res = ens.predict_topk([[1]], k=2)
    fused = {d["diagnosis_en"]: d["probability"] for d in res}

    total = sum(fused.values())
    assert total == pytest.approx(1.0, rel=1e-6)

    # oczekiwane log-średnie (przed normalizacją)
    g_flu = math.sqrt(0.8 * 0.6)
    g_cold = math.sqrt(0.2 * 0.4)
    exp_flu = g_flu / (g_flu + g_cold)

    assert fused["flu"] == pytest.approx(exp_flu, rel=5e-2)

def test_idempotence_same_distributions():
    """
    Gdy XGB i River mają ten sam rozkład, ensemble zachowuje ranking
    i wartości ~oryginalne.
    """
    top = [
        {"diagnosis_en": "flu", "probability": 0.7},
        {"diagnosis_en": "cold", "probability": 0.3},
    ]
    ens = _make_ens(top_x=top, top_r=top, wx=0.5, wr=0.5)
    res = ens.predict_topk([[1]], k=2)
    fused = {d["diagnosis_en"]: d["probability"] for d in res}

    assert fused["flu"] > fused["cold"]
    assert fused["flu"] == pytest.approx(0.7, abs=5e-2)
    assert fused["cold"] == pytest.approx(0.3, abs=5e-2)


def test_monotonicity_increasing_confidence_keeps_rank():
    """
    Zwiększenie pewności dla klasy nie może zepchnąć jej niżej w rankingu.
    """
    top_r = [
        {"diagnosis_en": "flu", "probability": 0.6},
        {"diagnosis_en": "cold", "probability": 0.4},
    ]

    top_x_low = [
        {"diagnosis_en": "flu", "probability": 0.55},
        {"diagnosis_en": "cold", "probability": 0.45},
    ]
    top_x_high = [
        {"diagnosis_en": "flu", "probability": 0.9},
        {"diagnosis_en": "cold", "probability": 0.1},
    ]

    ens1 = _make_ens(top_x=top_x_low, top_r=top_r, wx=0.7, wr=0.3)
    ens2 = _make_ens(top_x=top_x_high, top_r=top_r, wx=0.7, wr=0.3)

    fused1 = {d["diagnosis_en"]: d["probability"] for d in ens1.predict_topk([[1]], k=2)}
    fused2 = {d["diagnosis_en"]: d["probability"] for d in ens2.predict_topk([[1]], k=2)}

    assert fused1["flu"] > fused1["cold"]
    assert fused2["flu"] > fused2["cold"]
    assert fused2["flu"] >= fused1["flu"]


def test_normalization_sum_is_one():
    top_x = [
        {"diagnosis_en": "a", "probability": 0.9},
        {"diagnosis_en": "b", "probability": 0.1},
    ]
    top_r = [
        {"diagnosis_en": "a", "probability": 0.4},
        {"diagnosis_en": "b", "probability": 0.6},
    ]
    ens = _make_ens(top_x, top_r)
    res = ens.predict_topk([[1]], k=2)
    fused = {d["diagnosis_en"]: d["probability"] for d in res}

    assert sum(fused.values()) == pytest.approx(1.0, rel=1e-6)


def test_no_negative_weights_in_conf_signal():
    """
    _conf_signal nie powinien zwracać ujemnych wag / sygnałów.
    """
    m = {"a": 0.9, "b": 0.1}
    s = EnsembleTopK._conf_signal(m)  # type: ignore[arg-type]
    assert s >= 0.0

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from core.state import AppState
import heapq
import time
import math
import logging

log = logging.getLogger("symptomchecker.ensemble")

# ── konfiguracja ───────────────────────────────────────────────────────────────
try:
    from config import (
        ENSEMBLE_W_XGB,
        ENSEMBLE_W_RIVER,
        ENSEMBLE_K_FUSE,
        ENSEMBLE_MIN_XGB_WEIGHT,
        ENSEMBLE_MIN_RIVER_WEIGHT,
        ENSEMBLE_CONF_GAP,
    )
except Exception:
    ENSEMBLE_W_XGB = 0.8
    ENSEMBLE_W_RIVER = 0.2
    ENSEMBLE_K_FUSE = 9
    ENSEMBLE_MIN_XGB_WEIGHT = 0.55
    ENSEMBLE_MIN_RIVER_WEIGHT = 0.15
    ENSEMBLE_CONF_GAP = 0.20

_EPS = 1e-12


class EnsembleTopK:
    """
    Ensemble XGB + River.

    Zasada:
      - bierzemy Top-K z XGB i z River,
      - normalizujemy obie listy do rozkładu prawdopodobieństwa,
      - liczymy sygnał "confidence" z (1 - entropia) i top-margin,
      - mieszamy wagi:
          * bazowo ENSEMBLE_W_XGB / ENSEMBLE_W_RIVER,
          * przesuwamy w stronę bardziej pewnego modelu,
          * XGB i River mają minimalny udział (MIN_*_WEIGHT),
      - fuzja w log-space: p = exp(wx*log(px) + wr*log(pr)),
      - normalizacja i Top-K.

    Publiczne API:
      - predict_topk(vec, k)  – to idzie na główny UI
      - predict_split(vec, k) – osobno XGB / River / Ensemble + meta (overlay).

    Uwaga: w testach możesz nadpisać self.wx / self.wr i wtedy
    używane są dokładnie te wagi (po renormalizacji), bez dodatkowej
    dynamiki – dzięki temu testy typu „log-space weighted mean”
    wychodzą jak trzeba.
    """

    def __init__(self, state: AppState):
        self.st = state
        self.wx_base = float(ENSEMBLE_W_XGB if ENSEMBLE_W_XGB is not None else 0.8)
        self.wr_base = float(ENSEMBLE_W_RIVER if ENSEMBLE_W_RIVER is not None else 0.2)
        self.kfuse = int(ENSEMBLE_K_FUSE if ENSEMBLE_K_FUSE is not None else 9)
        self.min_wx = float(ENSEMBLE_MIN_XGB_WEIGHT if ENSEMBLE_MIN_XGB_WEIGHT is not None else 0.55)
        self.min_wr = float(ENSEMBLE_MIN_RIVER_WEIGHT if ENSEMBLE_MIN_RIVER_WEIGHT is not None else 0.15)
        self.conf_gap = float(ENSEMBLE_CONF_GAP if ENSEMBLE_CONF_GAP is not None else 0.20)

    # ── utils ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_map(items: List[Dict[str, Any]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for d in items or []:
            if not isinstance(d, dict):
                continue
            name = str(
                d.get("diagnosis_en")
                or d.get("diagnosis")
                or d.get("label")
                or ""
            ).strip().lower()
            if not name:
                continue
            p = d.get("probability", d.get("prob", d.get("score", 0.0)))
            try:
                out[name] = float(p)
            except Exception:
                pass
        return out

    @staticmethod
    def _normalize(m: Dict[str, float]) -> Dict[str, float]:
        s = sum(v for v in m.values() if v > 0)
        if s <= 0:
            return {k: 0.0 for k in m}
        inv = 1.0 / s
        return {k: (v if v > 0 else 0.0) * inv for k, v in m.items()}

    @staticmethod
    def _entropy(ps: List[float]) -> float:
        ps = [p for p in ps if p > 0]
        n = len(ps)
        if n <= 1:
            return 0.0
        h = -sum(p * math.log(p + _EPS) for p in ps)
        h_max = math.log(n) + _EPS
        return float(h / h_max)

    @staticmethod
    def _top_margin(ps: List[float]) -> float:
        if not ps:
            return 0.0
        srt = sorted(ps, reverse=True)
        if len(srt) == 1:
            return srt[0]
        return max(0.0, srt[0] - srt[1])

    @staticmethod
    def _count_active(vec: List[List[int]]) -> int:
        if not vec or not vec[0]:
            return 0
        return sum(1 for v in vec[0] if int(v) > 0)

    # ── sygnał pewności ────────────────────────────────────────────────────────
    @staticmethod
    def _conf_signal(m: Dict[str, float]) -> float:
        """
        Publiczny sygnał "confidence" (0–1) – używany w testach,
        nie powinien nigdy zwracać wartości ujemnych.
        """
        if not m:
            return 0.0
        ps = list(m.values())
        ent = EnsembleTopK._entropy(ps)
        mar = EnsembleTopK._top_margin(ps)
        conf = 0.5 * (1.0 - ent) + 0.5 * mar
        return max(0.0, min(1.0, conf))

    @staticmethod
    def _confidence(m: Dict[str, float]) -> float:
        return EnsembleTopK._conf_signal(m)

    def _safe_topk(self, attr: str, vec: List[List[int]], k: int) -> List[Dict[str, Any]]:
        mdl = getattr(self.st, attr, None)
        if not mdl:
            return []
        try:
            return mdl.predict_topk(vec, k=k) or []
        except Exception as e:
            log.warning("[Ensemble] %s.predict_topk failed: %s", attr, e)
            return []

    def _compute_weights(self, mx: Dict[str, float], mr: Dict[str, float]) -> Tuple[float, float, str]:
        conf_x = self._confidence(mx)
        conf_r = self._confidence(mr)

        raw_x = self.wx_base * (0.5 + conf_x)
        raw_r = self.wr_base * (0.5 + conf_r)
        denom = (raw_x + raw_r) or 1e-6
        wx = raw_x / denom
        wr = raw_r / denom

        wx = max(self.min_wx, min(1.0, wx))
        wr = max(self.min_wr, min(1.0, wr))
        denom2 = (wx + wr) or 1e-6
        wx /= denom2
        wr /= denom2

        gated_to = "none"
        if (conf_x - conf_r) >= self.conf_gap:
            gated_to = "xgb"
        elif (conf_r - conf_x) >= self.conf_gap:
            gated_to = "river"

        return wx, wr, gated_to

    # ── główne API ─────────────────────────────────────────────────────────────
    def predict_topk(self, vec: List[List[int]], k: int = 3) -> List[Dict[str, Any]]:
        """
        Główna ścieżka – to jest używane do top_diagnoses na UI.
        ZAWSZE robi fuzję XGB + River (oprócz sytuacji, gdy jeden model nie zwraca nic).

        W trybie testowym:
        - jeśli na instancji ustawiono atrybuty self.wx i self.wr,
          to używamy tych wag (po renormalizacji) zamiast dynamiki.
        """
        k_each = max(self.kfuse, int(k))
        t0 = time.perf_counter()
        top_x = self._safe_topk("xgb", vec, k=k_each)
        t1 = time.perf_counter()
        top_r = self._safe_topk("river", vec, k=k_each)
        t2 = time.perf_counter()

        if not top_x and not top_r:
            log.info("[Ensemble] xgb=%.3fs river=%.3fs (both empty)", t1 - t0, t2 - t1)
            return []

        mx = self._normalize(self._to_map(top_x)) if top_x else {}
        mr = self._normalize(self._to_map(top_r)) if top_r else {}

        # tylko jeden model → nie ma co fuzjować
        if mx and not mr:
            best = heapq.nlargest(int(k), mx.items(), key=lambda kv: kv[1])
            return [{"diagnosis_en": n, "probability": float(p)} for n, p in best]
        if mr and not mx:
            best = heapq.nlargest(int(k), mr.items(), key=lambda kv: kv[1])
            return [{"diagnosis_en": n, "probability": float(p)} for n, p in best]

        conf_x = self._confidence(mx)
        conf_r = self._confidence(mr)

        # tryb testowy – ręczne wagi wx/wr
        manual_weights = hasattr(self, "wx") and hasattr(self, "wr")
        if manual_weights:
            wx_dyn = float(getattr(self, "wx"))
            wr_dyn = float(getattr(self, "wr"))
            if wx_dyn < 0.0:
                wx_dyn = 0.0
            if wr_dyn < 0.0:
                wr_dyn = 0.0
            denom_w = (wx_dyn + wr_dyn) or 1.0
            wx_dyn /= denom_w
            wr_dyn /= denom_w
            gated_to = "manual"
        else:
            wx_dyn, wr_dyn, gated_to = self._compute_weights(mx, mr)

        # obie listy → fuzja log-space
        keys = set(mx) | set(mr)
        scores: Dict[str, float] = {}
        for name in keys:
            px = max(mx.get(name, _EPS), _EPS)
            pr = max(mr.get(name, _EPS), _EPS)
            scores[name] = math.exp(wx_dyn * math.log(px) + wr_dyn * math.log(pr))

        combined = self._normalize(scores)
        k_final = int(min(max(k, 1), len(combined)))
        best = heapq.nlargest(k_final, combined.items(), key=lambda kv: kv[1])

        t3 = time.perf_counter()
        log.info(
            "[Ensemble] xgb=%.3fs river=%.3fs fuse=%.3fs | wx=%.2f wr=%.2f conf_x=%.2f conf_r=%.2f gated_to=%s",
            t1 - t0, t2 - t1, t3 - t2, wx_dyn, wr_dyn, conf_x, conf_r, gated_to
        )

        return [{"diagnosis_en": name, "probability": float(prob)} for name, prob in best]

    def predict_split(self, vec: List[List[int]], k: int = 3) -> Dict[str, Any]:
        """
        Overlay diagnostyczny – pokazuje osobno XGB / River / Ensemble + meta.
        Tu robimy fuzję tak samo jak w predict_topk.
        """
        k_each = max(self.kfuse, int(k))
        t0 = time.perf_counter()
        top_x = self._safe_topk("xgb", vec, k=k_each) or []
        t1 = time.perf_counter()
        top_r = self._safe_topk("river", vec, k=k_each) or []
        t2 = time.perf_counter()

        mx = self._normalize(self._to_map(top_x)) if top_x else {}
        mr = self._normalize(self._to_map(top_r)) if top_r else {}

        n_active = self._count_active(vec)
        conf_x = self._confidence(mx) if mx else 0.0
        conf_r = self._confidence(mr) if mr else 0.0

        # brak obu
        if not mx and not mr:
            log.info("[Ensemble] xgb=%.3fs river=%.3fs (both empty)", t1 - t0, t2 - t1)
            return {
                "xgb_top": [],
                "river_top": [],
                "ensemble_top": [],
                "meta": {
                    "conf_x": 0.0,
                    "conf_r": 0.0,
                    "wx_dyn": 0.0,
                    "wr_dyn": 0.0,
                    "ent_x": 0.0,
                    "ent_r": 0.0,
                    "n_active": n_active,
                    "gated_to": "none",
                },
            }

        # tylko jeden model
        if mx and not mr:
            out_x = [
                {"diagnosis_en": n, "probability": float(p)}
                for n, p in heapq.nlargest(int(k), mx.items(), key=lambda kv: kv[1])
            ]
            return {
                "xgb_top": out_x,
                "river_top": [],
                "ensemble_top": out_x,
                "meta": {
                    "conf_x": conf_x,
                    "conf_r": 0.0,
                    "wx_dyn": 1.0,
                    "wr_dyn": 0.0,
                    "ent_x": self._entropy(list(mx.values())),
                    "ent_r": 0.0,
                    "n_active": n_active,
                    "gated_to": "xgb",
                },
            }

        if mr and not mx:
            out_r = [
                {"diagnosis_en": n, "probability": float(p)}
                for n, p in heapq.nlargest(int(k), mr.items(), key=lambda kv: kv[1])
            ]
            return {
                "xgb_top": [],
                "river_top": out_r,
                "ensemble_top": out_r,
                "meta": {
                    "conf_x": 0.0,
                    "conf_r": conf_r,
                    "wx_dyn": 0.0,
                    "wr_dyn": 1.0,
                    "ent_x": 0.0,
                    "ent_r": self._entropy(list(mr.values())),
                    "n_active": n_active,
                    "gated_to": "river",
                },
            }

        # obie listy → fuzja jak w predict_topk
        manual_weights = hasattr(self, "wx") and hasattr(self, "wr")
        if manual_weights:
            wx_dyn = float(getattr(self, "wx"))
            wr_dyn = float(getattr(self, "wr"))
            if wx_dyn < 0.0:
                wx_dyn = 0.0
            if wr_dyn < 0.0:
                wr_dyn = 0.0
            denom_w = (wx_dyn + wr_dyn) or 1.0
            wx_dyn /= denom_w
            wr_dyn /= denom_w
            gated_to = "manual"
        else:
            wx_dyn, wr_dyn, gated_to = self._compute_weights(mx, mr)

        keys = set(mx) | set(mr)
        scores: Dict[str, float] = {}
        for name in keys:
            px = max(mx.get(name, _EPS), _EPS)
            pr = max(mr.get(name, _EPS), _EPS)
            scores[name] = math.exp(wx_dyn * math.log(px) + wr_dyn * math.log(pr))

        combined = self._normalize(scores)

        out_x = [
            {"diagnosis_en": n, "probability": float(p)}
            for n, p in heapq.nlargest(int(k), mx.items(), key=lambda kv: kv[1])
        ]
        out_r = [
            {"diagnosis_en": n, "probability": float(p)}
            for n, p in heapq.nlargest(int(k), mr.items(), key=lambda kv: kv[1])
        ]
        out_e = [
            {"diagnosis_en": n, "probability": float(p)}
            for n, p in heapq.nlargest(int(k), combined.items(), key=lambda kv: kv[1])
        ]

        t3 = time.perf_counter()
        log.info(
            "[Ensemble/SPLIT] xgb=%.3fs river=%.3fs fuse=%.3fs | wx=%.2f wr=%.2f conf_x=%.2f conf_r=%.2f gated_to=%s",
            t1 - t0,
            t2 - t1,
            t3 - t2,
            wx_dyn,
            wr_dyn,
            conf_x,
            conf_r,
            gated_to,
        )

        return {
            "xgb_top": out_x,
            "river_top": out_r,
            "ensemble_top": out_e,
            "meta": {
                "conf_x": float(conf_x),
                "conf_r": float(conf_r),
                "wx_dyn": float(wx_dyn),
                "wr_dyn": float(wr_dyn),
                "ent_x": float(self._entropy(list(mx.values()))),
                "ent_r": float(self._entropy(list(mr.values()))),
                "n_active": int(n_active),
                "gated_to": gated_to,
            },
        }

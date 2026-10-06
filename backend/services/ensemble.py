from __future__ import annotations
from typing import List, Dict, Any, Tuple
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
    ENSEMBLE_K_FUSE = 0
    ENSEMBLE_MIN_XGB_WEIGHT = 0.55
    ENSEMBLE_MIN_RIVER_WEIGHT = 0.15
    ENSEMBLE_CONF_GAP = 0.20

_EPS = 1e-12
# "Wszystkie klasy": managery XGB/River przycinają k do liczby klas.
_ALL_CLASSES = 1_000_000


class EnsembleTopK:
    """
    Ensemble XGB + River.

      - z każdego modelu bierzemy PEŁNY rozkład po klasach (ENSEMBLE_K_FUSE=0),
        dzięki czemu klasa nie "znika" tylko dlatego, że w drugim modelu
        nie weszła do Top-K;
      - klasa nieznana jednemu z modeli dostaje minimum z jego rozkładu
        (neutralnie nisko, ale nie 1e-12, które zerowałoby wynik);
      - pewność modelu = 0.5 * (1 - znormalizowana entropia) + 0.5 * margines top1-top2;
      - wagi: bazowe ENSEMBLE_W_* skalowane pewnością; jeśli jeden model jest
        wyraźnie pewniejszy (różnica >= ENSEMBLE_CONF_GAP), dostaje maksymalny
        dopuszczalny udział (gating). Minimalne udziały: MIN_XGB / MIN_RIVER;
      - fuzja w log-space: p = exp(wx*log(px) + wr*log(pr)), normalizacja po wszystkich klasach.

    Publiczne API:
      - predict_topk(vec, k)  – główny UI
      - predict_split(vec, k) – osobno XGB / River / Ensemble + meta (overlay).
    """

    def __init__(self, state: Any):
        self.st = state
        self.wx_base = float(ENSEMBLE_W_XGB if ENSEMBLE_W_XGB is not None else 0.8)
        self.wr_base = float(ENSEMBLE_W_RIVER if ENSEMBLE_W_RIVER is not None else 0.2)
        self.kfuse = int(ENSEMBLE_K_FUSE if ENSEMBLE_K_FUSE is not None else 0)
        self.min_wx = float(ENSEMBLE_MIN_XGB_WEIGHT if ENSEMBLE_MIN_XGB_WEIGHT is not None else 0.55)
        self.min_wr = float(ENSEMBLE_MIN_RIVER_WEIGHT if ENSEMBLE_MIN_RIVER_WEIGHT is not None else 0.15)
        self.conf_gap = float(ENSEMBLE_CONF_GAP if ENSEMBLE_CONF_GAP is not None else 0.20)

    # ───────────────────────── helpers ─────────────────────────
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

    @staticmethod
    def _conf_signal(m: Dict[str, float]) -> float:
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

    @staticmethod
    def _floor(m: Dict[str, float]) -> float:
        pos = [v for v in m.values() if v > 0]
        return max(min(pos), _EPS) if pos else _EPS

    def _k_each(self, k: int) -> int:
        if self.kfuse and self.kfuse > 0:
            return max(self.kfuse, int(k))
        return _ALL_CLASSES

    def _safe_topk(self, attr: str, vec: List[List[int]], k: int) -> List[Dict[str, Any]]:
        mdl = getattr(self.st, attr, None)
        if not mdl or getattr(mdl, "model", None) is None:
            return []
        try:
            return mdl.predict_topk(vec, k=k) or []
        except Exception as e:
            log.warning("[Ensemble] %s.predict_topk failed: %s", attr, e)
            return []

    def _clamp_weights(self, wx: float) -> Tuple[float, float]:
        lo = self.min_wx
        hi = 1.0 - self.min_wr
        if lo > hi:  # niespójna konfiguracja: proporcjonalnie do minimów
            wx = self.min_wx / ((self.min_wx + self.min_wr) or 1.0)
        else:
            wx = max(lo, min(hi, wx))
        return wx, 1.0 - wx

    def _compute_weights(self, mx: Dict[str, float], mr: Dict[str, float]) -> Tuple[float, float, str]:
        conf_x = self._confidence(mx)
        conf_r = self._confidence(mr)

        raw_x = self.wx_base * (0.5 + conf_x)
        raw_r = self.wr_base * (0.5 + conf_r)
        wx = raw_x / ((raw_x + raw_r) or 1e-6)

        gated_to = "none"
        if (conf_x - conf_r) >= self.conf_gap:
            gated_to = "xgb"
            wx = 1.0  # po clampie: maksymalny dopuszczalny udział XGB
        elif (conf_r - conf_x) >= self.conf_gap:
            gated_to = "river"
            wx = 0.0  # po clampie: maksymalny dopuszczalny udział River

        wx, wr = self._clamp_weights(wx)
        return wx, wr, gated_to

    def _manual_weights(self) -> Tuple[float, float]:
        wx_dyn = max(0.0, float(getattr(self, "wx")))
        wr_dyn = max(0.0, float(getattr(self, "wr")))
        denom_w = (wx_dyn + wr_dyn) or 1.0
        return wx_dyn / denom_w, wr_dyn / denom_w

    @staticmethod
    def _top(m: Dict[str, float], k: int) -> List[Dict[str, Any]]:
        return [
            {"diagnosis_en": n, "probability": float(p)}
            for n, p in heapq.nlargest(int(max(1, k)), m.items(), key=lambda kv: kv[1])
        ]

    # ───────────────────────── fuzja ─────────────────────────
    def _fuse(self, vec: List[List[int]], k: int) -> Dict[str, Any]:
        k_each = self._k_each(k)
        t0 = time.perf_counter()
        top_x = self._safe_topk("xgb", vec, k=k_each)
        t1 = time.perf_counter()
        top_r = self._safe_topk("river", vec, k=k_each)
        t2 = time.perf_counter()

        mx = self._normalize(self._to_map(top_x)) if top_x else {}
        mr = self._normalize(self._to_map(top_r)) if top_r else {}

        n_active = self._count_active(vec)
        conf_x = self._confidence(mx) if mx else 0.0
        conf_r = self._confidence(mr) if mr else 0.0
        meta: Dict[str, Any] = {
            "conf_x": float(conf_x),
            "conf_r": float(conf_r),
            "wx_dyn": 0.0,
            "wr_dyn": 0.0,
            "ent_x": float(self._entropy(list(mx.values()))) if mx else 0.0,
            "ent_r": float(self._entropy(list(mr.values()))) if mr else 0.0,
            "n_active": int(n_active),
            "gated_to": "none",
        }

        if not mx and not mr:
            log.info("[Ensemble] xgb=%.3fs river=%.3fs (both empty)", t1 - t0, t2 - t1)
            return {"mx": {}, "mr": {}, "combined": {}, "meta": meta}

        if mx and not mr:
            meta.update({"wx_dyn": 1.0, "wr_dyn": 0.0, "gated_to": "xgb"})
            return {"mx": mx, "mr": {}, "combined": mx, "meta": meta}

        if mr and not mx:
            meta.update({"wx_dyn": 0.0, "wr_dyn": 1.0, "gated_to": "river"})
            return {"mx": {}, "mr": mr, "combined": mr, "meta": meta}

        if hasattr(self, "wx") and hasattr(self, "wr"):
            wx_dyn, wr_dyn = self._manual_weights()
            gated_to = "manual"
        else:
            wx_dyn, wr_dyn, gated_to = self._compute_weights(mx, mr)

        fx, fr = self._floor(mx), self._floor(mr)
        scores: Dict[str, float] = {}
        for name in set(mx) | set(mr):
            px = max(mx.get(name, fx), _EPS)
            pr = max(mr.get(name, fr), _EPS)
            scores[name] = math.exp(wx_dyn * math.log(px) + wr_dyn * math.log(pr))
        combined = self._normalize(scores)

        meta.update({"wx_dyn": float(wx_dyn), "wr_dyn": float(wr_dyn), "gated_to": gated_to})
        log.info(
            "[Ensemble] xgb=%.3fs river=%.3fs fuse=%.3fs | classes=%d | wx=%.2f wr=%.2f conf_x=%.2f conf_r=%.2f gated_to=%s",
            t1 - t0, t2 - t1, time.perf_counter() - t2, len(combined),
            wx_dyn, wr_dyn, conf_x, conf_r, gated_to,
        )
        return {"mx": mx, "mr": mr, "combined": combined, "meta": meta}

    # ───────────────────────── API ─────────────────────────
    def predict_distribution(self, vec: List[List[int]]) -> Dict[str, float]:
        """Pełny rozkład po klasach (używany m.in. do wyboru pytań uzupełniających)."""
        return dict(self._fuse(vec, k=1)["combined"])

    def predict_topk(self, vec: List[List[int]], k: int = 3) -> List[Dict[str, Any]]:
        res = self._fuse(vec, k)
        combined = res["combined"]
        if not combined:
            return []
        return self._top(combined, min(max(int(k), 1), len(combined)))

    def predict_split(self, vec: List[List[int]], k: int = 3) -> Dict[str, Any]:
        res = self._fuse(vec, k)
        mx, mr, combined = res["mx"], res["mr"], res["combined"]
        return {
            "xgb_top": self._top(mx, k) if mx else [],
            "river_top": self._top(mr, k) if mr else [],
            "ensemble_top": self._top(combined, k) if combined else [],
            "meta": res["meta"],
        }

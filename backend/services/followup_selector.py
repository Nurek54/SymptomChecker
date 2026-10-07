# backend/services/followup_selector.py
from __future__ import annotations
from typing import Dict, List, Optional, Set
import os
import logging
import threading

try:
    from config import FOLLOWUP_BLOCKLIST_TOKENS as _CFG_BLOCKLIST
except Exception:
    _CFG_BLOCKLIST = None

log = logging.getLogger("symptomchecker.followup")

# Ranking ważności cech liczymy raz na model (get_score przechodzi po wszystkich drzewach).
_RANK_CACHE: Dict[int, List[str]] = {}
_RANK_LOCK = threading.Lock()


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    return s.replace("_", " ")


def _importance_ranking(xgb_mgr) -> List[str]:
    """
    Cechy posortowane wg ważności XGBoost (total_gain). Booster nie ma atrybutu
    `feature_importances_` (to API sklearn), więc używamy get_score().
    Gdy się nie da - kolejność z listy cech.
    """
    feats = [str(f).strip().lower() for f in (getattr(xgb_mgr, "features", None) or [])]
    model = getattr(xgb_mgr, "model", None)
    key = id(model)
    with _RANK_LOCK:
        if key in _RANK_CACHE:
            return _RANK_CACHE[key]
    ranked = feats
    try:
        score: Dict[str, float] = {}
        if hasattr(model, "get_score"):
            score = model.get_score(importance_type="total_gain") or {}
        elif hasattr(model, "feature_importances_"):
            score = {f: float(v) for f, v in zip(feats, list(model.feature_importances_))}
        if score:
            def _val(i: int, f: str) -> float:
                # booster trenowany na numpy ma nazwy f0..fN, na DataFrame - nazwy kolumn
                return float(score.get(f, score.get(f"f{i}", 0.0)))
            ranked = [f for _, f in sorted(((-_val(i, f), f) for i, f in enumerate(feats)))]
    except Exception as e:
        log.warning("[Followup] Ranking ważności niedostępny: %s", e)
    with _RANK_LOCK:
        _RANK_CACHE[key] = ranked
    return ranked


class FollowupSelector:
    def __init__(self, state):
        self.st = state

        bl: Set[str] = set()
        if isinstance(_CFG_BLOCKLIST, (set, list, tuple)):
            bl |= {str(x).lower() for x in _CFG_BLOCKLIST}
        env = os.environ.get("FOLLOWUP_BLOCKLIST", "")
        if env:
            bl |= {t.strip().lower() for t in env.split(",") if t.strip()}
        self._blocklist = bl

    def _blocked(self, name: str) -> bool:
        n = (name or "").lower()
        return any(tok and tok in n for tok in self._blocklist)

    def _good_pl(self, feat_name: str) -> bool:
        matcher = self.st.matcher
        if not hasattr(matcher, "to_pl"):
            return True
        pl = matcher.to_pl(feat_name)
        return _normalize(pl) != _normalize(feat_name)

    def _candidates(self, sess_id: str, have: Set[str]) -> List[str]:
        sessions = self.st.sessions
        try:
            neg = set(sessions.get_negated(sess_id))
        except Exception:
            neg = set()
        try:
            asked = set(sessions.get_asked(sess_id))
        except Exception:
            asked = set(sessions.get_recent_followups(sess_id, n=50)) if hasattr(sessions, "get_recent_followups") else set()

        feats = _importance_ranking(self.st.xgb)
        matcher = self.st.matcher
        alive = getattr(matcher, "is_alive", None)
        out = [
            f for f in feats
            if f and f not in have and f not in neg and f not in asked and not self._blocked(f)
            and (alive is None or alive(f))  # nie pytamy o objawy, o których model nic nie wie
        ]
        # pytamy tylko o objawy z sensownym polskim tłumaczeniem
        with_pl = [c for c in out if self._good_pl(c)]
        return with_pl or out

    def choose_next(
        self,
        sess_id: str,
        current_syms: List[str],
        lang: str = "pl",
        posterior: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        have = {(s or "").lower().strip() for s in (current_syms or [])}
        candidates = self._candidates(sess_id, have)
        if not candidates:
            return None

        sessions = self.st.sessions
        try:
            reference = sorted(have | set(sessions.get_negated(sess_id)) | set(sessions.get_asked(sess_id)))
        except Exception:
            reference = sorted(have)

        # 1) najlepiej: pytanie, które najbardziej rozróżnia aktualnie prawdopodobne choroby
        stats = getattr(self.st, "symptom_stats", None)
        if stats is not None and posterior:
            try:
                best = stats.best_question(posterior, candidates, reference)
                if best:
                    return best
            except Exception as e:
                log.warning("[Followup] best_question błąd: %s", e)

        # 2) awaryjnie: najważniejsza cecha modelu, preferując kategorię już zgłoszonych objawów
        matcher = self.st.matcher
        get_cat = getattr(matcher, "get_category", None)
        if callable(get_cat) and have:
            active = {get_cat(s) for s in have} - {"Other", "Ogólny"}
            if active:
                in_cat = [c for c in candidates if get_cat(c) in active]
                if in_cat:
                    return in_cat[0]
        return candidates[0]

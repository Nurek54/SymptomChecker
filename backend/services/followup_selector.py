# backend/services/followup_selector.py
from __future__ import annotations
from typing import List, Optional, Set
import os

from core.state import AppState

try:
    from config import FOLLOWUP_BLOCKLIST_TOKENS as _CFG_BLOCKLIST
except Exception:
    _CFG_BLOCKLIST = None


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    return s.replace("_", " ")


class FollowupSelector:
    def __init__(self, state: AppState):
        self.st = state

        # bloklista z configa i env
        bl: Set[str] = set()
        if isinstance(_CFG_BLOCKLIST, (set, list, tuple)):
            bl |= {str(x).lower() for x in _CFG_BLOCKLIST}
        env = os.environ.get("FOLLOWUP_BLOCKLIST", "")
        if env:
            bl |= {t.strip().lower() for t in env.split(",") if t.strip()}
        self._blocklist = bl
        feats = [str(f).strip().lower() for f in (self.st.xgb.features or [])]
        try:
            imps = list(getattr(self.st.xgb.model, "feature_importances_", []))
        except Exception:
            imps = []
        if imps and len(imps) == len(feats):
            pairs = sorted(zip(feats, imps), key=lambda x: x[1], reverse=True)
            self._ranked_feats: List[str] = [f for f, _ in pairs]
        else:
            self._ranked_feats = feats

    def _recent_asked(self, sess_id: str, n: int = 3) -> Set[str]:
        if hasattr(self.st.sessions, "get_recent_followups"):
            try:
                return set(self.st.sessions.get_recent_followups(sess_id, n=n))
            except Exception:
                return set()
        return set()

    def _blocked(self, name: str) -> bool:
        n = (name or "").lower()
        return any(tok and tok in n for tok in self._blocklist)

    def _good_pl(self, feat_name: str) -> bool:
        matcher = self.st.matcher
        if not hasattr(matcher, "to_pl"):
            return True
        pl = matcher.to_pl(feat_name)
        return _normalize(pl) != _normalize(feat_name)

    def choose_next(self, sess_id: str, current_syms: List[str], lang: str = "pl") -> Optional[str]:
        have = set((s or "").lower().strip() for s in (current_syms or []))
        try:
            neg = set(self.st.sessions.get_negated(sess_id))
        except Exception:
            neg = set()
        asked_recent = self._recent_asked(sess_id, n=3)

        matcher = self.st.matcher
        get_cat = getattr(matcher, "get_category", None)
        active_cats = {get_cat(s) for s in have} if callable(get_cat) else set()

        # zbuduj kandydatów bez wywołań modelu
        candidates: List[str] = []
        for f in self._ranked_feats:
            if not f:
                continue
            if f in have or f in neg or f in asked_recent or self._blocked(f):
                continue
            candidates.append(f)
            if len(candidates) >= 80:  # twardy limit rozważanych kandydatów
                break

        if not candidates:
            return None

        # preferuj te z „aktywnych” kategorii (jeśli są znane)
        if active_cats and callable(get_cat):
            in_cat = [c for c in candidates if get_cat(c) in active_cats]
            if in_cat:
                candidates = in_cat

        # dla PL preferuj takie z sensownym tłumaczeniem
        if (lang or "pl").lower().startswith("pl"):
            with_pl = [c for c in candidates if self._good_pl(c)]
            if with_pl:
                candidates = with_pl

        # ostatecznie: pierwszy z listy (już przefiltrowanej wg rangi)
        return candidates[0] if candidates else None

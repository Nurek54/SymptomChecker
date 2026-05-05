from __future__ import annotations
import re
import random
from typing import List, Dict, Any
from config import STOP_PHRASES  # <-- NOWE

# ——— frazy kończące rozmowę (regex fallback) ———
STOP_PATTERNS = [
    r"\bto wszystko\b",
    r"\bto już wszystko\b",
    r"\bkoniec\b",
    r"\bnie mam (już )?więcej objawów\b",
    r"\bstop\b",
]

# ——— maskowanie danych wrażliwych w logach ———

# bardzo prosty regex na e-mail
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 11 cyfr – np. PESEL
PESEL_RE = re.compile(r"\b\d{11}\b")


def mask_sensitive(msg: str) -> str:
    """
    Zastępuje w tekście:
      - adresy e-mail → ***
      - 11-cyfrowe numery (np. PESEL) → ***.
    Używane w testach R3 i do logów.
    """
    masked = EMAIL_RE.sub("***", msg or "")
    masked = PESEL_RE.sub("***", masked)
    return masked


def contains_stop_phrase(text: str) -> bool:
    t = (text or "").strip().lower()
    # 1) krótkie frazy z configu (PL/EN)
    for p in (STOP_PHRASES or []):
        p = (p or "").strip().lower()
        if p and p in t:
            return True
    # 2) regex fallback
    return any(re.search(p, t) for p in STOP_PATTERNS)


_BAD_INTROS = [
    r"^assistant\s*:?",
    r"^system\s*:?",
    r"^user\s*:?",
    r"^as an ai\b",
    r"^jako model (językowy|sztucznej inteligencji)\b",
]


def sanitize_llm_output(text: str) -> str:
    t = (text or "").strip()
    t = "\n".join(
        ln for ln in t.splitlines()
        if not any(re.search(p, (ln or "").lower()) for p in _BAD_INTROS)
    ).strip()
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def ensure_summary_only(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*#{1,6}\s*.*$", "", t, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def validate_llm_summary(
    text: str,
    diag_names_allowed: List[str],
    symptoms_for_display: List[str]
) -> bool:
    t = _norm(text)
    if any(pat in t for pat in ("### system", "### kontekst", "### instrukcja", "assistant:", "user:")):
        return False
    if diag_names_allowed:
        ok = False
        for name in diag_names_allowed[:3]:
            n = _norm(name)
            if n and n in t:
                ok = True
                break
        if not ok:
            return False
    if symptoms_for_display:
        any_sym = any(_norm(s) and _norm(s) in t for s in symptoms_for_display)
        if not any_sym:
            return False
    return True


def _is_pl(lang: str) -> bool:
    return True


def join_with_conjunction(items: List[str], conj_pl: str = "i") -> str:
    items = [x for x in (items or []) if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {conj_pl} " + items[-1]


def seed_everywhere(seed: int = 42) -> None:
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    random.seed(seed)

# backend/services/lang.py
from __future__ import annotations
import re
from typing import Sequence
from config import YES_TOKENS, NO_TOKENS

_PL_DIACRITICS = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")

_PL_HINT_WORDS = {
    "nie", "tak", "jest", "mam", "mnie", "moje", "moja", "mam",
    "boli", "bol", "bardzo", "troche", "troche", "trochę",
    "od", "do", "dla", "ze", "albo", "lub", "oraz",
    "gardlo", "gardło", "kaszel", "goraczka", "gorączka",
    "katar", "duszność", "nudnosci", "nudności",
}

_EN_HINT_WORDS = {
    "the", "and", "or", "have", "has", "had", "with", "without",
    "fever", "cough", "sore", "throat", "nausea", "headache",
    "pain", "feel", "feeling", "since", "yesterday", "today",
}

_WORD_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+")


def detect_lang(s: str, default: str = "pl") -> str:
    if not s:
        return default

    text = str(s).lower().strip()
    if not text:
        return default

    if _PL_DIACRITICS.search(text):
        return "pl"

    tokens = set(_WORD_RE.findall(text))
    if not tokens:
        return default

    pl_hits = len(tokens & _PL_HINT_WORDS)
    en_hits = len(tokens & _EN_HINT_WORDS)

    if pl_hits > en_hits:
        return "pl"
    if en_hits > pl_hits:
        return "en"
    return default


def _contains_token_whole(s: str, tokens: Sequence[str]) -> bool:
    if not s or not tokens:
        return False
    pat = r"\b(?:%s)\b" % "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
    return bool(re.search(pat, s.strip().lower()))


def is_yes(s: str) -> bool:
    return _contains_token_whole(s, YES_TOKENS)


def is_no(s: str) -> bool:
    return _contains_token_whole(s, NO_TOKENS)

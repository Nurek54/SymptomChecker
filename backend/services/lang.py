# backend/services/lang.py
import re
from typing import Sequence
from config import YES_TOKENS, NO_TOKENS

def detect_lang(s: str) -> str:
    # Wymuszamy polski niezależnie od wejścia
    return "pl"


def _contains_token_whole(s: str, tokens: Sequence[str]) -> bool:
    if not s or not tokens:
        return False
    pat = r"\b(?:%s)\b" % "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
    return bool(re.search(pat, s.strip().lower()))

def is_yes(s: str) -> bool:
    return _contains_token_whole(s, YES_TOKENS)

def is_no(s: str) -> bool:
    return _contains_token_whole(s, NO_TOKENS)

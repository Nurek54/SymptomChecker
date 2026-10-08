# backend/services/unmodeled_symptoms.py
"""
Objawy rozpoznawane w tekście, których model (XGB/River) nie zna, bo nie ma ich w danych
treningowych. Nie trafiają do wektora cech, ale:
  - pacjent dostaje potwierdzenie, że objaw został zauważony (zamiast "nie rozpoznałem"),
  - trafiają do opisu dla LLM i do oceny pilności (np. światłowstręt + gorączka + ból głowy).
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Pattern, Tuple


def _fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", (text or "").lower().replace("ł", "l"))
    return "".join(ch for ch in t if not unicodedata.combining(ch))


# (etykieta PL, wzorce na tekście bez polskich znaków, wzorce z wbudowanym "nie" w znaczeniu twierdzącym)
_UNMODELED: List[Tuple[str, List[Pattern[str]], List[Pattern[str]]]] = [
    (
        "światłowstręt",
        [re.compile(p) for p in (
            r"swiatlowstret\w*",
            r"fotofobi\w*",
            r"(?:nad)?wrazliw\w*\s+na\s+(?:swiatlo|swiatla)\w*",
            r"swiatl\w*\s+(?:mnie\s+)?(?:razi|razilo|razace|drazni|przeszkadza|boli|meczy)\w*",
            r"razi\w*\s+(?:mnie\s+)?(?:w\s+oczy\s+)?swi[ae]tl\w*",
            r"bol\w*\s+(?:mnie\s+)?(?:oczy\s+|glowa\s+)?(?:od|przy)\s+swi[ae]tl\w*",
            r"(?:oczy|oko|glowa)\s+(?:mnie\s+)?bol\w*\s+(?:od|przy|na)\s+swi[ae]tl\w*",
        )],
        [re.compile(p) for p in (
            r"nie\s+(?:moge|mozna|da\s+sie)\s+patrzec\s+na\s+swiatl\w*",
            r"nie\s+(?:znosz\w*|toleruj\w*)\s+swiatl\w*",
        )],
    ),
]

_CLAUSE_SPLIT = re.compile(r"[.;!?,\n]|\bale\b|\bnatomiast\b")
_NEGATION = re.compile(r"\b(?:nie|bez|brak|zadnego|zadnej)\b")


def extract_unmodeled(text: str) -> Tuple[List[str], List[str]]:
    """Zwraca (obecne, zaprzeczone) etykiety PL objawów spoza modelu."""
    present: List[str] = []
    negated: List[str] = []
    for clause in _CLAUSE_SPLIT.split(_fold(text)):
        for label, patterns, positive_with_nie in _UNMODELED:
            if any(p.search(clause) for p in positive_with_nie):
                if label not in present:
                    present.append(label)
                continue
            m = next((m for m in (p.search(clause) for p in patterns) if m), None)
            if not m:
                continue
            target = negated if _NEGATION.search(clause[: m.start()]) else present
            if label not in target:
                target.append(label)
    negated = [n for n in negated if n not in present]
    return present, negated

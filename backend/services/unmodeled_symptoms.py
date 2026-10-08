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


# Wzorce na tekście bez polskich znaków. Rodzaje:
#   "noun"  rzeczownik ("światłowstręt"): zaprzeczony tylko w dopełniaczu po przeczeniu
#           ("nie mam światłowstrętu", "bez fotofobii"); mianownik po "nie mam gorączki i ..." jest obecny
#   "verb"  opis czasownikiem ("razi mnie światło"): zaprzeczony, gdy "nie" stoi tuż przed nim
#   "pos"   zwroty z "nie" w znaczeniu twierdzącym ("nie mogę patrzeć na światło")
#   "neg"   jawne zaprzeczenie w środku zwrotu ("światło mnie nie razi")
_UNMODELED: List[Tuple[str, List[Tuple[str, Pattern[str]]]]] = [
    (
        "światłowstręt",
        [(kind, re.compile(p)) for kind, p in (
            ("pos", r"nie\s+(?:moge|mozna|da\s+sie)\s+patrzec\s+na\s+swi[ae]tl\w*"),
            ("pos", r"nie\s+(?:znosz\w*|toleruj\w*)\s+swi[ae]tl\w*"),
            ("neg", r"swi[ae]tl\w*\s+(?:mnie\s+)?nie\s+(?:razi|drazni|przeszkadza|boli|meczy)\w*"),
            ("noun", r"swiatlowstret\w*"),
            ("noun", r"fotofobi\w*"),
            ("verb", r"(?:nad)?wrazliw\w*\s+na\s+swi[ae]tl\w*"),
            ("verb", r"swi[ae]tl\w*\s+(?:mnie\s+)?(?:razi|razilo|razace|drazni|przeszkadza|boli|meczy)\w*"),
            ("verb", r"razi\w*\s+(?:mnie\s+)?(?:w\s+oczy\s+)?swi[ae]tl\w*"),
            ("verb", r"bol\w*\s+(?:mnie\s+)?(?:oczy\s+|glowa\s+)?(?:od|przy)\s+swi[ae]tl\w*"),
            ("verb", r"(?:oczy|oko|glowa)\s+(?:mnie\s+)?bol\w*\s+(?:od|przy|na)\s+swi[ae]tl\w*"),
        )],
    ),
]

_CLAUSE_SPLIT = re.compile(r"[.;!?,\n]|\bale\b|\bnatomiast\b|\bza\s+to\b")
_NEG_WORD = re.compile(r"\b(?:nie|bez|brak\w*|zadnego|zadnej|ani)\b")
# odmiana rzeczowników w dopełniaczu (forma po przeczeniu): światłowstrętu, fotofobii
_GENITIVE = re.compile(r"(?:swiatlowstretu|fotofobii)\b")


def _negated(kind: str, clause: str, m: "re.Match[str]") -> bool:
    before = clause[: m.start()]
    if kind == "noun":
        return bool(_GENITIVE.match(clause, m.start())) and bool(_NEG_WORD.search(before))
    # "verb": przeczenie najwyżej dwa słowa przed zwrotem ("nie razi mnie światło", "nie jestem wrażliwy")
    words = before.split()[-3:]
    return any(w in ("nie", "bez", "ani") for w in words)


def extract_unmodeled(text: str) -> Tuple[List[str], List[str]]:
    """Zwraca (obecne, zaprzeczone) etykiety PL objawów spoza modelu."""
    present: List[str] = []
    negated: List[str] = []
    for clause in _CLAUSE_SPLIT.split(_fold(text)):
        for label, patterns in _UNMODELED:
            for kind, p in patterns:
                m = p.search(clause)
                if not m:
                    continue
                if kind == "pos":
                    is_neg = False
                elif kind == "neg":
                    is_neg = True
                else:
                    is_neg = _negated(kind, clause, m)
                target = negated if is_neg else present
                if label not in target:
                    target.append(label)
                break   # pierwszy pasujący wzorzec rozstrzyga dla tego fragmentu
    negated = [n for n in negated if n not in present]
    return present, negated

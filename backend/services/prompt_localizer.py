# services/prompt_localizer.py
from __future__ import annotations
import os
import re
import json
from typing import List, Dict, Any, Optional

from services.symptom_matcher import SymptomMatcher

try:
    from config import DISEASES_DICT_PATH
except Exception:
    DISEASES_DICT_PATH = "data/diseases_pl_en.json"


_NO_INFLECT_EXACT: set[str] = {}

# Ręcznie napisane pytania uzupełniające dla każdej cechy modelu. Automatyczne
# "Czy masz {nazwa}?" dawało błędy (np. "Czy masz reakcja alergiczna?",
# "Czy masz symptom anxiety?") albo pytało o inny objaw niż cecha, bo kilka cech
# dzieli polską etykietę ("foul_smell_of_urine" -> "ciemny mocz").
FOLLOWUP_QUESTIONS_PATH = os.environ.get(
    "FOLLOWUP_QUESTIONS_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "followup_questions_pl.json"),
)
_FOLLOWUP_QUESTIONS: Optional[Dict[str, str]] = None


def _followup_questions() -> Dict[str, str]:
    global _FOLLOWUP_QUESTIONS
    if _FOLLOWUP_QUESTIONS is None:
        try:
            with open(FOLLOWUP_QUESTIONS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            _FOLLOWUP_QUESTIONS = {str(k).strip().lower(): str(v).strip() for k, v in data.items() if v}
        except Exception:
            _FOLLOWUP_QUESTIONS = {}
    return _FOLLOWUP_QUESTIONS

_PHRASE_HINTS: tuple[str, ...] = (
    "ból", "bol", "uczucie", "pieczenie", "swędzenie", "świąd",
    "brak", "utrata", "zaburzenie", "uczucie", "drętwienie",
    "w ", "na ", "pod ", "przy ", "z ", "ze "
)

_IRREGULAR_ACC: Dict[str, str] = {}

_F_ENDINGS: tuple[str, ...] = (
    "ka", "ska", "cka", "żka", "ńska", "lka", "rka", "zka", "dka", "ga", "cha",
    "a",
)

_WORD_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-]+")


def _clean_label(label: str) -> str:
    if not label:
        return ""
    s = str(label).strip()
    s = re.sub(r"[?!.。…]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_phrase(name_pl: str) -> bool:
    if not name_pl:
        return False
    if " " in name_pl:
        return True
    low = name_pl.lower()
    return any(h in low for h in _PHRASE_HINTS)


def _accusative_heuristic_single_word(word: str) -> str:
    if not word:
        return word
    w = word.strip()
    lw = w.lower()
    if lw in _NO_INFLECT_EXACT:
        return w
    if lw in _IRREGULAR_ACC:
        mapped = _IRREGULAR_ACC[lw]
        return mapped.capitalize() if w[0].isupper() else mapped
    if not _WORD_RE.fullmatch(w):
        return w
    for end in _F_ENDINGS:
        if lw.endswith(end):
            if lw.endswith("a"):
                base = w[:-1]
                repl = "Ę" if w[-1].isupper() else "ę"
                return base + repl
            return w
    return w


def _to_accusative_minimal(name_pl: str) -> str:
    name = _clean_label(name_pl)
    if not name:
        return name
    if _is_phrase(name):
        return name
    return _accusative_heuristic_single_word(name)


class PromptLocalizer:
    def __init__(self, matcher: SymptomMatcher, diseases_json_path: Optional[str] = None):
        self.matcher = matcher
        self._dis_en2pl: Dict[str, str] = {}
        self._dis_xgb2pl: Dict[str, str] = {}
        path = diseases_json_path or DISEASES_DICT_PATH
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for row in data or []:
                    pl = str(row.get("pl") or "").strip()
                    if not pl:
                        continue
                    xgb = row.get("xgb_label")
                    if isinstance(xgb, list):
                        for xu in xgb:
                            k = str(xu or "").strip().lower()
                            if k:
                                self._dis_xgb2pl[k] = pl
                    else:
                        k = str(xgb or "").strip().lower()
                        if k:
                            self._dis_xgb2pl[k] = pl
                    en = str(row.get("en") or "").strip().lower()
                    if en:
                        self._dis_en2pl[en] = pl
            except Exception:
                self._dis_en2pl = {}
                self._dis_xgb2pl = {}

    def localize_symptoms(self, sym_xgb_list: List[str], lang: str = "pl") -> List[str]:
        out: List[str] = []
        for s in (sym_xgb_list or []):
            if not s:
                continue
            try:
                pl = self.matcher.to_pl(s)
                out.append(pl if pl else str(s).replace("_", " "))
            except Exception:
                out.append(str(s).replace("_", " "))
        return out

    def _diag_pl(self, label: str) -> str:
        if not label:
            return ""
        k = str(label).strip().lower()
        if k in self._dis_xgb2pl:
            return self._dis_xgb2pl[k]
        if k in self._dis_en2pl:
            return self._dis_en2pl[k]
        return str(label).replace("_", " ")

    def localize_disease(self, label: str, lang: str = "pl") -> str:
        return self._diag_pl(label)

    def format_diag_summary(self, top: List[Dict[str, Any]], lang: str = "pl") -> str:
        """
        Zwraca jednowierszowe podsumowanie Top-3 w formie:
        'Diagnoza A (65%), Diagnoza B (20%), Diagnoza C (15%)'.
        Bez numerowania (1., 2., 3.), żeby nie zachęcać LLM do robienia list.
        """
        items: List[str] = []
        for d in (top or []):
            name_raw = str(d.get("diagnosis_en") or d.get("label") or "")
            name_pl = self._diag_pl(name_raw)

            prob_val = d.get("probability", d.get("prob", d.get("score")))
            pct_str = ""
            if prob_val is not None:
                try:
                    p = float(prob_val)
                    # jeśli wygląda na 0–1 → traktuj jako prawdopodobieństwo
                    if 0.0 <= p <= 1.0:
                        pct = int(round(p * 100))
                    else:
                        pct = int(round(p))
                    pct_str = f" ({pct}%)"
                except Exception:
                    pct_str = ""

            items.append(f"{name_pl}{pct_str}")

        return ", ".join(items)

    def followup_question(
        self,
        sym_xgb: str,
        lang: str = "pl",
        **_: Any,
    ) -> str:
        curated = _followup_questions().get(str(sym_xgb or "").strip().lower())
        if curated:
            return curated
        try:
            name_pl = self.matcher.to_pl(sym_xgb) or str(sym_xgb).replace("_", " ")
        except Exception:
            name_pl = str(sym_xgb).replace("_", " ")
        name_pl = _to_accusative_minimal(name_pl)
        return f"Czy masz {name_pl}?"

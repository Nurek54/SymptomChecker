# services/chat_service.py
from __future__ import annotations
import logging
import re
import time
from typing import List, Dict, Any, Sequence, Optional, Tuple

from core.state import AppState
from schemas import ChatRequest, ChatResponse
from config import (
    MIN_SYMPTOMS,
    LLM_MAX_NEW_TOKENS,
    LLM_MAX_NEW_TOKENS_CHAT,
    LLM_MIN_NEW_TOKENS_CHAT,
    LLM_TEMP,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_REPETITION_PENALTY,
    LLM_STOP_SEQUENCES,
    GLOBAL_SEED,
    TOPN_DIAGNOSES,
)

try:
    from config import STOP_FOLLOWUPS_AT
except Exception:
    STOP_FOLLOWUPS_AT = 4

try:
    from config import PREDICT_ONLY_ON_NEW
except Exception:
    PREDICT_ONLY_ON_NEW = True

try:
    from config import PROMPT_HISTORY_FETCH
except Exception:
    PROMPT_HISTORY_FETCH = 20

try:
    from config import PROMPT_HISTORY_TURNS
except Exception:
    PROMPT_HISTORY_TURNS = 8

try:
    from config import PROMPT_STRIP_SUMMARY
except Exception:
    PROMPT_STRIP_SUMMARY = False

from services.followup_selector import FollowupSelector
from services.lang import detect_lang, is_yes, is_no
from services.prompt_localizer import PromptLocalizer
from services.symptom_matcher import SymptomMatcher
from services.prompt_builder import (
    build_prompt,
    build_prompt_strict,
    _classify_intent_pl as _intent_pl,
)

try:
    from services.prompt_builder import _suggest_triage as _triage_hint  # type: ignore
except Exception:
    _triage_hint = None

from services.ensemble import EnsembleTopK

from utils import (
    contains_stop_phrase,
    sanitize_llm_output,
    ensure_summary_only,
)

log = logging.getLogger("symptomchecker.chat_service")

def _normalize_topk(top: Any) -> List[Dict[str, Any]]:
    items = top or []
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = items[0]
    out: List[Dict[str, Any]] = []
    for d in items:
        if isinstance(d, dict):
            diag = d.get("diagnosis_en") or d.get("diagnosis") or d.get("label") or str(d)
            prob = d.get("probability", d.get("prob", d.get("score", 0.0)))
            try:
                prob = float(prob)
            except Exception:
                prob = 0.0
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            diag, prob = str(d[0]), float(d[1])
        else:
            diag, prob = str(d), 0.0
        out.append({"diagnosis_en": str(diag).strip().lower(), "probability": prob})
    return out


def _normalize_history(hist: Sequence[Any]) -> List[Dict[str, str]]:
    norm: List[Dict[str, str]] = []
    for m in hist or []:
        if isinstance(m, dict):
            role = str(m.get("role", "user"))
            content = str(m.get("content", "") or "")
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            role = str(m[0])
            content = str(m[1])
        else:
            role = "user"
            content = str(m)
        norm.append({"role": role, "content": content})
    return norm


def _looks_like_summary(text: str) -> bool:
    if not text:
        return False
    needles = [
        r"\bNajbardziej prawdopodobna\b",
        r"\bInne możliwe przyczyny\b",
        r"\bUwaga: .*Top-3\b",
        r"\bWizyta u lekarza\b",
        r"\bZalecana konsultacja\b",
        r"\bMożesz obserwować\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in needles)


def _filter_history_for_prompt(history: List[Dict[str, str]], keep: int, strip_summaries: bool) -> List[Dict[str, str]]:
    short = history[-keep:] if keep > 0 else history[:]
    if not strip_summaries:
        return short
    filtered: List[Dict[str, str]] = []
    for m in short:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "assistant" and _looks_like_summary(content):
            continue
        filtered.append(m)
    return filtered


def _split_yesno_and_extra(text: str) -> Tuple[Optional[bool], str]:
    t = (text or "").strip()
    if not t:
        return None, ""
    m = re.match(r"^\s*(tak|nie|yes|no|yep|yeah|nope)\b[ ,.:;!?-]*", t, flags=re.IGNORECASE)
    if not m:
        return None, ""
    key = m.group(1).lower()
    val = True if key in ("tak", "yes", "yep", "yeah") else False
    extra = t[m.end():].strip(" ,.;:!-")
    return val, extra

_DOSE_PAT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|ml|mcg|µg|g|iu|j\.?m\.?|kropli|tabletek|tabl\.?|caps?\.?|razy\s*(?:dziennie|w\s*ciągu|na\s*dobę)?|x\s*\d+)\b",
    flags=re.IGNORECASE,
)
_FREQ_PAT = re.compile(r"\bco\s*\d+\s*(?:h|godzin[ya]?|godzinach?)\b", flags=re.IGNORECASE)


def _strip_doses_only(text: str) -> str:
    return text or ""


def _strip_doses_legacy(text: str) -> str:
    if not text:
        return text
    out = _DOSE_PAT.sub("", text)
    out = _FREQ_PAT.sub("", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _strip_allergy_speculation(text: str, have_allergy_symptom: bool) -> str:
    if have_allergy_symptom:
        return text
    out = re.sub(r"(alergen(y|ów)? pokarmow(e|ych)|nadwrażliwość na pokarmy|nietolerancja pokarmowa)[^.]*[.]", "", text, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _dedup_sentences(text: str, max_sentences: int | None = None) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())

    def _shingles(s: str) -> set[str]:
        toks = re.findall(r"\w+", s.lower())
        if len(toks) < 3:
            return {" ".join(toks)} if toks else set()
        return {" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)}

    seen_shingles: list[set[str]] = []
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        sh = _shingles(p)
        is_dup = False
        for prev in seen_shingles:
            if not sh or not prev:
                continue
            inter = len(sh & prev)
            denom = max(1, min(len(sh), len(prev)))
            if inter / denom >= 0.75:
                is_dup = True
                break
        if is_dup:
            continue
        seen_shingles.append(sh)
        out.append(p)
        if max_sentences is not None and len(out) >= max_sentences:
            break
    return " ".join(out)


def _intent_sentence_cap(intent: str) -> int:
    return {
        "otc": 25,
        "triage": 15,
        "advice": 30,
        "diagnosis": 20,
        "summary_points": 20,
        "prophylaxis": 25,
        "autosummary": 20,
    }.get(intent, 25)

def _chat_max_new_tokens() -> int:
    cap = int(LLM_MAX_NEW_TOKENS_CHAT)
    base = int(LLM_MAX_NEW_TOKENS)
    return min(base, cap) if cap > 0 else base

_LEADER_CLAIM_PATTERNS = [
    re.compile(r"najbardziej\s+prawdopodobn[aey]\s+(?:przyczyn[aą]|diagnoz[aą])\s+(?:jest|to|byłoby|byłaby|mogłoby?\s+być)\s+([^.;,\n]+)", re.I),
    re.compile(r"najprawdopodobniej(?:szą)?\s+(?:przyczyn[aą]|diagnoz[aą])\s+(?:jest|to|byłaby|mogłoby?\s+być)\s+([^.;,\n]+)", re.I),
    re.compile(r"^\s*(?:to|wygląda\s+na|wstępnie\s+wygląda\s+to\s+na)\s+([^.;,\n]+?)\s+(?:to|jest)\s+(?:najbardziej|najprawdopodobniej)", re.I),
]

def _llm_claims_other_top1(text: str, leader_pl: str, top3_names_pl: List[str]) -> Optional[str]:
    if not text or not leader_pl:
        return None
    leader_norm = leader_pl.strip().lower()

    others_norm = [n.strip().lower() for n in top3_names_pl if n and n.strip().lower() != leader_norm]
    if not others_norm:
        return None

    text_low = text.lower()

    if leader_norm in text_low:
        return None

    for pat in _LEADER_CLAIM_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        claimed = (m.group(1) or "").strip().lower()
        if not claimed:
            continue
        for other in others_norm:
            if other in claimed or claimed in other:
                return f"odpowiedź wskazała '{m.group(1).strip()}' jako najbardziej prawdopodobną zamiast '{leader_pl}'"

    if "jednakże" in text_low or "jednak " in text_low:
        # split na "jednak" i sprawdź drugą część
        for sep in ("jednakże", "jednak "):
            if sep in text_low:
                tail = text_low.split(sep, 1)[1]
                for other in others_norm:
                    if other in tail and "najbardziej prawdopodobn" in tail:
                        return f"odpowiedź odwraca hierarchię — używa 'jednak' i wskazuje '{other}' jako prawdopodobniejszą"

    return None


def _looks_truncated(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    if not s:
        return False

    last = s[-1]

    ends_well = last in '.!?…'
    if not ends_well and last in '")]' and len(s) >= 2 and s[-2] in '.!?…':
        ends_well = True

    if not ends_well:
        return True

    sentences = re.split(r"(?<=[.!?…])\s+", s)
    if not sentences:
        return False

    last_sentence = sentences[-1].strip()
    if not last_sentence:
        return False

    words = re.findall(r"[a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", last_sentence)

    if len(sentences) == 1:
        if len(words) < 2:
            return True
    else:
        if len(words) < 3:
            return True

    KNOWN_ABBREVS = {
        "tj", "tzw", "np", "itd", "itp", "tzn", "etc",
        "godz", "min", "sek", "tys", "mln", "mld",
        "ul", "al", "pl", "dr", "prof", "mgr", "inż",
        "pn", "wt", "śr", "czw", "pt", "sob", "nd",
        "tabl", "kaps", "amp", "ml", "mg", "kg",
    }

    body = re.sub(r"[.!?…\s)\]\"]+$", "", last_sentence)
    last_word_match = re.search(r"(\S+)$", body)
    if not last_word_match:
        return False
    last_word = last_word_match.group(1)
    bare = re.sub(r"[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]", "", last_word).lower()

    if not bare:
        return False
    if bare in KNOWN_ABBREVS:
        return False

    DANGLING_ADJ = {
        "najbliższy", "najbliższa", "najbliższe", "najbliższego", "najbliższej",
        "najbliższych", "najbliższym", "najbliższymi",
        "ostatni", "ostatnia", "ostatnie", "ostatniego", "ostatniej",
        "ostatnich", "ostatnim", "ostatnimi",
        "kilku", "kilka", "kilkoma", "kilkudziesięciu",
        "następny", "następna", "następne", "następnego", "następnej", "następnych",
        "poprzedni", "poprzednia", "poprzednie", "poprzedniego", "poprzedniej", "poprzednich",
        "który", "która", "które", "którego", "której", "których", "którym", "którymi",
        "taki", "taka", "takie", "takiego", "takiej", "takich",
        "jaki", "jaka", "jakie", "jakiego", "jakiej", "jakich",
        "ten", "ta", "to", "tego", "tej", "tych", "tym", "tymi",
        "duży", "duża", "duże", "dużego", "dużej", "dużych",
        "mały", "mała", "małe", "małego", "małej", "małych",
        "wysoki", "wysoka", "wysokie", "wysokiego", "wysokiej", "wysokich",
        "niski", "niska", "niskie", "niskiego", "niskiej", "niskich",
    }
    PREPOSITIONS = {
        "w", "we", "z", "ze", "do", "od", "na", "po", "przed", "przez", "przy",
        "nad", "pod", "obok", "wobec", "ponad", "poza", "między", "pomiędzy",
        "ciągu", "okresie",  # "w ciągu X", "w okresie X"
    }

    if bare in DANGLING_ADJ:
        # sprawdź ~3 słowa wstecz — czy jest tam przyimek?
        all_words = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", last_sentence)
        if len(all_words) >= 2:
            tail_words = [w.lower() for w in all_words[-4:-1]]  # 3 słowa przed ostatnim
            if any(w in PREPOSITIONS for w in tail_words):
                return True

    POLISH_VOWELS = set("aeiouyąęóAEIOUYĄĘÓ")
    POLISH_FREQ_END = set("ćńłjknmtszwiAĆŃŁJKNMTSZWI")  # częste końcówki

    POLISH_COMMON_ENDINGS = {
        # samogłoski na końcu
        "ka", "ki", "ko", "ku", "ek", "ik", "ka", "ce", "ci", "go", "gi",
        "na", "ne", "ni", "no", "nu", "ny", "ńy", "ń",
        "ta", "te", "ty", "tu", "to", "ć",
        "wa", "we", "wi", "wo", "wy", "ów", "ów", "ów",
        "la", "le", "li", "lu", "ło", "łe", "ły", "ł",
        "sa", "se", "si", "so", "su", "sy", "ść",
        "ja", "je", "ji", "jo", "ju", "j", "ej",
        "ra", "re", "ri", "ro", "ru", "ry", "rz",
        "da", "de", "dy", "do", "du", "dź",
        "ma", "me", "mi", "mo", "mu", "my",
        "ba", "be", "bi", "bo", "bu", "by",
        "pa", "pe", "pi", "po", "pu", "py",
        "fa", "fy",
        "ga", "ge", "gi", "go", "gu", "gy",
        "ha", "he", "hi", "ho", "hu", "hy",
        "ca", "ce", "cy", "cz", "ć",
        "za", "ze", "zi", "zo", "zu", "zy", "ź", "ż",
        "ść", "śń", "śm", "śl", "ś",
        # częste wieloliterowe końcówki
        "ość", "ość", "ący", "ącą", "ące",
        "iej", "owy", "owa", "owe", "ami", "ach",
        "ego", "emu", "ymi", "ami",
    }
    POLISH_VOWEL_ENDINGS = {"a", "e", "i", "o", "u", "y", "ą", "ę", "ó"}

    if 3 <= len(bare) <= 6:
        last_one = bare[-1]
        last_two = bare[-2:] if len(bare) >= 2 else ""
        last_three = bare[-3:] if len(bare) >= 3 else ""

        if last_one in POLISH_VOWEL_ENDINGS:
            return False
        if last_two in POLISH_COMMON_ENDINGS or last_three in POLISH_COMMON_ENDINGS:
            return False
        if last_one in {"ć", "ń", "ł", "ś", "ź", "ż", "j", "m", "w", "r", "t", "s", "n", "k", "l", "p", "z", "c", "d"}:
            CONS = set("bcćdfghjklłmnńprstwxzźż")
            if len(bare) >= 3 and all(c in CONS for c in bare[-3:]):
                return True
            return False
        return True

    return False

_LIST_OPENER_PATTERNS = [
    re.compile(r"\b(?:proponuj[eę]|zalecam|sugeruj[eę])\s+(?:nast[eę]puj[aą]ce|poni[zż]sze|kilka)\s+krok", re.I),
    re.compile(r"\boto\s+(?:co|kilka|nast[eę]puj[aą]ce)\b", re.I),
    re.compile(r":\s*\n?\s*(?:1\.|-|\u2022|\*)\s", re.I),  # ":\n1." lub ":\n-"
    re.compile(r"(?:^|[.!?]\s+)(?:1\.|2\.|3\.)\s", re.I),  # numeracja zaczynająca zdanie
]

def _has_list_opener(text: str) -> bool:
    if not text:
        return False
    for p in _LIST_OPENER_PATTERNS:
        if p.search(text):
            return True
    return False

def _lists_to_prose(text: str) -> str:
    if not text:
        return text
    out = text
    out = re.sub(
        r"((?:proponuj[eę]|zalecam|sugeruj[eę]|oto|przedstawiam|w\s+pierwszej\s+kolejno[sś]ci)[^.:\n]{1,80}):\s*\n?",
        r"\1. ",
        out,
        flags=re.IGNORECASE,
    )

    connectors = {
        1: "Po pierwsze",
        2: "Po drugie",
        3: "Po trzecie",
        4: "Po czwarte",
        5: "Po piąte",
        6: "Po szóste",
        7: "Po siódme",
        8: "Po ósme",
    }

    def _num_to_prose(m: re.Match) -> str:
        num = int(m.group(1))
        title = m.group(2)  # nazwa punktu (np. "Odpoczynek")
        sep = m.group(3)    # ":" / "." / "\n"
        connector = connectors.get(num, "Następnie")
        title_lower = title[0].lower() + title[1:] if len(title) > 1 else title.lower()
        # ":" → " — " (myślnik, naturalna proza), "." / "\n" → ". "
        if sep == ":":
            return f"{connector}, {title_lower} — "
        else:
            return f"{connector}, {title_lower}. "

    out = re.sub(
        r"(?:^|(?<=\n)|(?<=\.\s)|(?<=:\s))(\d{1,2})[.)]\s+([A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s\-]{1,60}?)([:.\n])",
        _num_to_prose,
        out,
    )
    out = re.sub(
        r"(?:^|\n)\s*[-•*]\s+([A-ZĄĆĘŁŃÓŚŹŻ])",
        r" \1",
        out,
    )

    out = re.sub(r"\n{2,}", " ", out)
    out = re.sub(r"\n", " ", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+\.", ".", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"\.\s*\.", ".", out)  # podwójne kropki

    return out.strip()


def _trim_truncated_tail(text: str) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    if not parts:
        return text

    good = []
    for p in parts:
        if not p or not p.strip():
            continue
        if _looks_truncated(p):
            break
        good.append(p)

    if not good:
        return text

    result = " ".join(good).strip()
    # Upewnij się, że kończy kropką
    if result and result[-1] not in ".!?…)]\"":
        result += "."
    return result

class ChatService:
    def __init__(self, state: AppState):
        self.st = state
        self.selector = FollowupSelector(state)
        self.loc = PromptLocalizer(state.matcher)
        self.ens = EnsembleTopK(state)

        self._last_vec: Dict[str, Tuple[Tuple[int, ...], int, List[Dict[str, Any]]]] = {}
        self._expand_cache: Dict[str, List[str]] = {}

    def _guard_leader_consistency(self, text: str, leader_pl: Optional[str], leader_prob: Optional[float], *, enforce: bool) -> str:
        if not enforce or not text or not leader_pl:
            return text
        try:
            lp = leader_pl.strip().lower()
            if lp in text.lower():
                return text
            tail = f" Główna z rozważanych przyczyn to {leader_pl}."
            return (text.rstrip(" .") + ".").rstrip() + tail
        except Exception:
            return text

    def _join_pl(self, items: List[str]) -> str:
        items = [i for i in (items or []) if i]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " i " + items[-1]

    def _top3_update_note(self, lang: str, prev: List[str], curr: List[str]) -> str:
        return ""

    def _fallback_choose_next(self, sess_id: str, current_syms: List[str]) -> Optional[str]:
        xgb = self.st.xgb
        have = set(s.lower() for s in current_syms)
        try:
            imps = list(getattr(xgb.model, "feature_importances_", []))
        except Exception:
            imps = []
        feats = list(self.st.xgb.features)
        if imps and len(imps) == len(feats):
            feats = [f for f, _ in sorted(zip(feats, imps), key=lambda x: x[1], reverse=True)]
        try:
            neg = set(self.st.sessions.get_negated(sess_id))
        except Exception:
            neg = set()
        asked_recent = (
            set(self.st.sessions.get_recent_followups(sess_id, n=3))
            if hasattr(self.st.sessions, "get_recent_followups")
            else set()
        )
        for f in feats:
            fl = str(f).strip().lower()
            if fl not in have and fl not in neg and fl not in asked_recent:
                return fl
        return None

    def _deterministic_summary(
        self,
        sym_for_display: List[str],
        top: List[Dict[str, Any]],
        lang: str,
        *,
        patient_age: Optional[int],
        history_text: str,
    ) -> str:
        joined = self._join_pl(sym_for_display)
        s1 = f"Zgłoszone objawy: {joined}." if joined else "Brak potwierdzonych objawów."
        if top and len(top) >= 1:
            first = top[0]
            diag_en = first.get("diagnosis_en") or ""
            prob = float(first.get("probability", 0.0) or 0.0)
            diag_pl = self.loc.localize_disease(diag_en, lang)
            pct = int(round(prob * 100)) if prob <= 1.0 else int(round(prob))
            s2 = f"Najbardziej prawdopodobna przyczyna to {diag_pl} ({pct}%)."
        else:
            s2 = "Nie udało się ustalić najbardziej prawdopodobnej przyczyny."

        s3 = ""
        if top and len(top) > 1:
            others = []
            for d in top[1:]:
                de = d.get("diagnosis_en") or ""
                p = float(d.get("probability", 0.0) or 0.0)
                dp = self.loc.localize_disease(de, lang)
                pct = int(round(p * 100)) if p <= 1.0 else int(round(p))
                others.append(f"{dp} ({pct}%)")
            s3 = "Inne możliwe przyczyny to " + self._join_pl(others) + "."

        if _triage_hint is not None:
            tri_level, tri_window = _triage_hint(sym_for_display, patient_age, history_text=history_text)
            s4 = f"Pilność: {tri_level}; zalecane: {tri_window}."
        else:
            if top and len(top) >= 1 and float(top[0].get("probability", 0.0) or 0.0) >= 0.8:
                s4 = "Pilność: wysoka; zalecana pilna ocena teraz."
            elif top and len(top) >= 1 and float(top[0].get("probability", 0.0) or 0.0) >= 0.4:
                s4 = "Pilność: średnia; kontakt z lekarzem w ciągu 24 godzin."
            else:
                s4 = "Pilność: niska; samoobserwacja i konsultacja planowa w 48–72 h."

        parts = [p for p in (s1, s2, s3, s4) if p]
        text = " ".join(parts)
        text = ensure_summary_only(text)
        return text

    def _build_feature_vector(self, selected_canons: List[str]) -> List[int]:
        feats = [str(f).strip().lower() for f in (self.st.xgb.features or [])]
        expanded: set[str] = set()
        for s in (selected_canons or []):
            key = str(s).strip().lower()
            if key in self._expand_cache:
                expanded |= set(self._expand_cache[key])
                continue
            try:
                alts = list(self.st.matcher.expand_to_xgb(s) or [])
            except Exception:
                alts = [key]
            self._expand_cache[key] = alts
            expanded |= set(alts)
        return [1 if f in expanded else 0 for f in feats]

    def _llm_ready(self, llm) -> bool:
        if not llm:
            return False
        be = getattr(llm, "backend", None)
        if be == "ctransformers":
            return llm.model is not None
        return False

    def _safe_stop_sequences(self) -> Tuple[str, ...]:
        raw = list(LLM_STOP_SEQUENCES or [])
        blacklist = {"\nUser:", "\nSystem:", "User:", "System:"}
        return tuple(s for s in raw if s not in blacklist)

    def _ensemble_topk_cached(self, sess_id: str, vec: List[List[int]], k: int = 3) -> List[Dict[str, Any]]:
        vt: Tuple[int, ...] = tuple(vec[0] if vec else [])
        cached = self._last_vec.get(sess_id)
        if cached and cached[0] == vt and cached[1] == int(k):
            return cached[2]
        try:
            raw_top = self.ens.predict_topk(vec, k=k) or []
        except Exception as e:
            log.exception("Ensemble.predict_topk error: %s", e)
            raw_top = []
        top = _normalize_topk(raw_top)
        self._last_vec[sess_id] = (vt, int(k), top)
        return top

    def _get_cached_top(self, sess_id: str) -> Optional[List[Dict[str, Any]]]:
        c = self._last_vec.get(sess_id)
        return c[2] if c else None

    # ———————————————————————————————————————————————————————————————
    #          G Ł Ó W N A   P E T L A
    # ———————————————————————————————————————————————————————————————
    def handle_chat(self, req: ChatRequest, patient_age: int | None = None) -> ChatResponse:
        t_req = time.perf_counter()

        xgb = self.st.xgb
        llm = self.st.llm
        matcher: SymptomMatcher = self.st.matcher
        sessions = self.st.sessions

        if not (xgb and xgb.model):
            raise RuntimeError("Model XGB niezaładowany.")
        if matcher is None:
            raise RuntimeError("Matcher niezaładowany.")
        if not self._llm_ready(llm):
            raise RuntimeError("LLM niezaładowany.")

        if not sessions.exists(req.session_id):
            sessions.init_session(req.session_id)

        user_text = (req.user_input or "").strip()
        sessions.append_user(req.session_id, user_text)
        raw_history = sessions.get_recent_history(req.session_id, max_messages=int(PROMPT_HISTORY_FETCH))
        history = _normalize_history(raw_history)
        sess_symptoms = sessions.get_symptoms(req.session_id)

        syms_before = set(sess_symptoms)

        # — follow-up Tak/Nie —
        pending = sessions.get_pending_symptom(req.session_id)
        allow_extract = True
        answered_pending = False

        if pending:
            yn_val, extra_tail = _split_yesno_and_extra(user_text)
            if yn_val is not None:
                if yn_val:
                    sess_symptoms.add(pending)
                else:
                    sessions.add_negated_symptom(req.session_id, pending)
                sessions.clear_pending_symptom(req.session_id)
                sessions.mark_followup_asked(req.session_id, pending)
                answered_pending = True

                if extra_tail and len(extra_tail) >= 3 and re.search(r"[a-zA-Ząćęłńóśźż]", extra_tail, re.I):
                    try:
                        if hasattr(matcher, "extract_fast"):
                            fast_syms = matcher.extract_fast(extra_tail)  # type: ignore[attr-defined]
                        else:
                            fast_syms = matcher.extract(extra_tail)
                    except Exception as e:
                        log.exception("Matcher.extract_fast błąd: %s", e)
                        fast_syms = []
                    if fast_syms:
                        sessions.add_symptoms(req.session_id, fast_syms)
                else:
                    allow_extract = False
            else:
                has_yes = is_yes(user_text)
                has_no = is_no(user_text)
                if has_yes or has_no:
                    if has_yes:
                        sess_symptoms.add(pending)
                    else:
                        sessions.add_negated_symptom(req.session_id, pending)
                    sessions.clear_pending_symptom(req.session_id)
                    sessions.mark_followup_asked(req.session_id, pending)
                    answered_pending = True

                    parts = re.split(r"(?:^|\b)(tak|nie|yes|no|yep|yeah|nope)\b[ ,.:;!?-]*", user_text, flags=re.I)
                    tail = ""
                    if len(parts) >= 4:
                        tail = "".join(parts[3:]).strip(" ,.;:!-")
                    if tail and len(tail) >= 3 and re.search(r"[a-zA-Ząćęłńóśźż]", tail, re.I):
                        try:
                            if hasattr(matcher, "extract_fast"):
                                fast_syms2 = matcher.extract_fast(tail)  # type: ignore[attr-defined]
                            else:
                                fast_syms2 = matcher.extract(tail)
                        except Exception as e:
                            log.exception("Matcher.extract_fast błąd: %s", e)
                            fast_syms2 = []
                        if fast_syms2:
                            sessions.add_symptoms(req.session_id, fast_syms2)
                    else:
                        allow_extract = False

        # wczesny powrót po samym tak/nie
        if answered_pending and not allow_extract:
            sess_symptoms = sessions.get_symptoms(req.session_id)
            count = len(sess_symptoms)
            if count < int(MIN_SYMPTOMS):
                t_sel = time.perf_counter()
                try:
                    next_sym = self.selector.choose_next(req.session_id, sorted(list(sess_symptoms)), lang="pl")
                except Exception as e:
                    log.exception("FollowupSelector error: %s", e)
                    next_sym = self._fallback_choose_next(req.session_id, sorted(list(sess_symptoms)))
                log.info("TIMING select_followup=%.3fs", time.perf_counter() - t_sel)

                follow = self.loc.followup_question(next_sym, "pl") if next_sym else "Czy to wszystkie objawy?"
                if next_sym:
                    sessions.set_pending_symptom(req.session_id, next_sym)
                    sessions.mark_followup_asked(req.session_id, next_sym)
                sessions.append_assistant(req.session_id, follow)
                log.info("TIMING early_yesno total=%.3fs", time.perf_counter() - t_req)
                return ChatResponse(response=follow, symptoms=sorted(list(sess_symptoms)))

        # ekstrakcja objawów
        t_ext = time.perf_counter()
        new_syms: List[str] = []
        short_or_noise = len(user_text) < 3 or not re.search(r"[a-zA-Ząćęłńóśźż]", user_text, re.I)

        if allow_extract and not short_or_noise:
            try:
                if len(user_text) < 12 and hasattr(matcher, "extract_fast"):
                    new_syms = matcher.extract_fast(user_text)  # type: ignore[attr-defined]
                else:
                    new_syms = matcher.extract(user_text)
            except Exception as e:
                log.exception("Matcher.extract błąd: %s", e)
                new_syms = []
            if new_syms:
                sessions.add_symptoms(req.session_id, new_syms)
        log.info("TIMING extract=%.3fs", time.perf_counter() - t_ext)

        # stan po aktualizacji
        sess_symptoms = sessions.get_symptoms(req.session_id)
        count = len(sess_symptoms)
        changed_vec = set(sess_symptoms) != syms_before
        log.info("DEBUG new_syms=%s | symptoms=%s (n=%d) | changed_vec=%s",
                 new_syms, sorted(list(sess_symptoms))[:12], count, changed_vec)

        if count >= int(STOP_FOLLOWUPS_AT) and pending:
            sessions.clear_pending_symptom(req.session_id)

        user_stop = contains_stop_phrase(user_text)
        enough = count >= int(MIN_SYMPTOMS)
        lang = detect_lang(user_text, default="pl")

        if user_stop and not enough:
            t_fu = time.perf_counter()
            try:
                next_sym = self.selector.choose_next(req.session_id, sorted(list(sess_symptoms)), lang=lang)
            except Exception as e:
                log.exception("FollowupSelector error: %s", e)
                next_sym = self._fallback_choose_next(req.session_id, sorted(list(sess_symptoms)))
            if next_sym:
                follow = self.loc.followup_question(next_sym, lang)
                sessions.set_pending_symptom(req.session_id, next_sym)
                sessions.mark_followup_asked(req.session_id, next_sym)
            else:
                follow = "Czy to wszystkie objawy?"
            sessions.append_assistant(req.session_id, follow)
            log.info("TIMING followup_only=%.3fs total=%.3fs", time.perf_counter() - t_fu, time.perf_counter() - t_req)
            return ChatResponse(response=follow, symptoms=sorted(list(sess_symptoms)))

        # — predykcja (Ensemble) + LLM —
        if enough:
            t_pred = time.perf_counter()
            k = int(TOPN_DIAGNOSES)

            if PREDICT_ONLY_ON_NEW and not changed_vec:
                cached_top = self._get_cached_top(req.session_id)
                if cached_top is not None:
                    top = cached_top
                    log.info("PREDICT decision=SKIP reason=no_change cache_hit=True")
                    log.info("TIMING predict_ensemble=%.3fs (skipped)", 0.0)
                else:
                    vec = [self._build_feature_vector(sorted(list(sess_symptoms)))]
                    top = self._ensemble_topk_cached(req.session_id, vec, k=k)
                    log.info("PREDICT decision=RUN reason=first_time_with_min")
                    log.info("TIMING predict_ensemble=%.3fs", time.perf_counter() - t_pred)
            else:
                vec = [self._build_feature_vector(sorted(list(sess_symptoms)))]
                top = self._ensemble_topk_cached(req.session_id, vec, k=k)
                log.info("PREDICT decision=RUN reason=%s", "changed_vec" if changed_vec else "policy")
                log.info("TIMING predict_ensemble=%.3fs", time.perf_counter() - t_pred)

            curr_top_names = [d.get("diagnosis_en", str(d)) for d in (top or [])]
            prev_top_names = sessions.get_last_top3(req.session_id)
            sessions.set_last_top3(req.session_id, curr_top_names)

            sym_for_display = self.loc.localize_symptoms(sorted(list(sess_symptoms)), lang)
            diag_summary_display = self.loc.format_diag_summary(top, lang)

            history_for_prompt = _filter_history_for_prompt(
                history, keep=int(PROMPT_HISTORY_TURNS), strip_summaries=bool(PROMPT_STRIP_SUMMARY)
            )

            leader_en = (top[0]["diagnosis_en"] if top else None)
            leader_pl = self.loc.localize_disease(leader_en, lang) if leader_en else None
            leader_prob = float(top[0].get("probability") or 0.0) if top else None

            prompt = build_prompt(
                history_for_prompt,
                sym_for_display,
                need_more=False,
                diag_summary=diag_summary_display,
                patient_age=patient_age,
            )

            t_gen = time.perf_counter()
            safe_stops = self._safe_stop_sequences()
            max_tokens = _chat_max_new_tokens()

            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                min_new_tokens=int(LLM_MIN_NEW_TOKENS_CHAT),
                temperature=max(float(LLM_TEMP), 0.72),
                top_p=max(float(LLM_TOP_P), 0.92),
                top_k=int(LLM_TOP_K) if LLM_TOP_K else 40,
                repetition_penalty=max(float(LLM_REPETITION_PENALTY), 1.1),
                stop_sequences=safe_stops,
                seed=int(GLOBAL_SEED),
                continue_until_complete=True,
                max_continuations=2,
            )

            try:
                llm_resp = llm.generate(prompt, **gen_kwargs)
            except TypeError:
                gen_kwargs.pop("top_k", None)
                gen_kwargs.pop("repetition_penalty", None)
                gen_kwargs.pop("min_new_tokens", None)
                try:
                    llm_resp = llm.generate(prompt, **gen_kwargs)
                except Exception as e:
                    log.exception("Błąd LLM (ctransformers, fallback): %s", e)
                    llm_resp = ""
            except Exception as e:
                log.exception("Błąd LLM (ctransformers): %s", e)
                llm_resp = ""

            # PUSTO? → retry bez stopów
            if not (llm_resp or "").strip():
                retry_kwargs = dict(gen_kwargs)
                retry_kwargs["stop_sequences"] = tuple()
                try:
                    llm_resp = llm.generate(prompt, **retry_kwargs)
                    log.info("LLM retry: no-stops fallback used")
                except TypeError:
                    retry_kwargs.pop("top_k", None)
                    retry_kwargs.pop("repetition_penalty", None)
                    try:
                        llm_resp = llm.generate(prompt, **retry_kwargs)
                    except Exception as e:
                        log.exception("Błąd LLM (retry no-stops, fallback): %s", e)
                        llm_resp = ""
                except Exception as e:
                    log.exception("Błąd LLM (retry no-stops): %s", e)
                    llm_resp = ""
            log.info("TIMING llm_generate=%.3fs (max_tokens=%d, stops=%s)",
                     time.perf_counter() - t_gen, max_tokens, list(safe_stops))
            est_tokens = int((time.perf_counter() - t_gen) * 50)
            response_chars = len((llm_resp or "").strip())
            log.info("LLM_OUTPUT_STATS chars=%d est_tokens=%d budget_used=%d%% min_required=%d",
                     response_chars, est_tokens,
                     int(100 * est_tokens / max(1, max_tokens)),
                     int(LLM_MIN_NEW_TOKENS_CHAT))

            top3_names_pl_for_check: List[str] = []
            for d in (top or []):
                en_name = (d.get("diagnosis_en") or "").strip().lower()
                pl_name = self.loc.localize_disease(en_name, lang)
                if pl_name:
                    top3_names_pl_for_check.append(pl_name)
            rejection_reason: Optional[str] = None
            intent_check = _intent_pl(history_for_prompt[-1]["content"] if history_for_prompt else (req.user_input or ""))
            should_check_for_regen = intent_check not in {"advice", "otc", "prophylaxis"}

            if should_check_for_regen and leader_pl:
                if intent_check in {"diagnosis", "summary", "autosummary"}:
                    rr = _llm_claims_other_top1(llm_resp, leader_pl, top3_names_pl_for_check)
                    if rr:
                        rejection_reason = rr

            if rejection_reason is None and _looks_truncated(llm_resp):
                rejection_reason = "odpowiedź została urwana w połowie zdania"

            if rejection_reason:
                log.warning("LLM regen reason: %s. Original (first 200): %r",
                            rejection_reason, (llm_resp or "")[:200])
                log.info("LLM regen START — to może potrwać +5-15s, ale tylko jeden retry")

                leader_pct_int: Optional[int] = None
                if leader_prob is not None:
                    try:
                        if 0.0 <= float(leader_prob) <= 1.0:
                            leader_pct_int = int(round(float(leader_prob) * 100))
                        else:
                            leader_pct_int = int(round(float(leader_prob)))
                    except Exception:
                        leader_pct_int = None

                strict_prompt = build_prompt_strict(
                    history_for_prompt,
                    sym_for_display,
                    diag_summary=diag_summary_display,
                    patient_age=patient_age,
                    leader_pl=leader_pl,
                    leader_pct=leader_pct_int,
                    rejection_reason=rejection_reason,
                )
                strict_kwargs = dict(gen_kwargs)
                strict_kwargs["temperature"] = 0.55
                strict_kwargs["top_p"] = 0.85
                strict_kwargs["max_new_tokens"] = min(int(max_tokens), 512)

                t_regen = time.perf_counter()
                regen_resp = ""
                try:
                    regen_resp = llm.generate(strict_prompt, **strict_kwargs)
                except TypeError:
                    strict_kwargs.pop("top_k", None)
                    strict_kwargs.pop("repetition_penalty", None)
                    try:
                        regen_resp = llm.generate(strict_prompt, **strict_kwargs)
                    except Exception as e:
                        log.exception("LLM regen błąd (fallback): %s", e)
                        regen_resp = ""
                except Exception as e:
                    log.exception("LLM regen błąd: %s", e)
                    regen_resp = ""

                log.info("TIMING llm_regen=%.3fs", time.perf_counter() - t_regen)

                regen_ok = bool(regen_resp and regen_resp.strip())
                if regen_ok and leader_pl and intent_check in {"diagnosis", "summary", "autosummary"}:
                    if _llm_claims_other_top1(regen_resp, leader_pl, top3_names_pl_for_check):
                        regen_ok = False
                if regen_ok and _looks_truncated(regen_resp):
                    regen_ok = False

                if regen_ok:
                    log.info("LLM regen accepted")
                    llm_resp = regen_resp
                else:
                    log.info("LLM regen rejected — keeping original")

            # post-processing
            llm_resp = sanitize_llm_output(llm_resp)
            llm_resp = ensure_summary_only(llm_resp)

            llm_resp = _lists_to_prose(llm_resp)

            llm_resp = re.sub(r"\(\s?0[,.]\d{1,3}\s?\)", "", llm_resp)
            llm_resp = re.sub(r"\(\s?(\d{1,3})\s?%\)\s*\(\s?0[,.]\d{1,3}\s?\)", r"(\1%)", llm_resp)

            llm_resp = _strip_doses_only(llm_resp)

            allergy_markers = {"allergic_reaction", "hives", "urticaria", "rash", "itching", "angioedema", "wheezing"}
            have_allergy = any(s in allergy_markers for s in (self.st.sessions.get_symptoms(req.session_id) or []))
            llm_resp = _strip_allergy_speculation(llm_resp, have_allergy)

            last_user_msg = history_for_prompt[-1]["content"] if history_for_prompt else (req.user_input or "")
            intent = _intent_pl(last_user_msg or "")
            llm_resp = _dedup_sentences(llm_resp, max_sentences=_intent_sentence_cap(intent))

            llm_resp = self._guard_leader_consistency(llm_resp, leader_pl, leader_prob, enforce=(intent == "diagnosis"))

            if not llm_resp.strip():
                llm_resp = self._deterministic_summary(
                    sym_for_display, top, lang, patient_age=patient_age,
                    history_text=" ".join((m.get("content") or "") for m in history_for_prompt)
                )

            note = self._top3_update_note(lang, prev_top_names, curr_top_names)
            if note:
                llm_resp = (note + llm_resp).strip()
                llm_resp = ensure_summary_only(llm_resp)

            llm_resp = _trim_truncated_tail(llm_resp)

            top_with_pl: List[Dict[str, Any]] = []
            for d in (top or []):
                en_name = (d.get("diagnosis_en") or "").strip().lower()
                dd = dict(d)
                dd["diagnosis_pl"] = self.loc.localize_disease(en_name, lang)
                top_with_pl.append(dd)

            diagnosis_en = top_with_pl[0]["diagnosis_en"] if top_with_pl else None
            diagnosis_pl = self.loc.localize_disease(diagnosis_en, lang) if diagnosis_en else None

            sessions.append_assistant(req.session_id, llm_resp)
            log.info("TIMING total=%.3fs", time.perf_counter() - t_req)
            # FIX #1: diagnosis_en/_pl są teraz w schemacie ChatResponse
            return ChatResponse(
                response=llm_resp,
                symptoms=sorted(list(sess_symptoms)),
                diagnosis_en=diagnosis_en,
                diagnosis_pl=diagnosis_pl,
                top_diagnoses=top_with_pl,
            )

        if count < int(MIN_SYMPTOMS):
            t_sel2 = time.perf_counter()
            try:
                next_sym = self.selector.choose_next(req.session_id, sorted(list(sess_symptoms)), lang=lang)
            except Exception as e:
                log.exception("FollowupSelector error: %s", e)
                next_sym = self._fallback_choose_next(req.session_id, sorted(list(sess_symptoms)))
            log.info("TIMING select_followup=%.3fs", time.perf_counter() - t_sel2)
        else:
            next_sym = None

        if not next_sym:
            follow = "Czy to wszystkie objawy? Jeśli tak, napisz 'to wszystko'."
            sessions.append_assistant(req.session_id, follow)
            log.info("TIMING total=%.3fs", time.perf_counter() - t_req)
            return ChatResponse(response=follow, symptoms=sorted(list(sess_symptoms)))

        follow = self.loc.followup_question(next_sym, lang)
        sessions.set_pending_symptom(req.session_id, next_sym)
        sessions.mark_followup_asked(req.session_id, next_sym)
        sessions.append_assistant(req.session_id, follow)
        log.info("TIMING total=%.3fs", time.perf_counter() - t_req)
        return ChatResponse(response=follow, symptoms=sorted(list(sess_symptoms)))
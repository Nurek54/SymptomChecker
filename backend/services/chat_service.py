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
    STOP_FOLLOWUPS_AT,
    FOLLOWUP_CONFIDENCE,
    PREDICT_ONLY_ON_NEW,
    SUMMARY_MODE,
)

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
from services.prompt_localizer import PromptLocalizer
from services.symptom_matcher import SymptomMatcher
from services.prompt_builder import (
    build_prompt,
    build_prompt_strict,
    _classify_intent_pl as _intent_pl,
    _suggest_triage as _triage_hint,
)
from services.prompt_helpers import _user_text
from services.unmodeled_symptoms import extract_unmodeled
from services.ensemble import EnsembleTopK

from utils import (
    contains_stop_phrase,
    sanitize_llm_output,
    ensure_summary_only,
)

log = logging.getLogger("symptomchecker.chat_service")

# Wszystkie teksty dla pacjenta, słowniki objawów/chorób i model językowy (Bielik)
# są polskie, więc odpowiadamy zawsze po polsku.
LANG = "pl"

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



# słowa, które mogą poprzedzać "tak"/"nie" bez zmiany sensu: "niestety tak", "no tak", "raczej nie"
_ANSWER_LEAD = r"(?:(?:no|niestety|zdecydowanie|raczej|chyba|oczywi[sś]cie|ale[zż]|w[lł]a[sś]nie|na\s+pewno|szczerze\s+m[oó]wi[aą]c)[\s,]+)?"
_YES_RE = re.compile(
    r"^\s*" + _ANSWER_LEAD +
    r"(?:tak|ta|yes|yep|yeah|jasne|oczywi[sś]cie|owszem|pewnie|zgadza\s+si[eę]|jak\s+najbardziej|potwierdzam|dok[lł]adnie)\b"
    # po polsku "no" na początku zwykle znaczy "tak": "no tak", "no, mam", "no boli"
    r"|^\s*no[\s,]+(?:mam|boli|bol[aą]|jest|czuj[eę]|wyst[eę]puje)\b",
    re.IGNORECASE,
)
# "no" samo w sobie = "nie" (np. po angielsku), ale "no tak" = "tak"; "oczywiście nie" = "nie"
_NO_RE = re.compile(r"^\s*" + _ANSWER_LEAD + r"(?:nie|nope)\b|^\s*no\s*[.!]*\s*$", re.IGNORECASE)
# krótkie odpowiedzi twierdzące bez "tak" na pytania typu "Czy boli Cię gardło?" / "Czy masz kaszel?".
# Tylko jako cała wiadomość: "mam gorączkę i kaszel" to opis objawów, nie odpowiedź "tak".
_YES_SHORT_RE = re.compile(
    r"^\s*(?:mam|boli|bolą|czuję|odczuwam|występuje|zdarza\s+si[eę]|bywa|czasem|czasami|"
    r"trochę|troszkę|lekko|niestety|mhm)\s*[.!]*\s*$",
    re.IGNORECASE,
)
_UNKNOWN_RE = re.compile(
    r"^\s*(?:nie\s+wiem|nie\s+jestem\s+pew\w*|nie\s+pami[eę]tam|trudno\s+powiedzie[cć]|nie\s+mam\s+poj[eę]cia)\b",
    re.IGNORECASE,
)


def _parse_yes_no(text: str) -> Optional[str]:
    """
    Odpowiedź na pytanie "Czy masz X?": "yes" / "no" / "unknown" albo None,
    jeśli wiadomość nie jest odpowiedzią tak/nie (to zwykły opis objawów).
    """
    t = (text or "").strip()
    if not t:
        return None
    if _UNKNOWN_RE.match(t):
        return "unknown"
    if _NO_RE.match(t):
        return "no"
    if _YES_RE.match(t) or _YES_SHORT_RE.match(t):
        return "yes"
    return None


# Polityka leków: bez dawek. Zdania z dawkowaniem usuwamy w całości
# (wycinanie samych liczb zostawiało urwane zdania).
_DOSE_SENTENCE_PAT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|µg|ml|j\.?\s?m\.?|iu|tabletk\w*|tabl\.?|kapsu\w*|kaps\.?|kropl\w*|saszet\w*)\b"
    r"|\bco\s*\d+\s*(?:h|godz\w*)\b"
    r"|\b\d+\s*(?:razy|x)\s*(?:dziennie|na\s+dob[eę]|w\s+ci[aą]gu\s+doby)\b",
    flags=re.IGNORECASE,
)


def _drop_dose_sentences(text: str) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [p for p in parts if not _DOSE_SENTENCE_PAT.search(p)]
    return " ".join(kept).strip() if kept else text

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
                        return f"odpowiedź odwraca hierarchię: używa 'jednak' i wskazuje '{other}' jako prawdopodobniejszą"

    return None



_DANGLING_END_WORDS = {
    "w", "we", "z", "ze", "do", "od", "na", "po", "przed", "przez", "przy", "nad", "pod",
    "o", "u", "dla", "bez", "i", "oraz", "lub", "albo", "a", "ale", "że", "bo", "czy",
    "jak", "gdy", "jeśli", "jeżeli", "który", "która", "które", "którego", "której",
    "których", "jest", "są", "się", "ten", "ta", "tego", "tej", "ciągu", "okresie",
}
_DANGLING_ADJ = {
    "najbliższy", "najbliższa", "najbliższe", "najbliższego", "najbliższej", "najbliższych",
    "kilku", "kilka", "następnych", "następnego", "następnej", "kolejnych", "ostatnich",
}
_PREPOSITIONS = {"w", "we", "z", "ze", "do", "od", "na", "po", "przed", "przez", "przy", "ciągu", "okresie"}


def _looks_truncated(text: str) -> bool:
    """
    Odpowiedź urwana w połowie zdania: brak znaku końca zdania albo zdanie
    kończy się przyimkiem/spójnikiem ("...w ciągu.", "...oraz.") lub
    zawieszonym określeniem po przyimku ("...w ciągu najbliższych.").
    Krótkie, poprawne zdania ("Zdrowia życzę!", "Nie.") NIE są urwane.
    """
    s = (text or "").strip()
    if not s:
        return False
    last = s[-1]
    ends_well = last in ".!?…" or (last in "\")]" and len(s) >= 2 and s[-2] in ".!?…")
    if not ends_well:
        return True

    sentences = re.split(r"(?<=[.!?…])\s+", s)
    last_sentence = sentences[-1].strip() if sentences else s
    words = re.findall(r"[a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", last_sentence.lower())
    if not words:
        return False
    if words[-1] in _DANGLING_END_WORDS:
        return True
    if words[-1] in _DANGLING_ADJ and any(w in _PREPOSITIONS for w in words[-4:-1]):
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
        # ":" → ": ", "." / "\n" → ". "
        if sep == ":":
            return f"{connector}, {title_lower}: "
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



def _trim_truncated_tail(text: str, max_drop: int = 2) -> str:
    """Usuwa TYLKO urwane zdania z końca odpowiedzi (najwyżej `max_drop`)."""
    if not text:
        return text
    parts = [p for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]
    if not parts:
        return text
    dropped = 0
    while parts and dropped < max_drop and _looks_truncated(parts[-1]):
        parts.pop()
        dropped += 1
    if not parts:
        return text
    result = " ".join(parts).strip()
    if result and result[-1] not in ".!?…)]\"":
        result += "."
    return result


def _objaw_locative(n: int) -> str:
    """"na 1 objawie", "na 2 objawach"."""
    return "objawie" if n == 1 else "objawach"


class ChatService:
    def __init__(self, state: AppState):
        self.st = state
        self.selector = FollowupSelector(state)
        self.loc = PromptLocalizer(state.matcher)
        self.ens = EnsembleTopK(state)

    # ───────────────────────── helpers ─────────────────────────
    def _guard_leader_consistency(self, text: str, leader_pl: Optional[str], *, enforce: bool) -> str:
        if not enforce or not text or not leader_pl:
            return text
        if leader_pl.strip().lower() in text.lower():
            return text
        return (text.rstrip(" .") + ".").rstrip() + f" Główna z rozważanych przyczyn to {leader_pl}."

    @staticmethod
    def _join_pl(items: List[str]) -> str:
        items = [i for i in (items or []) if i]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " i " + items[-1]

    def _known_features(self, items: Optional[Sequence[str]]) -> List[str]:
        feats = {str(f).strip().lower() for f in (self.st.xgb.features or [])}
        return sorted({str(s).strip().lower() for s in (items or []) if str(s).strip().lower() in feats})

    def _build_feature_vector(self, selected_canons: List[str]) -> List[int]:
        feats = [str(f).strip().lower() for f in (self.st.xgb.features or [])]
        expanded: set[str] = set()
        for s in (selected_canons or []):
            key = str(s).strip().lower()
            try:
                expanded |= set(self.st.matcher.expand_to_xgb(key) or [key])
            except Exception:
                expanded.add(key)
        return [1 if f in expanded else 0 for f in feats]

    def _llm_ready(self, llm) -> bool:
        if not llm:
            return False
        if getattr(llm, "backend", None) == "ctransformers":
            return llm.model is not None
        return False

    def _safe_stop_sequences(self) -> Tuple[str, ...]:
        raw = list(LLM_STOP_SEQUENCES or [])
        blacklist = {"\nUser:", "\nSystem:", "User:", "System:"}
        return tuple(s for s in raw if s not in blacklist)

    def _predict_top(self, sess_id: str, symptoms: List[str], k: int) -> List[Dict[str, Any]]:
        """Top-k z ensemble; wynik cache'owany w sesji i liczony ponownie tylko po zmianie wektora."""
        vec = self._build_feature_vector(sorted(symptoms))
        negated = sorted(self.st.sessions.get_negated(sess_id))
        # klucz cache: wektor objawów + objawy zaprzeczone (one też zmieniają Top-3)
        vt = tuple(vec) + (-1,) + tuple(negated)
        cached = self.st.sessions.get_last_pred(sess_id)
        if PREDICT_ONLY_ON_NEW and cached and cached[0] == vt and cached[1] == int(k):
            log.info("PREDICT decision=SKIP reason=no_change")
            return [dict(d) for d in cached[2]]
        try:
            raw_top = self.ens.predict_topk([vec], k=k, negated=negated) or []
        except Exception as e:
            log.exception("Ensemble.predict_topk error: %s", e)
            raw_top = []
        top = _normalize_topk(raw_top)
        self.st.sessions.set_last_pred(sess_id, vt, int(k), top)
        return top

    def _next_question(self, sess_id: str, symptoms: List[str]) -> Optional[str]:
        """
        Objaw, o który warto zapytać: największy oczekiwany przyrost informacji względem
        rozkładu chorób. Rozkład liczymy z P(objaw | choroba) dla objawów obecnych i zaprzeczonych
        (naiwny Bayes), bo model traktuje "nie wiadomo" jak "nie ma", więc po odpowiedzi "nie"
        jego rozkład się nie zmienia i kolejne pytania byłyby wybierane po omacku.
        Symulacja na zbiorze testowym (2 objawy na start, 3 pytania): trafność Top-1 0.45 -> 0.56.
        """
        posterior: Optional[Dict[str, float]] = None
        negated = self.st.sessions.get_negated(sess_id)
        stats = getattr(self.st, "symptom_stats", None)
        if symptoms and stats is not None and hasattr(stats, "nb_posterior"):
            try:
                posterior = stats.nb_posterior(symptoms, negated)
            except Exception as e:
                log.warning("Rozkład NB dla pytań niedostępny: %s", e)
        if posterior is None and symptoms:
            try:
                posterior = self.ens.predict_distribution(
                    [self._build_feature_vector(sorted(symptoms))], negated=negated)
            except Exception as e:
                log.warning("Posterior dla pytań niedostępny: %s", e)
        try:
            return self.selector.choose_next(sess_id, sorted(symptoms), lang=LANG, posterior=posterior)
        except Exception as e:
            log.exception("FollowupSelector error: %s", e)
            return None

    def _ask(self, sess_id: str, feat: str, symptoms: List[str], prefix: str = "", suffix: str = "") -> ChatResponse:
        question = self.loc.followup_question(feat, LANG)
        self.st.sessions.set_pending_symptom(sess_id, feat)
        self.st.sessions.mark_followup_asked(sess_id, feat)
        text = " ".join(p for p in (prefix.strip(), question, suffix.strip()) if p)
        self.st.sessions.append_assistant(sess_id, text)
        return ChatResponse(response=text, symptoms=sorted(symptoms))

    def _say(self, sess_id: str, text: str, symptoms: List[str]) -> ChatResponse:
        self.st.sessions.append_assistant(sess_id, text)
        return ChatResponse(response=text, symptoms=sorted(symptoms))

    def _deterministic_summary(
        self,
        sym_for_display: List[str],
        top: List[Dict[str, Any]],
        *,
        patient_age: Optional[int],
        history_text: str,
    ) -> str:
        joined = self._join_pl(sym_for_display)
        s1 = f"Zgłoszone objawy: {joined}." if joined else "Brak potwierdzonych objawów."
        if top:
            first = top[0]
            diag_pl = self.loc.localize_disease(first.get("diagnosis_en") or "", LANG)
            prob = float(first.get("probability", 0.0) or 0.0)
            pct = int(round(prob * 100)) if prob <= 1.0 else int(round(prob))
            s2 = f"Najbardziej prawdopodobna przyczyna to {diag_pl} ({pct}%)."
        else:
            s2 = "Nie udało się ustalić najbardziej prawdopodobnej przyczyny."

        s3 = ""
        if top and len(top) > 1:
            others = []
            for d in top[1:]:
                p = float(d.get("probability", 0.0) or 0.0)
                pct = int(round(p * 100)) if p <= 1.0 else int(round(p))
                others.append(f"{self.loc.localize_disease(d.get('diagnosis_en') or '', LANG)} ({pct}%)")
            s3 = "Inne możliwe przyczyny to " + self._join_pl(others) + "."

        # pilność wynika z objawów, wieku i czasu trwania, a NIE z pewności modelu
        tri_level, tri_window = _triage_hint(sym_for_display, patient_age, history_text=history_text)
        s4 = f"Pilność: {tri_level}; zalecane: {tri_window}."
        return ensure_summary_only(" ".join(p for p in (s1, s2, s3, s4) if p))

    def _generate(self, llm, prompt: str, **kwargs) -> str:
        try:
            return llm.generate(prompt, **kwargs) or ""
        except Exception as e:
            log.exception("Błąd LLM: %s", e)
            return ""

    # ───────────────────────── diagnoza + odpowiedź LLM ─────────────────────────
    def _diagnose(
        self,
        req: ChatRequest,
        sess_symptoms: List[str],
        history: List[Dict[str, str]],
        patient_age: Optional[int],
        *,
        low_confidence: bool,
        use_llm: bool,
        t_req: float,
        note: str = "",
    ) -> ChatResponse:
        sessions = self.st.sessions
        sid = req.session_id
        llm = self.st.llm
        sessions.clear_pending_symptom(sid)

        k = int(TOPN_DIAGNOSES)
        t_pred = time.perf_counter()
        top = self._predict_top(sid, list(sess_symptoms), k)
        log.info("TIMING predict_ensemble=%.3fs", time.perf_counter() - t_pred)
        sessions.set_last_top3(sid, [d.get("diagnosis_en", "") for d in top])

        sym_sorted = sorted(sess_symptoms)
        # objawy spoza modelu (np. światłowstręt) nie zmieniają Top-3, ale idą do opisu i oceny pilności
        sym_for_display = self.loc.localize_symptoms(sym_sorted, LANG) + sessions.get_extra(sid)
        diag_summary_display = self.loc.format_diag_summary(top, LANG)
        history_for_prompt = _filter_history_for_prompt(
            history, keep=int(PROMPT_HISTORY_TURNS), strip_summaries=bool(PROMPT_STRIP_SUMMARY)
        )
        user_history_text = _user_text(history_for_prompt)

        leader_en = top[0]["diagnosis_en"] if top else None
        leader_pl = self.loc.localize_disease(leader_en, LANG) if leader_en else None
        leader_prob = float(top[0].get("probability") or 0.0) if top else None
        last_user_msg = history_for_prompt[-1]["content"] if history_for_prompt else (req.user_input or "")
        intent = _intent_pl(last_user_msg or "")

        llm_resp = ""
        if use_llm:
            prompt = build_prompt(
                history_for_prompt,
                sym_for_display,
                need_more=False,
                diag_summary=diag_summary_display,
                patient_age=patient_age,
            )
            max_tokens = _chat_max_new_tokens()
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                min_new_tokens=int(LLM_MIN_NEW_TOKENS_CHAT),
                temperature=float(LLM_TEMP),
                top_p=float(LLM_TOP_P),
                top_k=int(LLM_TOP_K) if LLM_TOP_K else 40,
                repetition_penalty=float(LLM_REPETITION_PENALTY),
                stop_sequences=self._safe_stop_sequences(),
                seed=int(GLOBAL_SEED),
                continue_until_complete=True,
                max_continuations=2,
            )

            t_gen = time.perf_counter()
            llm_resp = self._generate(llm, prompt, **gen_kwargs)
            if not llm_resp.strip():
                # pusta odpowiedź: jedna próba bez sekwencji stopu
                llm_resp = self._generate(llm, prompt, **{**gen_kwargs, "stop_sequences": tuple()})
                log.info("LLM retry: no-stops fallback used")
            log.info("TIMING llm_generate=%.3fs (max_tokens=%d)", time.perf_counter() - t_gen, max_tokens)

            top3_names_pl = [n for n in (self.loc.localize_disease((d.get("diagnosis_en") or "").lower(), LANG) for d in top) if n]
            rejection_reason: Optional[str] = None
            if leader_pl and intent in {"diagnosis", "summary", "autosummary"}:
                rejection_reason = _llm_claims_other_top1(llm_resp, leader_pl, top3_names_pl)
            if rejection_reason is None and _looks_truncated(llm_resp):
                rejection_reason = "odpowiedź została urwana w połowie zdania"

            if rejection_reason:
                log.warning("LLM regen reason: %s. Original (first 200): %r", rejection_reason, llm_resp[:200])
                leader_pct_int: Optional[int] = None
                if leader_prob is not None:
                    leader_pct_int = int(round(leader_prob * 100)) if 0.0 <= leader_prob <= 1.0 else int(round(leader_prob))
                strict_prompt = build_prompt_strict(
                    history_for_prompt,
                    sym_for_display,
                    diag_summary=diag_summary_display,
                    patient_age=patient_age,
                    leader_pl=leader_pl,
                    leader_pct=leader_pct_int,
                    rejection_reason=rejection_reason,
                )
                strict_kwargs = {
                    **gen_kwargs,
                    "temperature": min(float(LLM_TEMP), 0.55),
                    "top_p": min(float(LLM_TOP_P), 0.85),
                    "max_new_tokens": min(int(max_tokens), 512),
                }
                t_regen = time.perf_counter()
                regen_resp = self._generate(llm, strict_prompt, **strict_kwargs)
                log.info("TIMING llm_regen=%.3fs", time.perf_counter() - t_regen)

                regen_ok = bool(regen_resp.strip())
                if regen_ok and leader_pl and intent in {"diagnosis", "summary", "autosummary"}:
                    if _llm_claims_other_top1(regen_resp, leader_pl, top3_names_pl):
                        regen_ok = False
                if regen_ok and _looks_truncated(regen_resp):
                    regen_ok = False
                if regen_ok:
                    log.info("LLM regen accepted")
                    llm_resp = regen_resp
                else:
                    log.info("LLM regen rejected - keeping original")

            # post-processing
            llm_resp = sanitize_llm_output(llm_resp)
            llm_resp = ensure_summary_only(llm_resp)
            llm_resp = _lists_to_prose(llm_resp)
            llm_resp = re.sub(r"\(\s?(\d{1,3})\s?%\)\s*\(\s?0[,.]\d{1,3}\s?\)", r"(\1%)", llm_resp)
            llm_resp = re.sub(r"\(\s?0[,.]\d{1,3}\s?\)", "", llm_resp)
            llm_resp = _drop_dose_sentences(llm_resp)

            allergy_markers = {"allergic_reaction", "hives", "urticaria", "rash", "itching", "angioedema", "wheezing"}
            llm_resp = _strip_allergy_speculation(llm_resp, any(s in allergy_markers for s in sess_symptoms))
            llm_resp = _dedup_sentences(llm_resp, max_sentences=_intent_sentence_cap(intent))
            llm_resp = self._guard_leader_consistency(llm_resp, leader_pl, enforce=(intent == "diagnosis"))
            llm_resp = _trim_truncated_tail(llm_resp)

        if not llm_resp.strip():
            llm_resp = self._deterministic_summary(
                sym_for_display, top, patient_age=patient_age, history_text=user_history_text,
            )

        # zastrzeżenie raz (przy pierwszej ocenie orientacyjnej), a nie w każdej kolejnej odpowiedzi
        caveat_given = any(
            (m.get("role") == "assistant") and "Uwaga: ocena opiera się" in (m.get("content") or "")
            for m in history
        )
        if low_confidence and not caveat_given:
            n = len(sess_symptoms)
            llm_resp = (
                f"Uwaga: ocena opiera się tylko na {n} {_objaw_locative(n)} (zalecane minimum to {int(MIN_SYMPTOMS)}), "
                f"więc jest orientacyjna i mało pewna. " + llm_resp
            ).strip()

        top_with_pl: List[Dict[str, Any]] = []
        for d in top:
            dd = dict(d)
            dd["diagnosis_pl"] = self.loc.localize_disease((d.get("diagnosis_en") or "").strip().lower(), LANG)
            top_with_pl.append(dd)
        diagnosis_en = top_with_pl[0]["diagnosis_en"] if top_with_pl else None
        diagnosis_pl = top_with_pl[0]["diagnosis_pl"] if top_with_pl else None

        if note:
            llm_resp = f"{note} {llm_resp}".strip()
        sessions.append_assistant(sid, llm_resp)
        # ocena podana: kolejne wiadomości (np. nowy objaw) ją aktualizują, bez ponownego dopytywania
        sessions.set_stopped(sid, True)
        log.info("TIMING total=%.3fs", time.perf_counter() - t_req)
        return ChatResponse(
            response=llm_resp,
            symptoms=sym_sorted,
            diagnosis_en=diagnosis_en,
            diagnosis_pl=diagnosis_pl,
            top_diagnoses=top_with_pl,
        )

    # ───────────────────────────────────────────────────────────────
    #          G Ł Ó W N A   P Ę T L A
    # ───────────────────────────────────────────────────────────────
    def handle_chat(self, req: ChatRequest, patient_age: int | None = None) -> ChatResponse:
        t_req = time.perf_counter()

        xgb = self.st.xgb
        matcher: SymptomMatcher = self.st.matcher
        sessions = self.st.sessions
        sid = req.session_id
        use_llm = str(SUMMARY_MODE or "llm").strip().lower() != "template"

        if not (xgb and xgb.model):
            raise RuntimeError("Model XGB niezaładowany.")
        if matcher is None:
            raise RuntimeError("Matcher niezaładowany.")
        if use_llm and not self._llm_ready(self.st.llm):
            raise RuntimeError("LLM niezaładowany.")

        if not sessions.exists(sid):
            sessions.init_session(sid)
        # backend mógł zostać zrestartowany - odtwarzamy objawy, które pamięta frontend.
        # Sprawdzamy "pusta sesja", a nie "brak sesji": router (claim_owner) tworzy ją wcześniej.
        if not sessions.get_history(sid) and not sessions.get_symptoms(sid):
            restored = self._known_features(getattr(req, "client_symptoms", None))
            if restored:
                sessions.set_symptoms(sid, restored)
                log.info("Sesja %s odtworzona z danych klienta (%d objawów).", sid, len(restored))

        user_text = (req.user_input or "").strip()
        sessions.append_user(sid, user_text)
        history = _normalize_history(
            sessions.get_recent_history(sid, max_messages=int(PROMPT_HISTORY_FETCH))
        )

        # 1) odpowiedź na pytanie "Czy masz X?"
        pending = sessions.get_pending_symptom(sid)
        answer = _parse_yes_no(user_text) if pending else None
        if pending and answer is not None:
            if answer == "yes":
                sessions.add_symptoms(sid, [pending])
            elif answer == "no":
                sessions.add_negated_symptom(sid, pending)
            sessions.clear_pending_symptom(sid)

        # 2) ekstrakcja objawów z całej wiadomości (z kontekstem przeczeń:
        #    "nie mam gorączki, ale mam kaszel" -> gorączka zaprzeczona, kaszel dodany)
        t_ext = time.perf_counter()
        new_syms: List[str] = []
        neg_syms: List[str] = []
        noise = len(user_text) < 3 or not re.search(r"[a-zA-Ząćęłńóśźż]", user_text, re.I)
        if not noise:
            try:
                if hasattr(matcher, "extract_detailed"):
                    new_syms, neg_syms = matcher.extract_detailed(user_text)
                else:
                    new_syms = matcher.extract(user_text)
            except Exception as e:
                log.exception("Matcher.extract błąd: %s", e)
            for f in neg_syms:
                sessions.add_negated_symptom(sid, f)
            if new_syms:
                sessions.add_symptoms(sid, new_syms)
            if pending and pending in (set(new_syms) | set(neg_syms)):
                sessions.clear_pending_symptom(sid)
        # objawy, których model nie zna (np. "światłowstręt"): potwierdzamy, że je widzimy
        extra_new: List[str] = []
        extra_neg: List[str] = []
        if not noise:
            try:
                extra_pos, extra_neg = extract_unmodeled(user_text)
                extra_new = sessions.update_extra(sid, extra_pos, extra_neg)
            except Exception as e:
                log.warning("extract_unmodeled błąd: %s", e)
        note = ""
        if extra_new:
            note = (f"Zanotowałem: {self._join_pl(extra_new)}. Model nie ma "
                    f"{'tego objawu' if len(extra_new) == 1 else 'tych objawów'} w danych, więc nie zmienia to "
                    f"listy Top-3, ale uwzględnię {'go' if len(extra_new) == 1 else 'je'} w opisie i ocenie pilności.")
        log.info("TIMING extract=%.3fs", time.perf_counter() - t_ext)

        sess_symptoms = sorted(sessions.get_symptoms(sid))
        count = len(sess_symptoms)
        user_stop = contains_stop_phrase(user_text)
        recognized = bool(new_syms or neg_syms or answer is not None or extra_new or extra_neg)

        # 2b) było pytanie, a odpowiedź nie jest ani "tak/nie", ani opisem objawów: pytamy jeszcze raz
        #     o to samo (wcześniej pytanie przepadało, a pacjent dostawał kolejne)
        if pending and sessions.get_pending_symptom(sid) == pending and not recognized and not user_stop:
            question = self.loc.followup_question(pending, LANG)
            msg = (f"Nie jestem pewien, jak zrozumieć tę odpowiedź. {question} "
                   "Wystarczy „tak”, „nie” albo „nie wiem”.")
            return self._say(sid, msg, sess_symptoms)
        log.info("DEBUG new=%s neg=%s answer=%s | symptoms=%s (n=%d) stop=%s",
                 new_syms, neg_syms, answer, sess_symptoms[:12], count, user_stop)

        asked_n = sessions.get_followup_count(sid)

        # 3a) ocena była już podana albo pacjent zakończył zbieranie objawów: aktualizujemy ocenę
        if count > 0 and sessions.is_stopped(sid):
            return self._diagnose(req, sess_symptoms, history, patient_age,
                                  low_confidence=count < int(MIN_SYMPTOMS), use_llm=use_llm, t_req=t_req, note=note)

        # 3a') pacjent napisał już raz "to wszystko" i dostał zapowiedź jednego pytania: po odpowiedzi
        #      podajemy ocenę zamiast kolejnych pytań
        if count > 0 and sessions.is_stop_warned(sid):
            sessions.set_stopped(sid, True)
            return self._diagnose(req, sess_symptoms, history, patient_age,
                                  low_confidence=count < int(MIN_SYMPTOMS), use_llm=use_llm, t_req=t_req, note=note)

        # 3b) dość objawów: oceniamy od razu, jeśli model jest pewny; inaczej dopytujemy o objaw,
        #     który najlepiej rozróżnia prawdopodobne choroby (najwyżej STOP_FOLLOWUPS_AT pytań)
        if count >= int(MIN_SYMPTOMS):
            if user_stop or asked_n >= int(STOP_FOLLOWUPS_AT):
                return self._diagnose(req, sess_symptoms, history, patient_age,
                                      low_confidence=False, use_llm=use_llm, t_req=t_req, note=note)
            top = self._predict_top(sid, sess_symptoms, int(TOPN_DIAGNOSES))
            p1 = float(top[0].get("probability") or 0.0) if top else 0.0
            if p1 >= float(FOLLOWUP_CONFIDENCE):
                log.info("FOLLOWUP stop: confident p1=%.2f after %d questions", p1, asked_n)
                return self._diagnose(req, sess_symptoms, history, patient_age,
                                      low_confidence=False, use_llm=use_llm, t_req=t_req, note=note)
            next_sym = self._next_question(sid, sess_symptoms)
            if not next_sym:
                return self._diagnose(req, sess_symptoms, history, patient_age,
                                      low_confidence=False, use_llm=use_llm, t_req=t_req, note=note)
            log.info("FOLLOWUP ask=%s p1=%.2f asked=%d", next_sym, p1, asked_n)
            if asked_n == 0:
                prefix = "Zanim podam ocenę, dopytam jeszcze o kilka rzeczy, żeby ją zawęzić."
            else:
                prefix = "" if recognized else "Nie rozpoznałem w tej wiadomości nowych objawów."
            return self._ask(sid, next_sym, sess_symptoms, prefix=f"{note} {prefix}".strip())

        # 4) brak jakichkolwiek objawów (rozpoznanych przez model)
        if count == 0:
            if extra_new:
                msg = "Opisz proszę, co jeszcze Ci dolega, np. ból głowy, gorączka, nudności."
            elif user_stop:
                msg = ("Nie mam jeszcze żadnych objawów do oceny. Opisz proszę własnymi słowami, "
                       "co Ci dolega, np. gorączka, kaszel, ból głowy.")
            elif neg_syms or answer == "no":
                msg = "Rozumiem. Jakie objawy masz w tej chwili? Opisz je proszę własnymi słowami."
            else:
                msg = ("Nie rozpoznałem w tej wiadomości objawów. Opisz proszę, co Ci dolega, "
                       "np. „od wczoraj mam gorączkę i boli mnie gardło”.")
            return self._say(sid, f"{note} {msg}".strip(), sess_symptoms)

        # 5) 0 < objawów < MIN_SYMPTOMS
        if user_stop:
            if sessions.is_stop_warned(sid) or asked_n >= int(STOP_FOLLOWUPS_AT):
                sessions.set_stopped(sid, True)
                return self._diagnose(req, sess_symptoms, history, patient_age,
                                      low_confidence=True, use_llm=use_llm, t_req=t_req, note=note)
            sessions.set_stop_warned(sid, True)
            next_sym = self._next_question(sid, sess_symptoms)
            if not next_sym:
                sessions.set_stopped(sid, True)
                return self._diagnose(req, sess_symptoms, history, patient_age,
                                      low_confidence=True, use_llm=use_llm, t_req=t_req, note=note)
            have = self._join_pl(self.loc.localize_symptoms(sess_symptoms, LANG))
            prefix = (f"Do wiarygodnej wstępnej oceny potrzebuję co najmniej {int(MIN_SYMPTOMS)} objawów, "
                      f"a na razie mam {count}: {have}. Jeszcze jedno pytanie.")
            suffix = "Jeśli nie masz innych objawów, napisz ponownie „to wszystko”, a podam ocenę orientacyjną."
            return self._ask(sid, next_sym, sess_symptoms, prefix=f"{note} {prefix}".strip(), suffix=suffix)

        if asked_n >= int(STOP_FOLLOWUPS_AT):
            # dość dopytywania - podajemy ocenę orientacyjną zamiast pytać w nieskończoność
            sessions.set_stopped(sid, True)
            return self._diagnose(req, sess_symptoms, history, patient_age,
                                  low_confidence=True, use_llm=use_llm, t_req=t_req, note=note)

        t_sel = time.perf_counter()
        next_sym = self._next_question(sid, sess_symptoms)
        log.info("TIMING select_followup=%.3fs", time.perf_counter() - t_sel)
        if not next_sym:
            sessions.set_stopped(sid, True)
            return self._diagnose(req, sess_symptoms, history, patient_age,
                                  low_confidence=True, use_llm=use_llm, t_req=t_req, note=note)

        prefix = "" if recognized else "Nie rozpoznałem w tej wiadomości nowych objawów."
        log.info("TIMING total=%.3fs", time.perf_counter() - t_req)
        return self._ask(sid, next_sym, sess_symptoms, prefix=f"{note} {prefix}".strip())

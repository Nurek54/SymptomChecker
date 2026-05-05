# services/chat_service.py
from __future__ import annotations
import logging, re, time
from typing import List, Dict, Any, Sequence, Optional, Tuple

from core.state import AppState
from schemas import ChatRequest, ChatResponse
from config import (
    MIN_SYMPTOMS, LLM_MAX_NEW_TOKENS, LLM_TEMP, LLM_TOP_P,
    LLM_STOP_SEQUENCES, GLOBAL_SEED,
    TOPN_DIAGNOSES
)

# Fallbacki opcji konfiguracyjnych
try:
    from config import STOP_FOLLOWUPS_AT
except Exception:
    STOP_FOLLOWUPS_AT = 4

try:
    from config import PREDICT_ONLY_ON_NEW
except Exception:
    PREDICT_ONLY_ON_NEW = True  # predykcja tylko po nowych objawach

# Nowe (opcjonalne) sterowanie historią do promptu
try:
    from config import PROMPT_HISTORY_FETCH  # ile wiadomości pobrać z sesji
except Exception:
    PROMPT_HISTORY_FETCH = 20

try:
    from config import PROMPT_HISTORY_TURNS  # ile zostawić po filtrze
except Exception:
    PROMPT_HISTORY_TURNS = 8

try:
    from config import PROMPT_STRIP_SUMMARY  # czy wycinać wcześniejsze podsumowania
except Exception:
    PROMPT_STRIP_SUMMARY = False

from services.followup_selector import FollowupSelector
from services.lang import detect_lang, is_yes, is_no
from services.prompt_localizer import PromptLocalizer
from services.symptom_matcher import SymptomMatcher
from services.prompt_builder import build_prompt, _classify_intent_pl as _intent_pl
# użyjemy heurystyki triage z prompt_buildera dla deterministycznego fallbacku
try:
    from services.prompt_builder import _suggest_triage as _triage_hint  # type: ignore
except Exception:
    _triage_hint = None  # fallback niżej

from services.ensemble import EnsembleTopK

from utils import (
    contains_stop_phrase,
    sanitize_llm_output,
    ensure_summary_only,
)

log = logging.getLogger("symptomchecker.chat_service")


# ====== Normalizacje i pomocnicze =================================================

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
            role = str(m[0]); content = str(m[1])
        else:
            role = "user"; content = str(m)
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
    # krótka historia = szybsze i mniej echa; zostawiamy ostatnie 'keep' wpisów
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


# ====== Post-processing jakości i bezpieczeństwa ================================

# wzorce dawek/jednostek do usunięcia (zostawiamy NAZWY substancji!)
_DOSE_PAT = re.compile(
    r"(?:(?:\b|\s)(?:\d+(?:[.,]\d+)?)\s?(?:mg|ml|mcg|µg|g|%|iu|j\.?m\.?|kropli|tabletek|tabl\.?|caps?\.?|spray(?:e|u)?|razy|x)\b(?:[^.;\n]{0,40})?)",
    flags=re.IGNORECASE
)

def _strip_doses_only(text: str) -> str:
    if not text:
        return text
    out = _DOSE_PAT.sub("", text)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out

def _strip_allergy_speculation(text: str, have_allergy_symptom: bool) -> str:
    if have_allergy_symptom:
        return text
    # wytnij fragmenty o alergenach pokarmowych itd., gdy brak objawów alergicznych
    out = re.sub(r"(alergen(y|ów)? pokarmow(e|ych)|nadwrażliwość na pokarmy|nietolerancja pokarmowa)[^.]*[.]", "", text, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out

def _dedup_sentences(text: str, max_sentences: int | None = None) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    seen = set()
    out = []
    for p in parts:
        key = re.sub(r"\s+", " ", p.strip().lower())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p.strip())
        if max_sentences is not None and len(out) >= max_sentences:
            break
    return " ".join(out)

def _intent_sentence_cap(intent: str) -> int:
    # twarde limity zdań → czytelność i unikanie lania wody
    return {
        "otc": 2,
        "triage": 3,
        "advice": 4,
        "diagnosis": 4,
        "summary_points": 5,
        "prophylaxis": 4,
        "autosummary": 4,
    }.get(intent, 4)


# ====== Główna usługa czatu ======================================================

class ChatService:
    def __init__(self, state: AppState):
        self.st = state
        self.selector = FollowupSelector(state)
        self.loc = PromptLocalizer(state.matcher)
        self.ens = EnsembleTopK(state)  # XGB+River

        # Cache predykcji per sesja: (ostatni wektor, k, top)
        self._last_vec: Dict[str, Tuple[Tuple[int, ...], int, List[Dict[str, Any]]]] = {}
        # Cache ekspansji kanonów -> cechy XGB
        self._expand_cache: Dict[str, List[str]] = {}

    # — post-guard: Top-1 tylko, gdy to ma sens (diagnosis) —
    def _guard_leader_consistency(self, text: str, leader_pl: Optional[str], leader_prob: Optional[float], *, enforce: bool) -> str:
        if not enforce or not text or not leader_pl:
            return text
        try:
            lp = leader_pl.strip().lower()
            pat = re.compile(r"(najbardziej\s+prawdopodobn[aey]\s+(?:przyczyna|diagnoza)\s+to\s+)([^.;\n]+)", re.I)

            pct = ""
            if leader_prob is not None:
                try:
                    pct_val = int(round(leader_prob * 100)) if 0.0 <= leader_prob <= 1.0 else int(round(leader_prob))
                    pct = f" ({pct_val}%)"
                except Exception:
                    pct = ""

            correct = f"Najbardziej prawdopodobna przyczyna to {leader_pl}{pct}"

            m = pat.search(text)
            if m:
                mentioned = (m.group(2) or "").strip().lower()
                if lp not in mentioned:
                    text = pat.sub(lambda _: correct, text, count=1)
            else:
                text = correct + ". " + text.lstrip()

            first_stop = re.search(r"[.!?]\s", text)
            first_sentence = text[:first_stop.end()] if first_stop else text
            if lp not in first_sentence.lower():
                text = correct + ". " + text.lstrip()

            return text
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
        if not prev or prev == curr:
            return ""
        prev_disp = [self.loc.localize_disease(x, lang) for x in (prev or [])]
        curr_disp = [self.loc.localize_disease(x, lang) for x in (curr or [])]
        # jedna fraza, bez nowych linii – zostaje w jednym akapicie
        return (
            "Uwaga: po nowych objawach zaktualizowano listę możliwych przyczyn (Top-3); "
            f"poprzednio: {', '.join(prev_disp)}; teraz: {', '.join(curr_disp)}. "
        )

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
        asked_recent = set(self.st.sessions.get_recent_followups(sess_id, n=3)) if hasattr(self.st.sessions, "get_recent_followups") else set()
        for f in feats:
            fl = str(f).strip().lower()
            if fl not in have and fl not in neg and fl not in asked_recent:
                return fl
        return None

    def _deterministic_summary(self, sym_for_display: List[str], top: List[Dict[str, Any]], lang: str,
                               *, patient_age: Optional[int], history_text: str) -> str:
        """
        Fallback gdy LLM nie odpowie: jeden akapit + triage z heurystyką z prompt_buildera.
        """
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

        # Triage: preferujemy heurystykę z prompt_buildera
        if _triage_hint is not None:
            tri_level, tri_window = _triage_hint(sym_for_display, patient_age, history_text=history_text)
            s4 = f"Pilność: {tri_level}; zalecane: {tri_window}."
        else:
            # bardzo prosty fallback (gdyby ktoś usunął _triage_hint)
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

    # — pomocnicze: bezpieczne STOP-y tylko na czas generacji —
    def _safe_stop_sequences(self) -> Tuple[str, ...]:
        raw = list(LLM_STOP_SEQUENCES or [])
        blacklist = {"\nUser:", "\nSystem:", "User:", "System:"}
        return tuple(s for s in raw if s not in blacklist)

    # — Ensemble (XGB+River) Top-K + cache —
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

        # — follow-up Tak/Nie (+ ewentualny „ogon”) —
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

        # wczesny powrót po samym tak/nie (bez „ogona”)
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

        # ekstrakcja objawów (z bramką)
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
        lang = detect_lang(user_text)  # PL domyślnie

        # tylko follow-up („to wszystko”) przy zbyt małej liczbie objawów
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

            # >>>>>>> HISTORIA DO PROMPTU: więcej i bez wycinania podsumowań
            history_for_prompt = _filter_history_for_prompt(
                history, keep=int(PROMPT_HISTORY_TURNS), strip_summaries=bool(PROMPT_STRIP_SUMMARY)
            )

            # anchor dla Top-1 (do strażnika spójności)
            leader_en = (top[0]["diagnosis_en"] if top else None)
            leader_pl = self.loc.localize_disease(leader_en, lang) if leader_en else None
            leader_prob = float(top[0].get("probability") or 0.0) if top else None

            # prompt (PL → PL)
            prompt = build_prompt(
                history_for_prompt,
                sym_for_display,
                need_more=False,
                diag_summary=diag_summary_display,
                patient_age=patient_age,
            )

            # — generacja —
            t_gen = time.perf_counter()
            safe_stops = self._safe_stop_sequences()
            try:
                llm_resp = llm.generate(
                    prompt,
                    max_new_tokens=min(int(LLM_MAX_NEW_TOKENS), 256),
                    temperature=float(LLM_TEMP),
                    top_p=float(LLM_TOP_P),
                    stop_sequences=safe_stops,
                    seed=int(GLOBAL_SEED),
                )
            except Exception as e:
                log.exception("Błąd LLM (ctransformers): %s", e)
                llm_resp = ""

            # PUSTO? —> jedno powtórzenie BEZ STOPÓW
            if not (llm_resp or "").strip():
                try:
                    llm_resp = llm.generate(
                        prompt,
                        max_new_tokens=min(int(LLM_MAX_NEW_TOKENS), 256),
                        temperature=float(LLM_TEMP),
                        top_p=float(LLM_TOP_P),
                        stop_sequences=tuple(),  # brak stopów
                        seed=int(GLOBAL_SEED),
                    )
                    log.info("LLM retry: no-stops fallback used")
                except Exception as e:
                    log.exception("Błąd LLM (retry no-stops): %s", e)
                    llm_resp = ""
            log.info("TIMING llm_generate=%.3fs (stops=%s)", time.perf_counter() - t_gen, list(safe_stops))

            # — post-processing odpowiedzi —
            llm_resp = sanitize_llm_output(llm_resp)
            llm_resp = ensure_summary_only(llm_resp)

            # dublowane (0,46) vs (46%)
            llm_resp = re.sub(r"\(\s?0[,.]\d{1,3}\s?\)", "", llm_resp)
            llm_resp = re.sub(r"\(\s?(\d{1,3})\s?%\)\s*\(\s?0[,.]\d{1,3}\s?\)", r"(\1%)", llm_resp)

            # usuń dawki/jednostki, ale zostaw NAZWY substancji
            llm_resp = _strip_doses_only(llm_resp)

            # filtr spekulacji o alergiach jeśli brak objawów alergicznych
            allergy_markers = {"allergic_reaction", "hives", "urticaria", "rash", "itching", "angioedema", "wheezing"}
            have_allergy = any(s in allergy_markers for s in (self.st.sessions.get_symptoms(req.session_id) or []))
            llm_resp = _strip_allergy_speculation(llm_resp, have_allergy)

            # limit liczby zdań wg intencji + odduplikowanie
            last_user_msg = history_for_prompt[-1]["content"] if history_for_prompt else (req.user_input or "")
            intent = _intent_pl(last_user_msg or "")
            llm_resp = _dedup_sentences(llm_resp, max_sentences=_intent_sentence_cap(intent))

            # wymuś Top-1 tylko jeśli pytanie było o DIAGNOZĘ
            llm_resp = self._guard_leader_consistency(llm_resp, leader_pl, leader_prob, enforce=(intent == "diagnosis"))

            if not llm_resp.strip():
                # deterministyczny fallback z triage-heurystyką
                llm_resp = self._deterministic_summary(
                    sym_for_display, top, lang, patient_age=patient_age,
                    history_text=" ".join((m.get("content") or "") for m in history_for_prompt)
                )

            note = self._top3_update_note(lang, prev_top_names, curr_top_names)
            if note:
                llm_resp = (note + llm_resp).strip()
                llm_resp = ensure_summary_only(llm_resp)  # upewnij się, że pozostał jeden akapit

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
            return ChatResponse(
                response=llm_resp,
                symptoms=sorted(list(sess_symptoms)),
                diagnosis_en=diagnosis_en,
                diagnosis_pl=diagnosis_pl,
                top_diagnoses=top_with_pl
            )

        # — follow-up kiedy < MIN_SYMPTOMS —
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

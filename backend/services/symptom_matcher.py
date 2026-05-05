# backend/services/symptom_matcher.py
from __future__ import annotations
import os
import re
import json
import unicodedata
from typing import List, Dict, Set, Optional

from config import (
    SYMPTOMS_DICT_PATH, EMB_USE_DICTIONARY, EMB_RULE_FIRST, EMB_RULE_MIN_LEN,
    NEGATION_PATTERNS,
)

MIN_TOKEN_LEN = 3
FUZZY_MIN_SCORE_TOKEN = 80
FUZZY_MIN_SCORE_PHRASE = 78
NEG_WINDOW_CHARS = 14

# ───── Generyczne (nie są objawami same w sobie) ─────
GENERIC_SYMPTOM_WORDS: Set[str] = {
    "objaw","objawy","objawow","objawami","objawach","objawie",
    "symptom","symptomy","symptomow","symptomami","symptomach",
}

# ───── Słowa pytające / modalne ─────
QUESTION_WORDS: Set[str] = {
    "jak","jaki","jaka","jakie","jacy","jakich","jakiego","jakiej","jakim","jakimi",
    "ktory","ktora","ktore","ktorzy","ktorych","ktorego","ktorej","ktorym","którą",
    "gdzie","kiedy","dlaczego","po","co","ile","kto","czy",
    "what","which","how","where","when","why","who"
}

# ───── Meta-intencje (wykrywanie pytań „o poradę”, nie o objawy) ─────
META_INTENT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bzalecen"),                 # zalecenia, zalecenie
    re.compile(r"\bpilnos"),                  # pilność
    re.compile(r"\bleki\s*(bez\s*recepty|otc)?\b"),
    re.compile(r"\bprofilaktyk"),
    re.compile(r"\bpodsum"),                  # podsumuj/podsumowanie
    re.compile(r"\bstres"),                   # streść
    re.compile(r"\bnajbardziej\s+prawdopodobn"),
    re.compile(r"\bdiagnoz"),
    re.compile(r"\bocen\b"),                  # oceń
    re.compile(r"\bpowiedz\b"),
    re.compile(r"\bjaka?\s+chorob"),
]

# Tokeny nie-medyczne do bramki leksykalnej
MED_BLACKLIST_TOKENS = {
    "leki","lek","otc","dawka","dawki","profilaktyka","zalecenia"
} | GENERIC_SYMPTOM_WORDS | QUESTION_WORDS

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


def _canonize(txt: str) -> str:
    t = (txt or "").lower()
    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"[^a-z0-9żźćńółęąś\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _strip_accents(text: str) -> str:
    if not text:
        return text
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _has_diacritics(s: str) -> bool:
    return any(ch in "ąćęłńóśźż" for ch in s)


_STOP = {
    "mam","mnie","mi","nie","to","jest","są","sa","i","oraz","a","czy","że","ze","w","na","u","z","od","do","się","sie",
    "and","or","the","a","an","of","to","in","on","at","for","with","without","have","has","had"
} | GENERIC_SYMPTOM_WORDS | QUESTION_WORDS
MED_WHITELIST = {"bol"}  # 'ból' po kanonizacji


def _tokens(txt: str, min_len: int = MIN_TOKEN_LEN) -> List[str]:
    noacc = _strip_accents(_canonize(txt))
    return [t for t in noacc.split() if t in MED_WHITELIST or (len(t) >= min_len and t not in _STOP)]


def _all_generic(text: str) -> bool:
    toks = [_strip_accents(t) for t in _canonize(text).split()]
    return bool(toks) and all(t in GENERIC_SYMPTOM_WORDS for t in toks)


def _approx_in(needle: str, hay: str) -> bool:
    n0 = _strip_accents(_canonize(needle))
    if n0 in GENERIC_SYMPTOM_WORDS or n0 in QUESTION_WORDS or len(n0) < 6:
        return False
    n = n0
    h = _strip_accents(_canonize(hay))
    if not n or not h:
        return False
    variants = {n}
    if len(n) > 3: variants.add(n[:-1])
    if len(n) > 4: variants.add(n[:-2])
    variants |= {n.replace("ę","e"), n.replace("ą","a")}
    if len(n) > 3: variants |= {n[:-1]+"e", n[:-1]+"a"}
    return any(v and v in h for v in variants)


def _split_phrases(txt: str) -> List[str]:
    t = " " + _canonize(txt) + " "
    t = re.sub(r"\s+(?:oraz|albo|lub|i|a także|plus|and|or)\s+", " | ", t)
    parts = re.split(r"[|,;/.]+", t)
    return [p.strip() for p in parts if p.strip()]


_NEG_PATTERNS_NORM: Set[str] = {_strip_accents(_canonize(p)) for p in (NEGATION_PATTERNS or [])}


def _has_negation(fragment: str, alias_like: str) -> bool:
    frag = " " + _strip_accents(_canonize(fragment)) + " "
    ali  = " " + _strip_accents(_canonize(alias_like)) + " "
    idx = frag.find(ali)
    if idx == -1:
        return False
    window = frag[max(0, idx - NEG_WINDOW_CHARS): idx + 1]
    return any(neg in window for neg in _NEG_PATTERNS_NORM if neg)


class SymptomMatcher:
    """
    XGB-aware matcher (rules-only):
    - aliasy/contains/approx/fuzzy + negacje,
    - bez modeli embeddingowych.
    """

    def __init__(self, features: List[str], device: str = "cpu") -> None:
        self.features: List[str] = [str(f).strip().lower() for f in (features or [])]
        self._feat_set: Set[str] = set(self.features)

        self.alias_to_feature: Dict[str, str] = {}
        self._alias_regex: Optional[re.Pattern] = None
        self._alias_keys: List[str] = []
        self._alias_key_to_feat: Dict[str, str] = {}
        self._feat_to_aliases: Dict[str, Set[str]] = {}

        self._feat2alts: Dict[str, Set[str]] = {}
        self._feat_category: Dict[str, str] = {}
        self._xgb2pl: Dict[str, str] = {}
        self._pl2xgb: Dict[str, str] = {}

        self._med_tokens: Set[str] = set()

        if EMB_USE_DICTIONARY and SYMPTOMS_DICT_PATH and os.path.exists(SYMPTOMS_DICT_PATH):
            ok = self._build_alias_index(SYMPTOMS_DICT_PATH)
            if not ok:
                self._build_default_alias_index_from_features()
        else:
            self._build_default_alias_index_from_features()

        self.ready = True

    # ───────────────────────── budowa indeksów ─────────────────────────

    def _build_default_alias_index_from_features(self) -> None:
        added = 0
        for canon in sorted(self._feat_set):
            display_pl = canon.replace("_", " ")
            self._xgb2pl[canon] = display_pl
            self._pl2xgb[_canonize(display_pl)] = canon
            self._feat_category[canon] = "Other"
            self._feat2alts[canon] = {canon}

            aliases: Set[str] = {canon, _canonize(display_pl)}
            aliases_noacc = {_strip_accents(a) for a in list(aliases)}
            aliases |= aliases_noacc

            for a in aliases:
                self.alias_to_feature[a] = canon
                self._feat_to_aliases.setdefault(canon, set()).add(a)
                self._alias_keys.append(a)
                self._alias_key_to_feat[a] = canon
                for tok in a.split():
                    tt = _strip_accents(tok)
                    if len(tt) >= 4 and tt not in _STOP and tt not in MED_BLACKLIST_TOKENS:
                        self._med_tokens.add(tt)

            added += 1

        alias_uniq = sorted(set(self.alias_to_feature.keys()), key=len, reverse=True)
        alias_parts: List[str] = []
        for a in alias_uniq:
            if " " in a or "-" in a or "/" in a:
                alias_parts.append(re.escape(a))
            else:
                alias_parts.append(rf"\b{re.escape(a)}\b")
        if alias_parts:
            try:
                self._alias_regex = re.compile("(" + "|".join(alias_parts) + ")", flags=re.IGNORECASE)
            except re.error:
                self._alias_regex = None

        print(f"[DICT/FALLBACK] features={len(self._feat_set)} | aliasów={len(self.alias_to_feature)} | med_tokens={len(self._med_tokens)}")

    def _build_alias_index(self, dict_path: str) -> bool:
        try:
            data = json.load(open(dict_path, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[DICT] Błąd wczytywania {dict_path}: {e}")
            return False

        added = 0
        for row in (data or []):
            pl_label = (row.get("pl") or "").strip()
            cat_raw  = (row.get("category") or "").strip() or "Other"
            xgb = row.get("xgb_label")

            if isinstance(xgb, str):
                xgb_list = [xgb.strip().lower()] if xgb else []
            else:
                xgb_list = [str(x).strip().lower() for x in (xgb or []) if x]

            canon = next((x for x in xgb_list if x in self._feat_set), (xgb_list[0] if xgb_list else ""))
            if not canon or canon not in self._feat_set:
                continue

            display_pl = pl_label or canon.replace("_", " ")
            self._xgb2pl[canon] = display_pl
            self._pl2xgb[_canonize(display_pl)] = canon
            self._feat_category[canon] = cat_raw

            alts = {x for x in xgb_list if x in self._feat_set} or {canon}
            self._feat2alts[canon] = alts

            aliases: Set[str] = set(alts)
            en_raw = (row.get("en") or "").strip()
            if en_raw:
                aliases.add(_canonize(en_raw))
            if display_pl:
                aliases.add(_canonize(display_pl))
            for a in (row.get("synonyms_en") or []):
                if a: aliases.add(_canonize(a))
            for a in (row.get("synonyms_pl") or []):
                if a: aliases.add(_canonize(a))

            for a in aliases:
                if not a or len(a) < int(EMB_RULE_FIRST and EMB_RULE_MIN_LEN or 0):
                    continue
                if _all_generic(a):
                    continue

                a_no = _strip_accents(a)
                add_noacc = (len(a) > 4)

                self.alias_to_feature[a] = canon
                self._feat_to_aliases.setdefault(canon, set()).add(a)

                if add_noacc:
                    self.alias_to_feature[a_no] = canon
                    self._feat_to_aliases[canon].add(a_no)
                    self._alias_keys.append(a_no)
                    self._alias_key_to_feat[a_no] = canon

                for tok in a.split():
                    tt = _strip_accents(tok)
                    if len(tt) >= 4 and tt not in _STOP and tt not in MED_BLACKLIST_TOKENS:
                        self._med_tokens.add(tt)

            added += 1

        alias_uniq = sorted(set(self.alias_to_feature.keys()), key=len, reverse=True)
        alias_parts: List[str] = []
        for a in alias_uniq:
            if " " in a or "-" in a or "/" in a:
                alias_parts.append(re.escape(a))
            else:
                alias_parts.append(rf"\b{re.escape(a)}\b")
        if alias_parts:
            try:
                self._alias_regex = re.compile("(" + "|".join(alias_parts) + ")", flags=re.IGNORECASE)
            except re.error:
                self._alias_regex = None

        print(f"[DICT] Wpisów:{added} | aliasów:{len(self.alias_to_feature)} | keys:{len(self._alias_keys)} | cat:{len(self._feat_category)} | med_tokens:{len(self._med_tokens)}")
        return True

    # ───────────────────────── translacje/kategorie ─────────────────────────

    def to_pl(self, xgb_or_alias: str) -> str:
        k = _canonize(xgb_or_alias)
        canon = self.alias_to_feature.get(k, k)
        return self._xgb2pl.get(canon, canon.replace("_", " "))

    def to_en(self, text_pl_or_alias: str) -> Optional[str]:
        k = _canonize(text_pl_or_alias)
        if k in self._pl2xgb:
            return self._pl2xgb[k]
        return self.alias_to_feature.get(_strip_accents(k))

    def get_category(self, xgb_or_alias: str) -> str:
        canon = self.alias_to_feature.get(_canonize(xgb_or_alias), _canonize(xgb_or_alias))
        return self._feat_category.get(canon, "Other")

    # ───────────────────────── fuzzy/negacje ─────────────────────────

    def _fuzzy_on_fragment(self, frag: str) -> Set[str]:
        hits: Set[str] = set()
        toks = _tokens(frag)
        if not toks:
            return hits

        grams = []
        L = len(toks)
        for n in (1, 2, 3):
            for i in range(L - n + 1):
                grams.append(" ".join(toks[i:i+n]))
        grams = grams[:20]

        grams = [g for g in grams if not _all_generic(g)]
        grams = [g for g in grams if (" " in g) or (len(g) >= 5)]

        # ⬇️ n-gram musi zawierać ≥1 med-token (lub whitelisted 'bol')
        grams_keep = []
        for g in grams:
            gtoks = [t for t in _tokens(g) if t not in MED_BLACKLIST_TOKENS]
            if any(t in self._med_tokens or t in MED_WHITELIST for t in gtoks):
                grams_keep.append((g, set(gtoks)))
        grams = grams_keep

        if not grams or not self._alias_keys:
            return hits

        def _jaccard(a: Set[str], b: Set[str]) -> float:
            if not a and not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b) or 1
            return inter / union

        if _HAS_RAPIDFUZZ:
            for g, gtoks_set in grams:
                base_thr = FUZZY_MIN_SCORE_TOKEN if " " not in g else FUZZY_MIN_SCORE_PHRASE
                # delikatna dynamika progu wg długości
                lg = len(g)
                if lg <= 5:
                    base_thr += 8
                elif lg >= 12:
                    base_thr += 4

                for cand, score_set, _ in rf_process.extract(
                    g, self._alias_keys, scorer=rf_fuzz.token_set_ratio, limit=4
                ):
                    # dodatkowe metryki dla stabilizacji
                    score_sort = rf_fuzz.token_sort_ratio(g, cand)
                    score_part = rf_fuzz.partial_ratio(g, cand)
                    score = max(score_set, score_sort)

                    if score < base_thr or score_part < (base_thr - 5):
                        continue

                    # wymagaj wspólnego niegenerycznego tokenu
                    ctoks = set(t for t in _tokens(cand) if t not in MED_BLACKLIST_TOKENS)
                    if not (gtoks_set & ctoks):
                        continue

                    # dodatkowy bezpiecznik: prefiks/jaccard
                    if " " in g:
                        if _jaccard(gtoks_set, ctoks) < 0.34:
                            continue
                    else:
                        # jednowyrazowe: prefiks ≥3 znaki (po od-akcentowaniu)
                        g0 = _strip_accents(g).split()[0]
                        c0 = _strip_accents(cand).split()[0]
                        if len(g0) < 3 or len(c0) < 3 or not (g0[:3] == c0[:3]):
                            continue

                    feat = self._alias_key_to_feat.get(cand)
                    if feat:
                        hits.add(feat)
        else:
            import difflib
            for g, gtoks_set in grams:
                base_thr = FUZZY_MIN_SCORE_TOKEN if " " not in g else FUZZY_MIN_SCORE_PHRASE
                lg = len(g)
                if lg <= 5:
                    base_thr += 8
                elif lg >= 12:
                    base_thr += 4

                for c in difflib.get_close_matches(g, self._alias_keys, n=4, cutoff=0.0):
                    score = int(100 * difflib.SequenceMatcher(a=g, b=c).ratio())
                    if score < base_thr:
                        continue
                    ctoks = set(t for t in _tokens(c) if t not in MED_BLACKLIST_TOKENS)
                    if not (gtoks_set & ctoks):
                        continue
                    if " " in g:
                        if _jaccard(gtoks_set, ctoks) < 0.34:
                            continue
                    else:
                        g0 = _strip_accents(g).split()[0]
                        c0 = _strip_accents(c).split()[0]
                        if len(g0) < 3 or len(c0) < 3 or not (g0[:3] == c0[:3]):
                            continue
                    feat = self._alias_key_to_feat.get(c)
                    if feat:
                        hits.add(feat)
        return hits

    def _has_neg_for_feature(self, fragment: str, feat_canon: str) -> bool:
        for ali in self._feat_to_aliases.get(_canonize(feat_canon), set()):
            if _has_negation(fragment, ali):
                return True
        return False

    # ───────────────────────── Meta-bramka ─────────────────────────

    def _looks_like_meta(self, text: str) -> bool:
        tx = _strip_accents(_canonize(text))
        has_med = any(tok in self._med_tokens for tok in tx.split() if tok not in MED_BLACKLIST_TOKENS)
        if has_med:
            return False
        return any(p.search(tx) for p in META_INTENT_PATTERNS)

    # ───────────────────────── RULES-ONLY ─────────────────────────

    def _extract_rules_only(self, raw: str) -> List[str]:
        if not (raw or "").strip():
            return []
        if self._looks_like_meta(raw):
            return []

        found: Set[str] = set()
        for frag in _split_phrases(raw):
            frag_toks = [t for t in _tokens(frag) if t not in MED_BLACKLIST_TOKENS]
            if not any(t in self._med_tokens or t in MED_WHITELIST for t in frag_toks):
                continue

            fn = _canonize(frag)
            fn_no = _strip_accents(fn)
            frag_hits: Set[str] = set()

            # 1) regex aliasów
            if self._alias_regex is not None:
                for m in self._alias_regex.finditer(fn + " " + fn_no):
                    alias = m.group(1).lower()
                    if len(alias) <= 4 and _has_diacritics(alias):
                        if re.search(rf"\b{re.escape(alias)}\b", fn, flags=re.IGNORECASE) is None:
                            continue
                    if _all_generic(alias) or alias in QUESTION_WORDS:
                        continue
                    feat = self.alias_to_feature.get(alias)
                    if feat and not _has_negation(frag, alias):
                        frag_hits.add(feat)

            # 2) contains
            if not frag_hits:
                for alias, feat in self.alias_to_feature.items():
                    if _all_generic(alias) or alias in QUESTION_WORDS:
                        continue
                    if len(alias) <= 4 and _has_diacritics(alias):
                        if re.search(rf"\b{re.escape(alias)}\b", fn, flags=re.IGNORECASE) is None:
                            continue
                        if not _has_negation(frag, alias):
                            frag_hits.add(feat)
                    else:
                        if alias in fn or alias in fn_no:
                            if not _has_negation(frag, alias):
                                frag_hits.add(feat)

            # 3) approx (dla dłuższych aliasów)
            if not frag_hits:
                for alias, feat in self.alias_to_feature.items():
                    if _all_generic(alias) or alias in QUESTION_WORDS or len(alias) < 6:
                        continue
                    if not any(t in self._med_tokens for t in _tokens(alias)):
                        continue
                    if _approx_in(alias, fn_no) and not _has_negation(frag, alias):
                        frag_hits.add(feat)

            # 4) fuzzy
            if not frag_hits:
                for feat_canon in self._fuzzy_on_fragment(fn_no):
                    if not self._has_neg_for_feature(frag, feat_canon):
                        frag_hits.add(feat_canon)

            found |= frag_hits

        return sorted(found)

    def extract_fast(self, text: str) -> List[str]:
        if not getattr(self, "ready", False):
            return []
        return self._extract_rules_only(text)

    def extract(self, text: str) -> List[str]:
        if not getattr(self, "ready", False):
            return []
        raw = (text or "").strip()
        if not raw:
            return []
        if self._looks_like_meta(raw):
            return []
        return self._extract_rules_only(raw)

    def expand_to_xgb(self, canon_or_alias: str) -> Set[str]:
        k = _canonize(canon_or_alias)
        canon = self.alias_to_feature.get(k, k)
        return set(self._feat2alts.get(canon, {canon}))

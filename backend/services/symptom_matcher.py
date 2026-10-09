"""Rule-based extraction of model symptom features (present and negated) from Polish/English text."""
from __future__ import annotations
import os
import re
import json
import unicodedata
import logging
from typing import List, Dict, Set, Optional, Tuple

log = logging.getLogger("symptomchecker.symptom_matcher")

from config import (
    SYMPTOMS_DICT_PATH, EMB_USE_DICTIONARY, EMB_RULE_FIRST, EMB_RULE_MIN_LEN,
    NEGATION_PATTERNS,
    MATCHER_MIN_TOKEN_LEN, MATCHER_NEG_WINDOW_CHARS,
)

MIN_TOKEN_LEN = MATCHER_MIN_TOKEN_LEN
NEG_WINDOW_CHARS = MATCHER_NEG_WINDOW_CHARS
FUZZY_MIN_SCORE_TOKEN = 80
FUZZY_MIN_SCORE_PHRASE = 78

GENERIC_SYMPTOM_WORDS: Set[str] = {
    "objaw","objawy","objawow","objawami","objawach","objawie",
    "symptom","symptomy","symptomow","symptomami","symptomach",
}

QUESTION_WORDS: Set[str] = {
    "jak","jaki","jaka","jakie","jacy","jakich","jakiego","jakiej","jakim","jakimi",
    "ktory","ktora","ktore","ktorzy","ktorych","ktorego","ktorej","ktorym","którą",
    "gdzie","kiedy","dlaczego","po","co","ile","kto","czy",
    "what","which","how","where","when","why","who"
}

# Stems of questions about the conversation itself (advice, urgency, summary, diagnosis),
# matched on accent-stripped text; such messages are not symptom reports.
META_INTENT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bzalecen"),
    re.compile(r"\bpilnos"),
    re.compile(r"\bleki\s*(bez\s*recepty|otc)?\b"),
    re.compile(r"\bprofilaktyk"),
    re.compile(r"\bpodsum"),
    re.compile(r"\bstres"),
    re.compile(r"\bnajbardziej\s+prawdopodobn"),
    re.compile(r"\bdiagnoz"),
    re.compile(r"\bocen\b"),
    re.compile(r"\bpowiedz\b"),
    re.compile(r"\bjaka?\s+chorob"),
]

# Non-medical tokens excluded from the lexical gate.
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
    # Polish l-with-stroke has no NFD decomposition, so map it manually (so "glowa" matches)
    text = text.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _has_diacritics(s: str) -> bool:
    return any(ch in "ąćęłńóśźż" for ch in s)


_STOP = {
    "mam","mnie","mi","nie","to","jest","są","sa","i","oraz","a","czy","że","ze","w","na","u","z","od","do","się","sie",
    "and","or","the","a","an","of","to","in","on","at","for","with","without","have","has","had"
} | GENERIC_SYMPTOM_WORDS | QUESTION_WORDS
MED_WHITELIST = {"bol"}  # "pain" after accent stripping; short but always medical


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


_CONJ_RE = re.compile(r"\s+(?:oraz|albo|lub|i|a także|a takze|plus|and|or)\s+")


def _split_pieces(txt: str) -> List[List[str]]:
    """
    Split text into pieces (boundaries: , ; . ! ? /) and each piece into sub-phrases joined
    by a conjunction (i, oraz, lub, ...). Keeping conjunction-joined phrases together is
    needed to carry negation: "nie mam goraczki i kaszlu" negates both.
    """
    raw = (txt or "").lower()
    pieces = re.split(r"[,;.!?/\n]+", raw)
    out: List[List[str]] = []
    for piece in pieces:
        c = _canonize(piece)
        if not c:
            continue
        subs = [x.strip() for x in _CONJ_RE.split(" " + c + " ") if x.strip()]
        if subs:
            out.append(subs)
    return out


# Lightweight Polish stemmer for inflection (goraczka/goraczki/goraczke, kaszel/kaszlu).
_STEM_SUFFIXES = sorted({
    "owalem", "owalam", "owalismy", "owaly", "owala", "owal", "owac",
    "ujesz", "ujemy", "ujac", "uje", "uja", "uj",
    "eniach", "eniem", "enia", "enie", "eniu",
    "ami", "ach", "ego", "emu", "ymi", "imi", "ych", "ich", "owi",
    "ow", "om", "em", "ej", "ym", "im", "ie", "ia", "iu", "io",
    "a", "e", "i", "y", "u", "o",
}, key=len, reverse=True)
_VOWELS = set("aeiouy")


def _stem(tok: str) -> str:
    t = _strip_accents((tok or "").lower())
    for suf in _STEM_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            break
    # Polish mobile "e": kaszel -> kaszl, so it matches kaszlu
    if len(t) >= 5 and t[-2] == "e" and t[-1] not in _VOWELS and t[-3] not in _VOWELS:
        t = t[:-2] + t[-1]
    return t


def _plain_tokens(txt: str) -> List[str]:
    return _strip_accents(_canonize(txt)).split()


# Negation handling
_NEG_PRE = {"nie", "bez", "brak", "ani", "zaprzeczam", "nigdy"}
_NEG_BREAKERS = {"wiem", "pewien", "pewna", "pamietam", "ale", "tylko", "czy", "tez", "rowniez"}
_NEG_POST_VERBS = {
    "mam", "ma", "mamy", "miewam", "wystepuje", "wystepuja", "wystepowal", "wystepowala",
    "wystepowalo", "odczuwam", "czuje", "stwierdzam", "zauwazylem", "zauwazylam",
    "pojawil", "pojawila", "pojawilo", "pojawia", "pojawily",
}
_RESOLVED_PREFIXES = (
    "minel", "minal", "ustapil", "ustapila", "ustapilo", "przeszl", "przeszed",
    "znikn", "znikl", "zeszl", "odpuscil",
)
_POSITIVE_MARKERS = {
    "mam", "mnie", "mi", "boli", "bola", "czuje", "odczuwam", "ale", "tylko",
    "natomiast", "jest", "sa", "dokucza", "dokuczaja", "pojawil", "pojawila", "pojawilo", "doszedl", "doszla",
}
# A whole piece like "dzis juz nie" or "teraz nie mam" negates the previous piece's symptoms.
_RESOLUTION_PIECE = re.compile(
    r"^(?:(?:a|ale|dzis|dzisiaj|teraz|obecnie|juz|to|on|ona|ono)\s+)*nie(?:\s+(?:mam|ma|juz|wystepuje))*$"
)


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
    def __init__(
        self,
        features: List[str],
        device: str = "cpu",
        feature_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self.features: List[str] = [str(f).strip().lower() for f in (features or [])]
        self._feat_set: Set[str] = set(self.features)
        # Training rows per feature (SymptomStats). When given:
        #  - features with 0 rows are skipped (the model knows nothing about them),
        #  - on alias collisions (one alias -> several features) the feature with more data wins.
        self._counts: Optional[Dict[str, int]] = (
            {str(k).strip().lower(): int(v) for k, v in feature_counts.items()} if feature_counts else None
        )

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
        self._short_accented: Set[str] = set()

        if EMB_USE_DICTIONARY and SYMPTOMS_DICT_PATH and os.path.exists(SYMPTOMS_DICT_PATH):
            ok = self._build_alias_index(SYMPTOMS_DICT_PATH)
            if not ok:
                self._build_default_alias_index_from_features()
        else:
            self._build_default_alias_index_from_features()

        self._build_stem_index()
        self.ready = True

    def _build_stem_index(self) -> None:
        """Alias index by stems, for matching independent of inflection and word order."""
        self._med_stems: Set[str] = {_stem(t) for t in self._med_tokens} | set(MED_WHITELIST)
        seen: Set[Tuple[str, Tuple[str, ...]]] = set()
        self._stem_index: List[Tuple[str, str, frozenset]] = []
        for alias, feat in self.alias_to_feature.items():
            if _all_generic(alias) or alias in QUESTION_WORDS:
                continue
            # negation words stay in the alias ("nie mam apetytu" != "mam apetyt")
            toks = [t for t in _plain_tokens(alias) if t not in _STOP or t in _NEG_PRE]
            if not toks:
                continue
            stems = tuple(sorted({_stem(t) for t in toks}))
            if len(stems) == 1 and len(stems[0]) < 4:
                continue  # too short a stem ("bol") would cause false hits
            key = (feat, stems)
            if key in seen:
                continue
            seen.add(key)
            self._stem_index.append((alias, feat, frozenset(stems)))

    def _count(self, f: str) -> int:
        return int(self._counts.get(f, 0)) if self._counts is not None else 0

    def is_alive(self, f: str) -> bool:
        """True if the feature has training data (every known feature when counts are unavailable)."""
        f = (f or "").strip().lower()
        if f not in self._feat_set:
            return False
        return self._counts is None or self._count(f) > 0

    def resolve_labels(self, labels: List[str]) -> Optional[str]:
        """Pick the model feature from dictionary labels: alive, with the most training data."""
        alive = [x for x in labels if self.is_alive(x)]
        if not alive:
            return None
        if self._counts is None:
            return alive[0]
        return max(alive, key=lambda x: (self._count(x), -alive.index(x)))

    def _set_alias(self, a: str, canon: str) -> bool:
        cur = self.alias_to_feature.get(a)
        if cur is not None and cur != canon and self._count(cur) > self._count(canon):
            return False
        if cur is not None and cur != canon:
            self._feat_to_aliases.get(cur, set()).discard(a)
        self.alias_to_feature[a] = canon
        self._feat_to_aliases.setdefault(canon, set()).add(a)
        return True

    def _build_default_alias_index_from_features(self) -> None:
        added = 0
        for canon in sorted(f for f in self._feat_set if self.is_alive(f)):
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

        log.info("DICT/FALLBACK features=%d | aliasów=%d | med_tokens=%d", len(self._feat_set), len(self.alias_to_feature), len(self._med_tokens))

    def _build_alias_index(self, dict_path: str) -> bool:
        try:
            data = json.load(open(dict_path, "r", encoding="utf-8"))
        except Exception as e:
            log.warning("DICT błąd wczytywania %s: %s", dict_path, e)
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

            canon = self.resolve_labels(xgb_list)
            if not canon:
                continue  # entry points only to unknown features or features without data

            display_pl = pl_label or canon.replace("_", " ")
            self._xgb2pl.setdefault(canon, display_pl)
            self._pl2xgb.setdefault(_canonize(display_pl), canon)
            self._feat_category.setdefault(canon, cat_raw)

            # With known counts only the chosen feature is activated (old, rare synonyms from
            # another data source would distort the vector). Without counts: all labels.
            alts = {canon} if self._counts is not None else ({x for x in xgb_list if self.is_alive(x)} or {canon})
            self._feat2alts.setdefault(canon, set()).update(alts)

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

                self._set_alias(a, canon)

                if add_noacc and self._set_alias(a_no, canon):
                    self._alias_keys.append(a_no)
                    self._alias_key_to_feat[a_no] = canon

                # Short one-word aliases with diacritics become another word when stripped
                # (anxiety vs. "lek" = drug, which is blacklisted), so only the accented form opens the gate
                if " " not in a and len(a) <= 4 and _has_diacritics(a):
                    self._short_accented.add(a)

                for tok in a.split():
                    tt = _strip_accents(tok)
                    if len(tt) >= 4 and tt not in _STOP and tt not in MED_BLACKLIST_TOKENS:
                        self._med_tokens.add(tt)

            added += 1

        # sync fuzzy keys with the final alias assignment (after collision resolution)
        self._alias_keys = sorted({k for k in self._alias_keys if k in self.alias_to_feature})
        self._alias_key_to_feat = {k: self.alias_to_feature[k] for k in self._alias_keys}

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

        log.info("DICT Wpisów:%d | aliasów:%d | keys:%d | cat:%d | med_tokens:%d", added, len(self.alias_to_feature), len(self._alias_keys), len(self._feat_category), len(self._med_tokens))
        return True

    def _canon_of(self, xgb_or_alias: str) -> str:
        """
        Model feature name for a feature or alias. Exact feature names are checked too, because
        _canonize turns "_" into a space, so e.g. "symptom_depression" (no "symptom depression"
        alias) would otherwise be dropped from the feature vector.
        """
        raw = str(xgb_or_alias or "").strip().lower()
        k = _canonize(raw)
        canon = self.alias_to_feature.get(k)
        if canon is not None:
            return canon            # alias points to a feature with data (e.g. abdominal_pain -> sharp_abdominal_pain)
        return raw if raw in self._feat_set else k

    # features without a Polish dictionary entry; otherwise English leaks into the summary
    _PL_FALLBACK = {"symptom_anxiety": "lęk", "symptom_indigestion": "niestrawność"}

    def to_pl(self, xgb_or_alias: str) -> str:
        canon = self._canon_of(xgb_or_alias)
        return self._xgb2pl.get(canon) or self._PL_FALLBACK.get(canon) or canon.replace("_", " ")

    def to_en(self, text_pl_or_alias: str) -> Optional[str]:
        k = _canonize(text_pl_or_alias)
        if k in self._pl2xgb:
            return self._pl2xgb[k]
        return self.alias_to_feature.get(_strip_accents(k))

    def get_category(self, xgb_or_alias: str) -> str:
        canon = self.alias_to_feature.get(_canonize(xgb_or_alias), _canonize(xgb_or_alias))
        return self._feat_category.get(canon, "Other")

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
                # stricter threshold for very short and very long grams
                lg = len(g)
                if lg <= 5:
                    base_thr += 8
                elif lg >= 12:
                    base_thr += 4

                for cand, score_set, _ in rf_process.extract(
                    g, self._alias_keys, scorer=rf_fuzz.token_set_ratio, limit=4
                ):
                    score_sort = rf_fuzz.token_sort_ratio(g, cand)
                    score_part = rf_fuzz.partial_ratio(g, cand)
                    score = max(score_set, score_sort)

                    if score < base_thr or score_part < (base_thr - 5):
                        continue

                    # require a shared non-generic token
                    ctoks = set(t for t in _tokens(cand) if t not in MED_BLACKLIST_TOKENS)
                    if not (gtoks_set & ctoks):
                        continue

                    # extra guard: Jaccard for phrases, shared 3-char prefix (accent-stripped) for single words
                    if " " in g:
                        if _jaccard(gtoks_set, ctoks) < 0.34:
                            continue
                    else:
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

    def _is_med_token(self, t: str) -> bool:
        return t in self._med_tokens or t in MED_WHITELIST or _stem(t) in self._med_stems

    def _looks_like_meta(self, text: str) -> bool:
        tx = _strip_accents(_canonize(text))
        has_med = any(self._is_med_token(tok) for tok in tx.split() if tok not in MED_BLACKLIST_TOKENS)
        if has_med:
            return False
        return any(p.search(tx) for p in META_INTENT_PATTERNS)

    # Match one phrase, returning {feature: alias}; strategies run from strictest to loosest.
    def _match_fragment(self, frag: str) -> Dict[str, str]:
        hits: Dict[str, str] = {}
        frag_toks = [t for t in _tokens(frag) if t not in MED_BLACKLIST_TOKENS]
        short_hit = bool(self._short_accented & set(_canonize(frag).split()))
        if not short_hit and not any(self._is_med_token(t) for t in frag_toks):
            return hits

        fn = _canonize(frag)
        fn_no = _strip_accents(fn)

        # 1) alias regex
        if self._alias_regex is not None:
            for m in self._alias_regex.finditer(fn + " " + fn_no):
                alias = m.group(1).lower()
                if len(alias) <= 4 and _has_diacritics(alias):
                    if re.search(rf"\b{re.escape(alias)}\b", fn, flags=re.IGNORECASE) is None:
                        continue
                if _all_generic(alias) or alias in QUESTION_WORDS:
                    continue
                feat = self.alias_to_feature.get(alias)
                if feat and feat not in hits:
                    hits[feat] = alias

        # 2) substring
        if not hits:
            for alias, feat in self.alias_to_feature.items():
                if _all_generic(alias) or alias in QUESTION_WORDS:
                    continue
                if len(alias) <= 4 and _has_diacritics(alias):
                    if re.search(rf"\b{re.escape(alias)}\b", fn, flags=re.IGNORECASE) is not None:
                        hits.setdefault(feat, alias)
                elif alias in fn or alias in fn_no:
                    hits.setdefault(feat, alias)

        # 3) stems, for inflection and word order ("boli mnie glowa", "39 stopni goraczki")
        frag_stems = {_stem(t) for t in _plain_tokens(frag)}
        stem_hits: List[Tuple[str, str, frozenset]] = [
            (alias, feat, stems) for alias, feat, stems in self._stem_index
            if feat not in hits and stems <= frag_stems
        ]
        for alias, feat, _stems in sorted(stem_hits, key=lambda x: -len(x[2])):
            hits.setdefault(feat, alias)

        # 4) approximate substring, longer aliases only
        if not hits:
            for alias, feat in self.alias_to_feature.items():
                if _all_generic(alias) or alias in QUESTION_WORDS or len(alias) < 6:
                    continue
                if not any(t in self._med_tokens for t in _tokens(alias)):
                    continue
                if _approx_in(alias, fn_no):
                    hits.setdefault(feat, alias)

        # 5) fuzzy, for typos
        if not hits:
            for feat_canon in self._fuzzy_on_fragment(fn_no):
                aliases = sorted(self._feat_to_aliases.get(feat_canon, {feat_canon}), key=len)
                hits.setdefault(feat_canon, aliases[0] if aliases else feat_canon)

        # a more specific alias wins ("bol w klatce piersiowej" over "bol")
        if len(hits) > 1:
            stems_of = {f: {_stem(t) for t in _plain_tokens(a) if t not in _STOP} for f, a in hits.items()}
            hits = {
                f: a for f, a in hits.items()
                if not any(o != f and stems_of[f] and stems_of[f] < stems_of[o] for o in hits)
            }
        return hits

    # Negation state of a hit: "pre", "post" or None.
    @staticmethod
    def _negation_state(frag: str, alias: str) -> Optional[str]:
        toks = _plain_tokens(frag)
        a_toks = _plain_tokens(alias)
        if any(t in _NEG_PRE for t in a_toks):
            # the alias itself contains a negation ("brak apetytu"): it is part of the symptom
            a_stems = {_stem(t) for t in a_toks}
        else:
            a_stems = {_stem(t) for t in a_toks if t not in _STOP} or {_stem(t) for t in a_toks}
        idxs = [i for i, t in enumerate(toks) if _stem(t) in a_stems]
        if not idxs:
            return "pre" if _has_negation(frag, alias) else None
        start, end = min(idxs), max(idxs)

        # before the symptom: "nie mam goraczki", "bez kaszlu", "nie boli mnie glowa"
        for i in range(max(0, start - 3), start):
            if toks[i] in _NEG_PRE:
                between = toks[i + 1:start]
                if not any(b in _NEG_BREAKERS for b in between):
                    return "pre"

        # after the symptom: "goraczki nie mam", "kaszel? nie", "goraczka (juz) minela".
        # Note: "nie ustepuje" / "nie daje spac" mean the symptom persists.
        post = toks[end + 1:end + 4]
        if post:
            p = [t for t in post if t not in {"juz", "mi", "mnie", "to"}]
            if p and p[0] == "nie":
                nxt = p[1] if len(p) > 1 else None
                if nxt is None or nxt in _NEG_POST_VERBS:
                    return "post"
            if p and p[0].startswith(_RESOLVED_PREFIXES):
                return "post"
        return None

    def _extract_detailed(self, raw: str) -> Tuple[List[str], List[str]]:
        """Return (confirmed_symptoms, negated_symptoms)."""
        if not (raw or "").strip():
            return [], []
        if self._looks_like_meta(raw):
            return [], []

        pos: Set[str] = set()
        neg: Set[str] = set()
        prev_piece_pos: Set[str] = set()

        for piece in _split_pieces(raw):
            piece_norm = _strip_accents(" ".join(piece))
            if _RESOLUTION_PIECE.match(piece_norm):
                # "wczoraj mialem goraczke, dzis juz nie"
                neg |= prev_piece_pos
                pos -= prev_piece_pos
                prev_piece_pos = set()
                continue

            piece_pos: Set[str] = set()
            carry_neg = False
            for sub_idx, frag in enumerate(piece):
                hits = self._match_fragment(frag)
                toks = set(_plain_tokens(frag))
                has_positive = bool(toks & _POSITIVE_MARKERS)
                if not hits:
                    carry_neg = carry_neg and not has_positive
                    continue
                pre_neg_here = False
                for feat, alias in hits.items():
                    state = self._negation_state(frag, alias)
                    if state:
                        neg.add(feat)
                        pre_neg_here = pre_neg_here or state == "pre"
                    elif carry_neg and sub_idx > 0 and not has_positive:
                        # "nie mam goraczki i kaszlu": cough is negated too
                        neg.add(feat)
                    else:
                        piece_pos.add(feat)
                carry_neg = pre_neg_here or (carry_neg and not has_positive)

            pos |= piece_pos
            prev_piece_pos = piece_pos

        pos -= neg
        return sorted(pos), sorted(neg)

    def extract_detailed(self, text: str) -> Tuple[List[str], List[str]]:
        if not getattr(self, "ready", False):
            return [], []
        return self._extract_detailed(text)

    def _extract_rules_only(self, raw: str) -> List[str]:
        return self._extract_detailed(raw)[0]

    def extract_fast(self, text: str) -> List[str]:
        if not getattr(self, "ready", False):
            return []
        return self._extract_detailed(text)[0]

    def extract(self, text: str) -> List[str]:
        if not getattr(self, "ready", False):
            return []
        return self._extract_detailed(text)[0]

    def expand_to_xgb(self, canon_or_alias: str) -> Set[str]:
        canon = self._canon_of(canon_or_alias)
        return set(self._feat2alts.get(canon, {canon}))

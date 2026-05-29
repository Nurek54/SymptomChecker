# services/prompt_helpers.py
from __future__ import annotations

import re
from typing import List, Dict, Optional, Tuple

try:
    from config import PROMPT_SYMPTOMS_MAX_FOR_LLM as _CFG_MAX_SYM
    PROMPT_SYMPTOMS_MAX_FOR_LLM = int(_CFG_MAX_SYM)
except Exception:
    PROMPT_SYMPTOMS_MAX_FOR_LLM = 6

_HISTORY_TURNS_MAX = 6

_ADVICE_PAT = re.compile(r"\b(zaleceni[ae]|co (robic|robić)|jak (postepowac|postępować)|co mog[eę] zrobic)\b", re.I)
_TRIAGE_PAT = re.compile(r"\b(piln(?:e|a|y|osc|ość)|oce[nń]\s+piln|kiedy do lekarza|sor|pogotowie|112)\b", re.I)
_OTC_PAT    = re.compile(r"\b(lek[iy]?\s*otc|bez recepty|apteczn(?:e|a|y)|spray|aerozol|antyhistamin|mukolity|przeciwgor[aą]czk|przeciwb(?:o|ó)lw)\b", re.I)
_PROPH_PAT  = re.compile(r"\b(profilaktyk|zapobieg|jak (nie )?pogorszyc|jak unikac|jak się nie rozchorowac)\b", re.I)
_DIAG_PAT   = re.compile(r"\b(jaka?\s+chorob|co to (moze|może) byc|co najprawdopodobniej|najbardziej prawdopodobn[ae]\s+chorob)\b", re.I)
_SUMMARY_PAT = re.compile(r"\b(podsumuj|stre[sś]c|streszczenie)\b", re.I)


def _get_last_user_message(hist: List[Dict[str, str]]) -> str:
    for m in reversed(hist or []):
        if (m.get("role") or "").lower() == "user":
            return (m.get("content") or "").strip()
    return ""


def _classify_intent_pl(last_user_msg: str) -> str:
    t = (last_user_msg or "").lower()
    if _OTC_PAT.search(t):     return "otc"
    if _TRIAGE_PAT.search(t):  return "triage"
    if _PROPH_PAT.search(t):   return "prophylaxis"
    if _ADVICE_PAT.search(t):  return "advice"
    if _DIAG_PAT.search(t):    return "diagnosis"
    if _SUMMARY_PAT.search(t): return "summary"
    return "autosummary"

_RESP_SYM = re.compile(r"(kaszel|tachypno|duszno|świst|furcz|rzężen|oddech|plwocin|krwioplu)", re.I)
_PSY_SYM  = re.compile(r"(lęk|niepokój|panik|hiperwentyl)", re.I)
_RESP_DX  = re.compile(r"(płu|oskrzel|astma|krup|covid|gryp|pochp|odma|śródmiąższ|zapalenie płuc|zapalenie|angina|zapalenie zatok)", re.I)
_PSY_DX   = re.compile(r"(lęk|panik|hiperwentyl|zaburze(?:nie|nia)\s+lękow)", re.I)

_RED_FLAGS = re.compile(
    r"(krwioplu|krwaw.*plwocin|sinic|ból\s*klatki|silna\s*duszno|duszność\s*spoczynk|tachypnoe|"
    r"utrata\s*przytom|omdlen|splątanie|nagł[ay]\s*(niedowład|osłabienie|drętwienie)|"
    r"zaburzenia\s*mowy|sztywność\s*karku|wysok[aie]\s*gor[aą]czk[aie]|"
    r"wybroczyn|krwotok|silny\s*ból\s*brzucha|czarny\s*stolec|"
    r"brak\s*oddawania\s*moczu|ciężkie\s*odwodnienie|nagły\s*ból\s*głowy\s*najgorszy)",
    re.I,
)

_TEMP_VAL = re.compile(r"(\d{2}(?:[.,]\d)?)\s*(?:°|st|stopni)?\s*c", re.I)
_DUR_DAYS = re.compile(r"(?:od|ponad|>?)\s*(\d{1,2})\s*(?:dni|dnie|dzie[nń])", re.I)


def _symptom_clusters(symptoms: List[str]) -> Dict[str, bool]:
    t = " ".join(symptoms or [])
    return {
        "resp": bool(_RESP_SYM.search(t)),
        "psy":  bool(_PSY_SYM.search(t)),
        "red":  bool(_RED_FLAGS.search(t)),
    }


def _dx_domain(name: str) -> str:
    if _RESP_DX.search(name): return "resp"
    if _PSY_DX.search(name):  return "psy"
    return "other"

def _parse_top3(diag_summary: Optional[str]) -> List[Tuple[str, Optional[float]]]:
    out: List[Tuple[str, Optional[float]]] = []
    if not diag_summary:
        return out
    parts = [p.strip() for p in diag_summary.split(",") if p.strip()]
    for p in parts:
        m = re.match(r"(?P<name>.+?)\s*\((?P<pct>\d+(?:\.\d+)?)%\)\s*$", p)
        if m:
            out.append((m.group("name").strip(), float(m.group("pct")) / 100.0))
        else:
            out.append((p, None))
    return out


def _estimate_fit(diag_summary: Optional[str], symptoms: List[str]) -> Dict[str, str]:
    clusters = _symptom_clusters(symptoms)
    results: Dict[str, str] = {}
    for name, pct in _parse_top3(diag_summary):
        score = 0.0
        if pct is not None:
            if pct >= 0.75: score += 2.0
            elif pct >= 0.4: score += 1.0
            elif pct >= 0.2: score += 0.5
            else: score -= 0.5

        dom = _dx_domain(name)
        if dom == "resp":
            score += 1.0 if clusters["resp"] else -0.5
        elif dom == "psy":
            if clusters["psy"]:
                score += 1.0
            elif clusters["resp"]:
                score -= 0.5
        if clusters["red"] and dom == "resp":
            score += 0.5

        if score >= 2.0: label = "wysokie"
        elif score <= 0.0: label = "niskie"
        else: label = "średnie"
        results[name] = label
    return results

def _extract_temp_high(text: str) -> bool:
    """Wysoka gorączka ~≥39°C w tekście."""
    for m in _TEMP_VAL.finditer(text.lower()):
        try:
            v = float(m.group(1).replace(",", "."))
            if v >= 39.0:
                return True
        except Exception:
            pass
    return False

def _extract_duration_days(text: str) -> Optional[int]:
    m = _DUR_DAYS.search(text.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _age_risk_score(age: Optional[int]) -> float:
    if age is None: return 0.0
    if age < 1: return 2.0
    if age < 3: return 1.5
    if age < 12: return 0.5
    if age < 18: return 0.25
    if age < 65: return 0.0
    if age < 75: return 1.0
    return 1.5


def _suggest_triage(
    symptoms: List[str],
    patient_age: Optional[int],
    history_text: str = "",
) -> Tuple[str, str]:
    cl = _symptom_clusters(symptoms)
    text = " ".join(symptoms) + " " + history_text
    high_fever = _extract_temp_high(text)
    days = _extract_duration_days(text)
    age_score = _age_risk_score(patient_age)

    score = 0.0
    if cl["red"]: score += 3.0
    if high_fever: score += 1.5
    if cl["resp"]: score += 1.0
    if days is not None:
        if days >= 7: score += 1.0
        elif days >= 3: score += 0.5
    score += age_score

    if score >= 3.0:
        return "wysoka", "pilna ocena teraz — SOR/112 lub natychmiastowy kontakt z lekarzem"
    if score >= 1.5:
        return "średnia", "pilny kontakt z lekarzem dzisiaj lub w ciągu 24 h"
    return "niska", "samoobserwacja i konsultacja planowa w 48–72 h"

def _contains(words: List[str], text: str) -> bool:
    return any(w in text for w in words)


def _otc_hints(symptoms: List[str], patient_age: Optional[int]) -> str:
    t = " ".join(s.lower() for s in symptoms or [])
    parts: List[str] = []

    dry = _contains(["such", "drażni"], t) and not _contains(["plwocin", "mok", "produktyw", "odkrzt"], t)
    wet = _contains(["mok", "produktyw", "plwocin", "odkrzt"], t)

    if "kaszel" in t or dry or wet:
        if dry: parts.append("w kaszlu suchym sprawdzi się dekstrometorfan lub lewodropropizyna")
        if wet: parts.append("w kaszlu mokrym pomocna bywa acetylocysteina lub ambroksol")
        if not dry and not wet:
            parts.append("dobierz środek do rodzaju kaszlu: przeciwkaszlowy (suchy) lub mukolityk (mokry)")
    if _contains(["nos", "katar", "zatok"], t):
        parts.append("na niedrożność nosa rozważ krótkotrwałe użycie ksylometazoliny lub oksymetazoliny")
    if _contains(["gorącz", "temperatur", "dreszcz", "ból"], t):
        if patient_age is not None and patient_age < 12:
            parts.append("na ból i gorączkę preferuj paracetamol; unikaj ASA poniżej 12 lat")
        else:
            parts.append("na ból i gorączkę zastosuj paracetamol lub ibuprofen")
    if _contains(["gardł", "chrypk"], t):
        parts.append("na ból gardła pomocna bywa benzydamina w pastylkach lub roztworze")
    if _contains(["alerg", "kich", "świąd", "wodnist"], t):
        parts.append("na objawy alergiczne sprawdzi się cetyryzyna lub loratadyna")
    if _contains(["duszno", "tachypno"], t):
        parts.append("przy duszności lub tachypnoe rola OTC jest ograniczona i wskazana jest pilna ocena")
    if _contains(["biegun", "wymiot", "nudno"], t):
        parts.append("przy ryzyku odwodnienia użyj doustnych płynów nawadniających typu ORS")

    if not parts:
        return ""
    return "; ".join(parts) + "; stosuj zgodnie z ulotką"

_PRIOR_TRIAGE_PAT = re.compile(r"pilno(?:ść|sć)\s*(?:jest|:)?\s*(niska|średnia|srednia|wysoka)", re.I)
_PRIOR_TOP1_PAT   = re.compile(r"(?:diagnoza nr 1|najbardziej prawdopodobna|top-1)\s*[:\-]?\s*([^\s,.;:]+)", re.I)


def _extract_prior_summary(hist: List[Dict[str, str]]) -> str:
    prior_triage: Optional[str] = None
    prior_top1: Optional[str] = None
    for m in reversed(hist or []):
        txt = (m.get("content") or "")
        if prior_triage is None:
            mt = _PRIOR_TRIAGE_PAT.search(txt)
            if mt:
                val = mt.group(1).lower().replace("srednia", "średnia")
                if val in {"niska", "średnia", "wysoka"}:
                    prior_triage = val
        if prior_top1 is None:
            md = _PRIOR_TOP1_PAT.search(txt)
            if md:
                prior_top1 = md.group(1).strip()
        if prior_triage and prior_top1:
            break

    if not (prior_triage or prior_top1):
        return ""

    bits = []
    if prior_top1:   bits.append(f"wcześniej: diagnoza nr 1 = {prior_top1}")
    if prior_triage: bits.append(f"wcześniejsza pilność = {prior_triage}")
    return "; ".join(bits) + "."


def _top1_already_named(hist: List[Dict[str, str]], top1: Optional[str]) -> bool:
    if not top1:
        return False
    pat = re.compile(rf"\b{re.escape(top1)}\b", re.I)
    for m in hist or []:
        if (m.get("role") or "").lower() == "assistant" and pat.search(m.get("content") or ""):
            return True
    return False


def _style_variant_index(hist: List[Dict[str, str]]) -> int:
    base = sum(len((m.get("content") or "")) for m in hist or [])
    return base % 3


def _render_history_chatml(hist: List[Dict[str, str]]) -> str:
    if not hist:
        return ""
    tail = hist[-_HISTORY_TURNS_MAX:] if len(hist) > _HISTORY_TURNS_MAX else hist
    parts: List[str] = []
    for m in tail:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role in ("user", "human"):
            parts.append("<|im_start|>user\n" + content + "\n<|im_end|>\n")
        elif role in ("assistant", "bot"):
            parts.append("<|im_start|>assistant\n" + content + "\n<|im_end|>\n")
    return "".join(parts)

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items or []:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _trim_symptoms(symptoms: List[str]) -> List[str]:
    if not symptoms:
        return []
    return _dedup_keep_order(symptoms)[: max(1, int(PROMPT_SYMPTOMS_MAX_FOR_LLM))]

def _render_context(
    sym_for_llm: List[str],
    diag_summary: Optional[str],
    patient_age: Optional[int],
) -> str:
    parts: List[str] = []
    parts.append(
        "Wykryte objawy: " + (", ".join(sym_for_llm) + "." if sym_for_llm else "(brak potwierdzonych).")
    )
    parts.append(
        "Wyniki klasyfikatora (Top-3, malejąco): "
        + (diag_summary + "." if diag_summary else "(brak dostępnych wyników).")
    )
    if patient_age is not None:
        parts.append(f"Wiek pacjenta: {patient_age} lat.")
    return " ".join(parts)

def _extract_top1_name(diag_summary: Optional[str]) -> Optional[str]:
    if not diag_summary:
        return None
    head = diag_summary.split(",")[0].strip()
    name = head.split("(")[0].strip()
    return name or None

# prompt_builder.py — wersja z ulepszonym triage/dopasowaniem,
# tylko akapity (bez wypunktowań) i bez dawek leków (same nazwy substancji)

from __future__ import annotations

import re
from typing import List, Dict, Optional, Union, Tuple

# ——— konfiguracja pomocnicza ———
try:
    from config import PROMPT_SYMPTOMS_MAX_FOR_LLM as _CFG_MAX_SYM
    PROMPT_SYMPTOMS_MAX_FOR_LLM = int(_CFG_MAX_SYM)
except Exception:
    PROMPT_SYMPTOMS_MAX_FOR_LLM = 6  # krótki, celowy kontekst dla LLM

# Ile ostatnich wiadomości dołączamy do historii (żeby „pamiętał trochę”)
_HISTORY_TURNS_MAX = 6

# ========== ZASADY GLOBALNE (SYSTEM) ==========
SYSTEM_PROMPT_CORE = (
    "Jesteś empatycznym, rzeczowym asystentem do WSTĘPNEJ oceny objawów. "
    "Korzystaj WYŁĄCZNIE z «Kontekst». Nie dopisuj objawów ani diagnoz i nie zmieniaj ich nazw. "
    "Wymień tylko elementy obecne w kontekście. "
    "Język: polski; styl: kliniczny i naturalny; pełne zdania."
)

# ========== STRAŻNICY SPÓJNOŚCI ==========
CONSISTENCY_GUARDS = (
    "Trzymaj się «Top-3» (nie zmieniaj kolejności). "
    "Cytuj nazwy diagnoz dokładnie jak w «Top-3». "
    "Gdy nie podano procentów — nie podawaj liczb."
)

# ========== POLITYKI FORMATU ==========
FORMAT_POLICY = (
    "Format odpowiedzi: dokładnie JEDEN LUB DWA zwarte akapity; bez nagłówków, bez wypunktowań, bez numerowania, "
    "bez pustych linii i bez znaków nowej linii; elementy oddzielaj średnikami lub kropkami; "
    "nie używaj symboli typu '-', '•', '1.' ani etykiet list; "
    "nie parafrazuj nazewnictwa objawów/diagnoz; nie wstawiaj markdown."
)

FORMAT_POLICY_RELAXED = (
    "Format odpowiedzi: 2–3 krótkie akapity pisane ciągłym tekstem. "
    "Kategorycznie NIE używaj wypunktowań ani numerowanych list; "
    "nie poprzedzaj nazw leków numerami ani myślnikami; "
    "nie stosuj symboli '-', '•', '1.', '2)' itp. jako znaczników list; "
    "nie używaj markdownowych nagłówków ani złożonych struktur (tabel, sekcji); "
    "nazwy objawów i diagnoz cytuj dokładnie tak jak w kontekście."
)

# ========== POLITYKI LEKOWE / WIEK ==========
DRUG_POLICY = (
    "OTC: wolno podawać wyłącznie NAZWY SUBSTANCJI czynnych, bez żadnych liczb w ich opisie. "
    "Bezwzględnie zakazane są: dawki liczbowe (mg, g, ml, %, jednostki), liczby poprzedzające lek, "
    "formuły przypominające dawkowanie («x dziennie», «co 8 godzin», «co 6 h», «2 tabletki», «jedna dawka»), "
    "oraz nazwy marek handlowych. "
    "Zawsze wplataj nazwy substancji w normalne zdania opisowe, w których tłumaczysz, w jakiej sytuacji dana substancja może być pomocna. "
    "Nie twórz list, nie twórz skrótowych wyliczeń, nie używaj opisów w stylu katalogu. "
    "Odpowiedź o lekach powinna być rozwinięta, pełna i zawierać krótkie uzasadnienie, dlaczego dana substancja może pomagać "
    "(np. działanie przeciwbólowe, przeciwgorączkowe, przeciwzapalne, mukolityczne, przeciwhistaminowe). "
    "Po każdym wspomnieniu substancji dopisz frazę «stosuj zgodnie z ulotką». "
    "Substancje przykładowe: paracetamol, ibuprofen, dekstrometorfan, lewodropropizyna, acetylocysteina, ambroksol, "
    "ksylometazolina, oksymetazolina, cetyryzyna, loratadyna, benzydamina, ORS, diklofenak w żelu."
)

AGE_POLICY = (
    "Jeśli podano wiek: < 12 lat — bez kwasu acetylosalicylowego; 12–17 — nadal bez ASA; "
    "≥ 65 — większe ryzyko działań niepożądanych i interakcji, co powinno zwiększać ostrożność."
)

# ========== TRIAGE — definicje i styl odpowiedzi ==========
TRIAGE_POLICY_DETAIL = (
    "Ustal pilność na trzystopniowej skali i osadź ją w czasie: "
    "niska → samoobserwacja i ewentualna planowa konsultacja w 48–72 h; "
    "średnia → kontakt z lekarzem POZ dzisiaj lub w 24 h; "
    "wysoka → pilna ocena teraz (SOR/112), zwłaszcza gdy obecne czerwone flagi. "
    "Przykłady czerwonych flag (nie wylistowuj, wpleć naturalnie): silny, narastający ból w klatce piersiowej; nagła duszność spoczynkowa; krwioplucie; utrata przytomności lub splątanie; objawy ogniskowe neurologiczne; sztywność karku z wysoką gorączką; wysypka wybroczynowa; ciężkie odwodnienie lub brak oddawania moczu."
)

# ========== TRYBY (INTENCJE) ==========
TASK_DIAGNOSIS_EXPLICIT = (
    "Napisz 4–6 zdań w jednym akapicie: nazwij diagnozę nr 1 (z % jeśli jest), oceń jej dopasowanie do objawów jako wysokie/średnie/niskie, "
    "uzasadnij odwołując się do 1–2 objawów z «Wykryte objawy» oraz wskaż co ewentualnie nie pasuje; "
    "zbiorczo odnieś się do pozostałych pozycji z Top-3 (z oceną dopasowania); "
    "na koniec jedno zdanie o pilności z osadzeniem w czasie zgodnie z polityką triage."
)

TASK_DIAGNOSIS_IMPLICIT = (
    "Napisz 4–6 zdań w jednym akapicie: wyjaśnij dopasowanie diagnozy nr 1 do objawów (nazwij 1–2), "
    "zbiorczo skomentuj pozostałe opcje z Top-3 oraz zakończ zdaniem o pilności z ramą czasową (48–72 h / 24 h / natychmiast)."
)

TASK_ADVICE = (
    "Napisz 5–9 zdań w 2–3 akapitach. "
    "Podaj zalecenia wynikające z diagnozy nr 1 i objawów. "
    "Opisuj także działania niefarmakologiczne w formie ciągłych, pełnych zdań. "
    "Nazwy substancji OTC podawaj tylko w formie opisowej, bez marek, bez liczb, bez dawek i bez jakichkolwiek schematów dawkowania. "
    "W każdym przypadku dopisz frazę «stosuj zgodnie z ulotką». "
    "Wyjaśnij krótko, dlaczego dana substancja może pomagać (np. działanie przeciwbólowe, przeciwgorączkowe, przeciwzapalne, mukolityczne). "
    "Nie twórz żadnych list ani wyliczeń — tekst ma być opisowy. "
    "Uwzględnij wiek pacjenta, jeśli jest podany. "
    "Zakończ jasną informacją o pilności wraz z ramą czasową."
)

TASK_TRIAGE = (
    "Napisz 3–5 zdań w jednym akapicie: określ pilność (niska/średnia/wysoka) i powiąż ją z ramą czasową (48–72 h / 24 h / natychmiast), "
    "wspomnij 1–2 potencjalne czerwone flagi istotne dla tego obrazu, ale bez list; "
    "jeśli sytuacja jest graniczna, wskaż co powinno skłonić do szybszej oceny."
)

TASK_OTC = (
    "Napisz 4–7 zdań w dwóch akapitach. "
    "Wymień 2–4 nazwy substancji czynnych, które mogą łagodzić objawy, ale opisuj je naturalnie, w zdaniach. "
    "Nie podawaj żadnych dawek liczbowych, jednostek, częstotliwości ani ilości. "
    "Nie używaj list ani numerowania — nazwy substancji muszą być wplecione w opis. "
    "Przy każdej substancji dodaj frazę «stosuj zgodnie z ulotką». "
    "Uwzględnij wiek pacjenta. "
    "Zakończ krótką informacją o pilności."
)

TASK_PROPHYLAXIS = (
    "Napisz 5–8 zdań w 1–3 akapitach. "
    "Skup się na działaniach profilaktycznych, które zmniejszają ryzyko nawrotu takiego obrazu objawów w przyszłości "
    "oraz ograniczają ciężkość kolejnych epizodów (np. szczepienia, modyfikacja stylu życia, higiena, unikanie wyzwalaczy, kontrola chorób przewlekłych). "
    "Nie powtarzaj szczegółowych zaleceń leczenia ostrych objawów — jeśli o nich wspomnisz, zrób to maksymalnie w jednym, ogólnym zdaniu. "
    "Nie wypisuj działań w formie listy, tylko wpleć je w spójne zdania. "
    "Unikaj pustych ogólników typu «zdrowy tryb życia» bez doprecyzowania; wiąż każde zalecenie z kontekstem objawów lub diagnozy. "
    "Na końcu krótko przypomnij o pilności i samomonitorowaniu, ale bez rozbudowanego opisu ostrych stanów."
)

TASK_SUMMARY_PARAGRAPH = (
    "Napisz 4–6 zdań w jednym akapicie: streść zgłoszone objawy, wskaż diagnozę nr 1 (z % jeśli jest) i jej dopasowanie, "
    "wspomnij pozostałe opcje z Top-3, oceń krótko pilność i zaproponuj następny krok wraz z horyzontem czasowym."
)

# ========== KLASYFIKACJA INTENCJI ==========
_ADVICE_PAT = re.compile(r"\b(zaleceni[ae]|co (robic|robić)|jak (postepowac|postępować)|co mog[eę] zrobic)\b", re.I)
_TRIAGE_PAT = re.compile(r"\b(piln(?:e|a|y|osc|ość)|oce[nń]\s+piln|kiedy do lekarza|sor|pogotowie|112)\b", re.I)
_OTC_PAT    = re.compile(r"\b(lek[iy]?\s*otc|bez recepty|apteczn(?:e|a|y)|spray|aerozol|antyhistamin|mukolity|przeciwgor[aą]czk|przeciwb(?:o|ó)lw)\b", re.I)
_PROPH_PAT  = re.compile(r"\b(profilaktyk|zapobieg|jak (nie )?pogorszyc|jak unikac|jak się nie rozchorowac)\b", re.I)
_DIAG_PAT   = re.compile(r"\b(jaka?\s+chorob|co to (moze|może) byc|co najprawdopodobniej|najbardziej prawdopodobn[ae]\s+chorob)\b", re.I)
_SUMMARY_PAT= re.compile(r"\b(podsumuj|stre[sś]c|streszczenie)\b", re.I)

def _get_last_user_message(hist: List[Dict[str, str]]) -> str:
    for m in reversed(hist or []):
        if (m.get("role") or "").lower() == "user":
            return (m.get("content") or "").strip()
    return ""

def _classify_intent_pl(last_user_msg: str) -> str:
    t = (last_user_msg or "").lower()
    if _OTC_PAT.search(t):      return "otc"
    if _TRIAGE_PAT.search(t):   return "triage"
    if _PROPH_PAT.search(t):    return "prophylaxis"
    if _ADVICE_PAT.search(t):   return "advice"
    if _DIAG_PAT.search(t):     return "diagnosis"
    if _SUMMARY_PAT.search(t):  return "summary"
    return "autosummary"

# ========== HEURYSTYKI KLINICZNE (rozszerzone) ==========
# Objawy oddechowe / psychogenne
_RESP_SYM = re.compile(r"(kaszel|tachypno|duszno|świst|furcz|rzężen|oddech|plwocin|krwioplu)", re.I)
_PSY_SYM  = re.compile(r"(lęk|niepokój|panik|hiperwentyl)", re.I)

# Domeny diagnoz
_RESP_DX = re.compile(r"(płu|oskrzel|astma|krup|covid|gryp|pochp|odma|śródmiąższ|zapalenie płuc|zapalenie|angina|zapalenie zatok)", re.I)
_PSY_DX  = re.compile(r"(lęk|panik|hiperwentyl|zaburze(?:nie|nia)\s+lękow)", re.I)

# Czerwone flagi — rozbudowane
_RED_FLAGS = re.compile(
    r"(krwioplu|krwaw.*plwocin|sinic|ból\s*klatki|silna\s*duszno|duszność\s*spoczynk|tachypnoe|"
    r"utrata\s*przytom|omdlen|splątanie|nagł[ay]\s*(niedowład|osłabienie|drętwienie)|"
    r"zaburzenia\s*mowy|sztywność\s*karku|wysok[aie]\s*gor[aą]czk[aie]|"
    r"wybroczyn|krwotok|silny\s*ból\s*brzucha|czarny\s*stolec|"
    r"brak\s*oddawania\s*moczu|ciężkie\s*odwodnienie|nagły\s*ból\s*głowy\s*najgorszy)",
    re.I,
)

# Dodatkowe heurystyki: gorączka liczbowo, długość objawów
_TEMP_VAL = re.compile(r"(\d{2}(?:[.,]\d)?)\s*(?:°|st|stopni)?\s*c", re.I)
_DUR_DAYS = re.compile(r"(?:od|ponad|>?)\s*(\d{1,2})\s*(?:dni|dnie|dzie[nń])", re.I)

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

def _estimate_fit(diag_summary: Optional[str], symptoms: List[str]) -> Dict[str, str]:
    """
    Heurystyczna ocena dopasowania każdej diagnozy z Top-3:
    łączy info z procentów klasyfikatora i zbieżność domeny (resp/psy) z objawami.
    Zwraca mapę: nazwa_diag → 'niskie'/'średnie'/'wysokie'.
    """
    clusters = _symptom_clusters(symptoms)
    results: Dict[str, str] = {}
    for name, pct in _parse_top3(diag_summary):
        score = 0.0

        # 1) wkład z prawdopodobieństwa modelu
        if pct is not None:
            if pct >= 0.75:
                score += 2.0
            elif pct >= 0.4:
                score += 1.0
            elif pct >= 0.2:
                score += 0.5
            else:
                score -= 0.5

        dom = _dx_domain(name)

        # 2) zbieżność domeny diagnozy z obrazem objawów
        if dom == "resp":
            if clusters["resp"]:
                score += 1.0
            else:
                score -= 0.5
        elif dom == "psy":
            if clusters["psy"]:
                score += 1.0
            elif clusters["resp"] and not clusters["psy"]:
                score -= 0.5

        # 3) czerwone flagi wzmacniają dopasowanie diag. oddechowych
        if clusters["red"] and dom == "resp":
            score += 0.5

        # 4) mapowanie na etykietę
        if score >= 2.0:
            label = "wysokie"
        elif score <= 0.0:
            label = "niskie"
        else:
            label = "średnie"

        results[name] = label

    return results

def _extract_temp_high(text: str) -> bool:
    """Wysoka gorączka ~≥39°C rozpoznana w tekście."""
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
    """
    Heurystyczny wkład ryzyka z wieku:
    - <1 r.ż. bardzo wysokie,
    - 1–2 lata wysokie,
    - 3–11 umiarkowane,
    - 12–17 niewielkie,
    - 18–64 bazowe,
    - 65–74 podwyższone,
    - ≥75 wysokie.
    """
    if age is None:
        return 0.0
    if age < 1:
        return 2.0
    if age < 3:
        return 1.5
    if age < 12:
        return 0.5
    if age < 18:
        return 0.25
    if age < 65:
        return 0.0
    if age < 75:
        return 1.0
    return 1.5

def _suggest_triage(symptoms: List[str], patient_age: Optional[int], history_text: str = "") -> Tuple[str, str]:
    """
    Zwraca (poziom: 'niska'|'średnia'|'wysoka', okno_czasowe_str).
    Heurystyka:
      - budujemy wynik punktowy z czerwonych flag, gorączki, długości objawów, wieku i domeny oddechowej;
      - wysoka pilność przy sumarycznie dużym wyniku (np. czerwone flagi + wiek),
        średnia przy umiarkowanym wyniku, w pozostałych przypadkach niska.
    """
    cl = _symptom_clusters(symptoms)
    text = " ".join(symptoms) + " " + history_text
    high_fever = _extract_temp_high(text)
    days = _extract_duration_days(text)
    age_score = _age_risk_score(patient_age)

    score = 0.0

    # czerwone flagi dominują
    if cl["red"]:
        score += 3.0

    # wysoka gorączka
    if high_fever:
        score += 1.5

    # objawy oddechowe zwiększają pilność
    if cl["resp"]:
        score += 1.0

    # czas trwania
    if days is not None:
        if days >= 7:
            score += 1.0
        elif days >= 3:
            score += 0.5

    # wiek
    score += age_score

    # mapowanie wyniku na poziom
    if score >= 3.0:
        level = "wysoka"
        window = "pilna ocena teraz — SOR/112 lub natychmiastowy kontakt z lekarzem"
    elif score >= 1.5:
        level = "średnia"
        window = "pilny kontakt z lekarzem dzisiaj lub w ciągu 24 h"
    else:
        level = "niska"
        window = "samoobserwacja i konsultacja planowa w 48–72 h"

    return level, window

# ========== OTC HINTS (PL, substancje — bez marek/dawkowania) ==========
def _contains(words: List[str], text: str) -> bool:
    return any(w in text for w in words)

def _otc_hints(symptoms: List[str], patient_age: Optional[int]) -> str:
    """
    Generuje krótkie, wewnętrzne podpowiedzi OTC:
    wyłącznie nazwy substancji, bez dawek, schematów, ilości czy częstotliwości.
    """
    t = " ".join(s.lower() for s in symptoms or [])
    parts: List[str] = []

    dry = _contains(["such", "drażni"], t) and not _contains(["plwocin", "mok", "produktyw", "odkrzt"], t)
    wet = _contains(["mok", "produktyw", "plwocin", "odkrzt"], t)

    if "kaszel" in t or dry or wet:
        if dry:
            parts.append("w kaszlu suchym sprawdzi się dekstrometorfan lub lewodropropizyna")
        if wet:
            parts.append("w kaszlu mokrym pomocna bywa acetylocysteina lub ambroksol")
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

# ========== PAMIĘĆ: skrót poprzednich ocen ==========
_PRIOR_TRIAGE_PAT = re.compile(r"pilno(?:ść|sć)\s*(?:jest|:)?\s*(niska|średnia|srednia|wysoka)", re.I)
_PRIOR_TOP1_PAT   = re.compile(r"(?:diagnoza nr 1|najbardziej prawdopodobna|top-1)\s*[:\-]?\s*([^\s,.;:]+)", re.I)

def _extract_prior_summary(hist: List[Dict[str, str]]) -> str:
    """
    Bardzo krótki skrót z historii: poprzednia diagnoza nr 1 i pilność, jeśli uda się wydobyć.
    """
    prior_triage = None
    prior_top1 = None
    for m in reversed(hist or []):
        txt = (m.get("content") or "")
        if prior_triage is None:
            mt = _PRIOR_TRIAGE_PAT.search(txt)
            if mt:
                val = mt.group(1).lower().replace("srednia", "średnia")
                if val in {"niska","średnia","wysoka"}:
                    prior_triage = val
        if prior_top1 is None:
            md = _PRIOR_TOP1_PAT.search(txt)
            if md:
                prior_top1 = md.group(1).strip()
        if prior_triage and prior_top1:
            break

    if prior_triage or prior_top1:
        bits = []
        if prior_top1:  bits.append(f"wcześniej: diagnoza nr 1 = {prior_top1}")
        if prior_triage: bits.append(f"wcześniejsza pilność = {prior_triage}")
        return "; ".join(bits) + "."
    return ""

# ========== POMOCNICZE ==========
Msg = Union[Dict[str, str], tuple]

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items or []:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def _trim_symptoms(symptoms: List[str]) -> List[str]:
    return _dedup_keep_order(symptoms)[: max(1, int(PROMPT_SYMPTOMS_MAX_FOR_LLM))] if symptoms else []

def _render_context(sym_for_llm: List[str], diag_summary: Optional[str], patient_age: Optional[int]) -> str:
    parts = []
    parts.append("Wykryte objawy: " + (", ".join(sym_for_llm) + "." if sym_for_llm else "(brak potwierdzonych)."))
    parts.append(
        "Wyniki klasyfikatora (Top-3, malejąco): " + (diag_summary + "." if diag_summary else "(brak dostępnych wyników).")
    )
    if patient_age is not None:
        parts.append(f"Wiek pacjenta: {patient_age} lat.")
    return " ".join(parts)

def _extract_top1_name(diag_summary: Optional[str]) -> Optional[str]:
    if not diag_summary: return None
    head = diag_summary.split(",")[0].strip()
    name = head.split("(")[0].strip()
    return name or None

def _top1_already_named(hist: List[Dict[str, str]], top1: Optional[str]) -> bool:
    if not top1: return False
    pat = re.compile(rf"\b{re.escape(top1)}\b", re.I)
    for m in hist or []:
        if (m.get("role") or "").lower() == "assistant" and pat.search(m.get("content") or ""):
            return True
    return False

def _style_variant_index(hist: List[Dict[str, str]]) -> int:
    base = sum(len((m.get("content") or "")) for m in hist or [])
    return base % 3

def _render_history_chatml(hist: List[Dict[str, str]]) -> str:
    """
    Dołączamy tylko ostatnie _HISTORY_TURNS_MAX wiadomości (lekka pamięć),
    aby nie przepełniać kontekstu, ale zachować ciągłość.
    """
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

FALLBACK_POLICY = (
    "Jeżeli w «Kontekst» brakuje diagnoz lub objawów, używaj warunkowych sformułowań; nie dopisuj faktów i liczb."
)

# ========== GŁÓWNY BUILDER (CHATML) ==========
def build_prompt(
    history: List[Dict[str, str]],
    detected_symptoms: List[str],
    need_more: bool,  # nieużywane, zachowane dla kompatybilności
    *,
    diag_summary: Optional[str] = None,
    patient_age: Optional[int] = None,
) -> str:
    """
    Zwraca prompt w formacie ChatML, zakończony otwartym blokiem asystenta:
    <s><|im_start|>system ... <|im_end|>\n ... <|im_start|>assistant\n
    Model generuje do <|im_end|> (config: LLM_STOP_SEQUENCES = ['<|im_end|>']).
    """
    # — przygotowanie kontekstu —
    sym_for_llm = _trim_symptoms(detected_symptoms)
    context = _render_context(sym_for_llm, diag_summary, patient_age)
    last_user = _get_last_user_message(history)
    intent = _classify_intent_pl(last_user)

    # wybór polityki formatu (luźniejsza dla zaleceń/OTC/profilaktyki)
    if intent in {"advice", "otc", "prophylaxis"}:
        format_policy = FORMAT_POLICY_RELAXED
    else:
        format_policy = FORMAT_POLICY

    top1_name = _extract_top1_name(diag_summary)
    top1_named_before = _top1_already_named(history, top1_name)
    _ = _style_variant_index(history)  # ewentualnie do przyszłych wariantów stylu

    # dopasowanie i pilność (heurystycznie, bez zmiany numeracji)
    fit_map = _estimate_fit(diag_summary, sym_for_llm)
    fit_text = ""
    if diag_summary:
        pairs = [f"{name}: dopasowanie {fit_map.get(name, 'średnie')}" for name, _pct in _parse_top3(diag_summary)]
        fit_text = "; ".join(pairs) + "."

    # skrót historii (ostatnie ustalenia)
    prior_summary = _extract_prior_summary(history)

    # triage podpowiedź + okno czasowe (lepsze dostrojenie)
    triage_level, triage_window = _suggest_triage(
        sym_for_llm,
        patient_age,
        history_text=" ".join((m.get("content") or "") for m in history),
    )

    # OTC-hints dla intencji praktycznych (bez profilaktyki)
    otc_text = ""
    if intent in {"advice", "otc"}:
        otc_text = _otc_hints(sym_for_llm, patient_age)

    # wybór zadania
    if intent == "diagnosis":
        task = TASK_DIAGNOSIS_IMPLICIT if top1_named_before else TASK_DIAGNOSIS_EXPLICIT
    elif intent == "advice":
        task = TASK_ADVICE
    elif intent == "triage":
        task = TASK_TRIAGE
    elif intent == "otc":
        task = TASK_OTC
    elif intent == "prophylaxis":
        task = TASK_PROPHYLAXIS
    elif intent == "summary":
        task = TASK_SUMMARY_PARAGRAPH
    else:
        task = TASK_SUMMARY_PARAGRAPH

    flags = (
        f"Nazwa diagnozy nr 1: {'brak' if not top1_name else top1_name}; "
        f"wcześniej wymieniona: {'tak' if top1_named_before else 'nie'}; "
        f"sugerowana pilność (heurystyka): {triage_level}; "
        f"zalecane okno czasowe: {triage_window}."
    )

    # — polityki i kontekst (system) —
    sys_policy = (
        "### Zasady globalne (System)\n" + SYSTEM_PROMPT_CORE + "\n\n" +
        "### Spójność\n" + CONSISTENCY_GUARDS + "\n\n" +
        "### Format\n" + format_policy + "\n\n" +
        "### Leki/Wiek\n" + DRUG_POLICY + "\n" + AGE_POLICY + "\n\n" +
        "### Triage — zasady\n" + TRIAGE_POLICY_DETAIL + "\n\n" +
        "### Braki danych\n" + FALLBACK_POLICY
    )

    if intent in {"advice", "otc", "prophylaxis"}:
        format_hint = (
            "Odpowiedź ułóż w 1–3 krótkich akapitach; nie używaj wypunktowań ani numerowanych list, "
            "nie stosuj symboli '-', '•', '1.' jako znaczników list; "
            "podając leki OTC używaj wyłącznie nazw substancji, bez dawek i schematów dawkowania, zawsze z frazą «stosuj zgodnie z ulotką»."
        )
    else:
        format_hint = (
            "Najlepiej zachowaj jeden zwarty akapit bez nagłówków, wypunktowań i rozbudowanych list."
        )

    sys_context = (
        "### Flagi\n" + flags + "\n" +
        ("### Skrót historii\n" + prior_summary + "\n" if prior_summary else "") +
        "### Kontekst\n" + context + "\n" +
        (("### Ocena dopasowania (wewnętrzna)\n" + fit_text + "\n") if fit_text else "") +
        (("### Wskazówki OTC (wewnętrzne)\n" + otc_text + "\n") if otc_text else "") +
        "### Instrukcja\n" + task + "\n" +
        "Uwaga: wpleć ocenę pilności jako naturalne zdanie z ramą czasową (48–72 h / 24 h / natychmiast); "
        "jeśli dopasowanie diagnozy nr 1 jest 'niskie', powiedz to wprost (bez zmiany kolejności Top-3) i dostosuj pilność do obrazu i wieku; "
        + format_hint
    )

    # — historia rozmowy (ChatML) —
    hist_chatml = _render_history_chatml(history)

    # — finalny prompt —
    parts: List[str] = []
    parts.append("<s><|im_start|>system\n" + sys_policy + "\n<|im_end|>\n")
    parts.append("<|im_start|>system\n" + sys_context + "\n<|im_end|>\n")
    if hist_chatml:
        parts.append(hist_chatml)
    else:
        seed = last_user or "Odpowiedz zgodnie z instrukcją i kontekstem powyżej."
        parts.append("<|im_start|>user\n" + seed + "\n<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)

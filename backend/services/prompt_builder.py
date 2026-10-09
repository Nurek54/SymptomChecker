"""Builds ChatML prompts (Polish) for the Bielik LLM from symptoms, top-3 diagnoses and chat history."""
from __future__ import annotations
from typing import List, Dict, Optional, Union

from services.prompt_helpers import (
    PROMPT_SYMPTOMS_MAX_FOR_LLM,  # noqa: F401
    _classify_intent_pl,
    _get_last_user_message,
    _suggest_triage,
    _estimate_fit,
    _parse_top3,
    _otc_hints,
    _extract_prior_summary,
    _top1_already_named,
    _style_variant_index,
    _render_history_chatml,
    _trim_symptoms,
    _render_context,
    _extract_top1_name,
    _dedup_keep_order,
    _symptom_clusters,
    _dx_domain,
    _extract_temp_high,
    _extract_duration_days,
    _age_risk_score,
    _contains,
    _user_text,
)

SYSTEM_PROMPT = (
    "Jesteś polskim asystentem do wstępnej oceny objawów. "
    "Mów do pacjenta jak doświadczony lekarz w trakcie krótkiej rozmowy: "
    "rzeczowo, ale ciepło i naturalnie, pełnymi zdaniami. "
    "Wykorzystuj tylko informacje z sekcji Kontekst; niczego nie zmyślaj i nie dopisuj objawów. "
    "Możesz parafrazować nazwy objawów i diagnoz tak, żeby zdanie brzmiało po polsku naturalnie "
    "(np. zapalenie gardła zamiast pharyngitis); zachowaj jednak ich sens. "
    "Pisz wyłącznie ciągłą prozą w 1–3 akapitach. "
    "Nie używaj długich myślników (pauz); zamiast nich postaw przecinek, dwukropek albo kropkę. "
    "ABSOLUTNIE ZAKAZANE są listy w jakiejkolwiek formie: "
    "nie zaczynaj zdania od liczby z kropką (1., 2., 3.) ani od myślnika czy kropki, "
    "nie używaj słów-zwrotów rozpoczynających listę typu 'Proponuję następujące kroki:', "
    "'Oto co możesz zrobić:', 'Po pierwsze... Po drugie...', "
    "nie używaj nagłówków, markdownu, tabel ani znaczników listy. "
    "Każdą sugestię formułuj jako pełne, naturalne zdanie wplecione w akapit. "
    "WAŻNE: jeśli w Top-3 jest wyraźny lider z wysokim procentem (powyżej 50%), "
    "musi on być NAJBARDZIEJ PRAWDOPODOBNĄ przyczyną w Twojej odpowiedzi. "
    "Nie odwracaj hierarchii: pozycje 2 i 3 to ALTERNATYWY o niższym prawdopodobieństwie. "
    "Liczby procentowe podawaj w przybliżeniu (około 70%, znaczna większość prawdopodobieństwa), "
    "nie kopiuj surowych ułamków. "
    "Mów po polsku poprawnie gramatycznie; unikaj kalek angielskich i sztywnych formułek. "
    "ODPOWIADAJ ŚCIŚLE NA TO, O CO PACJENT PYTA; nie wykraczaj poza zakres pytania. "
    "Jeśli pacjent pyta o diagnozę, podaj diagnozę i krótkie uzasadnienie. NIE dopisuj "
    "porad o lekach OTC, NIE opisuj zaleceń domowych, NIE mów o pilności (chyba że pacjent "
    "wprost o nią pyta). Każdy z tych tematów jest osobnym pytaniem, na które pacjent może "
    "Cię zapytać oddzielnie. "
    "Naturalna długość odpowiedzi to 3–6 zdań, w 1–2 akapitach. "
    "Zawsze kończ pełnym zdaniem zakończonym kropką."
)

DRUG_POLICY = (
    "Gdy mówisz o lekach OTC, podawaj tylko nazwy substancji czynnych (paracetamol, ibuprofen, "
    "dekstrometorfan, loratadyna, cetyryzyna, ksylometazolina, ambroksol, acetylocysteina, "
    "benzydamina, ORS, lewodropropizyna), wplecione w normalne zdania z krótkim wyjaśnieniem "
    "działania. Nie podawaj dawek, jednostek, częstotliwości ani schematów dawkowania. "
    "Nie podawaj nazw marek. Po wymienieniu substancji raz dopisz frazę: stosuj zgodnie z ulotką. "
    "U dzieci poniżej 12 lat nie sugeruj kwasu acetylosalicylowego."
)

TRIAGE_GUIDELINE = (
    "Gdy mówisz o pilności, łącz ją z ramą czasową. Niska pilność oznacza samoobserwację "
    "i ewentualnie wizytę w 48–72 h, średnia kontakt z lekarzem dzisiaj lub w 24 h, "
    "a wysoka pilną ocenę teraz (SOR/112), zwłaszcza przy czerwonych flagach. "
    "Wpleć to w naturalne zdanie, nie ogłaszaj jak komunikat."
)

TASK_DIAGNOSIS = (
    "Wyjaśnij JEDYNIE jaka diagnoza najlepiej tłumaczy zgłoszone objawy. Powiedz to ludzkim "
    "językiem: nazwij chorobę po polsku, jeśli to możliwe, i krótko uzasadnij, które "
    "objawy ją wskazują. Możesz pokrótce wspomnieć inne pozycje z Top-3 jako alternatywy. "
    "NIE wymieniaj leków, NIE podawaj zaleceń domowych (odpoczynek, nawodnienie itp.), "
    "NIE opisuj pilności; to są osobne pytania, na które pacjent może zapytać oddzielnie. "
    "3–5 zdań w jednym akapicie."
)

TASK_DIAGNOSIS_FOLLOWUP = (
    "Pacjent dopytuje o tę samą diagnozę, którą wcześniej wymieniłeś. Skup się na "
    "wyjaśnieniu, dlaczego pasuje, jakie objawy ją potwierdzają, a co ewentualnie "
    "do niej nie pasuje. NIE opisuj pilności ani leków. 4–6 zdań w jednym akapicie."
)

# Intents whose answer should mention urgency (the other TASK_* prompts forbid it).
TRIAGE_INTENTS = {"triage", "summary", "autosummary"}

TASK_ADVICE = (
    "Powiedz pacjentowi konkretnie, co może zrobić w domu: działania niefarmakologiczne "
    "(odpoczynek, nawodnienie, pozycja, środowisko) opisz pełnymi zdaniami, bez list. "
    "NIE wymieniaj leków (od tego jest osobne pytanie), NIE powtarzaj diagnozy, "
    "NIE opisuj pilności. Skup się na praktycznych krokach pacjenta w domu. "
    "4–6 zdań w 1–2 akapitach."
)

TASK_TRIAGE = (
    "Oceń JEDYNIE pilność tej sytuacji. Powiedz, czy to niska, średnia czy wysoka pilność, "
    "i osadź to w czasie (48–72 h / dzisiaj / natychmiast). Wspomnij naturalnie o "
    "1–2 sytuacjach, które powinny skłonić pacjenta do szybszej oceny. "
    "NIE powtarzaj diagnozy, NIE wymieniaj leków, NIE rozpisuj zaleceń domowych. "
    "3–5 zdań, bez list."
)

TASK_OTC = (
    "Pacjent pyta o leki bez recepty. Wymień 2–4 nazwy substancji czynnych wplecione "
    "w zdania, z krótkim wyjaśnieniem działania każdej. Bez dawek, bez marek, bez "
    "częstotliwości. Po każdej dopisz: stosuj zgodnie z ulotką. Uwzględnij wiek. "
    "NIE powtarzaj diagnozy, NIE opisuj zaleceń niefarmakologicznych, NIE rozpisuj pilności. "
    "4–6 zdań w 1–2 akapitach."
)

TASK_PROPHYLAXIS = (
    "Skup się na profilaktyce: co zmniejszy ryzyko nawrotu lub powikłań tego "
    "obrazu objawów (np. szczepienia, higiena, modyfikacja stylu życia, kontrola "
    "chorób przewlekłych, unikanie wyzwalaczy). Konkretnie, w kontekście "
    "zgłoszonych objawów, bez ogólników w stylu zdrowy tryb życia. "
    "Bez list, ciągłym tekstem. 5–8 zdań."
)

TASK_SUMMARY = (
    "Streść w naturalnym tonie: zgłoszone objawy, najprawdopodobniejsza przyczyna "
    "i jej dopasowanie, pozostałe rozważane opcje, pilność i sugerowany następny krok. "
    "4–6 zdań w jednym akapicie."
)

FALLBACK_POLICY = (
    "Jeśli w Kontekście brakuje informacji, mów ostrożnie (np. wstępnie wygląda to na..., "
    "najprawdopodobniej...) zamiast podawać twarde fakty."
)

Msg = Union[Dict[str, str], tuple]


def build_prompt(
    history: List[Dict[str, str]],
    detected_symptoms: List[str],
    need_more: bool,  # unused, kept for API compatibility
    *,
    diag_summary: Optional[str] = None,
    patient_age: Optional[int] = None,
) -> str:
    sym_for_llm = _trim_symptoms(detected_symptoms)
    context = _render_context(sym_for_llm, diag_summary, patient_age)
    last_user = _get_last_user_message(history)
    intent = _classify_intent_pl(last_user)

    top1_name = _extract_top1_name(diag_summary)
    top1_named_before = _top1_already_named(history, top1_name)

    fit_map = _estimate_fit(diag_summary, sym_for_llm)
    fit_text = ""
    if diag_summary and fit_map:
        pairs = [
            f"{name}: dopasowanie {fit_map.get(name, 'srednie')}"
            for name, _pct in _parse_top3(diag_summary)
        ]
        if pairs:
            fit_text = "; ".join(pairs) + "."

    prior_summary = _extract_prior_summary(history)

    triage_level, triage_window = _suggest_triage(
        sym_for_llm,
        patient_age,
        history_text=_user_text(history),
    )

    otc_text = _otc_hints(sym_for_llm, patient_age) if intent in {"advice", "otc"} else ""

    task = {
        "diagnosis": (TASK_DIAGNOSIS_FOLLOWUP if top1_named_before else TASK_DIAGNOSIS),
        "advice": TASK_ADVICE,
        "triage": TASK_TRIAGE,
        "otc": TASK_OTC,
        "prophylaxis": TASK_PROPHYLAXIS,
        "summary": TASK_SUMMARY,
    }.get(intent, TASK_SUMMARY)

    hints_lines: List[str] = []
    if intent in TRIAGE_INTENTS:
        hints_lines.append(f"Sugerowana pilnosc: {triage_level} ({triage_window}).")
    if fit_text:
        hints_lines.append(f"Wewnetrzna ocena dopasowania: {fit_text}")
    if otc_text:
        hints_lines.append(f"Mozesz nawiazac do: {otc_text}")
    if prior_summary:
        hints_lines.append(f"Z poprzedniej rozmowy: {prior_summary}")
    if top1_named_before and top1_name:
        hints_lines.append(
            f"Pacjent zna juz diagnoze {top1_name}; nie powtarzaj jej jak nowej."
        )
    hints_text = "\n".join(hints_lines) or "Brak dodatkowych wskazowek."

    sys_block = (
        SYSTEM_PROMPT + "\n\n"
        + DRUG_POLICY + "\n\n"
        + TRIAGE_GUIDELINE + "\n\n"
        + FALLBACK_POLICY
    )

    ctx_block = (
        "### Kontekst rozmowy\n" + context + "\n\n"
        "### Zadanie\n" + task + "\n\n"
        "### Wewnetrzne wskazowki (NIE pokazuj ich pacjentowi, NIE cytuj slowo w slowo)\n"
        + hints_text
    )

    hist_chatml = _render_history_chatml(history)

    parts: List[str] = []
    parts.append("<s><|im_start|>system\n" + sys_block + "\n<|im_end|>\n")
    parts.append("<|im_start|>system\n" + ctx_block + "\n<|im_end|>\n")
    if hist_chatml:
        parts.append(hist_chatml)
    else:
        seed = last_user or "Odpowiedz pacjentowi w naturalnym tonie zgodnie z zadaniem."
        parts.append("<|im_start|>user\n" + seed + "\n<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_prompt_strict(
    history: List[Dict[str, str]],
    detected_symptoms: List[str],
    *,
    diag_summary: Optional[str] = None,
    patient_age: Optional[int] = None,
    leader_pl: Optional[str] = None,
    leader_pct: Optional[int] = None,
    rejection_reason: str = "",
) -> str:
    sym_for_llm = _trim_symptoms(detected_symptoms)
    context = _render_context(sym_for_llm, diag_summary, patient_age)
    last_user = _get_last_user_message(history)
    intent = _classify_intent_pl(last_user)

    task = {
        "diagnosis": TASK_DIAGNOSIS,
        "advice": TASK_ADVICE,
        "triage": TASK_TRIAGE,
        "otc": TASK_OTC,
        "prophylaxis": TASK_PROPHYLAXIS,
        "summary": TASK_SUMMARY,
    }.get(intent, TASK_SUMMARY)

    triage_level, triage_window = _suggest_triage(
        sym_for_llm,
        patient_age,
        history_text=_user_text(history),
    )

    leader_clause = ""
    if leader_pl:
        pct_str = f" (około {leader_pct}%)" if leader_pct else ""
        leader_clause = (
            f"OBOWIĄZKOWE OTWARCIE: pierwsze zdanie odpowiedzi MUSI nazwać "
            f"jako najbardziej prawdopodobną przyczynę: {leader_pl}{pct_str}. "
            f"NIE wolno Ci powiedzieć, że to inna diagnoza jest najbardziej prawdopodobna. "
            f"Pozostałe pozycje z Top-3 wymień tylko jako ALTERNATYWY o mniejszym prawdopodobieństwie. "
        )

    rejection_clause = ""
    if rejection_reason:
        rejection_clause = (
            f"Twoja poprzednia odpowiedź została odrzucona z powodu: {rejection_reason}. "
            f"Tym razem zrób to inaczej. "
        )

    strict_system = (
        "Jesteś polskim asystentem do wstępnej oceny objawów. "
        "Piszesz krótką wypowiedź w naturalnej polszczyźnie, w 1–3 akapitach ciągłym tekstem. "
        "ABSOLUTNIE NIE wolno Ci tworzyć list, używać numeracji (1., 2., 3.), myślników "
        "ani zwrotów typu 'proponuję następujące kroki', 'oto co możesz zrobić'. "
        "Każdą myśl formułujesz jako pełne zdanie. "
        "Kończ swoją wypowiedź pełną kropką i pełnym zdaniem; nie urywaj w środku myśli. "
        + rejection_clause
        + leader_clause
        + DRUG_POLICY + " "
        + TRIAGE_GUIDELINE
    )

    ctx_block = (
        "### Kontekst rozmowy\n" + context + "\n\n"
        "### Zadanie\n" + task
    )
    if intent in TRIAGE_INTENTS:
        ctx_block += (
            "\n\n"
            f"### Wewnetrzna wskazowka\nSugerowana pilnosc: {triage_level} ({triage_window}). "
            "Wpleć ją jako naturalne zdanie."
        )

    hist_chatml = _render_history_chatml(history)

    parts: List[str] = []
    parts.append("<s><|im_start|>system\n" + strict_system + "\n<|im_end|>\n")
    parts.append("<|im_start|>system\n" + ctx_block + "\n<|im_end|>\n")
    if hist_chatml:
        parts.append(hist_chatml)
    else:
        seed = last_user or "Odpowiedz pacjentowi w naturalnym tonie zgodnie z zadaniem."
        parts.append("<|im_start|>user\n" + seed + "\n<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


__all__ = [
    "build_prompt",
    "build_prompt_strict",
    "SYSTEM_PROMPT", "DRUG_POLICY", "TRIAGE_GUIDELINE",
    "TASK_DIAGNOSIS", "TASK_DIAGNOSIS_FOLLOWUP",
    "TASK_ADVICE", "TASK_TRIAGE", "TASK_OTC",
    "TASK_PROPHYLAXIS", "TASK_SUMMARY", "FALLBACK_POLICY",
    "_classify_intent_pl", "_get_last_user_message",
    "_suggest_triage", "_estimate_fit", "_parse_top3", "_otc_hints",
    "_extract_prior_summary", "_top1_already_named", "_style_variant_index",
    "_render_history_chatml", "_trim_symptoms", "_render_context",
    "_extract_top1_name", "_dedup_keep_order",
    "_symptom_clusters", "_dx_domain",
    "_extract_temp_high", "_extract_duration_days",
    "_age_risk_score", "_contains", "_user_text", "TRIAGE_INTENTS",
    "PROMPT_SYMPTOMS_MAX_FOR_LLM",
]
import type { SymptomDictItem } from "@/lib/types";

/**
 * Jeden wpis na cechę modelu (`en` jest kluczem w listach React).
 * Backend już grupuje wpisy po cesze; tu dodatkowo zabezpieczamy się przed duplikatami
 * (np. starsza wersja API): kolejne etykiety tej samej cechy trafiają do `aliases`.
 */
export function normalizeSymptomDict(items: SymptomDictItem[] | null | undefined): SymptomDictItem[] {
  const byEn = new Map<string, SymptomDictItem>();
  for (const raw of items ?? []) {
    const en = (raw?.en ?? "").trim().toLowerCase();
    if (!en) continue;
    const pl = (raw.pl ?? "").trim();
    const aliases = (raw.aliases ?? []).map((a) => (a ?? "").trim()).filter(Boolean);
    const prev = byEn.get(en);
    if (!prev) {
      byEn.set(en, { ...raw, en, pl, aliases });
      continue;
    }
    const seen = new Set([prev.pl.toLowerCase(), ...(prev.aliases ?? []).map((a) => a.toLowerCase())]);
    for (const a of [pl, ...aliases]) {
      if (a && !seen.has(a.toLowerCase())) {
        seen.add(a.toLowerCase());
        (prev.aliases ??= []).push(a);
      }
    }
  }
  return Array.from(byEn.values());
}

/**
 * Czy wpis pasuje do zapytania. Zwraca null (brak dopasowania), "" (pasuje etykieta główna)
 * albo alias, po którym znaleziono wpis (do pokazania obok etykiety).
 */
export function matchSymptom(it: SymptomDictItem, query: string, lang: "pl" | "en" = "pl"): string | null {
  const q = query.trim().toLowerCase();
  const label = (lang === "pl" ? it.pl || it.en : it.en).toLowerCase();
  if (!q || label.includes(q)) return "";
  if (lang === "pl") {
    const alias = (it.aliases ?? []).find((a) => a.toLowerCase().includes(q));
    if (alias) return alias;
  }
  return null;
}

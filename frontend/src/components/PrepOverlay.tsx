"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { listSymptomsDict } from "@/lib/api";
import { Calendar, Clock, Flame, Check, Clipboard, X } from "lucide-react";

/** Element słownika (z xgb_label do kanonizacji) */
type SymItem = {
  en: string;
  pl?: string;
  category?: string;
  xgb_label?: string;
};

type Props = {
  open: boolean;
  onClose: () => void;
};

const LS_KEY = "sc_prep_state_split_overlay_v1";
const MAX_SYMPTOMS = 12;
const MAX_NOTES = 300;

/* ───────── Utils ───────── */

function toXgbSlug(free: string): string {
  const basic = free
    .normalize("NFD")
    // @ts-ignore – unicode class
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "_")
    .replace(/_+/g, "_");
  return basic || "symptom";
}

function pluralPL(n: number, one: string, few: string, many: string): string {
  const mod10 = n % 10, mod100 = n % 100;
  if (n === 1) return one;
  if (mod10 >= 2 && mod10 <= 4 && !(mod100 >= 12 && mod100 <= 14)) return few;
  return many;
}

function unitLabel(unit: "hours" | "days" | "weeks", value: number | ""): string {
  const v = typeof value === "number" ? value : 0;
  if (unit === "hours") return pluralPL(v, "godzinę", "godziny", "godzin");
  if (unit === "days") return pluralPL(v, "dzień", "dni", "dni");
  return pluralPL(v, "tydzień", "tygodnie", "tygodni");
}

function formatLocalDateTime(localISO: string): string {
  if (!localISO) return "";
  try { return localISO.replace("T", " ").slice(0, 16); } catch { return localISO; }
}

function joinPolish(arr: string[]): string {
  if (arr.length === 0) return "";
  if (arr.length === 1) return arr[0];
  if (arr.length === 2) return `${arr[0]} i ${arr[1]}`;
  return `${arr.slice(0, -1).join(", ")} i ${arr[arr.length - 1]}`;
}

/* ───────── Komponent ───────── */

export default function PrepOverlay({ open, onClose }: Props) {
  // Izolowane od czatu: brak store’ów, tylko lokalny stan + localStorage.
  const [dict, setDict] = useState<SymItem[]>([]);
  const [loadingDict, setLoadingDict] = useState(true);

  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [severity, setSeverity] = useState<number>(5);
  const [durationValue, setDurationValue] = useState<number | "">("");
  const [durationUnit, setDurationUnit] = useState<"hours" | "days" | "weeks">("days");
  const [onset, setOnset] = useState<string>("");
  const [notes, setNotes] = useState<string>("");

  const [justSaved, setJustSaved] = useState(false);
  const savedTimer = useRef<number | null>(null);

  // zamykanie ESC
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // słownik (z AbortController)
  useEffect(() => {
    if (!open) return; // ładuj tylko gdy overlay otwarty
    const ac = new AbortController();
    (async () => {
      try {
        const arr = await (listSymptomsDict as any)?.({ signal: ac.signal }) ?? await listSymptomsDict?.();
        if (Array.isArray(arr)) {
          const normalized: SymItem[] = arr.map((it: any) => ({
            en: it.en,
            pl: it.pl ?? undefined,
            category: it.category ?? undefined,
            xgb_label: it.xgb_label ?? it.en,
          }));
          setDict(normalized);
        }
      } catch {
        // fallback: free text → slug
      } finally {
        if (!ac.signal.aborted) setLoadingDict(false);
      }
    })();
    return () => ac.abort();
  }, [open]);

  // odczyt stanu
  useEffect(() => {
    if (!open) return;
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      setSelected(s.selected ?? []);
      setSeverity(s.severity ?? 5);
      setDurationValue(s.durationValue ?? "");
      setDurationUnit(s.durationUnit ?? "days");
      setOnset(s.onset ?? "");
      setNotes(s.notes ?? "");
    } catch {}
  }, [open]);

  // zapis stanu
  useEffect(() => {
    if (!open) return;
    try {
      localStorage.setItem(
        LS_KEY,
        JSON.stringify({ selected, severity, durationValue, durationUnit, onset, notes })
      );
      setJustSaved(true);
      if (savedTimer.current) window.clearTimeout(savedTimer.current);
      savedTimer.current = window.setTimeout(() => setJustSaved(false), 1000);
    } catch {}
  }, [open, selected, severity, durationValue, durationUnit, onset, notes]);

  const byXgb = useMemo(() => {
    const m = new Map<string, SymItem>();
    for (const it of dict) m.set(it.xgb_label ?? it.en, it);
    return m;
  }, [dict]);

  function humanFor(xgb: string): string {
    const it = byXgb.get(xgb);
    return it?.pl || it?.en || xgb;
  }

  /* — Lista filtrowana (zamiast „sugestii”) — */
  const filtered = useMemo(() => {
    const s = query.trim().toLowerCase();
    if (!s) return dict.slice(0, 200); // pokaż pierwsze 200 przy pustym wyszukiwaniu
    return dict.filter((it) =>
      (it.pl ?? "").toLowerCase().includes(s) || (it.en ?? "").toLowerCase().includes(s)
    ).slice(0, 200);
  }, [query, dict]);

  function addByItem(it: SymItem) {
    const xgb = (it.xgb_label || it.en).trim();
    setSelected((prev) => {
      if (prev.includes(xgb)) return prev;
      if (prev.length >= MAX_SYMPTOMS) { toast(`Maksymalnie ${MAX_SYMPTOMS} objawów`); return prev; }
      return [...prev, xgb];
    });
    setQuery("");
  }

  function addSymptom(freeText: string) {
    const v = freeText.trim();
    if (!v) return;
    const hit =
      dict.find((it) => (it.pl ?? "").toLowerCase() === v.toLowerCase()) ??
      dict.find((it) => (it.en ?? "").toLowerCase() === v.toLowerCase());
    if (hit) return addByItem(hit);
    setSelected((prev) => {
      if (prev.length >= MAX_SYMPTOMS) { toast(`Maksymalnie ${MAX_SYMPTOMS} objawów`); return prev; }
      const x = toXgbSlug(v);
      return prev.includes(x) ? prev : [...prev, x];
    });
    setQuery("");
  }

  function removeSymptom(xgb: string) {
    setSelected((prev) => prev.filter((s) => s !== xgb));
  }

  function clearAll() {
    setSelected([]);
    setSeverity(5);
    setDurationValue("");
    setDurationUnit("days");
    setOnset("");
    setNotes("");
    try { localStorage.removeItem(LS_KEY); } catch {}
    toast("Wyczyszczono.");
  }

  function buildSentencePL(): string {
    const pieces: string[] = [];

    if (selected.length) {
      pieces.push(`Zgłaszam objawy: ${joinPolish(selected)}.`);
    } else {
      pieces.push("Nie wskazano objawów.");
    }

    if (durationValue) {
      pieces.push(`Dolegliwości trwają około ${durationValue} ${unitLabel(durationUnit, durationValue)}.`);
    }

    if (onset) {
      pieces.push(`Początek objawów: ${formatLocalDateTime(onset)}.`);
    }

    pieces.push(`Nasilenie subiektywne: ${severity}/10.`);

    if (notes.trim()) {
      pieces.push(`Dodatkowe informacje: ${notes.trim()}.`);
    }

    return pieces.join(" ");
  }

  async function copySentence() {
    const text = buildSentencePL();
    try {
      await navigator.clipboard.writeText(text);
      toast("Skopiowano podsumowanie.");
    } catch { prompt("Kopiuj ręcznie (Ctrl+C):", text); }
  }

  function toast(msg: string) {
    const el = document.createElement("div");
    el.textContent = msg;
    el.className =
      "fixed left-1/2 top-4 z-[999] -translate-x-1/2 rounded-full border border-base bg-card px-3 py-1.5 text-xs shadow-md";
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 1200);
  }

  /* — Overlay — */
  return (
    <div
      className={`fixed inset-0 z-[80] transition ${
        open ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"
      }`}
      role="dialog"
      aria-modal="true"
      aria-label="Przygotuj opis"
    >
      {/* półprzezroczysty backdrop – klik = zamknij */}
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />

      {/* PŁYTA NAKŁADKI – pełnoekranowa, split 50/50 (bez headera) */}
      <div className="absolute inset-4 z-[81] overflow-hidden rounded-2xl border border-base bg-surface shadow-2xl">
        <div className="grid h-full grid-cols-1 md:grid-cols-2">
          {/* LEWA — wejście (scrollowalna kolumna) */}
          <section className="relative flex h-full min-h-0 flex-col overflow-auto px-4 py-4 md:pr-6">
            <div className="mb-3 text-xs font-medium text-muted">Dane wejściowe</div>

            {/* Wyszukiwarka */}
            <div>
              <label className="mb-1 block text-[11px] text-muted">Wyszukaj objaw (PL lub EN) i kliknij, aby dodać</label>
              <div className="flex gap-2">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") addSymptom(query); }}
                  placeholder="np. ból głowy • headache…"
                  className="h-10 flex-1 rounded-xl border border-base bg-transparent px-3 text-sm outline-none"
                  autoFocus
                />
                <button onClick={() => addSymptom(query)} className="rounded-xl border border-base bg-card px-3 text-sm">
                  Dodaj
                </button>
              </div>
            </div>

            {/* Lista filtrowana (zamiast sugestii) */}
            <div className="mt-2">
              <label className="mb-1 block text-[11px] text-muted">Lista objawów</label>
              <ul className="max-h-64 overflow-auto rounded-xl border border-base">
                {loadingDict ? (
                  Array.from({ length: 8 }).map((_, i) => (
                    <li key={i} className="h-9 animate-pulse border-b border-base last:border-0" />
                  ))
                ) : filtered.length ? (
                  filtered.map((it) => (
                    <li
                      key={it.en}
                      className="flex cursor-pointer items-center justify-between gap-2 border-b border-base px-3 py-2 text-sm last:border-b-0 hover:bg-input"
                      onClick={() => addByItem(it)}
                      title={`Dodaj: ${it.pl || it.en}`}
                    >
                      <div className="min-w-0">
                        <div className="truncate">{it.pl || it.en}</div>
                        <div className="text-[11px] text-muted">
                          xgb: <span className="font-mono">{it.xgb_label || it.en}</span>
                          {it.category ? ` • ${it.category}` : ""}
                        </div>
                      </div>
                      <span className="text-[11px]">Dodaj</span>
                    </li>
                  ))
                ) : (
                  <li className="px-3 py-2 text-sm text-muted">Brak wyników.</li>
                )}
              </ul>
            </div>

            {/* Wybrane (chipsy) */}
            <div className="mt-3">
              <div className="mb-1 flex items-center justify-between">
                <label className="text-[11px] text-muted">Wybrane objawy (XGB)</label>
                <span className="text-[11px] text-muted">{selected.length}/{MAX_SYMPTOMS}</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {selected.length ? (
                  selected.map((xgb) => (
                    <span
                      key={xgb}
                      className="inline-flex items-center gap-2 rounded-full border border-base bg-input px-3 py-1 text-sm"
                      title={xgb}
                    >
                      <span className="truncate">{humanFor(xgb)}</span>
                      <span className="rounded-md bg-card px-1.5 py-0.5 font-mono text-[11px] opacity-80">{xgb}</span>
                      <button
                        className="text-muted hover:text-basecolor"
                        onClick={() => removeSymptom(xgb)}
                        aria-label={`Usuń ${xgb}`}
                      >
                        ×
                      </button>
                    </span>
                  ))
                ) : (
                  <span className="text-sm text-muted">Brak – dodaj powyżej.</span>
                )}
              </div>
            </div>

            {/* Pigułki */}
            <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
              <div>
                <div className="mb-1 flex items-center gap-1 text-[11px] text-muted">
                  <Flame className="h-3.5 w-3.5 opacity-70" /> Nasilenie (0–10)
                </div>
                <div className="flex items-center gap-3 rounded-xl border border-base bg-input p-2">
                  <input type="range" min={0} max={10} value={severity} onChange={(e) => setSeverity(+e.target.value)} className="flex-1" />
                  <span className="w-6 text-right text-sm">{severity}</span>
                </div>
              </div>

              <div>
                <div className="mb-1 flex items-center gap-1 text-[11px] text-muted">
                  <Clock className="h-3.5 w-3.5 opacity-70" /> Czas trwania
                </div>
                <div className="flex items-center gap-2 rounded-xl border border-base bg-input p-2">
                  <input
                    type="number" min={0} value={durationValue}
                    onChange={(e) => setDurationValue(e.target.value === "" ? "" : Number(e.target.value))}
                    className="h-9 w-20 rounded-lg border border-base bg-transparent px-2 text-sm outline-none"
                    placeholder="np. 3"
                  />
                  <select
                    value={durationUnit}
                    onChange={(e) => setDurationUnit(e.target.value as any)}
                    className="h-9 rounded-lg border border-base bg-transparent px-2 text-sm outline-none"
                  >
                    <option value="hours">godzin</option>
                    <option value="days">dni</option>
                    <option value="weeks">tygodni</option>
                  </select>
                </div>
              </div>

              <div>
                <div className="mb-1 flex items-center gap-1 text-[11px] text-muted">
                  <Calendar className="h-3.5 w-3.5 opacity-70" /> Początek objawów
                </div>
                <div className="rounded-xl border border-base bg-input p-2">
                  <input
                    type="datetime-local"
                    value={onset}
                    onChange={(e) => setOnset(e.target.value)}
                    className="h-9 w-full rounded-lg border border-base bg-transparent px-2 text-sm outline-none"
                  />
                </div>
              </div>
            </div>

            {/* Notatki (dopasowane, żeby się mieściło) */}
            <div className="mt-3">
              <div className="mb-1 flex items-center justify-between">
                <label className="text-[11px] text-muted">Dodatkowe informacje</label>
                <span className="text-[11px] text-muted">{`${notes.length}/${MAX_NOTES}`}</span>
              </div>
              <textarea
                rows={4}
                maxLength={MAX_NOTES}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Np. choroby przewlekłe, przyjmowane leki, alergie…"
                className="w-full resize-vertical rounded-xl border border-base bg-transparent px-3 py-2 text-sm outline-none"
              />
            </div>
          </section>

          {/* PRAWA — podgląd (scrollowalna) */}
          <section className="relative flex h-full min-h-0 flex-col overflow-auto px-4 py-4 md:pl-6">
            {/* pionowa linia akcentu między kolumnami */}
            <div
              aria-hidden
              className="pointer-events-none absolute left-0 top-0 hidden h-full w-[3px] md:block"
              style={{ background: "linear-gradient(180deg,#34d399,#38bdf8,#a78bfa)" }}
            />

            {/* Pasek akcji (przeniesione z headera) */}
            <div className="mb-3 flex items-center justify-end gap-2">
              <span
                className={`mr-auto inline-flex items-center gap-1 rounded-full border border-base px-2 py-0.5 text-[11px] transition ${
                  justSaved ? "opacity-100" : "opacity-40"
                }`}
                title="Autozapis"
              >
                <Check className="h-3.5 w-3.5" /> zapisano
              </span>

              <button
                onClick={copySentence}
                className="inline-flex items-center gap-1 rounded-xl bg-zinc-900 px-3 py-2 text-xs text-white hover:opacity-90 dark:bg-zinc-800"
                title="Skopiuj podsumowanie"
              >
                <Clipboard className="h-3.5 w-3.5" />
                Kopiuj
              </button>
              <button
                onClick={clearAll}
                className="rounded-xl border border-base bg-card px-3 py-2 text-xs"
                title="Wyczyść wszystko"
              >
                Wyczyść
              </button>
              <button
                onClick={onClose}
                className="rounded-xl border border-base bg-card p-2 text-xs"
                aria-label="Zamknij"
                title="Zamknij (Esc)"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mb-2 text-xs font-medium text-muted">Podgląd podsumowania</div>
            <div className="rounded-2xl border border-base bg-input/60 p-4 text-[15px] leading-relaxed shadow-sm">
              {buildSentencePL()}
            </div>

            <div className="mt-3 grid grid-cols-3 gap-3">
              <div className="rounded-xl border border-base bg-card p-2 text-xs">
                <div className="text-[11px] opacity-60">Objawów</div>
                <div className="text-lg font-semibold">{selected.length}</div>
              </div>
              <div className="rounded-xl border border-base bg-card p-2 text-xs">
                <div className="text-[11px] opacity-60">Nasilenie</div>
                <div className="text-lg font-semibold">{severity}/10</div>
              </div>
              <div className="rounded-xl border border-base bg-card p-2 text-xs">
                <div className="text-[11px] opacity-60">Czas trwania</div>
                <div className="text-lg font-semibold">
                  {durationValue ? `${durationValue} ${unitLabel(durationUnit, durationValue)}` : "—"}
                </div>
              </div>
            </div>

            <div className="mt-auto text-[11px] text-muted">
              Wskazówka: Enter dodaje wpisany objaw • Ctrl/Cmd+C po kliknięciu „Kopiuj”.
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

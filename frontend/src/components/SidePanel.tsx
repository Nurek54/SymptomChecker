"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { TopDiagnosis, SymptomDictItem, SymptomsPatchResponse } from "@/lib/types";
import { listSymptomsDict, patchSessionSymptoms } from "@/lib/api";
import { ActivitySquare, Plus, X, Search, Eraser, Info as InfoIcon } from "lucide-react";

type SymptomsProps = {
  sessionId: string;
  symptoms: string[]; // EN canonical
  onPatched: (res: SymptomsPatchResponse) => void;
  lang?: "pl" | "en";
};

/* ───────────────── Aurora akcent ───────────────── */
const AURORA      = "linear-gradient(120deg, rgba(52,211,153,.35), rgba(56,189,248,.35), rgba(167,139,250,.35))";
const AURORA_SOFT = "linear-gradient(120deg, rgba(52,211,153,.22), rgba(56,189,248,.22), rgba(167,139,250,.22))";
const AURORA_BAR  = "linear-gradient(90deg, rgba(52,211,153,.9), rgba(56,189,248,.9), rgba(167,139,250,.85))";

/* ───────────────── Lista – paginacja ───────────────── */
const PAGE_SIZE = 60;

/* ───────────────── Globalny cache słownika ───────────────── */
let SYM_CACHE: SymptomDictItem[] | null = null;
let SYM_CACHE_PROMISE: Promise<SymptomDictItem[]> | null = null;

async function getSymptomsDictCached(): Promise<SymptomDictItem[]> {
  if (SYM_CACHE) return SYM_CACHE;
  if (!SYM_CACHE_PROMISE) {
    SYM_CACHE_PROMISE = listSymptomsDict()
      .then((d) => (d || []).map((it) => ({ ...it, en: it.en.toLowerCase() })))
      .then((arr) => {
        SYM_CACHE = arr;
        return arr;
      })
      .catch(() => {
        SYM_CACHE = [];
        return [];
      })
      .finally(() => {
        SYM_CACHE_PROMISE = null;
      });
  }
  return SYM_CACHE_PROMISE;
}

/* ───────────────── helpers ───────────────── */
function pct(n?: number | null) {
  if (typeof n !== "number" || !isFinite(n)) return 0;
  return Math.round(Math.min(1, Math.max(0, n)) * 100);
}

function ProbBar({ value }: { value: number }) {
  const w = Math.max(0, Math.min(100, value));
  const gradient =
    w >= 80
      ? "linear-gradient(90deg, #34d399, #38bdf8)"            // emerald → cyan
      : w >= 40
      ? "linear-gradient(90deg, #38bdf8, #fbbf24)"            // cyan → amber
      : "linear-gradient(90deg, rgba(148,163,184,.6), rgba(148,163,184,.85))"; // gray subtle
  return (
    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-[rgb(var(--border))]">
      <div
        className="h-full rounded-full"
        style={{
          width: `${w}%`,
          background: gradient,
          transition: "width 600ms cubic-bezier(.4,0,.2,1), background 300ms",
        }}
      />
    </div>
  );
}

function Highlight({ text, query }: { text: string; query: string }) {
  const q = query.trim().toLowerCase();
  if (!q) return <>{text}</>;
  const i = text.toLowerCase().indexOf(q);
  if (i === -1) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark className="rounded bg-yellow-200/60 px-0.5 dark:bg-yellow-300/30">{text.slice(i, i + q.length)}</mark>
      {text.slice(i + q.length)}
    </>
  );
}

/* ───────────────── Utilities UI ───────────────── */
function TopShadow({ show }: { show: boolean }) {
  return (
    <div
      className={`pointer-events-none absolute left-0 right-0 top-0 h-5 transition-opacity ${show ? "opacity-100" : "opacity-0"}`}
      style={{ background: "linear-gradient(to bottom, rgba(0,0,0,.08), transparent)" }}
    />
  );
}
function BottomShadow({ show }: { show: boolean }) {
  return (
    <div
      className={`pointer-events-none absolute bottom-0 left-0 right-0 h-6 transition-opacity ${show ? "opacity-100" : "opacity-0"}`}
      style={{ background: "linear-gradient(to top, rgba(0,0,0,.08), transparent)" }}
    />
  );
}

function SkeletonRow() {
  return (
    <li className="relative px-3 py-2">
      <div className="h-5 w-full animate-pulse rounded-md bg-[rgb(var(--border))]/60" />
    </li>
  );
}

/* ───────────────── SymptomsCard ───────────────── */
export function SymptomsCard({ sessionId, symptoms, onPatched, lang = "pl" }: SymptomsProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [dict, setDict] = useState<SymptomDictItem[] | null>(SYM_CACHE); // ⬅️ start z cache
  const [loadingDict, setLoadingDict] = useState(!SYM_CACHE);
  const [busy, setBusy] = useState(false);

  const panelRef = useRef<HTMLDivElement | null>(null);
  const toggleBtnRef = useRef<HTMLButtonElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null); // scrollowalne miejsce z wynikami
  const [activeIdx, setActiveIdx] = useState<number>(-1);

  // Shadows
  const [atTop, setAtTop] = useState(true);
  const [atBottom, setAtBottom] = useState(false);

  // Paginacja (infinite scroll)
  const [visibleCount, setVisibleCount] = useState<number>(PAGE_SIZE);

  // Stabilny preload słownika (z cache)
  useEffect(() => {
    let cancelled = false;
    if (!dict) {
      setLoadingDict(true);
      getSymptomsDictCached()
        .then((data) => { if (!cancelled) setDict(data); })
        .finally(() => { if (!cancelled) setLoadingDict(false); });
    }
    return () => { cancelled = true; };
  }, [dict]);

  // Autofocus search gdy panel się otwiera
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [open]);

  // Zamknięcie po kliknięciu poza / Esc
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const n = e.target as Node;
      if (panelRef.current?.contains(n)) return;
      if (toggleBtnRef.current?.contains(n)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // Scroll shadows
  useEffect(() => {
    const el = listRef.current;
    if (!el || !open) return;
    const onScroll = () => {
      setAtTop(el.scrollTop <= 0);
      setAtBottom(el.scrollTop + el.clientHeight >= el.scrollHeight - 1);
    };
    onScroll();
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, [open]);

  // Zestaw zaznaczonych (lowercase)
  const selectedLC = useMemo(
    () => new Set((symptoms ?? []).map((s) => (s || "").toLowerCase())),
    [symptoms]
  );

  // Mapa EN→PL dla chipów
  const en2pl = useMemo(() => {
    const m = new Map<string, string>();
    (dict ?? []).forEach((it) => m.set(it.en.toLowerCase(), it.pl));
    return m;
  }, [dict]);

  // Chips (unikalne)
  const chips = useMemo(() => {
    const arr = (symptoms ?? []).map((en) => {
      const lc = (en || "").toLowerCase();
      const label = lang === "pl" ? (en2pl.get(lc) ?? lc) : lc;
      return { en: lc, label };
    });
    const seen = new Set<string>();
    return arr.filter(({ en }) => (seen.has(en) ? false : (seen.add(en), true)));
  }, [symptoms, en2pl, lang]);

  // Opcje w pickerze (pełna lista), a poniżej – cięcie do visibleCount
  const optionsAll = useMemo(() => {
    const q = query.trim().toLowerCase();
    const base = (dict ?? []).filter((it) => !selectedLC.has(it.en.toLowerCase()));
    if (!q) return base;
    return base.filter((it) => {
      const label = (lang === "pl" ? it.pl : it.en).toLowerCase();
      return label.includes(q);
    });
  }, [dict, selectedLC, query, lang]);

  // Reset paginacji przy każdej zmianie filtra/zbioru
  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
    setActiveIdx(-1);
  }, [query, selectedLC, dict, lang]);

  const options = useMemo(() => optionsAll.slice(0, visibleCount), [optionsAll, visibleCount]);

  // Infinite scroll – dociąganie przy dojechaniu do dołu
  useEffect(() => {
    if (!open) return;
    const el = listRef.current;
    if (!el) return;

    const onScroll = () => {
      const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (gap < 48 && options.length < optionsAll.length) {
        setVisibleCount((v) => Math.min(v + PAGE_SIZE, optionsAll.length));
      }
    };

    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, [open, options.length, optionsAll.length]);

  const add = async (en: string) => {
    if (busy) return;
    setBusy(true);
    try {
      const canon = (en || "").toLowerCase();
      const res = await patchSessionSymptoms(sessionId, { add: [canon], lang, current: symptoms ?? [] });
      onPatched(res);
      setQuery("");
      setActiveIdx(-1);
      setOpen(false);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (en: string) => {
    if (busy) return;
    setBusy(true);
    try {
      const canon = (en || "").toLowerCase();
      const res = await patchSessionSymptoms(sessionId, { remove: [canon], lang, current: symptoms ?? [] });
      onPatched(res);
    } finally {
      setBusy(false);
    }
  };

  const clearAll = async () => {
    if (busy || !symptoms?.length) return;
    setBusy(true);
    try {
      const res = await patchSessionSymptoms(sessionId, { replace: [], lang });
      onPatched(res);
    } finally {
      setBusy(false);
    }
  };

  const onSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!options.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => (i + 1) % options.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => (i - 1 + options.length) % options.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const pick = options[Math.max(0, activeIdx)];
      if (pick) add(pick.en);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  const canShowMore = options.length < optionsAll.length;

  return (
    <div
      className="relative rounded-2xl border border-base bg-card p-3 shadow-sm transition-shadow hover:shadow-md"
      aria-busy={busy}
    >
      {/* gradientowy hairline na górze karty */}
      <span aria-hidden className="absolute inset-x-0 top-0 h-px rounded-t-2xl" style={{ background: AURORA_SOFT }} />

      {/* Header */}
      <div className="flex items-center gap-2 text-sm font-semibold">
        <div className="relative grid h-6 w-6 place-items-center rounded-lg border border-base bg-input">
          <span aria-hidden className="absolute inset-0 rounded-lg opacity-70" style={{ background: AURORA_SOFT }} />
          <ActivitySquare className="relative h-3.5 w-3.5 opacity-80" />
        </div>
        <span>Zebrane objawy</span>
        <span className="ml-1 rounded-full border border-base bg-input px-2 py-0.5 text-[10px] text-muted">
          {chips.length}
        </span>
      </div>

      {/* cienki hairline pod tytułem */}
      <span aria-hidden className="mt-2 block h-px w-full rounded-full" style={{ background: AURORA_SOFT }} />

      {/* Actions */}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          ref={toggleBtnRef}
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls="symptom-picker"
          disabled={busy}
          className="group relative inline-flex h-8 items-center gap-1.5 overflow-visible rounded-xl border border-base bg-input px-3 text-xs font-medium shadow-sm transition-colors hover:bg-card focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-[rgb(var(--border))] disabled:opacity-60"
          title="Dodaj objaw"
        >
          <span
            aria-hidden
            className="pointer-events-none absolute -inset-1 rounded-[14px] opacity-0 blur-[6px] transition-opacity group-hover:opacity-90 group-focus-visible:opacity-90"
            style={{ background: AURORA }}
          />
          <Plus className="h-3.5 w-3.5" />
          <span className="relative">Dodaj</span>
        </button>

        {chips.length ? (
          <button
            onClick={clearAll}
            disabled={busy}
            className="group relative inline-flex h-8 items-center gap-1.5 overflow-visible rounded-xl border border-base bg-card px-3 text-xs shadow-sm transition-colors hover:bg-input focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-[rgb(var(--border))] disabled:opacity-60"
            title="Usuń wszystkie"
          >
            <span
              aria-hidden
              className="pointer-events-none absolute -inset-px rounded-[14px] blur-[6px] opacity-0 mix-blend-screen transition-opacity group-hover:opacity-70 group-focus-visible:opacity-70"
              style={{ background: AURORA }}
            />
            <Eraser className="h-3.5 w-3.5" />
            <span className="relative">Wyczyść</span>
          </button>
        ) : null}
      </div>

      {/* Chips — auto-szerokość do treści z limitem */}
      <div className="mt-3">
        {chips.length ? (
          <div className="flex flex-wrap items-start gap-1.5">
            {chips.map((s) => (
              <span
                key={s.en}
                className="relative inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-base bg-input px-2.5 py-1 text-[12px]"
                title={s.label}
              >
                <span aria-hidden className="h-2 w-2 shrink-0 rounded-full" style={{ background: AURORA_BAR }} />
                <span className="min-w-[2rem] max-w-[16rem] truncate font-medium">{s.label}</span>
                <button
                  onClick={() => remove(s.en)}
                  disabled={busy}
                  className="ml-0.5 shrink-0 rounded p-1 text-muted transition-colors hover:bg-card hover:text-basecolor focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-[rgb(var(--border))] disabled:opacity-50"
                  aria-label={`Usuń: ${s.label}`}
                  title="Usuń"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </span>
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-base bg-input/50 p-4 text-center">
            <p className="text-sm text-muted">Brak — dodaj objawy przyciskiem „Dodaj”.</p>
          </div>
        )}
      </div>

      {/* Click-catcher for closing */}
      {open && (
        <div className="fixed inset-0 z-[900] bg-transparent" aria-hidden="true" onClick={() => setOpen(false)} />
      )}

      {/* Picker panel */}
      <div
        id="symptom-picker"
        ref={panelRef}
        className={`absolute left-3 right-3 z-[1000] mt-2 origin-top rounded-2xl border border-base bg-card shadow-2xl ring-1 ring-black/5 transition-all
          ${open ? "pointer-events-auto translate-y-0 opacity-100" : "pointer-events-none -translate-y-1 opacity-0"}
        `}
        role="dialog"
        aria-label="Wybór objawów"
        aria-hidden={!open}
        style={{ maxHeight: "80vh" }}
      >
        <span aria-hidden className="absolute inset-x-0 top-0 h-px" style={{ background: AURORA_SOFT }} />

        <div className="flex max-h-[80vh] flex-col overflow-hidden rounded-2xl">
          {/* Search + hints (sticky header) */}
          <div className="relative sticky top-0 z-10 rounded-t-2xl">
            <span
              aria-hidden
              className="pointer-events-none absolute -inset-[2px] rounded-[20px] opacity-0 blur-[6px] transition-opacity focus-within:opacity-100"
              style={{ background: AURORA }}
            />
            <div className="relative z-[1] rounded-t-2xl border-b border-base bg-card/95 backdrop-blur">
              <div className="flex items-center gap-2 px-3 py-2">
                <Search className="h-4 w-4 opacity-70" />
                <input
                  ref={inputRef}
                  value={query}
                  onChange={(e) => { setQuery(e.target.value); setActiveIdx(-1); }}
                  onKeyDown={onSearchKeyDown}
                  placeholder="Szukaj objawu…"
                  className="h-8 flex-1 bg-transparent text-sm outline-none placeholder:text-muted"
                  aria-controls="symptom-options"
                  aria-autocomplete="list"
                />
                {query ? (
                  <button
                    onClick={() => { setQuery(""); setActiveIdx(-1); }}
                    className="rounded-lg border border-base bg-input px-2 py-1 text-[11px] hover:bg-card focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-[rgb(var(--border))]"
                    aria-label="Wyczyść wyszukiwanie"
                    title="Wyczyść"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                ) : null}
                <span
                  className="rounded-full border border-base bg-input px-2 py-0.5 text-[10px] text-muted"
                  aria-live="polite"
                >
                  {options.length}/{optionsAll.length}
                </span>
              </div>
              <div className="flex items-center justify-between gap-2 border-t border-base px-3 py-1.5 text-[10px] text-muted">
                <div className="flex items-center gap-1.5">
                  <span className="rounded border border-base bg-input px-1">↑</span>
                  <span className="rounded border border-base bg-input px-1">↓</span>
                  <span className="rounded border border-base bg-input px-1">Enter</span>
                  <span className="rounded border border-base bg-input px-1">Esc</span>
                </div>
                <span>Wyniki</span>
              </div>
            </div>
          </div>

          {/* Options — scrollowalna lista, z paginacją i „Pokaż więcej” */}
          <div
            ref={listRef}
            className="relative flex-1 overflow-y-auto overscroll-contain rounded-b-2xl"
            style={{ scrollbarWidth: "thin" }}
          >
            <TopShadow show={!atTop} />

            {loadingDict ? (
              <ul className="divide-y divide-[rgb(var(--border))]">
                {Array.from({ length: 10 }).map((_, i) => <SkeletonRow key={i} />)}
              </ul>
            ) : options.length ? (
              <>
                <ul id="symptom-options" role="listbox" className="divide-y divide-[rgb(var(--border))]">
                  {options.map((it, idx) => {
                    const labelRaw = lang === "pl" ? it.pl : it.en;
                    const label = labelRaw || it.en;
                    const cat = it.category ? String(it.category) : null;
                    const active = idx === activeIdx;
                    return (
                      <li
                        key={it.en}
                        role="option"
                        aria-selected={active}
                        className={`group relative cursor-pointer px-3 py-2 text-sm transition-colors
                          ${active ? "bg-input" : "hover:bg-input"}
                          focus-within:bg-input
                        `}
                        onMouseEnter={() => setActiveIdx(idx)}
                        onClick={() => add(it.en)}
                        title={label}
                      >
                        <span
                          aria-hidden
                          className="pointer-events-none absolute -inset-px z-0 rounded-xl opacity-0 blur-[4px] transition-opacity group-hover:opacity-50"
                          style={{ background: AURORA_SOFT }}
                        />
                        {active && (
                          <span
                            aria-hidden
                            className="absolute left-0 top-2 bottom-2 w-[3px] rounded-full"
                            style={{ background: AURORA_SOFT }}
                          />
                        )}
                        <div className="relative z-[1] flex items-center justify-between gap-3">
                          <span className="min-w-0 truncate">
                            <Highlight text={label} query={query} />
                          </span>
                          <div className="flex shrink-0 items-center gap-2">
                            {cat ? (
                              <span className="rounded-full border border-base bg-card px-2 py-0.5 text-[10px] text-muted">
                                {cat}
                              </span>
                            ) : null}
                            <button
                              onClick={(e) => { e.stopPropagation(); add(it.en); }}
                              className="inline-flex items-center gap-1 rounded-lg border border-base bg-card px-2 py-1 text-[11px] opacity-0 transition-opacity group-hover:opacity-100 data-[active=true]:opacity-100 hover:bg-input focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-[rgb(var(--border))]"
                              data-active={active}
                              aria-label={`Dodaj: ${label}`}
                              title="Dodaj"
                            >
                              <Plus className="h-3.5 w-3.5" />
                              Dodaj
                            </button>
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>

                {canShowMore && (
                  <div className="sticky bottom-0 z-10 border-t border-[rgb(var(--border))] bg-card/95 p-2 text-center backdrop-blur">
                    <button
                      className="inline-flex items-center gap-2 rounded-xl border border-base bg-input px-3 py-1.5 text-xs hover:bg-card focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-[rgb(var(--border))]"
                      onClick={() => setVisibleCount((v) => Math.min(v + PAGE_SIZE, optionsAll.length))}
                    >
                      Pokaż więcej ({options.length}/{optionsAll.length})
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className="p-6 text-center text-sm text-muted">
                Nic nie znaleziono dla „<span className="font-medium">{query}</span>”.
              </div>
            )}

            <BottomShadow show={!atBottom} />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ───────────────── TopDiagCard ───────────────── */
export function TopDiagCard({
  items,
  onInfo,
}: {
  items: TopDiagnosis[] | null;
  onInfo?: () => void;
}) {
  return (
    <div className="relative rounded-2xl border border-base bg-card p-3 shadow-sm transition-shadow hover:shadow-md">
      {/* hairline na górze karty */}
      <span aria-hidden className="absolute inset-x-0 top-0 h-px rounded-t-2x" style={{ background: AURORA_SOFT }} />

      <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
        <div className="relative grid h-6 w-6 place-items-center rounded-lg border border-base bg-input">
          <span aria-hidden className="absolute inset-0 rounded-lg opacity-70" style={{ background: AURORA_SOFT }} />
          <ActivitySquare className="relative h-3.5 w-3.5 opacity-80" />
        </div>
        <span className="mr-auto">Najbardziej prawdopodobne (Top-3)</span>

        {/* ⓘ ikonka informacji */}
        <button
          onClick={onInfo}
          className="rounded-lg border border-base bg-input p-1.5 text-[11px] transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
          title="Pokaż osobno XGBoost i River"
          aria-label="Pokaż osobno XGBoost i River"
        >
          <InfoIcon className="h-4 w-4" />
        </button>
      </div>

      {/* hairline pod tytułem */}
      <span aria-hidden className="mb-2 block h-px w-full rounded-full" style={{ background: AURORA_SOFT }} />

      {items && items.length ? (
        <ul className="space-y-2">
          {items.map((item, i) => {
            const d = item as TopDiagnosis & { label?: string };
            const label = d.diagnosis_pl ?? d.diagnosis_en ?? d.label ?? "-";
            const p = pct(d.probability);
            return (
              <li
                key={`${label}-${i}`}
                className="relative overflow-hidden rounded-xl border border-base bg-input p-2.5"
                title={label}
              >
                <span
                  aria-hidden
                  className="pointer-events-none absolute -inset-px rounded-xl opacity-0 blur-[4px] transition-opacity hover:opacity-70"
                  style={{ background: AURORA_SOFT }}
                />
                <div className="relative z-[1] flex items-center gap-2">
                  {}
                  <span
                    className={[
                      "grid h-6 w-6 place-items-center shrink-0 rounded-lg text-[11px] font-bold tabular-nums",
                      i === 0
                        ? "text-white shadow-sm border border-emerald-400/50"
                        : i === 1
                          ? "text-cyan-100 border border-cyan-400/60"
                          : "text-muted border border-base bg-card",
                    ].join(" ")}
                    style={
                      i === 0
                        ? { background: "linear-gradient(135deg, #34d399, #10b981)" }
                        : i === 1
                          ? { background: "linear-gradient(135deg, rgba(56,189,248,.35), rgba(56,189,248,.15))" }
                          : undefined
                    }
                  >
                    {i + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-3 text-[13px]">
                      <span className="truncate">{label}</span>
                      <span className="shrink-0 tabular-nums">{p ? `${p}%` : ""}</span>
                    </div>
                    {typeof d.probability === "number" && <ProbBar value={p} />}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      ) : (
        <div className="rounded-xl border border-dashed border-base bg-input/50 p-4 text-center">
          <p className="text-sm text-muted">Pojawią się po zebraniu wystarczającej liczby objawów.</p>
        </div>
      )}
    </div>
  );
}
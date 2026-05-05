"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/components/AuthProvider";
import {
    ArrowLeft,
    MessageSquare,
    Clock,
    RefreshCw,
    Trash2,
    Trash,
    Search,
    ChevronRight,
} from "lucide-react";

// ❗️Korzystamy z lib/api – retry 401 + Bearer + cookies + signals
import {
    listHistory,
    getConversation,
    deleteConversation as apiDeleteConversation,
    type HistoryItem as Conv,
    type HistoryDetail as ConvDetail,
    type HistorySymptom as Symptom,
    type HistoryDiagnosis as Diagnosis,
    prewarmAuth,
} from "@/lib/api";

/* ——— Aurora akcent (emerald → sky → violet) ——— */
const AURORA = "linear-gradient(120deg, rgba(52,211,153,.35), rgba(56,189,248,.35), rgba(167,139,250,.35))";
const AURORA_SOFT = "linear-gradient(120deg, rgba(52,211,153,.22), rgba(56,189,248,.22), rgba(167,139,250,.22))";
const AURORA_LINE = "linear-gradient(180deg, rgba(52,211,153,.25), rgba(56,189,248,.25), rgba(167,139,250,.25))";

type Msg = { role: "user" | "assistant"; content: string; created_at: string };

/* Helpers */
function timeAgo(iso: string) {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "—";
    const diff = (Date.now() - d.getTime()) / 1000;
    const rtf = new Intl.RelativeTimeFormat("pl", { numeric: "auto" });
    const units: Intl.RelativeTimeFormatUnit[] = ["second", "minute", "hour", "day", "week", "month", "year"];
    const steps = [60, 60, 24, 7, 4.345, 12, Number.POSITIVE_INFINITY];
    let val = Math.floor(diff);
    for (let i = 0; i < steps.length; i++) {
        if (val < steps[i]) return rtf.format(-val as any, units[i]);
        val = Math.floor(val / (steps[i] as number));
    }
    return rtf.format(-val as any, "year");
}
function pct(n?: number | null) {
    if (typeof n !== "number" || !isFinite(n)) return 0;
    return Math.round(Math.min(1, Math.max(0, n)) * 100);
}
function isSameDay(a: Date, b: Date) {
    return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}
function daysDiff(a: Date, b: Date) {
    return Math.floor((+a - +b) / (1000 * 60 * 60 * 24));
}

/* Skeleton list item */
function ListItemSkeleton() {
    return (
        <div className="relative pl-6">
            <div className="absolute left-[11px] top-0 h-full w-[2px]" style={{ background: AURORA_LINE }} />
            <div className="absolute left-[6px] top-2 h-3 w-3 rounded-full bg-[rgb(var(--border))]" />
            <div className="animate-pulse rounded-xl border border-base bg-card p-3 shadow-sm">
                <div className="h-3 w-4/5 rounded bg-[rgb(var(--input))]" />
                <div className="mt-2 flex items-center justify-between">
                    <div className="h-2.5 w-24 rounded bg-[rgb(var(--input))]" />
                    <div className="h-2.5 w-10 rounded bg-[rgb(var(--input))]" />
                </div>
            </div>
        </div>
    );
}

export default function HistoryPage() {
    const { user, openAuth, loading } = useAuth();

    const [items, setItems] = useState<Conv[]>([]);
    const [itemsLoading, setItemsLoading] = useState(false);

    const [current, setCurrent] = useState<ConvDetail | null>(null);
    const [busy, setBusy] = useState(false);
    const [err, setErr] = useState<string | null>(null);

    const [q, setQ] = useState("");

    // Abort Controllers – by nie „wisiały” żądania przy wyjściu z widoku
    const listAbort = useRef<AbortController | null>(null);
    const detailAbort = useRef<AbortController | null>(null);

    // dogrzej sesję po wejściu na stronę (chroni przed 401 po przełączeniu kart)
    useEffect(() => {
        void prewarmAuth();
    }, []);

    const safeTime = (iso?: string | null) => {
        const t = iso ? new Date(iso).getTime() : 0;
        return Number.isFinite(t) ? t : 0; // null → koniec listy
    };

    const loadList = async () => {
        setErr(null);
        setItemsLoading(true);
        listAbort.current?.abort();
        const ac = new AbortController();
        listAbort.current = ac;
        try {
            const arr = await listHistory(ac.signal);
            arr.sort((a, b) => safeTime(b.last_message_at) - safeTime(a.last_message_at));
            setItems(arr);
        } catch (e: any) {
            if (e?.name !== "AbortError") {
                setErr(e?.message || "Błąd");
                setItems([]);
            }
        } finally {
            setItemsLoading(false);
        }
    };

    const openConv = async (id: number) => {
        setBusy(true);
        setErr(null);
        detailAbort.current?.abort();
        const ac = new AbortController();
        detailAbort.current = ac;
        try {
            const detail = await getConversation(id, ac.signal);
            setCurrent(detail);
        } catch (e: any) {
            if (e?.name !== "AbortError") setErr(e?.message || "Błąd");
        } finally {
            setBusy(false);
        }
    };

    const delConv = async (id: number) => {
        if (!confirm("Usunąć tę rozmowę?")) return;
        try {
            await apiDeleteConversation(id);
            setCurrent(null);
            await loadList();
        } catch {
            /* ignore */
        }
    };

    const delAllConvs = async () => {
        if (!items.length) return;
        if (!confirm("Usunąć CAŁĄ historię rozmów? Tego nie można cofnąć.")) return;
        setItemsLoading(true);
        try {
            for (const it of items) {
                await apiDeleteConversation(it.id).catch(() => {});
            }
            setCurrent(null);
            await loadList();
        } finally {
            setItemsLoading(false);
        }
    };

    useEffect(() => {
        if (user) loadList();
    }, [user]);

    // posprzątaj przy wyjściu z widoku
    useEffect(() => {
        return () => {
            listAbort.current?.abort();
            detailAbort.current?.abort();
        };
    }, []);

    const filtered = useMemo(() => {
        const qq = q.trim().toLowerCase();
        if (!qq) return items;
        return items.filter((it) => {
            const last = it.last_message_at ? new Date(it.last_message_at).toLocaleString().toLowerCase() : "";
            return it.title.toLowerCase().includes(qq) || last.includes(qq) || String(it.message_count).includes(qq);
        });
    }, [items, q]);

    const grouped = useMemo(() => {
        const today: Conv[] = [];
        const week: Conv[] = [];
        const older: Conv[] = [];
        const now = new Date();
        for (const it of filtered) {
            const d = new Date(it.last_message_at ?? 0);
            if (!isNaN(d.getTime()) && isSameDay(d, now)) today.push(it);
            else if (!isNaN(d.getTime()) && daysDiff(now, d) < 7) week.push(it);
            else older.push(it);
        }
        return [
            { title: "Dzisiaj", arr: today },
            { title: "Ostatnie 7 dni", arr: week },
            { title: "Starsze", arr: older },
        ].filter((g) => g.arr.length);
    }, [filtered]);

    if (loading) return <main className="p-6 text-sm text-muted">Ładowanie…</main>;

    if (!user)
        return (
            <main className="flex min-h-[100svh] items-center justify-center bg-surface p-6 text-basecolor">
                <div className="w-full max-w-md rounded-2xl border border-base bg-card p-6 shadow-sm">
                    <h2 className="text-lg font-semibold">Historia rozmów</h2>
                    <p className="mt-1 text-sm text-muted">Aby zobaczyć historię, musisz się zalogować.</p>
                    <div className="mt-4 flex items-center gap-2">
                        <button onClick={() => openAuth("login")} className="btn-primary rounded-xl px-4 py-2 text-sm">
                            Zaloguj się
                        </button>
                        <Link href="/" className="inline-flex items-center gap-1 rounded-xl border border-base bg-card px-3 py-2 text-sm hover:brightness-110">
                            ← Wróć do czatu
                        </Link>
                    </div>
                </div>
            </main>
        );

    return (
        // 🔒 Globalny scroll OFF → tylko lokalne scrollbary
        <main className="flex h-[100svh] min-h-0 overflow-hidden bg-surface text-basecolor">
            {/* Sidebar */}
            <aside className="flex h-full min-h-0 w-[320px] shrink-0 flex-col border-r border-base">
                <div className="sticky top-0 z-10 border-b border-base bg-card px-3 py-3">
                    <div className="flex items-center justify-between">
                        <h2 className="text-sm font-semibold">Twoje rozmowy</h2>
                        <div className="flex items-center gap-2">
                            <span className="rounded-full border border-base bg-input px-2 py-0.5 text-[11px] text-muted">{items.length}</span>
                            <button
                                onClick={delAllConvs}
                                className="inline-flex items-center gap-1 rounded-xl border border-base bg-card px-2.5 py-1 text-[11px] text-rose-600 shadow-sm transition hover:brightness-110 disabled:opacity-50"
                                title="Usuń całą historię"
                                disabled={!items.length || itemsLoading}
                            >
                                <Trash className="h-3.5 w-3.5" />
                                Usuń wszystko
                            </button>
                        </div>
                    </div>
                    <span aria-hidden className="mt-2 block h-px w-full rounded-full" style={{ background: AURORA_SOFT }} />

                    <div className="relative mt-2 rounded-xl">
            <span
                aria-hidden
                className="pointer-events-none absolute -inset-[2px] rounded-2xl opacity-0 blur-[6px] transition-opacity focus-within:opacity-100"
                style={{ background: AURORA }}
            />
                        <div className="relative z-[1] flex items-center gap-2 rounded-xl border border-base bg-input px-2">
                            <Search className="h-4 w-4 opacity-70" />
                            <input
                                value={q}
                                onChange={(e) => setQ(e.target.value)}
                                placeholder="Szukaj w tytułach, dacie…"
                                className="h-8 flex-1 bg-transparent text-sm outline-none placeholder:text-muted"
                            />
                            <button onClick={loadList} className="rounded-lg p-1 text-muted transition hover:text-basecolor" title="Odśwież">
                                <RefreshCw className="h-4 w-4" />
                            </button>
                        </div>
                    </div>
                </div>

                <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
                    {itemsLoading ? (
                        <>
                            <ListItemSkeleton />
                            <ListItemSkeleton />
                            <ListItemSkeleton />
                        </>
                    ) : grouped.length ? (
                        grouped.map((g) => (
                            <div key={g.title}>
                                <div className="sticky top-0 z-0 mb-1 bg-surface/80 px-1 text-[11px] font-medium uppercase tracking-wide text-muted">
                                    {g.title}
                                </div>
                                <div className="relative space-y-2 pl-6">
                                    <div className="absolute left-[11px] top-0 h-full w-[2px]" style={{ background: AURORA_LINE }} />
                                    {g.arr.map((it) => {
                                        const active = current?.id === it.id;
                                        return (
                                            <button
                                                key={it.id}
                                                onClick={() => openConv(it.id)}
                                                className={`group relative w-full overflow-visible rounded-xl border bg-card p-3 text-left text-sm shadow-sm transition-all
                          ${active ? "border-neutral-300" : "border-base hover:-translate-y-0.5 hover:shadow-md"}`}
                                            >
                                                <span className="absolute left-[-6px] top-3 h-3 w-3 rounded-full border border-base bg-card" />
                                                <span
                                                    aria-hidden
                                                    className="pointer-events-none absolute left-[-10px] top-[7px] h-5 w-5 rounded-full opacity-0 blur-[6px] transition-opacity
                                     group-hover:opacity-70"
                                                    style={{ background: AURORA_SOFT }}
                                                />
                                                <span
                                                    aria-hidden
                                                    className={`pointer-events-none absolute -inset-px rounded-[inherit] opacity-0 transition-opacity
                                      ${active ? "opacity-100" : "group-hover:opacity-60"}`}
                                                    style={{ background: AURORA_SOFT }}
                                                />
                                                <div className="relative z-[1] flex items-start justify-between gap-2">
                                                    <div className="min-w-0">
                                                        <div className="truncate font-medium">{it.title}</div>
                                                        <div className="mt-1 flex items-center gap-2 text-xs text-muted">
                              <span className="inline-flex items-center gap-1">
                                <Clock className="h-3.5 w-3.5" />
                                  {it.last_message_at ? timeAgo(it.last_message_at) : "—"}
                              </span>
                                                            <span className="inline-flex items-center gap-1">
                                <MessageSquare className="h-3.5 w-3.5" />
                                                                {it.message_count}
                              </span>
                                                        </div>
                                                    </div>
                                                    <ChevronRight className="h-4 w-4 opacity-0 transition-opacity group-hover:opacity-100" />
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="rounded-xl border border-base bg-card p-4 text-center text-sm text-muted">
                            Brak rozmów {q ? "spełniających kryteria." : "do wyświetlenia."}
                        </div>
                    )}
                </div>
            </aside>

            {/* Detal */}
            <section className="min-w-0 flex-1 min-h-0 p-4 flex flex-col">
                <div className="mb-3 flex shrink-0 items-center justify-between gap-3">
                    <div className="flex min-w-0 items-center gap-3">
                        <button
                            onClick={() => setCurrent(null)}
                            disabled={!current}
                            className="inline-flex items-center gap-2 rounded-xl border border-base bg-card px-3 py-1.5 text-sm shadow-sm transition enabled:hover:brightness-110 disabled:opacity-60"
                            title="Powrót do listy"
                        >
                            <ArrowLeft className="h-4 w-4" />
                            Powrót
                        </button>

                        <div className="min-w-0">
                            <div className="truncate text-base font-semibold">{current ? current.title : "—"}</div>
                            <span aria-hidden className="mt-1 block h-px w-28 rounded-full" style={{ background: AURORA_SOFT }} />
                            <div className="mt-1 flex items-center gap-3 text-xs text-muted">
                <span className="inline-flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5" />
                    {current
                        ? timeAgo(current.messages[current.messages.length - 1]?.created_at ?? new Date().toISOString())
                        : "—"}
                </span>
                                <span className="inline-flex items-center gap-1">
                  <MessageSquare className="h-3.5 w-3.5" />
                                    {current ? current.messages.length : 0} wiadomości
                </span>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        {current && (
                            <button
                                onClick={() => delConv(current.id)}
                                className="inline-flex items-center gap-2 rounded-xl border border-base bg-card px-3 py-1.5 text-sm text-rose-600 shadow-sm transition hover:brightness-110"
                                title="Usuń tę rozmowę"
                            >
                                <Trash2 className="h-4 w-4" />
                                Usuń
                            </button>
                        )}
                        <Link
                            href="/"
                            className="inline-flex items-center gap-2 rounded-xl border border-base bg-card px-3 py-1.5 text-sm shadow-sm transition hover:brightness-110"
                            title="Wróć do czatu"
                        >
                            ← Wróć do czatu
                        </Link>
                    </div>
                </div>

                {!current ? (
                    <div className="grid flex-1 min-h-0 place-items-center">
                        <div className="w-full max-w-xl rounded-2xl border border-dashed border-base bg-card p-8 text-center shadow-sm">
                            <div className="mx-auto mb-3 h-10 w-10 rounded-full border border-base bg-input" />
                            <h3 className="text-base font-semibold">Wybierz rozmowę z listy</h3>
                            <p className="mt-1 text-sm text-muted">Po lewej stronie znajdziesz historię Twoich sesji.</p>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-1 min-h-0 flex-col gap-3">
                        <div className="grid min-h-0 grid-cols-1 items-stretch gap-3 md:grid-cols-2">
                            {/* Objawy (opcjonalne) */}
                            <div className="flex min-h-0 flex-col rounded-2xl border border-base bg-card shadow-sm">
                                <div className="flex items-center justify-between px-3 py-2">
                                    <div className="text-sm font-semibold">Wyłapane objawy</div>
                                    <span className="rounded-full border border-base bg-input px-2 py-0.5 text-[10px] text-muted">
                    {current.symptoms?.length ?? 0}
                  </span>
                                </div>
                                <span aria-hidden className="mx-2 mb-2 block h-px rounded-full" style={{ background: AURORA_SOFT }} />
                                <div className="px-2 pb-2">
                                    {busy ? (
                                        <div className="animate-pulse rounded-lg bg-[rgb(var(--input))] p-6" />
                                    ) : current.symptoms?.length ? (
                                        <div className="max-h-36 overflow-y-auto">
                                            <div className="flex flex-wrap gap-1.5">
                                                {current.symptoms.map((s: Symptom, i: number) => (
                                                    <span
                                                        key={i}
                                                        className="inline-flex items-center gap-1.5 rounded-full border border-base bg-input px-2.5 py-1 text-[11px]"
                                                        title={s.canonical ? `${s.text} (${s.canonical})` : s.text}
                                                    >
                            <span className="max-w-[12rem] truncate font-medium">{s.text}</span>
                                                        {s.canonical && <span className="rounded bg-card px-1 text-[10px] text-muted">{s.canonical}</span>}
                                                        {typeof s.score === "number" && (
                                                            <span className="rounded bg-card px-1 text-[10px]">{pct(s.score)}%</span>
                                                        )}
                          </span>
                                                ))}
                                            </div>
                                        </div>
                                    ) : (
                                        <p className="text-[12px] text-muted">Brak zarejestrowanych objawów w tej rozmowie.</p>
                                    )}
                                </div>
                            </div>

                            {/* Choroby (opcjonalne) */}
                            <div className="flex min-h-0 flex-col rounded-2xl border border-base bg-card shadow-sm">
                                <div className="flex items-center justify-between px-3 py-2">
                                    <div className="text-sm font-semibold">Możliwe choroby</div>
                                    <span className="rounded-full border border-base bg-input px-2 py-0.5 text-[10px] text-muted">
                    {current.diagnoses?.length ?? 0}
                  </span>
                                </div>
                                <span aria-hidden className="mx-2 mb-2 block h-px rounded-full" style={{ background: AURORA_SOFT }} />
                                <div className="px-2 pb-2">
                                    {busy ? (
                                        <div className="animate-pulse rounded-lg bg-[rgb(var(--input))] p-6" />
                                    ) : current.diagnoses?.length ? (
                                        <div className="max-h-56 overflow-y-auto pr-1">
                                            <ol className="space-y-1.5 text-sm">
                                                {current.diagnoses.map((d: Diagnosis, i: number) => {
                                                    const p = pct(d.score);
                                                    return (
                                                        <li key={i} className="rounded-lg border border-base bg-input p-2" title={d.label}>
                                                            <div className="flex items-center justify-between gap-3 text-[13px]">
                                                                <span className="min-w-0 truncate">{d.rank ? `${d.rank}. ` : ""}{d.label}</span>
                                                                <span className="shrink-0 text-[11px] tabular-nums">{p ? `${p}%` : ""}</span>
                                                            </div>
                                                            {typeof d.score === "number" && (
                                                                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-[rgb(var(--border))]">
                                                                    <div
                                                                        className="h-full rounded-full transition-[width]"
                                                                        style={{
                                                                            width: `${p}%`,
                                                                            background:
                                                                                "linear-gradient(90deg, rgba(52,211,153,.9), rgba(56,189,248,.9), rgba(167,139,250,.85))",
                                                                        }}
                                                                    />
                                                                </div>
                                                            )}
                                                        </li>
                                                    );
                                                })}
                                            </ol>
                                        </div>
                                    ) : (
                                        <p className="text-[12px] text-muted">Brak wyników triage w tej rozmowie.</p>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* CZAT */}
                        <div className="flex-1 min-h-0 overflow-y-auto rounded-2xl border border-base p-3">
                            <div className="space-y-4">
                                {current.messages.map((m: Msg, i: number) => {
                                    const isUser = m.role === "user";
                                    const when = new Date(m.created_at).toLocaleString();

                                    return (
                                        <div key={i} className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
                                            <div className={`mb-1 text-[11px] text-muted ${isUser ? "pr-2 text-right" : "pl-2 text-left"}`}>
                                                <span className="font-medium">{isUser ? "Użytkownik" : "Asystent"}</span> • {when}
                                            </div>

                                            <div
                                                className={[
                                                    "group relative prose max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm shadow-sm transition-all",
                                                    isUser
                                                        ? "rounded-br-md bg-bubble-user text-bubble-user border border-bubble-user hover:shadow-md"
                                                        : "rounded-bl-md bg-card text-basecolor border border-base hover:shadow-md",
                                                ].join(" ")}
                                            >
                                                {!isUser && (
                                                    <span
                                                        aria-hidden
                                                        className="absolute left-0 top-2 bottom-2 w-[3px] rounded-full opacity-90"
                                                        style={{ background: AURORA_SOFT }}
                                                    />
                                                )}
                                                {m.content}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                )}
                {err && <p className="mt-2 text-xs text-rose-600">{err}</p>}
            </section>
        </main>
    );
}

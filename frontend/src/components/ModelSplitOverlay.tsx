"use client";

import React from "react";
import { getTop3Split } from "@/lib/api";
import { X, ActivitySquare, Sigma, Info } from "lucide-react";

type TopDiagnosis = {
    diagnosis_en: string;
    probability: number;
    diagnosis_pl?: string | null;
};

type SplitResponse = {
    xgb_enabled: boolean;
    river_enabled: boolean;
    symptoms: string[];
    xgb_top: TopDiagnosis[];
    river_top: TopDiagnosis[];
    ensemble_top: TopDiagnosis[];
    meta: {
        conf_x?: number;
        conf_r?: number;
        wx_dyn?: number;
        wr_dyn?: number;
        n_active?: number;
        gated_to?: string;
        fusion?: string;
    };
};

type Props = {
    open: boolean;
    onClose: () => void;
    sessionId: string;
    k?: number;
};

const AURORA =
    "linear-gradient(120deg, rgba(52,211,153,.25), rgba(56,189,248,.25), rgba(167,139,250,.25))";
const AURORA_BAR = "linear-gradient(90deg, #34d399, #38bdf8 55%, #a78bfa)";

function pct(n?: number | null) {
    if (typeof n !== "number" || !isFinite(n)) return 0;
    return Math.round(Math.min(1, Math.max(0, n)) * 100);
}

function ProbBar({ value }: { value: number }) {
    const w = Math.max(0, Math.min(100, value));
    return (
        <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[rgb(var(--border))]">
            <div
                className="h-full rounded-full transition-[width]"
                style={{ width: `${w}%`, background: AURORA_BAR }}
            />
        </div>
    );
}

export default function ModelSplitOverlay({
                                              open,
                                              onClose,
                                              sessionId,
                                              k = 3,
                                          }: Props) {
    const [data, setData] = React.useState<SplitResponse | null>(null);
    const [loading, setLoading] = React.useState(false);
    const [err, setErr] = React.useState<string | null>(null);

    // fokus na "Zamknij" po otwarciu i powrót fokusu do przycisku, który otworzył okno
    const closeRef = React.useRef<HTMLButtonElement | null>(null);
    React.useEffect(() => {
        if (!open) return;
        const prev = document.activeElement as HTMLElement | null;
        const id = requestAnimationFrame(() => closeRef.current?.focus());
        return () => {
            cancelAnimationFrame(id);
            prev?.focus?.();
        };
    }, [open]);

    // ESC zamyka
    React.useEffect(() => {
        if (!open) return;
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [open, onClose]);

    // Ładowanie
    React.useEffect(() => {
        if (!open) return;
        const ac = new AbortController();
        setLoading(true);
        setErr(null);
        (async () => {
            try {
                const res = await getTop3Split(sessionId, k, ac.signal);
                setData(res);
            } catch (e: unknown) {
                setErr((e instanceof Error ? e.message : "") || "Błąd pobierania");
            } finally {
                if (!ac.signal.aborted) setLoading(false);
            }
        })();
        return () => ac.abort();
    }, [open, sessionId, k]);

    if (!open) return null;

    const xgbOff = !!(data && !data.xgb_enabled);
    const riverOff = !!(data && !data.river_enabled);

    return (
        <div
            className="fixed inset-0 z-[100] flex items-center justify-center p-3"
            role="dialog"
            aria-modal="true"
            aria-label="Podgląd predykcji modeli"
        >
            {/* tło zamyka okno (wcześniej przechwytywało kliknięcia i nic nie robiło) */}
            <div className="absolute inset-0 bg-black/40" onClick={onClose} aria-hidden />
            {/* na telefonie okno jest wyższe niż ekran: ograniczamy wysokość i przewijamy treść */}
            <div className="relative z-[101] flex max-h-[calc(100svh-1.5rem)] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-base bg-card/95 backdrop-blur-md shadow-2xl">
                {/* Header */}
                <header className="flex shrink-0 items-center gap-2 border-b border-base bg-card/80 px-4 py-2">
                    <span className="grid h-8 w-8 place-items-center rounded-full border border-base bg-input">
                        <Info className="h-4 w-4 opacity-80" />
                    </span>
                    <h2 className="text-sm font-semibold">
                        Podgląd Top-3: XGBoost, River i Ensemble
                    </h2>
                    <button
                        ref={closeRef}
                        onClick={onClose}
                        className="ml-auto rounded-lg border border-base bg-input p-1.5 text-xs transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                        aria-label="Zamknij"
                        title="Zamknij"
                    >
                        <X className="h-4 w-4" />
                    </button>
                </header>

                {/* Aurora hairline */}
                <div className="h-[3px] w-full shrink-0" style={{ background: AURORA }} />

                <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">

                {/* Trzy kolumny: XGBoost / River / Ensemble */}
                <div className="grid grid-cols-1 gap-3 p-3 md:grid-cols-3">
                    {/* XGBoost */}
                    <section className="rounded-2xl border border-base bg-card p-3 shadow-sm">
                        <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                            <div className="relative grid h-6 w-6 place-items-center rounded-lg border border-base bg-input">
                                <ActivitySquare className="relative h-3.5 w-3.5 opacity-80" />
                            </div>
                            <span>XGBoost (Top-3)</span>
                            {xgbOff ? (
                                <span className="ml-2 rounded-full border border-amber-300/40 bg-amber-50 px-2 py-0.5 text-[10px] text-amber-700 dark:bg-amber-900/20 dark:text-amber-300">
                                    Wyłączony
                                </span>
                            ) : null}
                        </div>
                        {loading ? (
                            <ul className="space-y-2">
                                {Array.from({ length: 3 }).map((_, i) => (
                                    <li
                                        key={i}
                                        className="h-9 animate-pulse rounded-xl border border-base bg-input"
                                    />
                                ))}
                            </ul>
                        ) : err ? (
                            <p className="text-sm text-rose-500">{err}</p>
                        ) : data?.xgb_top?.length ? (
                            <ul className="space-y-2">
                                {data.xgb_top.slice(0, 3).map((d, i) => (
                                    <li
                                        key={i}
                                        className="rounded-xl border border-base bg-input p-2.5"
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="min-w-0 truncate">
                                                {d.diagnosis_pl || d.diagnosis_en}
                                            </div>
                                            <div className="shrink-0 text-sm tabular-nums">
                                                {pct(d.probability)}%
                                            </div>
                                        </div>
                                        <ProbBar value={pct(d.probability)} />
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="text-sm text-muted">
                                {xgbOff ? "Model nieaktywny." : "Brak wyników."}
                            </p>
                        )}
                    </section>

                    {/* River */}
                    <section className="rounded-2xl border border-base bg-card p-3 shadow-sm">
                        <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                            <div className="relative grid h-6 w-6 place-items-center rounded-lg border border-base bg-input">
                                <Sigma className="relative h-3.5 w-3.5 opacity-80" />
                            </div>
                            <span>River (Top-3)</span>
                            {riverOff ? (
                                <span className="ml-2 rounded-full border border-amber-300/40 bg-amber-50 px-2 py-0.5 text-[10px] text-amber-700 dark:bg-amber-900/20 dark:text-amber-300">
                                    Wyłączony
                                </span>
                            ) : null}
                        </div>
                        {loading ? (
                            <ul className="space-y-2">
                                {Array.from({ length: 3 }).map((_, i) => (
                                    <li
                                        key={i}
                                        className="h-9 animate-pulse rounded-xl border border-base bg-input"
                                    />
                                ))}
                            </ul>
                        ) : err ? (
                            <p className="text-sm text-rose-500">{err}</p>
                        ) : data?.river_top?.length ? (
                            <ul className="space-y-2">
                                {data.river_top.slice(0, 3).map((d, i) => (
                                    <li
                                        key={i}
                                        className="rounded-xl border border-base bg-input p-2.5"
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="min-w-0 truncate">
                                                {d.diagnosis_pl || d.diagnosis_en}
                                            </div>
                                            <div className="shrink-0 text-sm tabular-nums">
                                                {pct(d.probability)}%
                                            </div>
                                        </div>
                                        <ProbBar value={pct(d.probability)} />
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="text-sm text-muted">
                                {riverOff ? "Model nieaktywny." : "Brak wyników."}
                            </p>
                        )}
                    </section>

                    {/* Ensemble */}
                    <section className="rounded-2xl border border-base bg-card p-3 shadow-sm">
                        <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                            <div className="relative grid h-6 w-6 place-items-center rounded-lg border border-base bg-input">
                                <ActivitySquare className="relative h-3.5 w-3.5 opacity-80" />
                            </div>
                            <span>Ensemble (Top-3)</span>
                        </div>
                        {loading ? (
                            <ul className="space-y-2">
                                {Array.from({ length: 3 }).map((_, i) => (
                                    <li
                                        key={i}
                                        className="h-9 animate-pulse rounded-xl border border-base bg-input"
                                    />
                                ))}
                            </ul>
                        ) : err ? (
                            <p className="text-sm text-rose-500">{err}</p>
                        ) : data?.ensemble_top?.length ? (
                            <ul className="space-y-2">
                                {data.ensemble_top.slice(0, 3).map((d, i) => (
                                    <li
                                        key={i}
                                        className="rounded-xl border border-base bg-input p-2.5"
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="min-w-0 truncate">
                                                {d.diagnosis_pl || d.diagnosis_en}
                                            </div>
                                            <div className="shrink-0 text-sm tabular-nums">
                                                {pct(d.probability)}%
                                            </div>
                                        </div>
                                        <ProbBar value={pct(d.probability)} />
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="text-sm text-muted">Brak wyników.</p>
                        )}
                    </section>
                </div>

                {/* Opis fuzji — bardzo krótki, prosty, z myślnikami */}
                <div className="border-t border-base bg-input/80 px-4 py-3 text-[13px] leading-relaxed text-[rgb(var(--foreground))]">
                    <p className="mb-1.5 flex items-center gap-2">
                        <span className="inline-flex h-5 w-5 items-center justify-center rounded-md border border-base bg-card">
                            <Sigma className="h-3 w-3 opacity-80" />
                        </span>
                        <span className="font-medium text-basecolor">
                            Jak powstaje wynik łączony?
                        </span>
                    </p>

                    <ul className="mt-1.5 space-y-1.5">
                        <li>
                            – Każdy z modeli osobno układa listę kilku chorób, które
                            najlepiej pasują do podanych objawów.
                        </li>
                        {data?.meta?.gated_to === "learned" ? (
                            <li>
                                – Serwer łączy oba rozkłady stałymi wagami:{" "}
                                <b>{Math.round((data.meta.wx_dyn ?? 0) * 100)}% XGBoost</b> i{" "}
                                <b>{Math.round((data.meta.wr_dyn ?? 0) * 100)}% River</b>. Wagi
                                dobrano na osobnym zbiorze danych tak, żeby podawane
                                prawdopodobieństwa jak najlepiej zgadzały się z rzeczywistością.
                            </li>
                        ) : (
                            <li>
                                – Serwer sprawdza, który model w danej chwili wydaje się
                                pewniejszy i miesza ich propozycje, dając trochę większy
                                wpływ temu bardziej pewnemu, ale wciąż uwzględnia oba.
                            </li>
                        )}
                        <li>
                            – Kolumna „Ensemble” pokazuje wynik po takim połączeniu:
                            choroby, co do których modele się zgadzają, lądują wyżej.
                            Jeśli jeden model nic nie zwróci, brany jest po prostu
                            wynik drugiego.
                        </li>
                    </ul>

                    {data?.meta ? (
                        <p className="mt-2 text-[12px]">
                            <span className="font-medium text-basecolor">
                                Jak czytać liczby poniżej:
                            </span>{" "}
                            udział XGBoost:{" "}
                            <b>{Math.round((data.meta.wx_dyn ?? 0) * 100)}%</b>, udział
                            River: <b>{Math.round((data.meta.wr_dyn ?? 0) * 100)}%</b>;
                            szacowana pewność XGBoost:{" "}
                            {Math.round((data.meta.conf_x ?? 0) * 100)}%, River:{" "}
                            {Math.round((data.meta.conf_r ?? 0) * 100)}%; aktywnych
                            objawów w tej sesji: {data.meta.n_active ?? 0}.
                        </p>
                    ) : null}
                </div>
                </div>
            </div>
        </div>
    );
}

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { healthcheck, getHealthDetailed } from "@/lib/api";
import type { HealthStatus } from "@/lib/api";
import {
    Sun,
    Moon,
    Wifi,
    WifiOff,
    Check,
    Ellipsis,
    Plus,
    Clock,
    ShieldAlert,
} from "lucide-react";
import UserMenu from "@/components/UserMenu";
import Logo from "@/components/Logo";
import { useAuth } from "@/components/AuthProvider"; // używamy tylko isGuest

/* ─────────────────────────── UI helpers ─────────────────────────── */

// mocniejszy, czytelny zielony dla OK
const OK_DOT_CLASSES = "bg-emerald-600 dark:bg-emerald-400";
const OK_CHIP_CLASSES =
    "border-emerald-400/80 bg-emerald-100 text-emerald-900 font-semibold " +
    "dark:border-emerald-400/60 dark:bg-emerald-900/40 dark:text-emerald-200";

function StatusDot({ state }: { state: "ok" | "starting" | "down" }) {
    const cls =
        state === "ok"
            ? OK_DOT_CLASSES
            : state === "starting"
                ? "bg-amber-500"
                : "bg-rose-500";
    return <span className={`inline-block h-2.5 w-2.5 rounded-full ${cls}`} />;
}

/** Płynne i BEZ-migowe przełączanie motywu */
function applyTheme(next: "light" | "dark") {
    const root = document.documentElement;
    root.classList.add("theme-changing");
    if (next === "dark") root.classList.add("dark");
    else root.classList.remove("dark");
    root.style.colorScheme = next;
    localStorage.setItem("theme", next);
    requestAnimationFrame(() => {
        requestAnimationFrame(() => root.classList.remove("theme-changing"));
    });
}

/* mały hook do wykrycia montażu */
function useMounted() {
    const [mounted, setMounted] = useState(false);
    useEffect(() => {
        setMounted(true);
    }, []);
    return mounted;
}

/* Zegar na żywo (PL) – bez SSR-owego czasu (zapobiega hydration mismatch) */
function LiveClock() {
    const mounted = useMounted();

    // SSR/first client render: null => rysujemy stałe placeholdery.
    const [now, setNow] = useState<Date | null>(null);

    useEffect(() => {
        // po zamontowaniu ustawiamy czas i interwał
        setNow(new Date());
        const id = setInterval(() => setNow(new Date()), 1000);
        return () => clearInterval(id);
    }, []);

    // placeholdery deterministyczne (identyczne po stronie serwera i przy 1. renderze klienta)
    if (!mounted || !now) {
        return (
            <span
                className="inline-flex items-center gap-1 rounded-full border border-base bg-card px-2 py-1 text-[11px] shadow-sm"
                title="Zegar systemowy (strefa: lokalna)"
                aria-live="off"
                suppressHydrationWarning
            >
                <Clock className="h-3.5 w-3.5 opacity-80" />
                <span className="hidden sm:inline font-medium">--.--.----</span>
                <span className="hidden sm:inline text-muted">•</span>
                <span className="tabular-nums">--:--:--</span>
            </span>
        );
    }

    // formatowanie dopiero po montażu (już tylko na kliencie)
    const dateStr = new Intl.DateTimeFormat("pl-PL", {
        weekday: "short",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
    }).format(now);

    const timeStr = new Intl.DateTimeFormat("pl-PL", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
    }).format(now);

    return (
        <span
            className="inline-flex items-center gap-1 rounded-full border border-base bg-card px-2 py-1 text-[11px] shadow-sm"
            title="Zegar systemowy (strefa: lokalna)"
            aria-live="off"
            suppressHydrationWarning
        >
            <Clock className="h-3.5 w-3.5 opacity-80" />
            <span className="hidden sm:inline font-medium">{dateStr}</span>
            <span className="hidden sm:inline text-muted">•</span>
            <span className="tabular-nums">{timeStr}</span>
        </span>
    );
}

export default function Header({ onNewChat }: { onNewChat: () => void }) {
    const [online, setOnline] = useState<boolean | null>(null);
    const [theme, setTheme] = useState<"light" | "dark">("light");

    // 🔐 Auth state (gość)
    const { isGuest } = useAuth();

    // Ping „online/offline”
    useEffect(() => {
        (async () => setOnline(await healthcheck()))();
    }, []);

    // Inicjalizacja motywu (tylko klient)
    useEffect(() => {
        const saved = localStorage.getItem("theme");
        const initial: "light" | "dark" = saved === "dark" ? "dark" : "light";
        setTheme(initial);
        applyTheme(initial);
    }, []);

    const setLight = () => {
        setTheme("light");
        applyTheme("light");
    };
    const setDark = () => {
        setTheme("dark");
        applyTheme("dark");
    };

    // Conversation Health
    const [health, setHealth] = useState<HealthStatus | null>(null);
    const [healthError, setHealthError] = useState(false);

    useEffect(() => {
        let alive = true;
        async function poll() {
            try {
                const h = await getHealthDetailed();
                if (!alive) return;
                setHealth(h);
                setHealthError(false);
            } catch {
                if (!alive) return;
                setHealthError(true);
            }
        }
        poll();
        const id = setInterval(poll, 10_000);
        return () => {
            alive = false;
            clearInterval(id);
        };
    }, []);

    const isOk = (v?: boolean) => (v === undefined ? true : !!v);

    // ── Status ogólny: offline ma priorytet nad resztą ──
    const overall: "ok" | "starting" | "down" =
        online === false
            ? "down"
            : healthError
                ? "down"
                : isOk(health?.xgb_ready) &&
                isOk(health?.matcher_ready) &&
                isOk(health?.llm_ready ?? undefined)
                    ? "ok"
                    : "starting";

    const overallLabel =
        online === false
            ? "Offline"
            : overall === "ok"
                ? "OK"
                : overall === "starting"
                    ? "Uruchamianie…"
                    : "Błąd";

    const Badge = ({ label, ready }: { label: string; ready?: boolean }) => (
        <span
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px]
        ${
                ready
                    ? OK_CHIP_CLASSES
                    : "border-[rgb(var(--border))] text-muted bg-card"
            }`}
            title={`${label}: ${ready ? "gotowe" : "uruchamianie…"}`}
        >
            {ready ? (
                <Check className="h-3.5 w-3.5" />
            ) : (
                <Ellipsis className="h-3.5 w-3.5" />
            )}
            {label}
        </span>
    );

    return (
        <header className="border-b border-base bg-card">
            <div className="flex items-center justify-between px-4 py-3 text-basecolor">
                {/* LEWA: logo + brand */}
                <div className="flex min-w-0 select-none items-center gap-3">
                    <div className="shrink-0 rounded-xl bg-gradient-to-br from-emerald-400/60 via-sky-400/60 to-violet-400/60 p-[1px] shadow-sm">
                        <div className="grid h-11 w-11 place-items-center rounded-[11px] border border-base bg-input overflow-hidden">
                            <Logo className="h-9 w-9 object-contain" />
                        </div>
                    </div>

                    <Link href="/" className="group block min-w-0" title="Strona główna">
                        <div className="flex items-center gap-2">
                            <h1 className="truncate text-[16px] font-extrabold tracking-tight">
                                Symptom
                                Checker
                            </h1>
                            <span className="opacity-80">
                                <StatusDot state={overall} />
                            </span>
                        </div>
                        <p className="truncate text-[11px] text-muted">
                            Asystent do wstępnej oceny objawów
                        </p>
                        <div className="mt-1 h-[2px] w-24 rounded-full bg-gradient-to-r from-emerald-400/70 via-sky-400/70 to-violet-400/70 transition-opacity group-hover:opacity-90" />
                    </Link>
                </div>

                {/* ŚRODEK: Pasek statusów (połączony: sieć + komponenty) */}
                <div
                    className="hidden min-w-0 items-center gap-2 rounded-full border border-base bg-input/70 px-2.5 py-1 text-xs shadow-sm backdrop-blur-sm sm:flex"
                    title="Stan połączenia i komponentów"
                >
                    <StatusDot state={overall} />
                    <span className="font-medium">{overallLabel}</span>
                    <span className="mx-1 text-muted">•</span>

                    {/* Sieć */}
                    <span
                        className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] ${
                            online
                                ? OK_CHIP_CLASSES
                                : online === false
                                    ? "border-rose-400/80 bg-rose-100 text-rose-900 font-semibold dark:border-rose-400/60 dark:bg-rose-900/40 dark:text-rose-200"
                                    : "border-[rgb(var(--border))] text-muted bg-card"
                        }`}
                        title={
                            online === null
                                ? "Sprawdzam połączenie…"
                                : online
                                    ? "Połączono"
                                    : "Brak połączenia"
                        }
                    >
                        {online ? (
                            <Wifi className="h-3.5 w-3.5" />
                        ) : online === false ? (
                            <WifiOff className="h-3.5 w-3.5" />
                        ) : (
                            <Ellipsis className="h-3.5 w-3.5" />
                        )}
                        Sieć
                    </span>

                    {/* Komponenty */}
                    <div className="flex min-w-0 flex-wrap items-center gap-1.5">
                        <Badge label="XGB / River" ready={health?.xgb_ready} />
                        <Badge label="Matcher" ready={health?.matcher_ready} />
                        <Badge label="LLM" ready={health?.llm_ready ?? undefined} />
                    </div>
                </div>

                {/* PRAWA: zegar, motyw, nowa rozmowa, informacja o gościu, user */}
                <div className="flex items-center gap-2 sm:gap-3">
                    <LiveClock />

                    <div className="inline-flex overflow-hidden rounded-full border border-base bg-input shadow-sm">
                        <button
                            onClick={setLight}
                            aria-pressed={theme === "light"}
                            className={`group flex items-center gap-1 px-2.5 py-1.5 text-sm transition-all ${
                                theme === "light"
                                    ? "bg-card text-basecolor shadow-[inset_0_0_0_1px_rgba(0,0,0,.03)]"
                                    : "text-muted hover:brightness-110 hover:-translate-y-0.5 active:translate-y-[1px]"
                            }`}
                            title="Motyw jasny"
                        >
                            <Sun
                                className={`h-4 w-4 ${
                                    theme === "light" ? "" : "opacity-70"
                                }`}
                            />
                            <span className="sr-only">Jasny</span>
                        </button>
                        <button
                            onClick={setDark}
                            aria-pressed={theme === "dark"}
                            className={`group flex items-center gap-1 px-2.5 py-1.5 text-sm transition-all ${
                                theme === "dark"
                                    ? "bg-card text-basecolor shadow-[inset_0_0_0_1px_rgba(0,0,0,.03)]"
                                    : "text-muted hover:brightness-110 hover:-translate-y-0.5 active:translate-y-[1px]"
                            }`}
                            title="Motyw ciemny"
                        >
                            <Moon
                                className={`h-4 w-4 ${
                                    theme === "dark" ? "" : "opacity-70"
                                }`}
                            />
                            <span className="sr-only">Ciemny</span>
                        </button>
                    </div>

                    {/* Nowa rozmowa */}
                    <button
                        onClick={onNewChat}
                        className="group relative inline-flex items-center gap-2 rounded-xl border border-base bg-card px-3 py-1.5 text-sm shadow-sm transition-all
                       hover:-translate-y-0.5 hover:shadow-md active:translate-y-[1px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                        title="Nowa sesja"
                    >
                        <span
                            aria-hidden
                            className="pointer-events-none absolute inset-0 -z-10 rounded-[14px] opacity-60 blur-[6px]"
                            style={{
                                background:
                                    "conic-gradient(from 120deg, rgba(16,185,129,.45), rgba(56,189,248,.45), rgba(139,92,246,.45), rgba(16,185,129,.45))",
                            }}
                        />
                        <span
                            aria-hidden
                            className="pointer-events-none absolute inset-0 -z-10 rounded-xl opacity-0 transition-opacity group-hover:opacity-10"
                            style={{ background: "linear-gradient(120deg, #fff, transparent)" }}
                        />
                        <Plus className="h-4 w-4 opacity-90 transition-transform group-hover:rotate-90" />
                        <span className="hidden sm:inline font-medium">Nowa rozmowa</span>
                    </button>

                    {/* 🔔 Informacja o trybie gościa (bez linku/akcji) */}
                    {isGuest && (
                        <div
                            className="hidden md:flex items-center gap-2 rounded-xl border border-amber-400/70 bg-amber-100/90 px-3 py-1.5 text-xs text-amber-900 shadow-sm
                         dark:border-amber-400/40 dark:bg-amber-900/30 dark:text-amber-200"
                            title="Tryb gościa — historia rozmów nie jest zapisywana."
                        >
                            <ShieldAlert className="h-4 w-4" />
                            <span className="font-medium">Tryb gościa</span>
                        </div>
                    )}

                    <UserMenu />
                </div>
            </div>

            {/* Wersja mobilna informacji o gościu – tylko komunikat */}
            {isGuest && (
                <div className="px-4 pb-3 md:hidden">
                    <div
                        className="flex items-center gap-2 rounded-xl border border-amber-400/70 bg-amber-100/90 px-3 py-2 text-xs text-amber-900 shadow-sm
                          dark:border-amber-400/40 dark:bg-amber-900/30 dark:text-amber-200"
                    >
                        <ShieldAlert className="h-4 w-4" />
                        <span className="font-medium">Tryb gościa</span>
                        <span className="text-muted">•</span>
                        <span>Historia nie będzie zapisywana</span>
                    </div>
                </div>
            )}
        </header>
    );
}

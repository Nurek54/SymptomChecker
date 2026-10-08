"use client";

import {
    useEffect,
    useRef,
    useState,
    useMemo,
    useCallback,
    type ElementType,
} from "react";
import { ChatMessage, ChatApiResponse, TopDiagUI } from "@/lib/types";
import { sendChat, errorMessage } from "@/lib/api";
import MessageBubble from "./MessageBubble";
import Header from "./Header";
import { SymptomsCard, TopDiagCard } from "./SidePanel";
import ModelSplitOverlay from "./ModelSplitOverlay";
import {
    ArrowDown,
    ArrowUpRight,
    Stethoscope,
    ClipboardList,
    AlertTriangle,
    Pill,
    ShieldCheck,
    Sparkles,
    ActivitySquare,
    ChevronUp,
    X,
} from "lucide-react";
import { useChatStore } from "@/store/chatStore";

function newSessionId() {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto)
        return crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

// Wyraźniejsza „aurora” (emerald → sky → violet)
const AURORA_SOFT = "linear-gradient(120deg, rgba(52,211,153,.25), rgba(56,189,248,.25), rgba(167,139,250,.25))";
const AURORA_HALO = "linear-gradient(120deg, rgba(52,211,153,.55), rgba(56,189,248,.55), rgba(167,139,250,.55))";

type QuickIntent = { label: string; text: string; icon: ElementType; tint: string };

export default function ChatUI() {
    // === SessionId ===
    const [sessionId, setSessionId] = useState<string>(() => {
        if (typeof window === "undefined") return "web";
        const key = "sc_session_id";
        let id = localStorage.getItem(key);
        if (!id) {
            id = newSessionId();
            localStorage.setItem(key, id);
        }
        return id;
    });

    // Overlay: split modeli
    const [showSplit, setShowSplit] = useState(false);

    // Telefon: panel objawów i Top-3 w wysuwanym arkuszu (na desktopie jest prawa kolumna)
    const [showPanel, setShowPanel] = useState(false);

    // Wąski ekran: krótszy placeholder (podpowiedź o Enter/Shift+Enter dotyczy klawiatury)
    const [narrow, setNarrow] = useState(false);
    useEffect(() => {
        const mq = window.matchMedia("(max-width: 639px)");
        const sync = () => setNarrow(mq.matches);
        sync();
        mq.addEventListener("change", sync);
        return () => mq.removeEventListener("change", sync);
    }, []);
    // arkusz: fokus na "Zamknij" po otwarciu, po zamknięciu wraca na pasek "Szczegóły"
    const sheetCloseRef = useRef<HTMLButtonElement | null>(null);
    useEffect(() => {
        if (!showPanel) return;
        const prev = document.activeElement as HTMLElement | null;
        const id = requestAnimationFrame(() => sheetCloseRef.current?.focus());
        return () => {
            cancelAnimationFrame(id);
            prev?.focus?.();
        };
    }, [showPanel]);
    useEffect(() => {
        if (!showPanel) return;
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") setShowPanel(false);
        };
        const mq = window.matchMedia("(min-width: 768px)");
        const onMq = () => mq.matches && setShowPanel(false);
        window.addEventListener("keydown", onKey);
        mq.addEventListener("change", onMq);
        return () => {
            window.removeEventListener("keydown", onKey);
            mq.removeEventListener("change", onMq);
        };
    }, [showPanel]);

    // Reakcja na logout: nadaj nową sesję i przewiń na dół (pustej listy)
    useEffect(() => {
        const onLogout = () => {
            const id = newSessionId();
            try {
                localStorage.setItem("sc_session_id", id);
            } catch {}
            setSessionId(id);
            setInput("");
            requestAnimationFrame(() => scrollToEnd(false));
        };
        window.addEventListener("sc:logout", onLogout as EventListener);
        return () => window.removeEventListener("sc:logout", onLogout as EventListener);
    }, []);

    // === Store: wiadomości + streszczenia (symptoms/top) ===
    const storedMsgs = useChatStore((s) => s.bySession[sessionId]);
    const addMsg = useChatStore((s) => s.addMessage);
    const setSummary = useChatStore((s) => s.setSummary);
    const summary = useChatStore((s) => s.summary[sessionId]);

    const EMPTY_LIST = useMemo(
        () => [] as { role: "user" | "assistant"; content: string; createdAt: string }[],
        []
    );

    const messages: ChatMessage[] = useMemo(
        () =>
            (storedMsgs ?? EMPTY_LIST).map((m) => ({
                role: m.role,
                content: m.content,
                createdAt: m.createdAt,
            })),
        [storedMsgs, EMPTY_LIST]
    );

    // Powitanie tylko raz
    useEffect(() => {
        const tid = setTimeout(() => {
            const arr = (useChatStore.getState().bySession[sessionId] ?? []) as typeof EMPTY_LIST;
            if (arr.length === 0) {
                addMsg(sessionId, {
                    role: "assistant",
                    content: "Cześć! Opisz proszę swoje objawy…",
                    createdAt: new Date().toISOString(),
                });
            }
        }, 100); // 100ms na hydrację persist'a; jest synchronous dla localStorage,
                 // ale w Strict Mode efekty mogą reentrować — opóźnienie chroni.
        return () => clearTimeout(tid);
    }, [sessionId, addMsg]);

    // === Lokalne ===
    const [input, setInput] = useState("");
    const [busy, setBusy] = useState(false);

    const scrollRef = useRef<HTMLDivElement>(null);
    const composerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const [showScrollBtn, setShowScrollBtn] = useState(false);

    // dynamiczna wysokość kompozytora (żeby przycisk był zawsze nad nim)
    const [composerH, setComposerH] = useState<number>(84); // sensowny default

    useEffect(() => {
        const el = composerRef.current;
        if (!el) return;

        const measure = () => setComposerH(el.offsetHeight || 84);
        measure();

        let ro: ResizeObserver | undefined;
        if (typeof window !== "undefined" && typeof window.ResizeObserver !== "undefined") {
            ro = new window.ResizeObserver(measure);
            ro.observe(el);
        }

        const onResize = () => measure();
        window.addEventListener("resize", onResize);

        return () => {
            ro?.disconnect();
            window.removeEventListener("resize", onResize);
        };
    }, []);

    // poprzednia długość listy (do wykrycia nowych wiadomości)
    const prevLenRef = useRef<number>(0);

    const isNearBottom = useCallback(
        (el: HTMLDivElement | null, threshold = 60) => {
            if (!el) return true;
            return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
        },
        []
    );

    const updateScrollBtn = useCallback(() => {
        const el = scrollRef.current;
        if (!el) return;
        setShowScrollBtn(!isNearBottom(el));
    }, [isNearBottom]);

    // ── Quick intents ──
    const quickIntents: QuickIntent[] = useMemo(
        () => [
            {
                label: "Choroba?",
                text: "Na podstawie moich objawów powiedz, jaka choroba jest najbardziej prawdopodobna i dlaczego.",
                icon: Stethoscope,
                tint: "text-emerald-400",
            },
            {
                label: "Zalecenia",
                text: "Proszę podaj mi konkretne zalecenia.",
                icon: ClipboardList,
                tint: "text-cyan-400",
            },
            {
                label: "Pilność",
                text: "Oceń pilność mojej sytuacji i powiedz, czy powinienem iść do lekarza.",
                icon: AlertTriangle,
                tint: "text-amber-400",
            },
            {
                label: "Leki OTC",
                text: "Jakie leki bez recepty mogą pomóc? Proszę o dawki i przeciwwskazania.",
                icon: Pill,
                tint: "text-cyan-400",
            },
            {
                label: "Profilaktyka",
                text: "Jak mogę zapobiegać pogorszeniu i jakie środki domowe polecacie?",
                icon: ShieldCheck,
                tint: "text-cyan-400",
            },
            {
                label: "Podsumuj",
                text: "Streść dotychczasową rozmowę w punktach i zaproponuj następny krok.",
                icon: Sparkles,
                tint: "text-violet-400",
            },
        ],
        []
    );

    const abortRef = useRef<AbortController | null>(null);

    // Anuluj trwające żądanie gdy komponent się odmonowuje albo zmienia się sesja.
    useEffect(() => {
        return () => {
            abortRef.current?.abort();
            abortRef.current = null;
        };
    }, [sessionId]);

    // Wspólna wysyłka
    const sendText = useCallback(
        async (text: string) => {
            const trimmed = text.trim();
            if (!trimmed) return;

            // v7: gdy backend wciąż liczy poprzednią odpowiedź — informujemy użytkownika.
            // Wcześniej cichaczem odrzucało (return bez śladu), więc użytkownik nie wiedział,
            // dlaczego druga wiadomość "nie zadziałała". Teraz pokazujemy assistantowi
            // krótki komunikat, żeby było widać, że żądanie zostało zauważone.
            if (busy) {
                addMsg(sessionId, {
                    role: "user",
                    content: trimmed,
                    createdAt: new Date().toISOString(),
                });
                addMsg(sessionId, {
                    role: "assistant",
                    content:
                        "Jeszcze pracuję nad poprzednią odpowiedzią. Daj mi chwilę i ponów pytanie, " +
                        "gdy zobaczysz, że skończyłem (znak ładowania zniknie).",
                    createdAt: new Date().toISOString(),
                });
                return;
            }

            addMsg(sessionId, {
                role: "user",
                content: trimmed,
                createdAt: new Date().toISOString(),
            });
            setBusy(true);

            abortRef.current?.abort(); // jakby coś zostało z poprzedniej iteracji
            const ctrl = new AbortController();
            abortRef.current = ctrl;

            const SAFETY_MS = 200_000; // 3:20 — trochę więcej niż timeout w apiFetch (180s)
            const safetyTimer = setTimeout(() => {
                if (abortRef.current === ctrl) {
                    ctrl.abort();
                }
            }, SAFETY_MS);

            try {
                const knownSymptoms = useChatStore.getState().summary[sessionId]?.symptoms ?? [];
                const data: ChatApiResponse = await sendChat(trimmed, sessionId, ctrl.signal, knownSymptoms);
                const reply = data.response || "(brak odpowiedzi)";

                addMsg(sessionId, {
                    role: "assistant",
                    content: String(reply),
                    createdAt: new Date().toISOString(),
                });

                const mappedTop: TopDiagUI[] = Array.isArray(data.top_diagnoses)
                    ? data.top_diagnoses.slice(0, 3).map((d) => ({
                        diagnosis_en: d.diagnosis_en ?? "",
                        probability: d.probability ?? 0,
                        diagnosis_pl: d.diagnosis_pl ?? null,
                    }))
                    : [];

                setSummary(sessionId, {
                    symptoms: Array.isArray(data.symptoms) ? data.symptoms : [],
                    topDiag: mappedTop,
                });
            } catch (e: unknown) {
                const msg = errorMessage(e);
                const isAbort =
                    (e as { name?: string } | null)?.name === "AbortError" ||
                    msg.toLowerCase().includes("timeout") ||
                    msg.toLowerCase().includes("anulowane");
                addMsg(sessionId, {
                    role: "assistant",
                    content: isAbort
                        ? "Backend nie odpowiedział na czas. Spróbuj jeszcze raz. Jeśli powtórzy się kilka razy, sprawdź logi serwera."
                        : `Niestety, coś poszło nie tak: ${msg || "nieznany błąd"}`,
                    createdAt: new Date().toISOString(),
                });
            } finally {
                clearTimeout(safetyTimer);
                if (abortRef.current === ctrl) {
                    abortRef.current = null;
                }
                setBusy(false);
            }
        },
        [busy, sessionId, addMsg, setSummary]
    );

    const send = async () => {
        const trimmed = input.trim();
        if (!trimmed || busy) return;
        setInput("");
        autosize();
        await sendText(trimmed);
    };

    // ——— Auto-scroll na sam dół PRZY KAŻDEJ nowej wiadomości ———
    useEffect(() => {
        const added = messages.length > prevLenRef.current;
        prevLenRef.current = messages.length;
        if (added) {
            requestAnimationFrame(() => {
                scrollToEnd(true);
                updateScrollBtn();
            });
        }
    }, [messages, updateScrollBtn]);

    // Aktualizuj widoczność przycisku na: scroll i resize
    const onScroll = () => updateScrollBtn();
    useEffect(() => {
        updateScrollBtn();
    }, [busy, updateScrollBtn]);
    useEffect(() => {
        const handler = () => updateScrollBtn();
        window.addEventListener("resize", handler);
        return () => window.removeEventListener("resize", handler);
    }, [updateScrollBtn]);

    // Stabilizacja scrolla: obserwuj zmiany wymiarów listy
    useEffect(() => {
        const el = scrollRef.current;
        if (!el) return;

        let ro: ResizeObserver | undefined;
        if (typeof window !== "undefined" && typeof window.ResizeObserver !== "undefined") {
            ro = new window.ResizeObserver(() => {
                if (isNearBottom(el)) scrollToEnd(false);
                updateScrollBtn();
            });
            ro?.observe(el);
        }

        return () => {
            ro?.disconnect();
        };
    }, [isNearBottom, updateScrollBtn]);

    const scrollToEnd = (smooth = false) => {
        const el = scrollRef.current;
        if (!el) return;
        const top = el.scrollHeight;
        el.scrollTo({ top, behavior: smooth ? "smooth" : "auto" });
    };

    const autosize = () => {
        const ta = inputRef.current;
        if (!ta) return;
        ta.style.height = "auto";
        const max = 160;
        ta.style.height = Math.min(ta.scrollHeight, max) + "px";
    };

    const newChat = () => {
        const id = newSessionId();
        localStorage.setItem("sc_session_id", id);
        setSessionId(id);
        setInput("");
        requestAnimationFrame(() => scrollToEnd());
    };

    const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    };

    const isEmptyState =
        messages.length <= 1 && (!messages[0] || messages[0].role === "assistant");

    // Lista przykładowych otwarć — short, conversational, klikalne wstawiają
    // tekst od razu do inputu (nie wysyła, user może edytować).
    const SUGGESTED_PROMPTS = [
        "Boli mnie głowa od 2 dni",
        "Mam gorączkę i kaszel",
        "Wysypka na ramieniu",
        "Ból brzucha po posiłku",
    ];

    // Zawartość panelu (prawa kolumna na desktopie, arkusz na telefonie)
    const renderPanel = () => (
        <div className="flex flex-col gap-3">
                <SymptomsCard
                    sessionId={sessionId}
                    symptoms={summary?.symptoms ?? []}
                    onPatched={(res) => {
                        const mapped: TopDiagUI[] = (res.top_diagnoses ?? [])
                            .slice(0, 3)
                            .map((d) => ({
                                diagnosis_en: d.diagnosis_en ?? "",
                                probability: d.probability ?? 0,
                                diagnosis_pl: d.diagnosis_pl ?? null,
                            }));
                        setSummary(sessionId, {
                            symptoms: res.symptoms,
                            topDiag: mapped,
                        });
                    }}
                    lang="pl"
                />
                <TopDiagCard
                    items={summary?.topDiag ?? null}
                    onInfo={() => {
                        setShowPanel(false);
                        setShowSplit(true);
                    }}
                />
        </div>
    );

    const symptomCount = summary?.symptoms?.length ?? 0;
    const top1 = summary?.topDiag?.[0] ?? null;

    // ====== RENDER ======
    return (
        <div className="flex h-full w-full flex-col overflow-hidden bg-surface text-basecolor">
            <Header onNewChat={newChat} />

            <div className="flex min-h-0 flex-1">
                {/* LEWA kolumna */}
                <section className="relative min-h-0 min-w-0 flex-1 border-base md:border-r">
                    {/* Subtelna pionowa poświata */}
                    <span
                        aria-hidden
                        className="pointer-events-none absolute inset-y-0 right-0 hidden w-px opacity-60 md:block"
                        style={{ background: AURORA_SOFT }}
                    />
                    <div className="flex h-full min-h-0 flex-col">
                        {/* Telefon: skrót objawów i Top-1, otwiera arkusz z pełnym panelem */}
                        <button
                            type="button"
                            onClick={() => setShowPanel(true)}
                            className="flex w-full shrink-0 items-center gap-2 border-b border-base bg-card px-4 py-2 text-left text-xs md:hidden"
                            aria-haspopup="dialog"
                            aria-expanded={showPanel}
                            aria-controls="mobile-panel"
                        >
                            <ActivitySquare className="h-4 w-4 shrink-0 opacity-70" />
                            <span className="shrink-0 font-semibold">Objawy: {symptomCount}</span>
                            <span className="text-muted">•</span>
                            <span className="min-w-0 flex-1 truncate text-muted">
                                {top1
                                    ? `${top1.diagnosis_pl || top1.diagnosis_en} ${Math.round((top1.probability ?? 0) * 100)}%`
                                    : "Top-3 pojawi się po ocenie"}
                            </span>
                            <span className="inline-flex shrink-0 items-center gap-0.5 font-medium">
                                Szczegóły
                                <ChevronUp className="h-3.5 w-3.5" />
                            </span>
                        </button>

                        {/* lista wiadomości */}
                        <div
                            ref={scrollRef}
                            onScroll={onScroll}
                            className="relative flex-1 space-y-4 overflow-y-auto px-4 py-4"
                            style={{
                                scrollbarGutter: "stable both-edges",
                                overscrollBehavior: "contain",
                                overflowAnchor: "none",
                            }}
                        >
                            {messages.map((m, i) => (
                                <MessageBubble
                                    key={i}
                                    role={m.role}
                                    content={m.content}
                                    createdAt={m.createdAt}
                                />
                            ))}

                            {}
                            {isEmptyState && !busy && (
                                <div className="ml-0 sm:ml-10 mt-1 flex flex-wrap gap-2 animate-in fade-in duration-500">
                                    {SUGGESTED_PROMPTS.map((p) => (
                                        <button
                                            key={p}
                                            onClick={() => {
                                                setInput(p);
                                                requestAnimationFrame(() => {
                                                    inputRef.current?.focus();
                                                    autosize();
                                                });
                                            }}
                                            className="group relative inline-flex items-center gap-1.5 rounded-full border border-base bg-card px-3 py-1.5 text-xs text-muted transition-all hover:border-emerald-400/40 hover:bg-input hover:text-basecolor focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/40"
                                            title="Kliknij, aby wstawić do pola"
                                        >
                                            <span className="text-[10px] opacity-50">→</span>
                                            {p}
                                        </button>
                                    ))}
                                </div>
                            )}

                            {}
                            {busy && (
                                <div className="flex w-full justify-start">
                                    <div className="mr-2 mt-0.5 hidden shrink-0 sm:block">
                                        <span
                                            className="relative grid h-8 w-8 place-items-center rounded-full border border-base shadow-sm"
                                            style={{
                                                background:
                                                    "linear-gradient(135deg, rgba(52,211,153,.18), rgba(56,189,248,.18) 55%, rgba(167,139,250,.18))",
                                            }}
                                            aria-hidden
                                        >
                                            <Stethoscope className="h-4 w-4 text-emerald-500 dark:text-emerald-300" />
                                        </span>
                                    </div>
                                    <div className="rounded-2xl rounded-bl-md border border-base bg-card px-4 py-3 shadow-sm">
                                        <span className="flex items-end gap-1" aria-label="Asystent pisze…">
                                            <span
                                                className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-400/80"
                                                style={{ animation: "sc-typing 1.4s ease-in-out infinite" }}
                                            />
                                            <span
                                                className="inline-block h-1.5 w-1.5 rounded-full bg-cyan-400/80"
                                                style={{
                                                    animation: "sc-typing 1.4s ease-in-out infinite",
                                                    animationDelay: "0.2s",
                                                }}
                                            />
                                            <span
                                                className="inline-block h-1.5 w-1.5 rounded-full bg-violet-400/80"
                                                style={{
                                                    animation: "sc-typing 1.4s ease-in-out infinite",
                                                    animationDelay: "0.4s",
                                                }}
                                            />
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* KOMPOZYTOR */}
                        <div
                            ref={composerRef}
                            className="relative border-base border-t p-3"
                        >
              <span
                  aria-hidden
                  className="pointer-events-none absolute -top-px left-0 right-0 h-px"
                  style={{ background: AURORA_SOFT }}
              />

                            {/* QUICK INTENTS */}
                            <div className="mb-2 w-full overflow-hidden pb-[6px]">
                                <div className="-mb-[6px] w-full overflow-x-auto overflow-y-hidden">
                                    <div className="mx-auto flex w-max min-w-full justify-start sm:justify-center">
                                        <div className="inline-flex items-center gap-1.5 whitespace-nowrap px-1">
                                            {quickIntents.map(({ label, text, icon: Icon, tint }) => (
                                                <button
                                                    key={label}
                                                    onClick={() => sendText(text)}
                                                    className="group relative inline-flex h-8 items-center gap-1.5 overflow-visible rounded-full border border-base bg-input px-3 text-[11px] font-medium text-basecolor shadow-sm transition-colors hover:border-neutral-300 hover:bg-card focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                                                    title={text}
                                                >
                          <span
                              aria-hidden
                              className="absolute -inset-1 z-0 rounded-full opacity-0 blur-md mix-blend-screen transition-opacity group-hover:opacity-90 group-focus-visible:opacity-90"
                              style={{ background: AURORA_HALO }}
                          />
                                                    <span
                                                        aria-hidden
                                                        className="absolute -inset-px z-0 rounded-full opacity-0 blur-[1px] transition-opacity group-hover:opacity-90 group-focus-visible:opacity-90"
                                                        style={{ background: AURORA_SOFT }}
                                                    />
                                                    <span className="relative z-10 inline-flex items-center gap-1.5 leading-none">
                            <Icon className={`h-4 w-4 shrink-0 -mt-[1px] ${tint}`} />
                                                        {label}
                          </span>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-end gap-2">
                                {/* INPUT */}
                                <div className="relative flex-1">
                  <span
                      aria-hidden
                      className="pointer-events-none absolute inset-0 -z-10 rounded-xl opacity-0 transition-opacity peer-focus:opacity-100"
                      style={{ background: AURORA_SOFT }}
                  />
                                    <textarea
                                        ref={inputRef}
                                        placeholder={
                                            narrow
                                                ? "Napisz swoje objawy…"
                                                : "Napisz swoje objawy… (Enter = wyślij, Shift+Enter = nowa linia)"
                                        }
                                        value={input}
                                        onChange={(e) => {
                                            setInput(e.target.value);
                                            autosize();
                                        }}
                                        onKeyDown={onKey}
                                        rows={2}
                                        className="peer min-h-[44px] max-h-40 w-full resize-none rounded-xl border border-base bg-input px-3 py-2 text-sm outline-none transition-all focus:bg-card focus:ring-2 focus:ring-neutral-400 hover:border-neutral-300"
                                    />
                                </div>

                                {/* SEND */}
                                <button
                                    onClick={send}
                                    disabled={busy || !input.trim()}
                                    className="btn-primary group relative inline-flex items-center gap-2 overflow-visible rounded-xl px-4 py-2 text-sm font-medium shadow-sm transition-all hover:shadow-md active:translate-y-[1px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300 disabled:cursor-not-allowed"
                                >
                  <span
                      aria-hidden
                      className="absolute -inset-1 z-0 rounded-xl opacity-0 blur-md mix-blend-screen transition-opacity group-hover:opacity-90"
                      style={{ background: AURORA_HALO }}
                  />
                                    <span className="relative z-10 inline-flex items-center gap-2">
                    Wyślij
                    <ArrowUpRight className="h-4 w-4 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
                  </span>
                                </button>
                            </div>

                            <p className="mt-2 text-center text-xs text-muted md:text-left">
                                To narzędzie ma charakter edukacyjny i nie zastępuje porady
                                medycznej.
                                {/* na telefonie podpis autora w tej linii (stały podpis z globals.css nachodziłby na tekst) */}
                                <span className="aurora-text ml-1 whitespace-nowrap font-semibold md:hidden">
                                    Autor: Grzegorz Sienkiewicz
                                </span>
                            </p>
                        </div>
                    </div>

                    {/* ⬇️ PRZYCISK „NA DÓŁ” */}
                    {showScrollBtn && (
                        <div
                            className="pointer-events-none absolute right-4 z-20"
                            style={{ bottom: (composerH ?? 84) + 12 }}
                        >
                            <button
                                onClick={() => scrollToEnd(true)}
                                className="group relative pointer-events-auto inline-flex items-center gap-1 overflow-visible rounded-full border border-base bg-card/95 px-3 py-1.5 text-xs shadow-md transition-colors hover:bg-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                                title="Przewiń na dół"
                                aria-label="Przewiń na dół"
                            >
                <span
                    aria-hidden
                    className="absolute -inset-1 z-0 rounded-full opacity-80 blur-md mix-blend-screen transition-opacity group-hover:opacity-100"
                    style={{ background: AURORA_HALO }}
                />
                                <span className="relative z-10 inline-flex items-center gap-1 leading-none">
                  <ArrowDown className="h-4 w-4 -mt-[1px]" /> Dół
                </span>
                            </button>
                        </div>
                    )}
                </section>

                {/* PRAWA kolumna (bez scrolla) */}
                <aside className="hidden h-full w-[440px] shrink-0 overflow-hidden md:block">
                    <div className="h-full p-3">{renderPanel()}</div>
                </aside>
            </div>

            {/* Telefon: arkusz z panelem objawów i Top-3 */}
            {showPanel && (
                <div className="fixed inset-0 z-50 md:hidden" role="dialog" aria-modal="true" aria-label="Objawy i Top-3" id="mobile-panel">
                    <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" onClick={() => setShowPanel(false)} aria-hidden />
                    <div
                        className="absolute inset-x-0 bottom-0 flex max-h-[88svh] flex-col rounded-t-2xl border-t border-base bg-surface shadow-2xl"
                        style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
                    >
                        <div className="flex items-center justify-between px-4 pb-1 pt-3">
                            <span className="text-sm font-semibold">Objawy i Top-3</span>
                            <button
                                ref={sheetCloseRef}
                                type="button"
                                onClick={() => setShowPanel(false)}
                                className="rounded-lg border border-base bg-card p-1.5 transition hover:bg-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
                                aria-label="Zamknij"
                                title="Zamknij"
                            >
                                <X className="h-4 w-4" />
                            </button>
                        </div>
                        <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-4 pt-1">{renderPanel()}</div>
                    </div>
                </div>
            )}

            {/* Overlay: podgląd Top-3 XGB vs River */}
            <ModelSplitOverlay
                open={showSplit}
                onClose={() => setShowSplit(false)}
                sessionId={sessionId}
            />
        </div>
    );
}

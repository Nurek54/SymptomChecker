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
import { sendChat } from "@/lib/api";
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
} from "lucide-react";
import Spinner from "./Spinner";
import { useChatStore } from "@/store/chatStore";

function newSessionId() {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto)
        return crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

// Wyraźniejsza „aurora” (emerald → sky → violet)
const AURORA = "linear-gradient(120deg, rgba(52,211,153,.35), rgba(56,189,248,.35), rgba(167,139,250,.35))";
const AURORA_SOFT = "linear-gradient(120deg, rgba(52,211,153,.25), rgba(56,189,248,.25), rgba(167,139,250,.25))";
const AURORA_HALO = "linear-gradient(120deg, rgba(52,211,153,.55), rgba(56,189,248,.55), rgba(167,139,250,.55))";

type QuickIntent = { label: string; text: string; icon: ElementType };

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
            })),
        [storedMsgs, EMPTY_LIST]
    );

    // Powitanie tylko raz
    useEffect(() => {
        const arr = (useChatStore.getState().bySession[sessionId] ?? []) as typeof EMPTY_LIST;
        if (arr.length === 0) {
            addMsg(sessionId, {
                role: "assistant",
                content: "Cześć! Opisz proszę swoje objawy…",
                createdAt: new Date().toISOString(),
            });
        }
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
        if (typeof window !== "undefined" && (window as any).ResizeObserver) {
            ro = new (window as any).ResizeObserver(measure);
            ro?.observe(el);
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
            },
            {
                label: "Zalecenia",
                text: "Proszę podaj mi konkretne zalecenia.",
                icon: ClipboardList,
            },
            {
                label: "Pilność",
                text: "Oceń pilność mojej sytuacji i powiedz, czy powinienem iść do lekarza.",
                icon: AlertTriangle,
            },
            {
                label: "Leki OTC",
                text: "Jakie leki bez recepty mogą pomóc? Proszę o dawki i przeciwwskazania.",
                icon: Pill,
            },
            {
                label: "Profilaktyka",
                text: "Jak mogę zapobiegać pogorszeniu i jakie środki domowe polecacie?",
                icon: ShieldCheck,
            },
            {
                label: "Podsumuj",
                text: "Streść dotychczasową rozmowę w punktach i zaproponuj następny krok.",
                icon: Sparkles,
            },
        ],
        []
    );

    // Wspólna wysyłka
    const sendText = useCallback(
        async (text: string) => {
            const trimmed = text.trim();
            if (!trimmed || busy) return;

            addMsg(sessionId, {
                role: "user",
                content: trimmed,
                createdAt: new Date().toISOString(),
            });
            setBusy(true);

            try {
                const data: ChatApiResponse = await sendChat(trimmed, sessionId);
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
            } catch (e: any) {
                addMsg(sessionId, {
                    role: "assistant",
                    content: `Niestety, coś poszło nie tak: ${
                        e?.message || "nieznany błąd"
                    }`,
                    createdAt: new Date().toISOString(),
                });
            } finally {
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
        if (typeof window !== "undefined" && (window as any).ResizeObserver) {
            ro = new (window as any).ResizeObserver(() => {
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
                        {/* lista wiadomości */}
                        <div
                            ref={scrollRef}
                            onScroll={onScroll}
                            className="relative flex-1 space-y-4 overflow-y-auto px-4 py-4"
                            style={{
                                scrollbarGutter: "stable both-edges",
                                overscrollBehavior: "contain",
                                overflowAnchor: "none" as any,
                            }}
                        >
                            {messages.map((m, i) => (
                                <MessageBubble
                                    key={i}
                                    role={m.role}
                                    content={m.content}
                                />
                            ))}

                            {busy && (
                                <div className="flex items-center gap-2 text-sm text-muted">
                  <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ background: AURORA }}
                  />
                                    <Spinner /> asystent myśli…
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
                                    <div className="min-w-full flex justify-center">
                                        <div className="inline-flex items-center gap-1.5 whitespace-nowrap px-1">
                                            {quickIntents.map(({ label, text, icon: Icon }) => (
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
                            <Icon className="h-4 w-4 shrink-0 -mt-[1px]" />
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
                                        placeholder="Napisz swoje objawy… (Enter = wyślij, Shift+Enter = nowa linia)"
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
                    <div className="h-full p-3">
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
                                items={summary?.topDiag ?? (null as any)}
                                onInfo={() => setShowSplit(true)}
                            />
                        </div>
                    </div>
                </aside>
            </div>

            {/* Overlay: podgląd Top-3 XGB vs River */}
            <ModelSplitOverlay
                open={showSplit}
                onClose={() => setShowSplit(false)}
                sessionId={sessionId}
            />
        </div>
    );
}

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

export type ChatMessageUI = {
    role: "user" | "assistant";
    content: string;
    createdAt: string; // ISO
};

export type TopDiagUI = {
    diagnosis_en: string;
    diagnosis_pl: string | null;
    probability: number;
};

type Summary = {
    symptoms: string[];
    topDiag: TopDiagUI[] | null;
};

type ChatState = {
    bySession: Record<string, ChatMessageUI[]>;
    summary: Record<string, Summary>;

    // ── akcje ──
    addMessage: (sessionId: string, msg: ChatMessageUI) => void;
    setSummary: (sessionId: string, part: Partial<Summary>) => void;
    clearSession: (sessionId: string) => void;
    clearSummary: (sessionId: string) => void;

    // FIX #16: AuthProvider wołał te metody — wcześniej ich nie było, więc
    // wszystkie wywołania po cichu robiły fallback. Teraz są na stałe.
    resetActive: () => void;
    clearAll: () => void;
};

/**
 * FIX #15: ujednolicenie storage'a.
 *
 * Wcześniej chatStore używał `sessionStorage` (czat znikał po zamknięciu zakładki),
 * a inne miejsca w aplikacji trzymały stan w `localStorage` (sc_session_id, sc_guest).
 * Konsekwencja: sesja istniała, ale wiadomości znikały — bardzo psuje UX.
 *
 * Decyzja: przechodzimy na `localStorage` dla całości, żeby było spójne.
 * Jeżeli kiedykolwiek będziesz chciał trzymać czat tylko w obrębie zakładki,
 * przełącz `useChatStore` z powrotem na sessionStorage — ale wtedy wyrzuć też
 * `sc_session_id` z localStorage, bo inaczej znów się rozjedzie.
 */
export const useChatStore = create<ChatState>()(
    persist(
        (set) => ({
            bySession: {},
            summary: {},

            addMessage: (sessionId, msg) =>
                set((s) => {
                    const list = s.bySession[sessionId] ?? [];
                    const last = list[list.length - 1];
                    if (last && last.role === msg.role && last.content === msg.content) return s;
                    return { bySession: { ...s.bySession, [sessionId]: [...list, msg] } };
                }),

            setSummary: (sessionId, part) =>
                set((s) => {
                    const prev: Summary = s.summary[sessionId] ?? { symptoms: [], topDiag: null };
                    const next: Summary = {
                        symptoms: part.symptoms ? [...part.symptoms] : prev.symptoms,
                        topDiag: part.topDiag ? [...part.topDiag] : prev.topDiag,
                    };
                    return { summary: { ...s.summary, [sessionId]: next } };
                }),

            clearSession: (sessionId) =>
                set((s) => ({ bySession: { ...s.bySession, [sessionId]: [] } })),

            clearSummary: (sessionId) =>
                set((s) => {
                    const copy = { ...s.summary };
                    delete copy[sessionId];
                    return { summary: copy };
                }),

            resetActive: () => {
                // bezpieczny no-op: nie znamy aktywnej sesji w storze, ale
                // AuthProvider używa tej metody jako sygnału "zacznij od zera"
                // — czyścimy całość, taniej i pewniej.
                set({ bySession: {}, summary: {} });
            },

            clearAll: () => {
                set({ bySession: {}, summary: {} });
            },
        }),
        {
            name: "sc-chat",
            // FIX #15: localStorage zamiast sessionStorage
            storage: createJSONStorage(() =>
                typeof window !== "undefined" ? window.localStorage : (undefined as any)
            ),
            partialize: (s) => ({ bySession: s.bySession, summary: s.summary }),
            version: 2,
            migrate: (persisted: any, fromVersion: number) => {
                // Przy migracji ze starego sessionStorage → nowego localStorage
                // nie mamy dostępu do starych danych w obrębie tego storage'a.
                // Zostawiamy puste — użytkownik dostanie pusty stan jednorazowo.
                if (fromVersion < 2 || !persisted) {
                    return { bySession: {}, summary: {} };
                }
                return persisted;
            },
        }
    )
);

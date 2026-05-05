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
  addMessage: (sessionId: string, msg: ChatMessageUI) => void;
  setSummary: (sessionId: string, part: Partial<Summary>) => void;
  clearSession: (sessionId: string) => void;
  clearSummary: (sessionId: string) => void;
};

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
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
    }),
    {
      name: "sc-chat",
      storage: createJSONStorage(() => sessionStorage),
      partialize: (s) => ({ bySession: s.bySession, summary: s.summary }),
    }
  )
);

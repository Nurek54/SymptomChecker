// src/lib/types.ts

export type Role = "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
  createdAt: string; // 👈 Dodane pole, aby komponenty UI (np. MessageBubble, ChatUI) miały dostęp do daty utworzenia wiadomości
}

export interface TopDiagnosis {
  diagnosis_en: string;
  probability: number;
  diagnosis_pl?: string | null; // opcjonalne z backendu
}

export interface ChatApiResponse {
  response: string;
  symptoms: string[];
  diagnosis_en?: string | null;
  diagnosis_pl?: string | null;
  top_diagnoses?: TopDiagnosis[] | null;
}

/** UI-owy, „twardy” typ do prawego panelu — jeśli potrzebujesz */
export type TopDiagUI = {
  diagnosis_en: string;
  probability: number;
  diagnosis_pl: string | null; // zawsze string|null (nie undefined)
};

/** Słownik symptomów dla pickera (EN + etykieta PL) */
export type SymptomDictItem = {
  en: string;
  pl: string;
  category?: string | null;
  /** inne etykiety PL tej samej cechy (np. "ból języka" przy "ból jamy ustnej") */
  aliases?: string[];
};

/** Odpowiedź po patchu objawów w sesji */
export type SymptomsPatchResponse = {
  symptoms: string[];          // EN canonical
  top_diagnoses: TopDiagnosis[];
  diagnosis_en?: string | null;
  diagnosis_pl?: string | null;
};
export type Role = "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
  createdAt: string;
}

export interface TopDiagnosis {
  diagnosis_en: string;
  probability: number;
  diagnosis_pl?: string | null;
}

export interface ChatApiResponse {
  response: string;
  symptoms: string[];
  diagnosis_en?: string | null;
  diagnosis_pl?: string | null;
  top_diagnoses?: TopDiagnosis[] | null;
}

/** Strict variant for the diagnosis side panel. */
export type TopDiagUI = {
  diagnosis_en: string;
  probability: number;
  diagnosis_pl: string | null; // never undefined
};

/** Symptom picker entry: EN model feature plus its PL label. */
export type SymptomDictItem = {
  en: string;
  pl: string;
  category?: string | null;
  /** Other PL labels for the same feature, e.g. "ból języka" for "ból jamy ustnej". */
  aliases?: string[];
};

/** Response to PATCHing the session's symptom list. */
export type SymptomsPatchResponse = {
  symptoms: string[];          // canonical EN feature names
  top_diagnoses: TopDiagnosis[];
  diagnosis_en?: string | null;
  diagnosis_pl?: string | null;
};
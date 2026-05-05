import axios, { AxiosHeaders, type InternalAxiosRequestConfig } from "axios";
import { emitUnauthorized } from "@/lib/authBus";
import { supabase } from "@/lib/supabaseClient";
import { ChatApiResponse, SymptomDictItem, SymptomsPatchResponse } from "./types";

export { supabase } from "@/lib/supabaseClient";

/** Podstawa URL API (FastAPI) */
export const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://localhost:8000";

/** axios – z cookies + Bearer (Supabase JWT) */
export const api = axios.create({
    baseURL: API_BASE,
    withCredentials: true,
});

/* ───────────────────────── helpers: token & retry ───────────────────────── */

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Szybkie pobranie tokenu (bez czekania) */
async function getTokenRaw(): Promise<string | null> {
    const { data } = await supabase.auth.getSession();
    return data.session?.access_token ?? null;
}

/** Po nawigacji Supabase może chwilę „nie mieć” sesji → krótki wait z retry */
async function getTokenWait(maxWaitMs = 2000, stepMs = 150): Promise<string | null> {
    const deadline = Date.now() + maxWaitMs;
    let tok = await getTokenRaw();
    while (!tok && Date.now() < deadline) {
        await sleep(stepMs);
        tok = await getTokenRaw();
    }
    return tok;
}

/** Interceptor requestów: Content-Type + Bearer */
api.interceptors.request.use(
    async (config: InternalAxiosRequestConfig) => {
        const headers = AxiosHeaders.from(config.headers || {});
        const isForm = typeof FormData !== "undefined" && config.data instanceof FormData;
        const method = (config.method || "GET").toUpperCase();

        if (!isForm && method !== "GET" && !headers.has("Content-Type")) {
            headers.set("Content-Type", "application/json");
        }

        const tok = await getTokenRaw(); // szybki — bez czekania
        if (tok) headers.set("Authorization", `Bearer ${tok}`);

        config.headers = headers;
        return config;
    },
    (error) => Promise.reject(error)
);

/** Deduplikacja exchange'ów, by nie wywoływać /auth/session równolegle */
let exchangePromise: Promise<boolean> | null = null;

/** Utrzymanie backendowego cookie sesji (export – do pre-warmu) */
export async function ensureBackendSession(): Promise<boolean> {
    if (exchangePromise) return exchangePromise;
    exchangePromise = (async () => {
        const access = await getTokenWait(2000);
        if (!access) return false;
        try {
            const r = await fetch(`${API_BASE}/auth/session`, {
                method: "POST",
                headers: { Authorization: `Bearer ${access}` },
                credentials: "include",
            });
            return r.ok;
        } catch {
            return false;
        }
    })();
    try {
        return await exchangePromise;
    } finally {
        exchangePromise = null;
    }
}

/** Interceptor odpowiedzi: 401 → /auth/session + retry 1× */
api.interceptors.response.use(
    (r) => r,
    async (err) => {
        const status = err?.response?.status;
        const cfg = err?.config as (InternalAxiosRequestConfig & { _retried?: boolean }) | undefined;

        if (status === 401 && cfg && !cfg._retried) {
            const ok = await ensureBackendSession();
            if (ok) {
                cfg._retried = true;
                return api(cfg);
            }
        }

        if (status === 401) emitUnauthorized();
        return Promise.reject(err);
    }
);

/** Zbuduj nagłówki dla fetch (Headers akceptuje każdy HeadersInit) */
async function buildHeaders(init?: RequestInit): Promise<Headers> {
    const h = new Headers(init?.headers);
    const method = (init?.method || "GET").toUpperCase();
    const isForm = init?.body instanceof FormData;

    if (!isForm && method !== "GET" && !h.has("Content-Type")) {
        h.set("Content-Type", "application/json");
    }

    const tok = await getTokenRaw();
    if (tok) h.set("Authorization", `Bearer ${tok}`);

    return h;
}

/** Ujednolicony wrapper na fetch (Bearer + COOKIES) z retry 1× po 401 */
export async function apiFetch<T = any>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = await buildHeaders(init);
    const doFetch = () =>
        fetch(`${API_BASE}${path}`, {
            ...init,
            headers,
            credentials: "include",
        });

    let res = await doFetch();

    if (res.status === 401) {
        const ok = await ensureBackendSession();
        if (ok) res = await doFetch();
    }

    const text = await res.text().catch(() => "");
    let data: any = null;
    try {
        data = text ? JSON.parse(text) : null;
    } catch {
        data = text || null;
    }

    if (!res.ok) {
        if (res.status === 401) emitUnauthorized();
        const detail =
            (data && (data.detail || data.message)) ||
            (typeof data === "string" ? data : "") ||
            res.statusText;
        throw new Error(`API ${res.status}: ${detail}`);
    }

    if (res.status === 204) return {} as T;
    return (data ?? {}) as T;
}

/* ─────────── PRE-WARM: publiczne API do „dogrzewania” sesji ─────────── */

/** Publiczny pre-warm (używany przez strażników/klient-bootstrap) */
export async function prewarmAuth(_force = false): Promise<boolean> {
    return ensureBackendSession();
}

/** Jednorazowa instalacja strażników focus/visibility/online */
export function installSessionGuards() {
    if (typeof window === "undefined") return;
    // uniknij wielokrotnej instalacji
    // @ts-ignore
    if ((window as any).__scGuardsInstalled) return;
    // @ts-ignore
    (window as any).__scGuardsInstalled = true;

    const warm = () => { void prewarmAuth(); };

    window.addEventListener("focus", warm);
    window.addEventListener("online", warm);
    document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") warm();
    });

    // szybkie dogrzanie po załadowaniu
    void prewarmAuth();
}

/* =======================
   ======== AUTH =========
   ======================= */

export async function exchangeSession(): Promise<any | null> {
    const access = await getTokenWait(2000);
    if (!access) return null;

    const r = await fetch(`${API_BASE}/auth/session`, {
        method: "POST",
        headers: { Authorization: `Bearer ${access}` },
        credentials: "include",
    });
    if (!r.ok) throw new Error(await r.text());
    return await r.json(); // UserOut
}

export async function login(email: string, password: string): Promise<void> {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw new Error(error.message);
    await exchangeSession();
}

export async function register(
    username: string,
    email: string,
    password: string,
    first_name?: string,
    last_name?: string,
    date_of_birth?: string, // "YYYY-MM-DD"
): Promise<void> {
    const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { username, first_name, last_name, date_of_birth } },
    });
    if (error) throw new Error(error.message);

    if (!data.session?.access_token) {
        await supabase.auth.signInWithPassword({ email, password }).catch(() => {});
    }
    await exchangeSession();
}

export async function logout(): Promise<void> {
    await supabase.auth.signOut();
    await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" }).catch(() => {});
}

export async function getMe<T = any>(): Promise<T | null> {
    try {
        const { data } = await api.get<T>("/auth/me");
        return (data as T) ?? null;
    } catch {
        return null;
    }
}

export async function me<T = any>(): Promise<T> {
    const { data } = await api.get<T>("/auth/me");
    return data;
}

/* =======================
   ========= CHAT ========
   ======================= */
export async function sendChat(user_input: string, session_id: string, signal?: AbortSignal): Promise<ChatApiResponse> {
    const raw = await apiFetch<any>("/chat", {
        method: "POST",
        body: JSON.stringify({ user_input, session_id }),
        signal,
    } as RequestInit);
    return {
        response: raw?.response ?? raw?.content ?? "",
        symptoms: Array.isArray(raw?.symptoms) ? raw.symptoms : [],
        top_diagnoses: Array.isArray(raw?.top_diagnoses) ? raw.top_diagnoses : null,
        diagnosis_en: raw?.diagnosis_en ?? null,
        diagnosis_pl: raw?.diagnosis_pl ?? null,
    };
}

/* =======================
   ======== DICTS ========
   ======================= */
export function listSymptomsDict(): Promise<SymptomDictItem[]> {
    return apiFetch<SymptomDictItem[]>("/dicts/symptoms");
}

/* =======================
   ===== SESSION OPS =====
   ======================= */
export function patchSessionSymptoms(
    sessionId: string,
    payload: { add?: string[]; remove?: string[]; replace?: string[]; lang?: "pl" | "en" },
    signal?: AbortSignal
): Promise<SymptomsPatchResponse> {
    return apiFetch<SymptomsPatchResponse>(`/sessions/${encodeURIComponent(sessionId)}/symptoms`, {
        method: "POST",
        body: JSON.stringify(payload ?? {}),
        signal,
    } as RequestInit);
}

/* =======================
   ======  TOP3 SPLIT ====
   ======================= */
export function getTop3Split(
    sessionId: string,
    k = 3,
    signal?: AbortSignal
): Promise<{
    xgb_enabled: boolean;
    river_enabled: boolean;
    symptoms: string[];
    xgb_top: { diagnosis_en: string; diagnosis_pl?: string; probability: number }[];
    river_top: { diagnosis_en: string; diagnosis_pl?: string; probability: number }[];
    ensemble_top: { diagnosis_en: string; diagnosis_pl?: string; probability: number }[];
    meta: { conf_x?: number; conf_r?: number; wx_dyn?: number; wr_dyn?: number; n_active?: number };
}> {
    return apiFetch(`/sessions/${encodeURIComponent(sessionId)}/top3-split?k=${k}`, { signal } as RequestInit);
}

/* =======================
   ========= HEALTH ======
   ======================= */
export async function healthcheck(): Promise<boolean> {
    try {
        const res = await fetch(`${API_BASE}/health`, { cache: "no-store", credentials: "include" });
        return res.ok;
    } catch {
        return false;
    }
}

export type HealthStatus = { xgb_ready?: boolean; matcher_ready?: boolean; llm_ready?: boolean };

export function getHealthDetailed(): Promise<HealthStatus> {
    return apiFetch<HealthStatus>("/health", { cache: "no-store" as any });
}

/* =======================
   ========= HISTORY =====
   ======================= */
export type HistorySymptom = {
    text: string;
    canonical?: string | null;
    lang?: string | null;
    score?: number | null;
};

export type HistoryDiagnosis = {
    label: string;
    score?: number | null;
    rank?: number | null;
};

export type HistoryItem = {
    id: number;
    title: string;
    /** może być null dla świeżych rozmów */
    last_message_at: string | null;
    message_count: number;
};

export type HistoryDetail = {
    id: number;
    title: string;
    messages: { role: "user" | "assistant"; content: string; created_at: string }[];
    symptoms: HistorySymptom[];
    diagnoses: HistoryDiagnosis[];
};

export function listHistory(signal?: AbortSignal): Promise<HistoryItem[]> {
    return apiFetch<HistoryItem[]>("/history", { signal } as RequestInit);
}

export function getConversation(convId: number, signal?: AbortSignal): Promise<HistoryDetail> {
    return apiFetch<HistoryDetail>(`/history/${convId}`, { signal } as RequestInit);
}

export async function deleteConversation(convId: number): Promise<void> {
    await apiFetch(`/history/${convId}`, { method: "DELETE" });
}
